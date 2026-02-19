"""Cascaded PID hover controller for the MuJoCo quadrotor.

Architecture:
    Position PID → desired acceleration (world frame)
    Acceleration → desired attitude + thrust (gravity feedforward + tilt compensation)
    Attitude P → desired body rates
    Rate PID → torques (inertia-scaled, I term in torque space for COM offset compensation)
    Normalize to [-1,1] for env.step()

Usage:
    python pid_controller.py                    # 5 episodes, rendered
    python pid_controller.py --no-render        # headless
    python pid_controller.py --plot             # save performance plots
    python pid_controller.py --episodes 20      # more episodes
"""

import argparse
import json
import os
import time
from collections import deque

import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

from envs import HoverEnv
from envs.trajectory_follow_env import TrajectoryFollowEnv
from utils.drone_config import (
    MASS, G, DT, MAX_TOTAL_THRUST, MAX_TORQUE,
    ARM_LENGTH as L, IXX, IYY, IZZ,
)

# Ensure top-level plots directory exists
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)
# also create pid and summary subfolders used by plotting
PID_PLOTS_DIR = os.path.join(PLOTS_DIR, "pid")
os.makedirs(PID_PLOTS_DIR, exist_ok=True)
os.makedirs(os.path.join(PID_PLOTS_DIR, "summary"), exist_ok=True)


def euler_to_rot_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return rotation matrix (body -> world) from Z-Y-X Euler angles.

    The matrix returned maps body-frame vectors to world-frame vectors:
        v_world = R @ v_body
    To transform world -> body use R.T.
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    R = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ], dtype=np.float64)

    return R


def world_to_body(vec: np.ndarray, roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Transform a 3D vector from world frame to body frame using Euler angles."""
    R = euler_to_rot_matrix(roll, pitch, yaw)
    return R.T @ vec


def body_to_world(vec: np.ndarray, roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Transform a 3D vector from body frame to world frame using Euler angles."""
    R = euler_to_rot_matrix(roll, pitch, yaw)
    return R @ vec


def angle_diff(target: float, source: float) -> float:
    """Shortest signed angular difference target - source in range [-pi, pi]."""
    return (target - source + np.pi) % (2.0 * np.pi) - np.pi

# ── Load default PID gains from pid_gains.json ──
_GAINS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pid_gains.json")
with open(_GAINS_PATH) as _f:
    DEFAULT_GAINS = json.load(_f)


class CascadedPIDController:
    """Cascaded PID controller: position → attitude → torques.

    Designed for the HoverEnv action space [thrust_norm, tau_x_norm, tau_y_norm, tau_z_norm]
    where each value is in [-1, 1].
    """

    def __init__(self, gains: dict | None = None):
        g = gains or DEFAULT_GAINS
        # Position XY gains
        self.kp_xy = g["position_xy"]["kp"]
        self.kd_xy = g["position_xy"]["kd"]
        self.ki_xy = g["position_xy"]["ki"]
        # Position Z gains
        self.kp_z = g["position_z"]["kp"]
        self.kd_z = g["position_z"]["kd"]
        self.ki_z = g["position_z"]["ki"]
        # Attitude gains
        self.kp_att = g["attitude"]["kp"]
        self.kd_att = g["attitude"]["kd"]
        # Yaw gains
        self.kp_yaw = g["yaw"]["kp"]
        self.kd_yaw = g["yaw"]["kd"]
        # Rate controller gains
        self.ki_rate_torque = g["rate"]["ki_torque"]
        self.rate_int_max = g["rate"]["integral_max"]
        # Limits
        lim = g["limits"]
        self.axy_max = lim["axy_max"]
        self.az_min = lim["az_min"]
        self.az_max = lim["az_max"]
        self.tilt_max = lim["tilt_max"]
        self.z_int_max = lim["z_integral_max"]
        self.xy_int_max = lim["xy_integral_max"]
        self.torque_motor_frac = lim["torque_motor_fraction"]
        self.torque_abs_max = lim["torque_abs_max"]
        self.yaw_torque_scale = lim["yaw_torque_scale"]
        # Integral states
        self.z_integral = 0.0
        self.xy_integral = np.zeros(2)
        self.rate_int_torque = np.zeros(3)  # accumulates in N·m (torque space)

    def reset(self):
        """Reset integral states (call on env.reset())."""
        self.z_integral = 0.0
        self.xy_integral = np.zeros(2)
        self.rate_int_torque = np.zeros(3)

    def compute(self, state: np.ndarray, target) -> tuple[np.ndarray, dict]:
        """Compute normalized action from physical state and target/trajectory.

        Args:
            state: 12D state [x,y,z, roll,pitch,yaw, vx,vy,vz, wx,wy,wz]
            target: can be one of:
                - 3D array-like position [x,y,z]
                - sequence (pos, vel, acc) where each is 3D
                - dict with keys 'pos', optional 'vel', optional 'acc'

        Returns:
            action: 4D normalized action [thrust, tau_x, tau_y, tau_z] in [-1,1]
            diag: Diagnostics dict with desired/actual rates and attitude setpoints
        """
        pos = state[0:3]
        roll, pitch, yaw = state[3:6]
        vel = state[6:9]
        wx, wy, wz = state[9:12]

        # --- parse target/trajectory input ---
        # default desired velocity/acceleration are zero if not provided
        if isinstance(target, dict):
            tgt_pos = np.asarray(target.get("pos", target.get("position", np.zeros(3))))
            tgt_vel = np.asarray(target.get("vel", target.get("velocity", np.zeros(3))))
            tgt_acc = np.asarray(target.get("acc", target.get("acceleration", np.zeros(3))))
        elif isinstance(target, (list, tuple)) and len(target) == 3:
            tgt_pos = np.asarray(target[0])
            tgt_vel = np.asarray(target[1])
            tgt_acc = np.asarray(target[2])
        else:
            targ_arr = np.asarray(target)
            if targ_arr.size == 3:
                tgt_pos = targ_arr
                tgt_vel = np.zeros(3)
                tgt_acc = np.zeros(3)
            elif targ_arr.size >= 9:
                tgt_pos = targ_arr[0:3]
                tgt_vel = targ_arr[3:6]
                tgt_acc = targ_arr[6:9]
            else:
                # fallback
                tgt_pos = np.asarray(target)
                tgt_vel = np.zeros(3)
                tgt_acc = np.zeros(3)

        # ── 1. Position PID → desired acceleration (world frame) ──
        pos_err = tgt_pos - pos

        # XY integral (compensates steady-state position bias from COM offset)
        self.xy_integral = np.clip(
            self.xy_integral + self.ki_xy * DT * pos_err[:2],
            -self.xy_int_max, self.xy_int_max,
        )
        # Use target velocity in D term: derivative of error = target_vel - current_vel
        ax = self.kp_xy * pos_err[0] + self.kd_xy * (tgt_vel[0] - vel[0]) + self.xy_integral[0]
        ay = self.kp_xy * pos_err[1] + self.kd_xy * (tgt_vel[1] - vel[1]) + self.xy_integral[1]
        az = self.kp_z * pos_err[2] + self.kd_z * (tgt_vel[2] - vel[2])

        # Z integral (altitude hold)
        self.z_integral = np.clip(
            self.z_integral + self.ki_z * DT * pos_err[2],
            -self.z_int_max, self.z_int_max,
        )
        az += self.z_integral

        # Add feedforward acceleration from trajectory
        ax += tgt_acc[0]
        ay += tgt_acc[1]
        az += tgt_acc[2]

        # Clamp accelerations
        ax = np.clip(ax, -self.axy_max, self.axy_max)
        ay = np.clip(ay, -self.axy_max, self.axy_max)
        az = np.clip(az, self.az_min, self.az_max)

        # ── 2. Acceleration → thrust + desired attitude ──
        cos_r, cos_p = np.cos(roll), np.cos(pitch)
        tilt = max(cos_r * cos_p, 0.5)  # prevent division blow-up
        thrust = MASS * (G + az) / tilt
        thrust = np.clip(thrust, 0.0, MAX_TOTAL_THRUST)

        # Motor-clipping-aware torque limit:
        # Each motor produces thrust/4. Max torque = motor_force × 2L.
        # Apply safety fraction to stay away from motor saturation.
        thrust_per_motor = thrust / 4.0
        max_tau = min(thrust_per_motor * 2.0 * L * self.torque_motor_frac,
                      self.torque_abs_max)

        # Rotate desired acceleration into body frame (full Euler rotation)
        acc_world = np.array([ax, ay, az], dtype=np.float64)
        acc_body = world_to_body(acc_world, roll, pitch, yaw)
        ax_b, ay_b, _ = acc_body

        # Desired roll/pitch from body-frame acceleration
        # NOTE: roll sign is NEGATED — positive roll → -Y force in XYZ Euler convention
        des_pitch = np.clip(np.arctan2(ax_b, G + az), -self.tilt_max, self.tilt_max)
        des_roll = np.clip(np.arctan2(-ay_b, G + az), -self.tilt_max, self.tilt_max)

        # ── 3. Attitude P → desired rates, Rate PID → torques ──
        # Outer attitude P: desired_rate = (Kp/Kd) * attitude_error
        des_wx = (self.kp_att / self.kd_att) * (des_roll - roll)
        des_wy = (self.kp_att / self.kd_att) * (des_pitch - pitch)

        # Yaw: point toward the target position in the world frame
        # desired yaw = atan2(target_y - y, target_x - x)
        des_yaw = np.arctan2(tgt_pos[1] - pos[1], tgt_pos[0] - pos[0])
        yaw_err = angle_diff(des_yaw, yaw)
        des_wz = (self.kp_yaw / self.kd_yaw) * yaw_err

        # Inner rate PID: P term (inertia-scaled) + I term (in torque space)
        rate_err = np.array([des_wx - wx, des_wy - wy, des_wz - wz])
        inertia = np.array([IXX, IYY, IZZ])
        kd = np.array([self.kd_att, self.kd_att, self.kd_yaw])
        tau_p = inertia * kd * rate_err

        # I term accumulates in torque space (N·m) to compensate COM offset bias
        self.rate_int_torque = np.clip(
            self.rate_int_torque + self.ki_rate_torque * DT * rate_err,
            -self.rate_int_max, self.rate_int_max,
        )
        tau = tau_p + self.rate_int_torque

        # Clamp torques (yaw gets less authority)
        tau[0] = np.clip(tau[0], -max_tau, max_tau)
        tau[1] = np.clip(tau[1], -max_tau, max_tau)
        tau[2] = np.clip(tau[2], -max_tau * self.yaw_torque_scale,
                         max_tau * self.yaw_torque_scale)

        # ── 4. Normalize to [-1, 1] ──
        thrust_norm = 2.0 * thrust / MAX_TOTAL_THRUST - 1.0
        action = np.array([
            thrust_norm,
            tau[0] / MAX_TORQUE,
            tau[1] / MAX_TORQUE,
            tau[2] / MAX_TORQUE,
        ], dtype=np.float32)

        diag = {
            "des_rate": np.array([des_wx, des_wy, des_wz]),
            "actual_rate": np.array([wx, wy, wz]),
            "des_att": np.array([des_roll, des_pitch, des_yaw]),
        }

        return np.clip(action, -1.0, 1.0), diag


def plot_episode(data: dict, episode_num: int, save_dir: str = "plots"):
    """Generate performance plots for a single PID evaluation episode."""
    os.makedirs(save_dir, exist_ok=True)

    t = np.array(data["times"])
    pos = np.array(data["positions"])
    tgt = np.array(data["targets"])
    att = np.rad2deg(np.array(data["attitudes"]))
    vel = np.array(data["velocities"])
    ang_vel = np.rad2deg(np.array(data["angular_velocities"]))
    motors = np.array(data["motor_commands"])
    actions = np.array(data["actions"])
    rewards = np.array(data["rewards"])
    pos_err = np.linalg.norm(pos - tgt, axis=1)
    des_rates = np.rad2deg(np.array(data["des_rates"]))
    actual_rates = np.rad2deg(np.array(data["actual_rates"]))
    rate_err = des_rates - actual_rates
    des_att = np.rad2deg(np.array(data["des_attitudes"]))

    fig, axes = plt.subplots(5, 2, figsize=(14, 16))
    fig.suptitle(f"PID Controller — Episode {episode_num}", fontsize=14)

    # Position tracking
    ax = axes[0, 0]
    for i, (label, color) in enumerate(zip(["x", "y", "z"], ["r", "g", "b"])):
        ax.plot(t, pos[:, i], color=color, label=label)
        ax.plot(t, tgt[:, i], color=color, linestyle="--", alpha=0.5)
    ax.set_ylabel("Position (m)")
    ax.set_title("Position (solid=UAV, dashed=target)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Position error
    ax = axes[0, 1]
    ax.plot(t, pos_err, color="k")
    ax.set_ylabel("Error (m)")
    ax.set_title("Position Error (Euclidean)")
    ax.grid(True, alpha=0.3)

    # Attitude: actual vs desired
    ax = axes[1, 0]
    for i, (label, color) in enumerate(zip(["roll", "pitch", "yaw"], ["r", "g", "b"])):
        ax.plot(t, att[:, i], color=color, label=label)
        ax.plot(t, des_att[:, i], color=color, linestyle="--", alpha=0.5)
    ax.set_ylabel("Angle (deg)")
    ax.set_title("Attitude (solid=actual, dashed=desired)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Rate tracking: commanded vs actual (per axis)
    ax = axes[1, 1]
    for i, (label, color) in enumerate(zip(["roll", "pitch", "yaw"], ["r", "g", "b"])):
        ax.plot(t, actual_rates[:, i], color=color, label=f"{label} actual")
        ax.plot(t, des_rates[:, i], color=color, linestyle="--", alpha=0.5,
                label=f"{label} cmd")
    ax.set_ylabel("Rate (deg/s)")
    ax.set_title("Rate Tracking (solid=actual, dashed=commanded)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Rate error per axis
    ax = axes[2, 0]
    for i, (label, color) in enumerate(zip(["roll", "pitch", "yaw"], ["r", "g", "b"])):
        ax.plot(t, rate_err[:, i], color=color, label=label, alpha=0.8)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_ylabel("Rate error (deg/s)")
    ax.set_title("Rate Controller Error (cmd - actual)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Rate error magnitude (Euclidean norm)
    ax = axes[2, 1]
    rate_err_norm = np.linalg.norm(rate_err, axis=1)
    ax.plot(t, rate_err_norm, color="k")
    ax.set_ylabel("|Rate error| (deg/s)")
    ax.set_title("Rate Error Magnitude")
    ax.grid(True, alpha=0.3)

    # Motor commands
    ax = axes[3, 0]
    for i in range(4):
        ax.plot(t, motors[:, i], label=f"M{i+1}")
    ax.set_ylabel("Thrust (N)")
    ax.set_title("Motor Commands")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Torque actions
    ax = axes[3, 1]
    for i, label in enumerate(["roll torque", "pitch torque", "yaw torque"]):
        ax.plot(t, actions[:, i + 1], label=label)
    ax.set_ylabel("Action (normalized)")
    ax.set_title("Actions: Torques")
    ax.set_ylim(-1.1, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Rewards
    ax = axes[4, 0]
    ax.plot(t, rewards, color="k")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Reward")
    ax.set_title("Step Reward")
    ax.grid(True, alpha=0.3)

    # Thrust action
    ax = axes[4, 1]
    ax.plot(t, actions[:, 0], label="thrust", color="k")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Action (normalized)")
    ax.set_title("Action: Thrust")
    ax.set_ylim(-1.1, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    filepath = os.path.join(save_dir, f"pid_episode_{episode_num}.png")
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  Plot saved to {filepath}")

    # --- Additional figure: per-axis position errors and attitude errors ---
    # Position component errors (signed) and attitude errors (deg, shortest wrap)
    pos_comp_err = pos - tgt  # signed error: UAV - target
    # Wrap attitude error to [-180, 180]
    att_err = (des_att - att + 180.0) % 360.0 - 180.0

    fig2, ax2 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    # Position component errors
    ax = ax2[0]
    for i, (label, color) in enumerate(zip(["x", "y", "z"], ["r", "g", "b"])):
        ax.plot(t, pos_comp_err[:, i], color=color, label=f"{label} error")
    ax.axhline(0.0, color="k", linewidth=0.5)
    ax.set_ylabel("Position error (m)")
    ax.set_title("Position Component Errors (UAV - target)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Attitude errors (roll, pitch, yaw)
    ax = ax2[1]
    for i, (label, color) in enumerate(zip(["roll", "pitch", "yaw"], ["r", "g", "b"])):
        ax.plot(t, att_err[:, i], color=color, label=f"{label} err")
    ax.axhline(0.0, color="k", linewidth=0.5)
    ax.set_ylabel("Angle error (deg)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Attitude Errors (desired - actual, wrapped)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig2.tight_layout()
    filepath2 = os.path.join(save_dir, f"pid_episode_{episode_num}_errors.png")
    fig2.savefig(filepath2, dpi=150)
    plt.close(fig2)
    print(f"  Error plot saved to {filepath2}")


def _update_visuals(viewer, drone_pos, target_pos, trail, pos_err):
    """Update custom visualization elements in the MuJoCo viewer.

    Draws:
        - Green translucent sphere at target position
        - Error line from drone to target (green=close, red=far)
        - Blue trajectory trail showing recent flight path
        - Ground shadow marker below the drone
    """
    scn = viewer.user_scn
    scn.ngeom = 0  # reset custom geoms each frame

    # 1. Target sphere (green, translucent)
    if scn.ngeom < scn.maxgeom:
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([0.05, 0, 0]),
            pos=np.asarray(target_pos, dtype=np.float64),
            mat=np.eye(3).flatten(),
            rgba=np.array([0.2, 1.0, 0.2, 0.5]),
        )
        scn.ngeom += 1

    # 2. Error line (drone → target), color-coded by distance
    if scn.ngeom < scn.maxgeom:
        mujoco.mjv_connector(
            scn.geoms[scn.ngeom],
            mujoco.mjtGeom.mjGEOM_LINE,
            0.003,
            np.asarray(drone_pos, dtype=np.float64),
            np.asarray(target_pos, dtype=np.float64),
        )
        err_ratio = min(pos_err / 1.0, 1.0)
        scn.geoms[scn.ngeom].rgba[:] = [err_ratio, 1.0 - err_ratio, 0.0, 0.8]
        scn.ngeom += 1

    # 3. Trajectory trail (blue dots)
    for pt in trail:
        if scn.ngeom >= scn.maxgeom:
            break
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([0.008, 0, 0]),
            pos=np.asarray(pt, dtype=np.float64),
            mat=np.eye(3).flatten(),
            rgba=np.array([0.2, 0.5, 1.0, 0.4]),
        )
        scn.ngeom += 1

    # 4. Ground shadow (small disc below drone)
    if scn.ngeom < scn.maxgeom:
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=np.array([0.04, 0.04, 0.001]),
            pos=np.array([drone_pos[0], drone_pos[1], 0.001]),
            mat=np.eye(3).flatten(),
            rgba=np.array([0.3, 0.3, 0.3, 0.4]),
        )
        scn.ngeom += 1


def evaluate(num_episodes: int = 5, render: bool = True, plot: bool = False,
             gains_file: str | None = None, use_trajectory_env: bool = False):
    """Evaluate the PID controller on HoverEnv.

    Args:
        num_episodes: Number of episodes to run.
        render: Whether to render the MuJoCo viewer.
        plot: Whether to save performance plots.
        gains_file: Optional path to a JSON file with custom PID gains.
    """
    # Load gains
    gains = DEFAULT_GAINS
    if gains_file and os.path.exists(gains_file):
        with open(gains_file) as f:
            gains = json.load(f)
        print(f"Loaded gains from {gains_file}")

    controller = CascadedPIDController(gains)
    env = TrajectoryFollowEnv() if use_trajectory_env else HoverEnv()

    episode_rewards = []
    episode_lengths = []
    episode_errors = []
    all_ep_data = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        controller.reset()
        total_reward = 0.0
        step_count = 0
        done = False
        errors = []
        trail = deque(maxlen=200)

        ep_data = {
            "times": [], "positions": [], "targets": [],
            "attitudes": [], "velocities": [], "angular_velocities": [],
            "motor_commands": [], "actions": [], "rewards": [],
            "des_rates": [], "actual_rates": [], "des_attitudes": [],
        }

        viewer = None
        if render:
            viewer = mujoco.viewer.launch_passive(env.model, env.data)
            viewer.cam.lookat[:] = [0.0, 0.0, 0.8]
            viewer.cam.distance = 7.0
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -25

        state = info["state"]
        # Build trajectory tuple (pos, vel, acc) from env info (fallback to zeros)
        tgt_pos = info.get("target")
        if tgt_pos is None:
            tgt_pos = info.get("target_pos", np.zeros(3))
        tgt_vel = info.get("target_vel", np.zeros(3))
        tgt_acc = info.get("target_acc", np.zeros(3))

        print(f"\n--- Episode {ep + 1}/{num_episodes} ---")
        print(f"  Start:  [{state[0]:.2f}, {state[1]:.2f}, {state[2]:.2f}]")
        print(f"  Target: [{tgt_pos[0]:.2f}, {tgt_pos[1]:.2f}, {tgt_pos[2]:.2f}]")

        while not done:
            state = info["state"]
            # pull trajectory info for this step
            tgt_pos = info.get("target", info.get("target_pos", np.zeros(3)))
            tgt_vel = info.get("target_vel", np.zeros(3))
            tgt_acc = info.get("target_acc", np.zeros(3))
            traj = (tgt_pos, tgt_vel, tgt_acc)
            action, diag = controller.compute(state, traj)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step_count += 1

            drone_pos = info["state"][:3]
            pos_err = float(np.linalg.norm(drone_pos - np.asarray(tgt_pos)))
            errors.append(pos_err)

            # Record trail every 5 steps to avoid too many geoms
            if step_count % 5 == 0:
                trail.append(drone_pos.copy())

            if plot:
                s = info["state"]
                ep_data["times"].append(step_count * DT)
                ep_data["positions"].append(s[:3].copy())
                ep_data["targets"].append(np.asarray(tgt_pos).copy())
                ep_data["attitudes"].append(s[3:6].copy())
                ep_data["velocities"].append(s[6:9].copy())
                ep_data["angular_velocities"].append(s[9:12].copy())
                ep_data["motor_commands"].append(info["motor_commands"].copy())
                ep_data["actions"].append(action.copy())
                ep_data["rewards"].append(reward)
                ep_data["des_rates"].append(diag["des_rate"].copy())
                ep_data["actual_rates"].append(diag["actual_rate"].copy())
                ep_data["des_attitudes"].append(diag["des_att"].copy())

            if render and viewer is not None and viewer.is_running():
                _update_visuals(viewer, drone_pos, np.asarray(tgt_pos), trail, pos_err)
                viewer.sync()
                time.sleep(DT)
            elif render and viewer is not None and not viewer.is_running():
                break

            if step_count % 100 == 0:
                s = info["state"]
                print(f"  Step {step_count}: pos=[{s[0]:.2f}, {s[1]:.2f}, {s[2]:.2f}], "
                      f"err={pos_err:.3f}m, reward={reward:.2f}")

        if viewer is not None:
            viewer.close()

        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)
        episode_errors.append(np.mean(errors))

        status = "TERMINATED" if terminated else "TRUNCATED (max steps)"
        print(f"  {status} after {step_count} steps")
        print(f"  Total reward: {total_reward:.2f}, mean error: {np.mean(errors):.3f}m")

        if plot and len(ep_data["times"]) > 0:
            plot_episode(ep_data, ep + 1, save_dir="./plots/pid")
            all_ep_data.append(ep_data)

    env.close()

    print("\n=== PID Evaluation Summary ===")
    print(f"Episodes:    {num_episodes}")
    print(f"Survival:    {sum(1 for l in episode_lengths if l >= 512)}/{num_episodes} "
          f"({100*sum(1 for l in episode_lengths if l >= 512)/num_episodes:.0f}%)")
    print(f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"Mean length: {np.mean(episode_lengths):.1f} +/- {np.std(episode_lengths):.1f}")
    print(f"Mean error:  {np.mean(episode_errors):.3f}m")

    # Summary plots across episodes (if requested)
    if plot and len(all_ep_data) > 0:
        summary_dir = os.path.join("./plots/pid", "summary")
        os.makedirs(summary_dir, exist_ok=True)

        # Mean position error per episode
        mean_pos_errors = []
        mean_abs_att_errors = []  # per-episode per-axis
        for epd in all_ep_data:
            pos = np.array(epd["positions"])
            tgt = np.array(epd["targets"])
            des_att = np.rad2deg(np.array(epd["des_attitudes"]))
            att = np.rad2deg(np.array(epd["attitudes"]))

            pos_err = np.linalg.norm(pos - tgt, axis=1)
            mean_pos_errors.append(np.mean(pos_err))

            att_err = (des_att - att + 180.0) % 360.0 - 180.0
            mean_abs_att_errors.append(np.mean(np.abs(att_err), axis=0))

        mean_pos_errors = np.array(mean_pos_errors)
        mean_abs_att_errors = np.array(mean_abs_att_errors)  # shape (n_eps, 3)

        # Plot mean position errors
        fig, ax = plt.subplots(figsize=(8, 4))
        eps = np.arange(1, len(mean_pos_errors) + 1)
        ax.bar(eps, mean_pos_errors, color="C0")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Mean position error (m)")
        ax.set_title("Mean Position Error per Episode")
        ax.grid(True, alpha=0.3)
        fpath = os.path.join(summary_dir, "mean_position_error_per_episode.png")
        fig.tight_layout()
        fig.savefig(fpath, dpi=150)
        plt.close(fig)
        print(f"  Summary saved to {fpath}")

        # Plot mean absolute attitude errors per axis (grouped bar)
        fig, ax = plt.subplots(figsize=(10, 4))
        width = 0.2
        x = np.arange(len(mean_abs_att_errors))
        labels = [f"Ep {i}" for i in eps]
        for i, (lbl, col) in enumerate(zip(["roll", "pitch", "yaw"], ["r", "g", "b"])):
            ax.bar(x + (i - 1) * width, mean_abs_att_errors[:, i], width=width, color=col, label=lbl)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Mean |angle error| (deg)")
        ax.set_title("Mean Absolute Attitude Error per Episode")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fpath2 = os.path.join(summary_dir, "mean_attitude_error_per_episode.png")
        fig.tight_layout()
        fig.savefig(fpath2, dpi=150)
        plt.close(fig)
        print(f"  Summary saved to {fpath2}")


def main():
    parser = argparse.ArgumentParser(description="PID hover controller evaluation")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of evaluation episodes")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable MuJoCo viewer")
    parser.add_argument("--plot", action="store_true",
                        help="Save performance plots to ./plots/pid/")
    parser.add_argument("--gains", type=str, default=None,
                        help="Path to custom PID gains JSON file")
    parser.add_argument("--traj", action="store_true",
                        help="Use trajectory-following environment (TrajectoryFollowEnv)")
    args = parser.parse_args()

    evaluate(
        num_episodes=args.episodes,
        render=not args.no_render,
        plot=args.plot,
        gains_file=args.gains,
        use_trajectory_env=args.traj,
    )


if __name__ == "__main__":
    main()
