"""SE(3) Geometric Controller for the MuJoCo quadrotor.

This controller uses differential geometric control on SE(3) (Special Euclidean group)
to avoid Euler angle singularities and provide global stability guarantees.

Architecture:
    Position/Velocity tracking → Desired acceleration (world frame)
    Acceleration + attitude → Desired rotation matrix (SO(3)) + thrust
    Attitude error on SO(3) → Desired body rates using geometric error
    Rate error → Torques via inertia-scaled feedback on so(3)
    Normalize to [-1,1] for env.step()

References:
    Lee, T., et al. (2010). "Geometric Tracking Control of a Quadrotor UAV on SE(3)"
    Mellinger, D. & Kumar, V. (2011). "Minimum snap trajectory generation and control"

Usage:
    python se3_geometric_controller.py                    # 5 episodes, rendered
    python se3_geometric_controller.py --no-render        # headless
    python se3_geometric_controller.py --plot             # save performance plots
    python se3_geometric_controller.py --episodes 20      # more episodes
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
SE3_PLOTS_DIR = os.path.join(PLOTS_DIR, "se3")
os.makedirs(SE3_PLOTS_DIR, exist_ok=True)
os.makedirs(os.path.join(SE3_PLOTS_DIR, "summary"), exist_ok=True)


def euler_to_rot_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return rotation matrix (body -> world) from Z-Y-X Euler angles.
    
    The matrix maps body-frame vectors to world-frame vectors:
        v_world = R @ v_body
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


def rot_matrix_to_euler(R: np.ndarray) -> tuple:
    """Extract Z-Y-X Euler angles from rotation matrix.
    
    Returns:
        (roll, pitch, yaw) in radians
    """
    pitch = np.arcsin(-R[2, 0])
    if np.abs(np.cos(pitch)) > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock: pitch near ±π/2
        roll = 0.0
        yaw = np.arctan2(-R[0, 1], R[1, 1])
    return roll, pitch, yaw


def vee(A: np.ndarray) -> np.ndarray:
    """Extract vector from skew-symmetric matrix: vee(skew(v)) = v.
    
    For a 3×3 skew-symmetric matrix:
        [  0  -v3   v2 ]
        [ v3    0  -v1 ]
        [-v2   v1    0 ]
    Returns [v1, v2, v3].
    """
    return np.array([A[2, 1], A[0, 2], A[1, 0]], dtype=np.float64)


def skew(v: np.ndarray) -> np.ndarray:
    """Create skew-symmetric matrix from vector: skew(v) @ u = v × u."""
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ], dtype=np.float64)


def log_so3(R: np.ndarray) -> np.ndarray:
    """Compute logarithm map on SO(3): returns the axis-angle representation as a vector.
    
    For R = exp(skew(ω)), returns ω such that ||ω|| = rotation angle,
    and ω/||ω|| = rotation axis.
    """
    # Clamp to avoid numerical issues with arccos
    trace = np.trace(R)
    angle = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    
    if angle < 1e-6:
        # Small rotation: use first-order approximation
        return vee(R)
    elif angle > np.pi - 1e-6:
        # Rotation near π: need special handling
        # Find the diagonal element closest to 1
        diag = np.diag(R)
        k = np.argmax(diag)
        col = R[:, k]
        col[k] = (1.0 + col[k]) / 2.0
        col = col / (np.linalg.norm(col) + 1e-10)
        col[k] = np.sqrt(col[k])
        omega_axis = col
        return angle * omega_axis
    else:
        # General case
        omega_axis = vee(R) / (2.0 * np.sin(angle))
        return angle * omega_axis


def exp_so3(omega: np.ndarray) -> np.ndarray:
    """Compute exponential map on SO(3): exp(skew(ω)) ≈ I + sin(θ)/θ * skew(ω) + (1-cos(θ))/θ² * skew(ω)²."""
    theta = np.linalg.norm(omega)
    if theta < 1e-6:
        # Small angle: use first-order approximation
        return np.eye(3) + skew(omega)
    else:
        axis = omega / theta
        s = np.sin(theta)
        c = np.cos(theta)
        return c * np.eye(3) + (1.0 - c) * np.outer(axis, axis) + s * skew(axis)


def attitude_error_so3(R_desired: np.ndarray, R_actual: np.ndarray) -> np.ndarray:
    """Compute geometric attitude error on SO(3) as a vector in R³.
    
    Error is defined as -1/2 * vee(R_d^T @ R_a - R_a^T @ R_d).
    This gives a vector pointing towards the desired rotation.
    """
    # Relative error: R_e = R_d^T @ R_a
    R_e = R_desired.T @ R_actual
    # Error vector in so(3): -1/2 * (R_e - R_e^T)
    error_matrix = R_e - R_e.T
    e_R = -0.5 * vee(error_matrix)
    return e_R


def angle_diff(target: float, source: float) -> float:
    """Shortest signed angular difference target - source in range [-pi, pi]."""
    return (target - source + np.pi) % (2.0 * np.pi) - np.pi


# Load default gains from pid_gains.json
_GAINS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pid_gains.json")
with open(_GAINS_PATH) as _f:
    DEFAULT_GAINS = json.load(_f)


class SE3GeometricController:
    """SE(3) Geometric Controller for quadrotor position and attitude tracking.
    
    Uses differential geometric control on SE(3) = R³ × SO(3) to track
    desired positions and velocities with guaranteed global stability.
    
    Key features:
    - Position tracking via desired acceleration (feedforward + feedback)
    - Attitude tracking via geometric error on SO(3) (no Euler angles)
    - Cascaded control structure: position → attitude → rates → torques
    - All gains are physically meaningful (SI units)
    """

    def __init__(self, gains: dict | None = None):
        """Initialize SE(3) controller with gains.
        
        Args:
            gains: Dictionary with gain parameters. If None, uses DEFAULT_GAINS.
                   Expected keys: position_xy, position_z, attitude, yaw, rate, limits
        """
        g = gains or DEFAULT_GAINS
        
        # Position XY gains
        self.kp_xy = g["position_xy"]["kp"]
        self.kd_xy = g["position_xy"]["kd"]
        self.ki_xy = g["position_xy"]["ki"]
        
        # Position Z gains
        self.kp_z = g["position_z"]["kp"]
        self.kd_z = g["position_z"]["kd"]
        self.ki_z = g["position_z"]["ki"]
        
        # Attitude gains (on SO(3))
        self.kp_att = g["attitude"]["kp"]      # Attitude proportional gain
        self.kd_att = g["attitude"]["kd"]      # Attitude derivative gain
        
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
        self.rate_int_torque = np.zeros(3)  # in torque space (N·m)
        
        print("SE(3) Geometric Controller initialized")
        print(f"  Position XY: Kp={self.kp_xy:.2f}, Kd={self.kd_xy:.2f}, Ki={self.ki_xy:.4f}")
        print(f"  Position Z:  Kp={self.kp_z:.2f}, Kd={self.kd_z:.2f}, Ki={self.ki_z:.4f}")
        print(f"  Attitude (SO3): Kp={self.kp_att:.2f}, Kd={self.kd_att:.2f}")

    def reset(self):
        """Reset integral states (call on env.reset())."""
        self.z_integral = 0.0
        self.xy_integral = np.zeros(2)
        self.rate_int_torque = np.zeros(3)

    def compute(self, state: np.ndarray, target) -> tuple[np.ndarray, dict]:
        """Compute normalized action using SE(3) geometric control.
        
        Args:
            state: 12D state [x,y,z, roll,pitch,yaw, vx,vy,vz, wx,wy,wz]
            target: position target or trajectory dict with pos/vel/acc
        
        Returns:
            action: 4D normalized action [thrust, tau_x, tau_y, tau_z] in [-1,1]
            diag: Diagnostics dict
        """
        # Parse state
        pos = state[0:3]
        roll, pitch, yaw = state[3:6]
        vel = state[6:9]
        wx, wy, wz = state[9:12]
        
        # Current rotation matrix (body to world)
        R_current = euler_to_rot_matrix(roll, pitch, yaw)
        
        # --- Parse target/trajectory input ---
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
            else:
                tgt_pos = np.asarray(target)
                tgt_vel = np.zeros(3)
                tgt_acc = np.zeros(3)
        
        # ═══ 1. POSITION CONTROL (world frame) ═══
        # Tracking error
        pos_err = tgt_pos - pos
        vel_err = tgt_vel - vel
        
        # Position integral (compensates COM bias)
        self.xy_integral = np.clip(
            self.xy_integral + self.ki_xy * DT * pos_err[:2],
            -self.xy_int_max, self.xy_int_max,
        )
        
        # Position-based acceleration command (PID + feedforward)
        ax = self.kp_xy * pos_err[0] + self.kd_xy * vel_err[0] + self.xy_integral[0]
        ay = self.kp_xy * pos_err[1] + self.kd_xy * vel_err[1] + self.xy_integral[1]
        
        # Z-axis control
        self.z_integral = np.clip(
            self.z_integral + self.ki_z * DT * pos_err[2],
            -self.z_int_max, self.z_int_max,
        )
        az = self.kp_z * pos_err[2] + self.kd_z * vel_err[2] + self.z_integral
        
        # Add feedforward acceleration
        ax += tgt_acc[0]
        ay += tgt_acc[1]
        az += tgt_acc[2]
        
        # Clamp accelerations
        ax = np.clip(ax, -self.axy_max, self.axy_max)
        ay = np.clip(ay, -self.axy_max, self.axy_max)
        az = np.clip(az, self.az_min, self.az_max)
        
        # ═══ 2. ATTITUDE CONTROL (SE(3) → SO(3) → torques) ═══
        # Compute desired rotation matrix from acceleration command
        # Reference frame: z_d points away from desired acceleration, aligned with thrust
        
        # Desired thrust direction (normalized acceleration with gravity)
        a_des = np.array([ax, ay, az], dtype=np.float64)
        thrust_vec = MASS * (a_des + np.array([0.0, 0.0, G]))  # world-frame thrust
        thrust_mag = np.linalg.norm(thrust_vec)
        thrust_mag = np.clip(thrust_mag, 0.1, MAX_TOTAL_THRUST)
        
        # Normalized thrust axis (body z-axis in world frame)
        z_axis = thrust_vec / (thrust_mag + 1e-10)
        
        # Desired yaw from trajectory tangent direction
        tgt_vel_xy = tgt_vel[:2]
        tgt_vel_norm = np.linalg.norm(tgt_vel_xy)
        if tgt_vel_norm > 1e-6:
            des_yaw = np.arctan2(tgt_vel_xy[1], tgt_vel_xy[0])
        else:
            des_yaw = yaw  # maintain current yaw
        
        # Construct desired rotation matrix using z-axis and yaw
        # z_d = normalized thrust direction in world frame
        # x_d = rotated by yaw angle in horizontal plane
        cy, sy = np.cos(des_yaw), np.sin(des_yaw)
        x_axis_world = np.array([cy, sy, 0.0])  # desired x-axis in world frame
        
        # Ensure z-axis is not parallel to x-axis body direction
        # y_d = z_d × x_d (right-hand rule)
        y_axis = np.cross(z_axis, x_axis_world)
        y_norm = np.linalg.norm(y_axis)
        
        if y_norm < 1e-3:
            # z_axis is parallel to x_axis: fallback
            y_axis = np.array([0.0, 0.0, 1.0])
        else:
            y_axis = y_axis / y_norm
        
        # x_d = y_d × z_d (orthogonalize)
        x_axis = np.cross(y_axis, z_axis)
        x_norm = np.linalg.norm(x_axis)
        if x_norm < 1e-3:
            x_axis = np.array([1.0, 0.0, 0.0])
        else:
            x_axis = x_axis / x_norm
        
        # Desired rotation matrix: columns are x_d, y_d, z_d
        R_desired = np.column_stack([x_axis, y_axis, z_axis])
        
        # Ensure orthogonality (use QR decomposition)
        R_desired, _ = np.linalg.qr(R_desired)
        if np.linalg.det(R_desired) < 0:
            R_desired *= -1
        
        # ═══ 3. ATTITUDE ERROR ON SO(3) ═══
        # Geometric attitude error: e_R = -1/2 * vee(R_d^T @ R_a - R_a^T @ R_d)
        e_R = attitude_error_so3(R_desired, R_current)
        
        # Extract current and desired angular velocities
        omega_current = np.array([wx, wy, wz])
        
        # Desired angular velocity: from proportional + derivative attitude control
        # ω_d = (Kp_att/Kd_att) * e_R (proportional to error)
        omega_desired = (self.kp_att / self.kd_att) * e_R
        
        # Angular velocity error
        omega_err = omega_desired - omega_current
        
        # ═══ 4. ANGULAR RATE CONTROL ═══
        # Inner rate control: τ = I * Kd_att * ω_err + ∫(Ki * ω_err)
        inertia = np.array([IXX, IYY, IZZ])
        tau_p = inertia * self.kd_att * omega_err
        
        # Integral term (in torque space)
        self.rate_int_torque = np.clip(
            self.rate_int_torque + self.ki_rate_torque * DT * omega_err,
            -self.rate_int_max, self.rate_int_max,
        )
        tau = tau_p + self.rate_int_torque
        
        # Motor-aware torque limit
        thrust_per_motor = thrust_mag / 4.0
        max_tau = min(thrust_per_motor * 2.0 * L * self.torque_motor_frac,
                      self.torque_abs_max)
        
        tau[0] = np.clip(tau[0], -max_tau, max_tau)
        tau[1] = np.clip(tau[1], -max_tau, max_tau)
        tau[2] = np.clip(tau[2], -max_tau * self.yaw_torque_scale,
                         max_tau * self.yaw_torque_scale)
        
        # ═══ 5. NORMALIZE TO [-1, 1] ═══
        thrust_norm = 2.0 * thrust_mag / MAX_TOTAL_THRUST - 1.0
        action = np.array([
            thrust_norm,
            tau[0] / MAX_TORQUE,
            tau[1] / MAX_TORQUE,
            tau[2] / MAX_TORQUE,
        ], dtype=np.float32)
        
        # Diagnostics
        diag = {
            "des_rate": omega_desired.copy(),
            "actual_rate": omega_current.copy(),
            "des_att": rot_matrix_to_euler(R_desired),
            "attitude_error": np.linalg.norm(e_R),
        }
        
        return np.clip(action, -1.0, 1.0), diag


def plot_episode(data: dict, episode_num: int, save_dir: str = "plots"):
    """Generate performance plots for a single SE(3) evaluation episode."""
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
    att_errors = np.array(data["attitude_errors"])

    fig, axes = plt.subplots(5, 2, figsize=(14, 16))
    fig.suptitle(f"SE(3) Geometric Controller — Episode {episode_num}", fontsize=14)

    # Position tracking
    ax = axes[0, 0]
    for i, (label, color) in enumerate(zip(["x", "y", "z"], ["r", "g", "b"])):
        ax.plot(t, pos[:, i], color=color, label=label, linewidth=2)
        ax.plot(t, tgt[:, i], color=color, linestyle="--", alpha=0.5, linewidth=1.5)
    ax.set_ylabel("Position (m)")
    ax.set_title("Position (solid=UAV, dashed=target)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Position error
    ax = axes[0, 1]
    ax.plot(t, pos_err, color="k", linewidth=2)
    ax.set_ylabel("Error (m)")
    ax.set_title("Position Error (Euclidean distance)")
    ax.grid(True, alpha=0.3)

    # Attitude: actual vs desired
    ax = axes[1, 0]
    for i, (label, color) in enumerate(zip(["roll", "pitch", "yaw"], ["r", "g", "b"])):
        ax.plot(t, att[:, i], color=color, label=label, linewidth=2)
        ax.plot(t, des_att[:, i], color=color, linestyle="--", alpha=0.5, linewidth=1.5)
    ax.set_ylabel("Angle (deg)")
    ax.set_title("Attitude (solid=actual, dashed=desired from SE(3))")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # SO(3) attitude error norm
    ax = axes[1, 1]
    ax.plot(t, att_errors, color="purple", linewidth=2)
    ax.set_ylabel("||e_R|| (geometric error)")
    ax.set_title("SO(3) Attitude Error Norm")
    ax.grid(True, alpha=0.3)

    # Rate tracking
    ax = axes[2, 0]
    for i, (label, color) in enumerate(zip(["roll", "pitch", "yaw"], ["r", "g", "b"])):
        ax.plot(t, actual_rates[:, i], color=color, label=f"{label} actual", linewidth=2)
        ax.plot(t, des_rates[:, i], color=color, linestyle="--", alpha=0.5, linewidth=1.5,
                label=f"{label} cmd")
    ax.set_ylabel("Rate (deg/s)")
    ax.set_title("Rate Tracking")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Rate error per axis
    ax = axes[2, 1]
    for i, (label, color) in enumerate(zip(["roll", "pitch", "yaw"], ["r", "g", "b"])):
        ax.plot(t, rate_err[:, i], color=color, label=label, alpha=0.8, linewidth=1.5)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_ylabel("Rate error (deg/s)")
    ax.set_title("Rate Controller Error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Motor commands
    ax = axes[3, 0]
    for i in range(4):
        ax.plot(t, motors[:, i], label=f"M{i+1}", linewidth=1.5)
    ax.set_ylabel("Thrust (N)")
    ax.set_title("Motor Commands")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Torque actions
    ax = axes[3, 1]
    for i, label in enumerate(["τ_x", "τ_y", "τ_z"]):
        ax.plot(t, actions[:, i + 1], label=label, linewidth=1.5)
    ax.set_ylabel("Action (normalized)")
    ax.set_title("Torque Actions")
    ax.set_ylim(-1.1, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Rewards
    ax = axes[4, 0]
    ax.plot(t, rewards, color="k", linewidth=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Reward")
    ax.set_title("Step Reward")
    ax.grid(True, alpha=0.3)

    # Thrust action
    ax = axes[4, 1]
    ax.plot(t, actions[:, 0], label="thrust", color="k", linewidth=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Action (normalized)")
    ax.set_title("Thrust Action")
    ax.set_ylim(-1.1, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    filepath = os.path.join(save_dir, f"se3_episode_{episode_num}.png")
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  Plot saved to {filepath}")

    # Position and attitude component errors
    pos_comp_err = pos - tgt
    att_err = (des_att - att + 180.0) % 360.0 - 180.0

    fig2, ax2 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax = ax2[0]
    for i, (label, color) in enumerate(zip(["x", "y", "z"], ["r", "g", "b"])):
        ax.plot(t, pos_comp_err[:, i], color=color, label=f"{label} error", linewidth=1.5)
    ax.axhline(0.0, color="k", linewidth=0.5)
    ax.set_ylabel("Position error (m)")
    ax.set_title("Position Component Errors")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = ax2[1]
    for i, (label, color) in enumerate(zip(["roll", "pitch", "yaw"], ["r", "g", "b"])):
        ax.plot(t, att_err[:, i], color=color, label=f"{label} err", linewidth=1.5)
    ax.axhline(0.0, color="k", linewidth=0.5)
    ax.set_ylabel("Angle error (deg)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Attitude Errors (Euler representation)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig2.tight_layout()
    filepath2 = os.path.join(save_dir, f"se3_episode_{episode_num}_errors.png")
    fig2.savefig(filepath2, dpi=150)
    plt.close(fig2)
    print(f"  Error plot saved to {filepath2}")


def _update_visuals(viewer, drone_pos, target_pos, trail, pos_err):
    """Update custom visualization elements in MuJoCo viewer."""
    scn = viewer.user_scn
    scn.ngeom = 0

    # Target sphere (green, translucent)
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

    # Error line (color-coded)
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

    # Trajectory trail (blue dots)
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

    # Ground shadow
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
             gains_file: str | None = None, use_trajectory_env: bool = False,
             nominal_voltage: float = 16.8, min_voltage: float = 13.2,
             voltage_drop_base_per_sec: float = 0.01,
             voltage_drop_load_per_sec: float = 0.08):
    """Evaluate the SE(3) controller."""
    # Load gains
    gains = DEFAULT_GAINS
    if gains_file and os.path.exists(gains_file):
        with open(gains_file) as f:
            gains = json.load(f)
        print(f"Loaded gains from {gains_file}")

    controller = SE3GeometricController(gains)
    env_kwargs = dict(
        nominal_voltage=nominal_voltage,
        min_voltage=min_voltage,
        voltage_drop_base_per_sec=voltage_drop_base_per_sec,
        voltage_drop_load_per_sec=voltage_drop_load_per_sec,
    )
    env = TrajectoryFollowEnv(**env_kwargs) if use_trajectory_env else HoverEnv(**env_kwargs)

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
            "attitude_errors": [],
        }

        viewer = None
        if render:
            viewer = mujoco.viewer.launch_passive(env.model, env.data)
            viewer.cam.lookat[:] = [0.0, 0.0, 0.8]
            viewer.cam.distance = 7.0
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -25

        state = info["state"]
        tgt_pos = info.get("target", info.get("target_pos", np.zeros(3)))
        tgt_vel = info.get("target_vel", np.zeros(3))
        tgt_acc = info.get("target_acc", np.zeros(3))

        print(f"\n--- Episode {ep + 1}/{num_episodes} ---")
        print(f"  Start:  [{state[0]:.2f}, {state[1]:.2f}, {state[2]:.2f}]")
        print(f"  Target: [{tgt_pos[0]:.2f}, {tgt_pos[1]:.2f}, {tgt_pos[2]:.2f}]")
        if "voltage" in info:
            print(f"  Voltage: {info['voltage']:.2f}V")

        while not done:
            state = info["state"]
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
                ep_data["des_attitudes"].append(np.array(diag["des_att"]))  # des_att is tuple, convert to array
                ep_data["attitude_errors"].append(diag["attitude_error"])

            if render and viewer is not None and viewer.is_running():
                _update_visuals(viewer, drone_pos, np.asarray(tgt_pos), trail, pos_err)
                viewer.sync()
                time.sleep(DT)
            elif render and viewer is not None and not viewer.is_running():
                break

            if step_count % 100 == 0:
                s = info["state"]
                print(f"  Step {step_count}: pos=[{s[0]:.2f}, {s[1]:.2f}, {s[2]:.2f}], "
                      f"err={pos_err:.3f}m, att_err={diag['attitude_error']:.4f}")

        if viewer is not None:
            viewer.close()

        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)
        episode_errors.append(np.mean(errors))

        status = "TERMINATED" if terminated else "TRUNCATED (max steps)"
        print(f"  {status} after {step_count} steps")
        print(f"  Total reward: {total_reward:.2f}, mean position error: {np.mean(errors):.3f}m")

        if plot and len(ep_data["times"]) > 0:
            plot_episode(ep_data, ep + 1, save_dir="./plots/se3")
            all_ep_data.append(ep_data)

    env.close()

    print("\n=== SE(3) Controller Evaluation Summary ===")
    print(f"Episodes:    {num_episodes}")
    print(f"Survival:    {sum(1 for l in episode_lengths if l >= 512)}/{num_episodes} "
          f"({100*sum(1 for l in episode_lengths if l >= 512)/num_episodes:.0f}%)")
    print(f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"Mean length: {np.mean(episode_lengths):.1f} +/- {np.std(episode_lengths):.1f}")
    print(f"Mean error:  {np.mean(episode_errors):.3f}m")

    # Summary plots
    if plot and len(all_ep_data) > 0:
        summary_dir = os.path.join("./plots/se3", "summary")
        os.makedirs(summary_dir, exist_ok=True)

        mean_pos_errors = []
        mean_att_errors = []
        for epd in all_ep_data:
            pos = np.array(epd["positions"])
            tgt = np.array(epd["targets"])
            pos_err = np.linalg.norm(pos - tgt, axis=1)
            mean_pos_errors.append(np.mean(pos_err))
            mean_att_errors.append(np.mean(epd["attitude_errors"]))

        mean_pos_errors = np.array(mean_pos_errors)
        mean_att_errors = np.array(mean_att_errors)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        eps = np.arange(1, len(mean_pos_errors) + 1)
        
        ax1.bar(eps, mean_pos_errors, color="C0")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Mean position error (m)")
        ax1.set_title("Position Tracking Error")
        ax1.grid(True, alpha=0.3)
        
        ax2.bar(eps, mean_att_errors, color="C1")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Mean ||e_R|| (SO(3) error)")
        ax2.set_title("Attitude Control Error")
        ax2.grid(True, alpha=0.3)
        
        fig.tight_layout()
        fpath = os.path.join(summary_dir, "summary_errors.png")
        fig.savefig(fpath, dpi=150)
        plt.close(fig)
        print(f"  Summary saved to {fpath}")


def main():
    parser = argparse.ArgumentParser(description="SE(3) Geometric controller evaluation")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of evaluation episodes")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable MuJoCo viewer")
    parser.add_argument("--plot", action="store_true",
                        help="Save performance plots to ./plots/se3/")
    parser.add_argument("--gains", type=str, default=None,
                        help="Path to custom PID gains JSON file")
    parser.add_argument("--traj", action="store_true",
                        help="Use trajectory-following environment")
    parser.add_argument("--nominal-voltage", type=float, default=8.4,
                        help="Battery nominal voltage in volts")
    parser.add_argument("--min-voltage", type=float, default=7.6,
                        help="Battery minimum voltage clamp in volts")
    parser.add_argument("--voltage-drop-base", type=float, default=0.01,
                        help="Base voltage drop rate in V/s")
    parser.add_argument("--voltage-drop-load", type=float, default=0.2,
                        help="Load-dependent voltage drop gain in V/s")
    args = parser.parse_args()

    evaluate(
        num_episodes=args.episodes,
        render=not args.no_render,
        plot=args.plot,
        gains_file=args.gains,
        use_trajectory_env=args.traj,
        nominal_voltage=args.nominal_voltage,
        min_voltage=args.min_voltage,
        voltage_drop_base_per_sec=args.voltage_drop_base,
        voltage_drop_load_per_sec=args.voltage_drop_load,
    )


if __name__ == "__main__":
    main()
