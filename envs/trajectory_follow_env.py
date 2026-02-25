import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import mujoco
import os
from scipy.interpolate import CubicSpline

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import QuadState, normalize, denormalize
from utils.drone_config import MAX_MOTOR_THRUST, MAX_TORQUE, YAW_TORQUE_COEFF, ARM_LENGTH


class TrajectoryFollowEnv(gym.Env):
    """Environment where the quadrotor must follow a smooth random trajectory.

    The trajectory is generated per-episode as a sum of sinusoids for each axis
    producing position, velocity and acceleration arrays for each timestep.
    The env returns `info` keys: "target" (pos), "target_vel", "target_acc".
    """

    def __init__(self, render_mode: str | None = None, max_motor_thrust: float = MAX_MOTOR_THRUST,
                 yaw_torque_coeff: float = YAW_TORQUE_COEFF, arm_length: float = ARM_LENGTH,
                 max_episode_steps: int = 2048, initial_state_bounds: Box | None = None,
                 traj_center_bounds: Box | None = None, traj_duration_seconds: float | None = 30.0,
                 nominal_voltage: float = 16.8, min_voltage: float = 13.2,
                 voltage_drop_base_per_sec: float = 0.01,
                 voltage_drop_load_per_sec: float = 0.08):
        super().__init__()
        self.max_motor_thrust = max_motor_thrust
        k = yaw_torque_coeff
        l = arm_length
        self.max_episode_steps = max_episode_steps
        # If set, use this total trajectory duration (seconds) when generating
        # the time base for the spline. If None, use model timestep * steps.
        self.traj_duration_seconds = traj_duration_seconds
        self.render_mode = render_mode
        self._viewer = None

        self.action_space = Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)

        # reuse hover env bounds for safety
        self._obs_bounds = Box(
            low=np.array([-4, -4, -2, -np.pi, -np.pi, -np.pi, -10, -10, -10, -6*np.pi, -6*np.pi, -6*np.pi], dtype=np.float32),
            high=np.array([4, 4, 2, np.pi, np.pi, np.pi, 10, 10, 10, 6*np.pi, 6*np.pi, 6*np.pi], dtype=np.float32),
        )

        self._initial_state_bounds = initial_state_bounds or Box(
            low=np.array([-1.5, -1.5, 0.1, -0.3, -0.3, -0.3, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5], dtype=np.float32),
            high=np.array([1.5, 1.5, 1.5, 0.3, 0.3, 0.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32),
        )

        # bounds where the trajectory center (mean position) will be sampled
        self._traj_center_bounds = traj_center_bounds or Box(
            low=np.array([-1.0, -1.0, 0.4], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.4], dtype=np.float32),
        )

        self._state_bounds = Box(
            low=np.array([-3, -3, 0.0, -np.pi, -np.pi, -np.pi, -10, -10, -10, -6*np.pi, -6*np.pi, -6*np.pi], dtype=np.float32),
            high=np.array([3, 3, 3, np.pi, np.pi, np.pi, 10, 10, 10, 6*np.pi, 6*np.pi, 6*np.pi], dtype=np.float32),
        )

        self.max_total_thrust = 4 * self.max_motor_thrust
        self.max_torque = MAX_TORQUE
        self._action_bounds = Box(
            low=np.array([0.0, -self.max_torque, -self.max_torque, -self.max_torque], dtype=np.float32),
            high=np.array([self.max_total_thrust, self.max_torque, self.max_torque, self.max_torque], dtype=np.float32),
        )

        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "model",
            "drone",
            "drone.xml"
        )
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.target_state = QuadState()
        self._state = QuadState(self._obs_bounds)
        self.dt = self.model.opt.timestep
        self.frame_skip = 1
        self._step_count = 0
        self._prev_action = np.zeros(4, dtype=np.float32)
        self.nominal_voltage = float(nominal_voltage)
        self.min_voltage = float(min_voltage)
        self.voltage_drop_base_per_sec = float(voltage_drop_base_per_sec)
        self.voltage_drop_load_per_sec = float(voltage_drop_load_per_sec)
        self.voltage = self.nominal_voltage

        A = np.array([
            [ 1,   1,   1,   1],
            [-l,  -l,  +l,  +l],
            [-l,  +l,  +l,  -l],
            [+k,  -k,  +k,  -k],
        ])
        self.A_inv = np.linalg.inv(A)

        # trajectory containers (filled in reset)
        self._traj_pos = None
        self._traj_vel = None
        self._traj_acc = None

    def _voltage_scale(self) -> float:
        scale = self.voltage / self.nominal_voltage
        return float(np.clip(scale, 0.0, 1.0))

    def _update_voltage(self, motor_commands: np.ndarray):
        load_ratio = float(np.mean(motor_commands) / max(self.max_motor_thrust, 1e-6))
        dV = (self.voltage_drop_base_per_sec + self.voltage_drop_load_per_sec * load_ratio) * self.dt
        self.voltage = float(np.clip(self.voltage - dV, self.min_voltage, self.nominal_voltage))

    def _mix_to_motors(self, thrust: float, tau_x: float, tau_y: float, tau_z: float):
        u = np.array([thrust, tau_x, tau_y, tau_z])
        F = self.A_inv @ u
        return np.clip(F, 0.0, self.max_motor_thrust)

    def _get_obs(self) -> np.ndarray:
        self._state.set_from_mujoco(self.data.qpos[:7], self.data.qvel[:6])
        obs = self._state.vec()
        # relative position to current trajectory setpoint
        obs[0:3] = self.target_state.position - self._state.position
        return normalize(obs, self._obs_bounds).astype(np.float32)

    def _get_reward(self) -> float:
        # reward based on Euclidean position error to instantaneous trajectory point
        pos_error = float(np.linalg.norm(self._state.position - self.target_state.position))
        return np.exp(-pos_error**2)

    def set_state(self, qpos, qvel):
        self.data.qpos[:7] = qpos
        self.data.qvel[:6] = qvel
        mujoco.mj_forward(self.model, self.data)

    def _is_terminated(self) -> bool:
        state_vec = self._state.vec()
        if not np.isfinite(state_vec).all():
            return True
        if not self._state_bounds.contains(state_vec):
            return True
        return False

    def step(self, action):
        self._prev_action = np.array(action, dtype=np.float32)
        physical_action = denormalize(action, self._action_bounds)
        thrust, tau_x, tau_y, tau_z = physical_action
        motor_commands = self._mix_to_motors(thrust, tau_x, tau_y, tau_z)
        voltage_scale = self._voltage_scale()
        motor_commands = np.clip(motor_commands * voltage_scale, 0.0, self.max_motor_thrust * voltage_scale)
        self._update_voltage(motor_commands)
        self.data.ctrl[:] = motor_commands
        mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        obs = self._get_obs()
        reward = self._get_reward()
        terminated = self._is_terminated()
        truncated = self._step_count >= self.max_episode_steps

        # current trajectory index
        idx = min(self._step_count - 1, self._traj_pos.shape[0] - 1)
        info = {
            "state": self._state.vec().copy(),
            "motor_commands": motor_commands.copy(),
            "target": self._traj_pos[idx].copy(),
            "target_vel": self._traj_vel[idx].copy(),
            "target_acc": self._traj_acc[idx].copy(),
            "voltage": float(self.voltage),
            "voltage_scale": float(voltage_scale),
        }

        return obs, reward, terminated, truncated, info

    def _sample_sinusoid_trajectory(self, center: np.ndarray, T_steps: int, start_pos: np.ndarray | None = None):
        """Generate smooth trajectory by interpolating random waypoints with a cubic spline.

        If `start_pos` is provided, the spline's first waypoint will equal the UAV start
        position so the trajectory begins at the drone's initial pose.

        Returns (pos, vel, acc) arrays shaped (T_steps, 3).
        """
        # Time base for spline interpolation. If user provided a desired total
        # duration for the trajectory, space the samples over that duration;
        # otherwise use the simulator timestep.
        if self.traj_duration_seconds is not None:
            t = np.linspace(0.0, float(self.traj_duration_seconds), T_steps)
        else:
            t = np.arange(T_steps) * self.dt
        T = t[-1] if T_steps > 0 else 0.0

        # number of waypoints (including start and end)
        n_wp = int(self.np_random.integers(3, 6))
        # sample waypoint times in [0, T]
        wp_times = np.linspace(0.0, T, n_wp)

        pos = np.zeros((T_steps, 3), dtype=np.float32)
        vel = np.zeros((T_steps, 3), dtype=np.float32)
        acc = np.zeros((T_steps, 3), dtype=np.float32)

        for axis in range(3):
            amp_scale = 0.6 if axis < 2 else 0.4
            # generate random offsets for waypoints
            wp_offsets = self.np_random.uniform(-amp_scale, amp_scale, size=(n_wp,))
            wp_positions = center[axis] + wp_offsets

            # if start_pos specified, enforce it at t=0
            if start_pos is not None:
                wp_positions[0] = float(start_pos[axis])

            cs = CubicSpline(wp_times, wp_positions, bc_type='natural')

            pos[:, axis] = cs(t)
            vel[:, axis] = cs.derivative(1)(t)
            acc[:, axis] = cs.derivative(2)(t)

        return pos, vel, acc

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._step_count = 0
        self._prev_action = np.zeros(4, dtype=np.float32)
        self.voltage = self.nominal_voltage
        mujoco.mj_resetData(self.model, self.data)

        # Randomize initial UAV state
        initial_state = QuadState()
        initial_state.random_reset(self.np_random, self._initial_state_bounds)
        qpos, qvel = initial_state.get_mujoco_state()
        self.set_state(qpos, qvel)

        # Sample a trajectory center and generate spline trajectory arrays
        center = self.np_random.uniform(self._traj_center_bounds.low, self._traj_center_bounds.high)
        N = self.max_episode_steps
        pos, vel, acc = self._sample_sinusoid_trajectory(center, N, start_pos=initial_state.position)
        self._traj_pos = pos
        self._traj_vel = vel
        self._traj_acc = acc

        # Set target to first trajectory point
        self.target_state = QuadState()
        self.target_state.state[0:3] = self._traj_pos[0]

        obs = self._get_obs()
        info = {"state": self._state.vec().copy(),
                "target": self._traj_pos[0].copy(),
                "target_vel": self._traj_vel[0].copy(),
                "target_acc": self._traj_acc[0].copy(),
            "voltage": float(self.voltage),
            "voltage_scale": float(self._voltage_scale()),
        }
        return obs, info

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
        elif self.render_mode == "rgb_array":
            renderer = mujoco.Renderer(self.model, height=480, width=640)
            renderer.update_scene(self.data)
            return renderer.render()
        return None

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
