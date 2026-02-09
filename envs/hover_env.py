import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import mujoco
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import QuadState, normalize, denormalize


class HoverEnv(gym.Env):
    """Quadrotor hovering environment."""
    def __init__(self, render_mode: str | None = None, max_motor_thrust: float = 13.0, yaw_torque_coeff: float = 0.0201,
                 arm_length: float = 0.039799, max_episode_steps: int = 512):
        super().__init__()
        # Params
        self.max_motor_thrust = max_motor_thrust
        k = yaw_torque_coeff
        l = arm_length
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self._viewer = None
        

        # Define action and observation spaces here
        self.action_space = Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)

        # Physical observation bounds (before normalization)
        self._obs_bounds = Box(
            low=np.array([-2, -2, 0.0, -np.pi, -np.pi, -np.pi, -10, -10, -10, -6*np.pi, -6*np.pi, -6*np.pi], dtype=np.float32),
            high=np.array([2, 2, 2, np.pi, np.pi, np.pi, 10, 10, 10, 6*np.pi, 6*np.pi, 6*np.pi], dtype=np.float32),
        )

        # Physical action bounds (before normalization) - thrust in N, torques in N⋅m
        self.max_total_thrust = 4 * self.max_motor_thrust  # 52 N
        self.max_torque = 0.5  # N⋅m
        self._action_bounds = Box(
            low=np.array([0.0, -self.max_torque, -self.max_torque, -self.max_torque], dtype=np.float32),
            high=np.array([self.max_total_thrust, self.max_torque, self.max_torque, self.max_torque], dtype=np.float32),
        )

        # Load MuJoCo model
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "model",
            "drone",
            "drone.xml"
        )
        # Load MuJoCo model and initialize simulation data
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Target state (hover at z=1m)
        self.target_state = QuadState()
        self.target_state.state[2] = 1.0  # Set z position to 1 meter

        # State tracker
        self._state = QuadState()
        self.dt = self.model.opt.timestep  # 0.01s
        self.frame_skip = 1  # Control at 100Hz
        self._step_count = 0

        # ── Mixer matrix ──
        A = np.array([
            [ 1,   1,   1,   1],      # thrust
            [-l,  -l,  +l,  +l],      # roll  torque (tau_x)
            [-l,  +l,  +l,  -l],      # pitch torque (tau_y)
            [+k,  -k,  +k,  -k],      # yaw   torque (tau_z) — from XML gear[5]
        ])
        self.A_inv = np.linalg.inv(A)

    def _mix_to_motors(self, thrust: float, tau_x: float, tau_y: float, tau_z: float):
        """
        Convert thrust and torques to motor commands using the mixer matrix.

        Args:
            thrust: Total thrust in N
            tau_x, tau_y, tau_z: Torques in N⋅m

        Returns:
            motor_commands: [F1, F2, F3, F4] in N
        """
        u = np.array([thrust, tau_x, tau_y, tau_z])
        F = self.A_inv @ u
        return np.clip(F, 0.0, self.max_motor_thrust)

    def _get_obs(self) -> np.ndarray:
        """Get current observation (normalized state)."""
        # Only read base body state (ignore propeller joints)
        self._state.set_from_mujoco(self.data.qpos[:7], self.data.qvel[:6])
        obs = self._state.vec()
        return normalize(obs, self._obs_bounds).astype(np.float32)

    def _get_reward(self) -> float:
        """Reward based on position error only, always positive so surviving = more reward."""
        pos_error = float(np.linalg.norm(self._state.position - self.target_state.position))
        return np.exp(-pos_error ** 2)
    
    def _is_terminated(self) -> bool:
        """Check if episode should terminate (state out of bounds or NaN)."""
        state_vec = self._state.vec()
        if not np.isfinite(state_vec).all():
            return True
        if not self._obs_bounds.contains(state_vec):
            return True
        return False
    
    def step(self, action):
        """
        Apply action to the simulation.

        Args:
            action: [thrust_norm, tau_x_norm, tau_y_norm, tau_z_norm] in [-1, 1]
        """
        # Denormalize: [-1, 1] → physical units [thrust in N, torques in N⋅m]
        physical_action = denormalize(action, self._action_bounds)
        thrust, tau_x, tau_y, tau_z = physical_action

        # Convert to motor commands via mix matrix
        motor_commands = self._mix_to_motors(thrust, tau_x, tau_y, tau_z)
        self.data.ctrl[:] = motor_commands

        # Step simulation
        mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        # Get observation and reward
        obs = self._get_obs()
        reward = self._get_reward()
        terminated = self._is_terminated()
        truncated = self._step_count >= self.max_episode_steps

        info = {
            "state": self._state.vec().copy(),
            "motor_commands": motor_commands.copy(),
        }

        return obs, reward, terminated, truncated, info

    def reset(self, seed: int | None = None, options: dict | None = None):
        """Reset environment to initial hover state.

        Args:
            seed: Random seed
            options: Additional reset options

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        self._step_count = 0

        # Reset MuJoCo state
        mujoco.mj_resetData(self.model, self.data)

        # Set initial state (start at ground level)
        initial_state = QuadState()
        qpos, qvel = initial_state.get_mujoco_state()
        # Only set base body state (first 7 qpos, first 6 qvel)
        # Propeller joints remain at default (0)
        self.data.qpos[:7] = qpos
        self.data.qvel[:6] = qvel
        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = {"state": self._state.vec().copy()}

        return obs, info

    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
        elif self.render_mode == "rgb_array":
            # Return RGB image from camera
            renderer = mujoco.Renderer(self.model, height=480, width=640)
            renderer.update_scene(self.data)
            return renderer.render()
        return None

    def close(self):
        """Close viewer if open."""
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None