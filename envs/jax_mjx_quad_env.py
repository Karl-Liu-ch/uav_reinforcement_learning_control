# Copyright 2025 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Trajectory-following quadrotor environment based on mjx_env.MjxEnv."""

from mujoco_playground._src import mjx_env
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import os
import sys

# Add parent directory to path for utils imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.drone_config import MAX_MOTOR_THRUST, MAX_TORQUE, YAW_TORQUE_COEFF, ARM_LENGTH


def _normalize_mjx_impl(impl):
	if impl is None:
		return None
	if not isinstance(impl, str):
		return impl

	impl_lower = impl.strip().lower()
	alias_map = {
		"gpu": "jax",
		"cuda": "jax",
		"tpu": "jax",
		"cpu": "c",
	}
	return alias_map.get(impl_lower, impl_lower)

class JaxMJXQuadEnv(mjx_env.MjxEnv):
	@property
	def action_size(self):
		# Agent outputs 4 values: [thrust, tau_x, tau_y, tau_z]
		return 4

	@property
	def mj_model(self):
		return self._mj_model

	@property
	def mjx_model(self):
		return self._mjx_model

	@property
	def xml_path(self):
		return self._xml_path

	def __init__(self, xml_path, config, config_overrides=None):
		super().__init__(config, config_overrides)
		self._mj_model = mujoco.MjModel.from_xml_path(xml_path)
		impl = _normalize_mjx_impl(getattr(config, "impl", None))
		self._mjx_model = mjx.put_model(self._mj_model, impl=impl)
		self._xml_path = xml_path
		self.max_episode_steps = config.max_episode_steps
		self.traj_duration_seconds = config.traj_duration_seconds
		self._step_count = 0
		self._traj_pos = None
		self._traj_vel = None
		self._traj_acc = None
		self.state = None
		self.rng = jax.random.PRNGKey(0)
		
		# Setup mixing matrix for thrust/torque to motor commands conversion
		self.max_motor_thrust = MAX_MOTOR_THRUST
		k = YAW_TORQUE_COEFF
		l = ARM_LENGTH
		self.max_total_thrust = 4 * self.max_motor_thrust
		self.max_torque = MAX_TORQUE
		
		# Mixing matrix: maps motor forces to [thrust, tau_x, tau_y, tau_z]
		# A @ [F1, F2, F3, F4] = [thrust, tau_x, tau_y, tau_z]
		A = jnp.array([
			[ 1,   1,   1,   1],
			[-l,  -l,  +l,  +l],
			[-l,  +l,  +l,  -l],
			[+k,  -k,  +k,  -k],
		])
		self.A_inv = jnp.linalg.inv(A)
		
		# Action bounds for [thrust, tau_x, tau_y, tau_z]
		self._action_low = jnp.array([0.0, -self.max_torque, -self.max_torque, -self.max_torque])
		self._action_high = jnp.array([self.max_total_thrust, self.max_torque, self.max_torque, self.max_torque])

	def _mix_to_motors(self, thrust, tau_x, tau_y, tau_z):
		"""Convert thrust and torques to individual motor commands.
		
		Args:
			thrust: Total thrust force (N)
			tau_x: Roll torque (N·m)
			tau_y: Pitch torque (N·m)
			tau_z: Yaw torque (N·m)
			
		Returns:
			Array of 4 motor forces clipped to [0, max_motor_thrust]
		"""
		u = jnp.array([thrust, tau_x, tau_y, tau_z])
		F = self.A_inv @ u
		return jnp.clip(F, 0.0, self.max_motor_thrust)
	
	def _denormalize_action(self, action_normalized):
		"""Denormalize action from [-1, 1] to physical bounds."""
		return (action_normalized + 1.0) / 2.0 * (self._action_high - self._action_low) + self._action_low

	def reset(self, seed=None, options=None):
		if seed is not None:
			self.rng = jax.random.PRNGKey(seed)
		self._step_count = 0
		self.state = mjx.make_data(self._mjx_model)
		self._traj_pos, self._traj_vel, self._traj_acc = self._sample_trajectory()
		obs = self._get_obs()
		info = {}
		return obs, info

	def step(self, action):
		"""Step the environment with agent action.
		
		Args:
			action: Normalized action from agent in [-1, 1] for [thrust, tau_x, tau_y, tau_z]
			
		Returns:
			obs, reward, terminated, truncated, info
		"""
		action = jnp.asarray(action)
		
		# Denormalize action from [-1, 1] to physical units
		physical_action = self._denormalize_action(action)
		thrust, tau_x, tau_y, tau_z = physical_action
		
		# Convert thrust and torques to motor commands through mixing matrix
		motor_commands = self._mix_to_motors(thrust, tau_x, tau_y, tau_z)
		
		# Apply motor commands to MuJoCo
		self.state = self.state.replace(ctrl=motor_commands)
		self.state = mjx.step(self._mjx_model, self.state)
		self._step_count += 1
		
		obs = self._get_obs()
		reward = self._get_reward()
		terminated = self._is_terminated()
		truncated = self._step_count >= self.max_episode_steps
		
		# Store motor commands in info for logging/debugging
		info = {
			'motor_commands': motor_commands,
			'thrust': thrust,
			'torques': jnp.array([tau_x, tau_y, tau_z]),
		}
		return obs, reward, terminated, truncated, info

	def _get_obs(self):
		return jnp.concatenate([self.state.qpos, self.state.qvel], axis=-1)

	def _get_reward(self):
		idx = min(self._step_count, self._traj_pos.shape[0] - 1)
		pos = self.state.qpos[:3]
		target = self._traj_pos[idx]
		pos_error = jnp.linalg.norm(pos - target)
		return jnp.exp(-pos_error ** 2)

	def _is_terminated(self):
		# 可根据状态范围等自定义终止条件
		return False

	def _sample_trajectory(self):
		T = self.max_episode_steps
		t = jnp.linspace(0, self.traj_duration_seconds, T)
		center = jnp.array([0.0, 0.0, 1.0])
		amp = jnp.array([0.5, 0.5, 0.2])
		freq = jnp.array([0.2, 0.15, 0.1])
		traj = center + amp * jnp.sin(2 * jnp.pi * freq * t[:, None])
		vel = jnp.gradient(traj, axis=0)
		acc = jnp.gradient(vel, axis=0)
		return traj, vel, acc
