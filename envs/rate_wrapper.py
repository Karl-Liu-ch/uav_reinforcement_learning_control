"""Rate control (CTBR) action wrapper for HoverEnv.

Converts policy outputs from [thrust, roll_rate, pitch_rate, yaw_rate]
to [thrust, tau_x, tau_y, tau_z] using a PID rate controller.
"""

import json
import os

import gymnasium as gym
import numpy as np

from utils.drone_config import IXX, IYY, IZZ, MAX_TORQUE

# Load rate wrapper defaults from pid_gains.json
_GAINS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pid_gains.json")
with open(_GAINS_PATH) as _f:
    _RATE_DEFAULTS = json.load(_f).get("rate_wrapper", {})

_DEFAULT_KD = _RATE_DEFAULTS.get("kd", [22.0, 22.0, 15.0])
_DEFAULT_KI = _RATE_DEFAULTS.get("ki_rate_torque", 0.02)
_DEFAULT_IMAX = _RATE_DEFAULTS.get("integral_max", 0.008)
_DEFAULT_MAX_RATE = _RATE_DEFAULTS.get("max_rate_deg", 360.0)


class RateControlWrapper(gym.ActionWrapper):
    """ActionWrapper that converts body-rate commands to torques via PID.

    Policy outputs: [thrust_norm, roll_rate_norm, pitch_rate_norm, yaw_rate_norm]
        all in [-1, 1].

    Rates are denormalized from [-1, 1] to [-max_rate, +max_rate] deg/s,
    converted to rad/s, then a per-axis PID computes torques which are
    re-normalized to [-1, 1] for the base env.

    Thrust passes through unchanged.

    Args:
        env: The base HoverEnv (or wrapped HoverEnv).
        max_rate: Maximum body rate in deg/s (symmetric for all axes).
        kd: Proportional gains (inertia-scaled) per axis [roll, pitch, yaw].
        ki_rate_torque: Integral gain in torque space.
        integral_max: Integral term clamp magnitude in N·m.
    """

    def __init__(
        self,
        env: gym.Env,
        max_rate: float = _DEFAULT_MAX_RATE,
        kd: np.ndarray | None = None,
        ki_rate_torque: float = _DEFAULT_KI,
        integral_max: float = _DEFAULT_IMAX,
    ):
        super().__init__(env)
        self.max_rate_rad = np.deg2rad(max_rate)

        # PID gains
        self.inertia = np.array([IXX, IYY, IZZ])
        self.kd = np.array(kd) if kd is not None else np.array(_DEFAULT_KD)
        self.ki_rate_torque = ki_rate_torque
        self.integral_max = integral_max

        # Integral state
        self._rate_int_torque = np.zeros(3)

        # dt from base env
        self._dt = self.unwrapped.dt

    def action(self, action: np.ndarray) -> np.ndarray:
        """Convert [thrust, rate_r, rate_p, rate_y] → [thrust, tau_x, tau_y, tau_z]."""
        thrust_norm = action[0]

        # Denormalize rates: [-1, 1] → [-max_rate_rad, +max_rate_rad]
        des_rates = action[1:4].astype(np.float64) * self.max_rate_rad

        # Current angular velocity (rad/s), updated by previous step/reset
        actual_rates = self.unwrapped._state.angular_velocity

        # Rate error
        rate_err = des_rates - actual_rates

        # P term (inertia-scaled)
        tau_p = self.inertia * self.kd * rate_err

        # I term (torque space, anti-windup)
        self._rate_int_torque += self.ki_rate_torque * self._dt * rate_err
        self._rate_int_torque = np.clip(
            self._rate_int_torque, -self.integral_max, self.integral_max
        )

        tau = tau_p + self._rate_int_torque

        # Normalize torques to [-1, 1]
        tau_norm = np.clip(tau / MAX_TORQUE, -1.0, 1.0)

        return np.array(
            [thrust_norm, tau_norm[0], tau_norm[1], tau_norm[2]], dtype=np.float32
        )

    def step(self, action):
        """Step the env and fix _prev_action to store the rate action (policy output)."""
        action = np.asarray(action, dtype=np.float32)
        obs, reward, terminated, truncated, info = super().step(action)
        # Overwrite so observation wrappers see the rate action, not torques
        self.unwrapped._prev_action = action.copy()
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset integral state, then delegate to base env."""
        self._rate_int_torque = np.zeros(3)
        return self.env.reset(**kwargs)
