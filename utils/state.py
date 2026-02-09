"""Quadrotor state management with quaternion/Euler conversion."""

import numpy as np
from scipy.spatial.transform import Rotation


class QuadState:
    """12D quadrotor state: [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz].

    Handles conversion between MuJoCo's quaternion representation and
    Euler angles (ZYX rotation sequence) for RL observation.

    State indices:
        0-2: Position (x, y, z)
        3-5: Attitude (roll, pitch, yaw) in radians
        6-8: Linear velocity (vx, vy, vz)
        9-11: Angular velocity (wx, wy, wz) in body frame
    """

    ROT_SEQ = "XYZ"  # Euler angle rotation sequence (intrinsic) - returns [roll, pitch, yaw]

    def __init__(self):
        self.state = np.zeros(12, dtype=np.float32)

    def set_from_mujoco(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        """Set state from MuJoCo qpos/qvel arrays.

        Args:
            qpos: MuJoCo position array [x, y, z, qw, qx, qy, qz]
            qvel: MuJoCo velocity array [vx, vy, vz, wx, wy, wz]
        """
        # Position
        self.state[0:3] = qpos[0:3]

        # Attitude: quaternion to Euler (MuJoCo uses wxyz, scipy uses xyzw)
        quat_wxyz = qpos[3:7]
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        # as_euler with 'xyz' returns [roll, pitch, yaw] directly
        self.state[3:6] = Rotation.from_quat(quat_xyzw).as_euler(self.ROT_SEQ.lower())

        # Velocities
        self.state[6:9] = qvel[0:3]
        self.state[9:12] = qvel[3:6]

    def get_mujoco_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Get MuJoCo qpos/qvel arrays from current state.

        Returns:
            Tuple of (qpos, qvel) arrays for MuJoCo
        """
        # Position
        pos = self.state[0:3]

        # Attitude: Euler to quaternion
        roll, pitch, yaw = self.state[3:6]
        quat_xyzw = Rotation.from_euler(self.ROT_SEQ.lower(), [roll, pitch, yaw]).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        qpos = np.concatenate([pos, quat_wxyz])
        qvel = self.state[6:12].copy()

        return qpos, qvel

    @property
    def position(self) -> np.ndarray:
        return self.state[0:3]

    @property
    def attitude(self) -> np.ndarray:
        return self.state[3:6]

    @property
    def velocity(self) -> np.ndarray:
        return self.state[6:9]

    @property
    def angular_velocity(self) -> np.ndarray:
        return self.state[9:12]

    def vec(self) -> np.ndarray:
        """Return state as a 12D vector."""
        return self.state.copy()

    def __repr__(self) -> str:
        return (
            f"QuadState(pos={self.position}, att={np.rad2deg(self.attitude)}, "
            f"vel={self.velocity}, ang_vel={self.angular_velocity})"
        )
