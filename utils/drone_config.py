"""Central drone physical parameters — single source of truth.

All Python modules should import from here instead of hardcoding values.
The MuJoCo XML (model/drone/drone.xml) must be kept in sync manually;
update ctrlrange and keyframe hover ctrl when changing MAX_MOTOR_THRUST.
"""

# ── Base parameters (change these to update everywhere) ──
MAX_MOTOR_THRUST = 13.0         # N per motor
ARM_LENGTH = 0.039799           # m
YAW_TORQUE_COEFF = 0.0201       # motor reaction-torque / thrust ratio
MASS = 0.2227                   # kg (frame 25g + battery/electronics 170g + propellers 27.7g)
G = 9.81                        # m/s²
DT = 0.01                       # s (MuJoCo timestep)
IXX = 4.16e-4                   # kg·m² (roll inertia, scaled from 6.44e-4)
IYY = 4.23e-4                   # kg·m² (pitch inertia, scaled from 6.54e-4)
IZZ = 5.37e-4                   # kg·m² (yaw inertia, scaled from 8.31e-4)

# ── Derived parameters ──
MAX_TOTAL_THRUST = 4 * MAX_MOTOR_THRUST                # N  (52.0)
MAX_TORQUE = 0.5                                        # N·m
HOVER_THRUST_PER_MOTOR = MASS * G / 4                   # N  (~0.545)
