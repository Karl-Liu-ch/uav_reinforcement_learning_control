import argparse
import importlib
import json
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np

from utils.drone_config import ARM_LENGTH, G, MASS, MAX_MOTOR_THRUST, MAX_TORQUE, YAW_TORQUE_COEFF


def rad(x):
    return np.deg2rad(x)


def wrap_angle_rad(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def quat_wxyz_to_euler_xyz(quat_wxyz):
    w, x, y, z = quat_wxyz
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float64)


def settling_time(t, err, band):
    inside = np.abs(err) < band
    for i in range(len(inside)):
        if np.all(inside[i:]):
            return t[i]
    return np.nan


def build_cmd_seq(total_steps, dt, hover_thrust, step_deg, axis):
    axis_to_idx = {"roll": 1, "pitch": 2, "yaw": 3}
    axis_idx = axis_to_idx[axis]
    step_rad = rad(step_deg)
    cmd_seq = []
    for k in range(total_steps):
        t = k * dt
        cmd = [hover_thrust, 0.0, 0.0, 0.0]
        if t < 2.0:
            pass
        elif t < 4.0:
            cmd[axis_idx] = step_rad
        cmd_seq.append(cmd)
    return cmd_seq


def default_model_path():
    return Path(__file__).resolve().parent / "model" / "drone" / "scene_attitude_only.xml"


def build_output_dir(output_root):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / f"attitude_pid_test_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="Attitude inner-loop PID tuning test with offscreen video recording")
    parser.add_argument("--model", type=str, default=str(default_model_path()), help="MuJoCo XML scene path")
    parser.add_argument("--duration", type=float, default=6.0, help="Simulation duration in seconds")
    parser.add_argument("--hover-thrust", type=float, default=MASS * G, help="Total thrust command (N)")
    parser.add_argument("--axis", type=str, default="roll", choices=["roll", "pitch", "yaw"], help="Body-angle step axis")
    parser.add_argument("--step-deg", type=float, default=10.0, help="Step command amplitude in degrees")
    parser.add_argument("--att-kp", type=float, nargs=3, default=[6.0, 5.5, 4.0], help="PID Kp for roll pitch yaw")
    parser.add_argument("--att-ki", type=float, nargs=3, default=[0.10, 0.08, 0.08], help="PID Ki for roll pitch yaw")
    parser.add_argument("--att-kd", type=float, nargs=3, default=[0.15, 0.13, 0.12], help="PID Kd for roll pitch yaw")
    parser.add_argument("--record", action="store_true", help="Record MP4 video")
    parser.add_argument("--fps", type=int, default=50, help="Video frame rate")
    parser.add_argument("--width", type=int, default=640, help="Video width")
    parser.add_argument("--height", type=int, default=480, help="Video height")
    parser.add_argument("--output-root", type=str, default="recordings/body_angle_pid", help="Root folder for recordings")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(Path(args.model).expanduser().resolve()))
    data = mujoco.MjData(model)

    dt = float(model.opt.timestep)
    total_steps = int(args.duration / dt)
    cmd_seq = build_cmd_seq(total_steps, dt, args.hover_thrust, args.step_deg, args.axis)

    kp = np.array(args.att_kp, dtype=np.float64)
    ki = np.array(args.att_ki, dtype=np.float64)
    kd = np.array(args.att_kd, dtype=np.float64)
    int_err = np.zeros(3, dtype=np.float64)
    int_limit = np.array([0.4, 0.4, 0.6], dtype=np.float64)

    a = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [-ARM_LENGTH, -ARM_LENGTH, +ARM_LENGTH, +ARM_LENGTH],
            [-ARM_LENGTH, +ARM_LENGTH, +ARM_LENGTH, -ARM_LENGTH],
            [+YAW_TORQUE_COEFF, -YAW_TORQUE_COEFF, +YAW_TORQUE_COEFF, -YAW_TORQUE_COEFF],
        ],
        dtype=np.float64,
    )
    a_inv = np.linalg.inv(a)

    rpy_log = []
    ref_log = []
    t_log = []

    run_dir = build_output_dir(args.output_root)
    metrics_path = run_dir / "metrics.json"
    video_path = run_dir / "attitude_pid_test.mp4"

    renderer = None
    writer = None
    render_step = max(int(round(1.0 / max(args.fps, 1) / dt)), 1)

    try:
        if args.record:
            try:
                imageio = importlib.import_module("imageio.v2")
            except Exception as exc:
                raise RuntimeError(
                    "Recording needs imageio. Install with: pip install imageio imageio-ffmpeg"
                ) from exc

            renderer = mujoco.Renderer(model, width=args.width, height=args.height)
            writer = imageio.get_writer(str(video_path), fps=args.fps)

        for k, cmd in enumerate(cmd_seq):
            desired_rpy = np.array(cmd[1:4], dtype=np.float64)
            quat = data.qpos[:4].copy()
            current_rpy = quat_wxyz_to_euler_xyz(quat)

            rpy_err = desired_rpy - current_rpy
            rpy_err[2] = wrap_angle_rad(rpy_err[2])

            body_rates = data.qvel[:3].copy()
            int_err = np.clip(int_err + rpy_err * dt, -int_limit, int_limit)

            tau = kp * rpy_err + ki * int_err - kd * body_rates
            tau = np.clip(tau, -MAX_TORQUE, MAX_TORQUE)
            u = np.array([cmd[0], tau[0], tau[1], tau[2]], dtype=np.float64)
            motors = np.clip(a_inv @ u, 0.0, MAX_MOTOR_THRUST)

            data.ctrl[:4] = motors
            mujoco.mj_step(model, data)

            t_now = k * dt
            rpy_log.append(current_rpy)
            ref_log.append(desired_rpy)
            t_log.append(t_now)

            if renderer is not None and writer is not None and (k % render_step == 0):
                renderer.update_scene(data)
                writer.append_data(renderer.render())

    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()

    rpy_log = np.array(rpy_log)
    ref_log = np.array(ref_log)
    t_log = np.array(t_log)

    mask = (t_log >= 2.0) & (t_log < 4.0)
    t = t_log[mask] - 2.0
    axis_to_metric_idx = {"roll": 0, "pitch": 1, "yaw": 2}
    axis_idx = axis_to_metric_idx[args.axis]
    y = rpy_log[mask, axis_idx]
    r = ref_log[mask, axis_idx]
    e = y - r
    target = r[0] if len(r) > 0 else 0.0

    steady_slice_start = int(0.8 * len(e)) if len(e) > 0 else 0
    steady_err = np.mean(e[steady_slice_start:]) if len(e) > 0 else np.nan
    overshoot = (np.max(y) - target) / max(abs(target), 1e-6) * 100.0 if len(y) > 0 else np.nan
    ts = settling_time(t, e, band=rad(2.0)) if len(t) > 0 else np.nan

    metrics = {
        "axis": args.axis,
        "step_deg": float(args.step_deg),
        "steady_err_deg": float(np.rad2deg(steady_err)),
        "overshoot_pct": float(overshoot),
        "settling_s": float(ts),
        "output_dir": str(run_dir),
        "video_path": str(video_path) if args.record else None,
        "model_path": str(Path(args.model).expanduser().resolve()),
        "att_kp": kp.tolist(),
        "att_ki": ki.tolist(),
        "att_kd": kd.tolist(),
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"steady_err_deg = {metrics['steady_err_deg']:.3f}")
    print(f"overshoot_pct  = {metrics['overshoot_pct']:.2f}")
    print(f"settling_s     = {metrics['settling_s']:.3f}")
    print(f"results_dir    = {run_dir}")
    if args.record:
        print(f"video_path     = {video_path}")


if __name__ == "__main__":
    main()