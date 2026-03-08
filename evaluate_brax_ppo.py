import argparse
import importlib
import json
import os
from datetime import datetime
from functools import partial

import jax
import mujoco
import numpy as np
from brax.io import model
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from brax.training.agents.ppo import networks as ppo_networks
from jax import numpy as jp

from train_brax_ppo import (
    JaxMJXQuadBraxEnv,
    QuadHoverBraxEnv,
    _default_xml_path,
    _ensure_mesh_assets_resolved,
)


def _load_summary(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_hidden_sizes(value: str) -> tuple[int, ...]:
    items = [part.strip() for part in value.split(",") if part.strip()]
    if not items:
        raise ValueError("Hidden layer sizes cannot be empty.")
    sizes = tuple(int(item) for item in items)
    if any(size <= 0 for size in sizes):
        raise ValueError("All hidden layer sizes must be positive integers.")
    return sizes


def _build_env(args):
    _ensure_mesh_assets_resolved(args.xml)
    if args.env == "jax_mjx_quad":
        return JaxMJXQuadBraxEnv(
            xml_path=args.xml,
            impl=args.impl,
            max_episode_steps=args.episode_length,
            traj_duration_seconds=args.traj_duration_seconds,
            action_min=args.action_min,
            action_max=args.action_max,
        )

    return QuadHoverBraxEnv(
        xml_path=args.xml,
        backend=args.backend,
        action_min=args.action_min,
        action_max=args.action_max,
    )


def _resolve_render_xml_path(xml_path: str, render_xml_arg: str | None) -> str:
    if render_xml_arg:
        return os.path.abspath(render_xml_arg)

    env_xml = os.path.abspath(xml_path)
    xml_dir = os.path.dirname(env_xml)
    scene_candidate = os.path.join(xml_dir, "scene.xml")
    if os.path.exists(scene_candidate):
        return scene_candidate
    return env_xml


def _coerce_action(action):
    if hasattr(action, "ndim") and action.ndim > 1:
        return action[0]
    return action


def _resolve_orbax_checkpoint_path(path: str) -> str:
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

    if os.path.exists(os.path.join(path, "_CHECKPOINT_METADATA")):
        return path

    step_dirs = []
    for name in os.listdir(path):
        if not name.isdigit():
            continue
        step_dir = os.path.join(path, name)
        if os.path.isdir(step_dir) and os.path.exists(os.path.join(step_dir, "_CHECKPOINT_METADATA")):
            step_dirs.append((int(name), step_dir))

    if not step_dirs:
        raise FileNotFoundError(f"No valid Orbax checkpoint found under: {path}")

    step_dirs.sort(key=lambda item: item[0])
    return step_dirs[-1][1]


def _load_network_config_from_checkpoint(checkpoint_dir: str) -> dict:
    config_path = os.path.join(checkpoint_dir, "ppo_network_config.json")
    if not os.path.exists(config_path):
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    kwargs = config.get("network_factory_kwargs", {})
    activation = kwargs.get("activation", None)
    if activation == "<PjitFunction of <function silu at 0x7f5bd6d24ea0>>":
        activation = "silu"

    return {
        "policy_hidden_sizes": kwargs.get("policy_hidden_layer_sizes", None),
        "value_hidden_sizes": kwargs.get("value_hidden_layer_sizes", None),
        "activation": activation,
    }


def _extract_position(state):
    pipeline_state = state.pipeline_state
    if hasattr(pipeline_state, "qpos"):
        return jp.asarray(pipeline_state.qpos[:3])
    if hasattr(pipeline_state, "q"):
        return jp.asarray(pipeline_state.q[:3])
    return jp.zeros((3,))


def _extract_target(state, env):
    if isinstance(getattr(state, "info", None), dict) and "traj_pos" in state.info:
        traj = state.info["traj_pos"]
        step_count = int(state.info.get("step_count", 0))
        idx = min(step_count, int(traj.shape[0]) - 1)
        return jp.asarray(traj[idx])

    if hasattr(env, "_target_pos"):
        return jp.asarray(env._target_pos)

    return jp.zeros((3,))


def _extract_qpos_qvel(state):
    pipeline_state = state.pipeline_state
    if hasattr(pipeline_state, "qpos") and hasattr(pipeline_state, "qvel"):
        return np.asarray(pipeline_state.qpos, dtype=np.float64), np.asarray(pipeline_state.qvel, dtype=np.float64)
    if hasattr(pipeline_state, "q") and hasattr(pipeline_state, "qd"):
        return np.asarray(pipeline_state.q, dtype=np.float64), np.asarray(pipeline_state.qd, dtype=np.float64)
    return None, None


def _make_tracking_camera(mj_model):
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(mj_model, cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = max(float(mj_model.stat.extent) * 2.0, 2.5)
    cam.azimuth = 135.0
    cam.elevation = -25.0
    cam.lookat[:] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return cam


def _save_plots(all_errors, all_actual_xyz, all_target_xyz, plot_dir):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"plot_warning=matplotlib_unavailable reason={exc}")
        return None

    os.makedirs(plot_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    non_empty_count = 0
    for idx, err in enumerate(all_errors):
        if len(err) == 0:
            continue
        non_empty_count += 1
        x = np.arange(len(err), dtype=np.int32)
        if len(err) == 1:
            ax.scatter(x, err, label=f"ep{idx}", s=40)
        else:
            ax.plot(x, err, label=f"ep{idx}", alpha=0.85, marker="o", markersize=2)

    ax.set_title(f"Trajectory Following Error per Step (non-empty episodes={non_empty_count})")
    ax.set_xlabel("Step")
    ax.set_ylabel("Position Error (L2)")
    if len(all_errors) <= 10:
        ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    curve_path = os.path.join(plot_dir, "trajectory_error_curve.png")
    fig.savefig(curve_path, dpi=160)
    plt.close(fig)

    flat = np.concatenate([np.asarray(e, dtype=np.float32) for e in all_errors if len(e) > 0])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(flat, bins=40)
    ax.set_title("Trajectory Error Distribution")
    ax.set_xlabel("Position Error (L2)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    hist_path = os.path.join(plot_dir, "trajectory_error_hist.png")
    fig.savefig(hist_path, dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    for idx, (actual_xyz, target_xyz) in enumerate(zip(all_actual_xyz, all_target_xyz)):
        if len(actual_xyz) == 0:
            continue
        actual_arr = np.asarray(actual_xyz, dtype=np.float32)
        target_arr = np.asarray(target_xyz, dtype=np.float32)
        if actual_arr.shape[0] == 1:
            ax.scatter(actual_arr[:, 0], actual_arr[:, 1], actual_arr[:, 2], s=40, label=f"actual_ep{idx}")
            ax.scatter(target_arr[:, 0], target_arr[:, 1], target_arr[:, 2], s=40, marker="x", label=f"target_ep{idx}")
        else:
            ax.plot(actual_arr[:, 0], actual_arr[:, 1], actual_arr[:, 2], label=f"actual_ep{idx}")
            ax.plot(target_arr[:, 0], target_arr[:, 1], target_arr[:, 2], linestyle="--", label=f"target_ep{idx}")

    ax.set_title("3D Trajectory: Actual vs Target")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if len(all_actual_xyz) <= 5:
        ax.legend(loc="best")
    fig.tight_layout()
    traj3d_path = os.path.join(plot_dir, "trajectory_3d_actual_vs_target.png")
    fig.savefig(traj3d_path, dpi=160)
    plt.close(fig)

    return {"curve": curve_path, "hist": hist_path, "traj3d": traj3d_path}


def main():
    parser = argparse.ArgumentParser(description="Evaluate Brax PPO checkpoint")
    parser.add_argument("--summary", type=str, default=None, help="Path to training_summary.json")
    parser.add_argument("--params", type=str, default=None, help="Path to ppo_params.msgpack")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Path to Orbax checkpoint dir or checkpoints root")

    parser.add_argument("--env", type=str, default="hover", choices=["hover", "jax_mjx_quad"])
    parser.add_argument("--xml", type=str, default=_default_xml_path())
    parser.add_argument("--render-xml", type=str, default=None, help="XML used only for video rendering (defaults to sibling scene.xml when available)")
    parser.add_argument("--backend", type=str, default="mjx", choices=["mjx", "generalized", "spring", "positional"])
    parser.add_argument("--impl", type=str, default="jax")
    parser.add_argument("--traj-duration-seconds", type=float, default=5.0)
    parser.add_argument("--episode-length", type=int, default=500)
    parser.add_argument("--action-min", type=float, default=0.0)
    parser.add_argument("--action-max", type=float, default=13.0)
    parser.add_argument("--policy-hidden-sizes", type=str, default=None)
    parser.add_argument("--value-hidden-sizes", type=str, default=None)
    parser.add_argument("--activation", type=str, default=None, choices=["silu", "relu", "tanh"])

    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--plots-dir", type=str, default="plots")
    parser.add_argument("--record-video", action="store_true", help="Record evaluation episodes to MP4")
    parser.add_argument("--video-fps", type=int, default=30, help="Recorded video FPS")
    parser.add_argument("--video-width", type=int, default=640, help="Recorded video width")
    parser.add_argument("--video-height", type=int, default=480, help="Recorded video height")
    parser.add_argument("--video-episodes", type=int, default=1, help="How many leading episodes to record")
    args = parser.parse_args()

    eval_run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_out_dir = os.path.join(os.path.abspath(args.plots_dir), f"brax_eval_{eval_run_tag}")
    os.makedirs(eval_out_dir, exist_ok=True)

    if args.summary:
        summary = _load_summary(args.summary)
        args.env = summary.get("env", args.env)
        args.xml = summary.get("xml", args.xml)
        args.backend = summary.get("backend", args.backend)
        args.impl = summary.get("impl", args.impl)
        args.traj_duration_seconds = summary.get("traj_duration_seconds", args.traj_duration_seconds)
        args.episode_length = int(summary.get("episode_length", args.episode_length))
        args.action_min = float(summary.get("action_min", args.action_min))
        args.action_max = float(summary.get("action_max", args.action_max))
        if args.params is None:
            args.params = summary.get("params_path", args.params)
        if args.checkpoint_path is None:
            args.checkpoint_path = summary.get("checkpoint_dir", args.checkpoint_path)
        if args.policy_hidden_sizes is None and summary.get("policy_hidden_sizes") is not None:
            args.policy_hidden_sizes = ",".join(str(v) for v in summary["policy_hidden_sizes"])
        if args.value_hidden_sizes is None and summary.get("value_hidden_sizes") is not None:
            args.value_hidden_sizes = ",".join(str(v) for v in summary["value_hidden_sizes"])
        if args.activation is None and summary.get("activation") is not None:
            args.activation = summary["activation"]

    render_xml_path = _resolve_render_xml_path(args.xml, args.render_xml)

    if args.params is None and args.checkpoint_path is None:
        raise ValueError("Provide --params or --checkpoint-path (or --summary containing one of them)")

    env = _build_env(args)

    if args.checkpoint_path is not None:
        resolved_ckpt = _resolve_orbax_checkpoint_path(args.checkpoint_path)
        print(f"Loading Orbax checkpoint: {resolved_ckpt}")
        params = ppo_checkpoint.load(resolved_ckpt)
        params_source = resolved_ckpt
        ckpt_net_cfg = _load_network_config_from_checkpoint(resolved_ckpt)
        if args.policy_hidden_sizes is None and ckpt_net_cfg.get("policy_hidden_sizes") is not None:
            args.policy_hidden_sizes = ",".join(str(v) for v in ckpt_net_cfg["policy_hidden_sizes"])
        if args.value_hidden_sizes is None and ckpt_net_cfg.get("value_hidden_sizes") is not None:
            args.value_hidden_sizes = ",".join(str(v) for v in ckpt_net_cfg["value_hidden_sizes"])
        if args.activation is None and ckpt_net_cfg.get("activation") is not None:
            args.activation = ckpt_net_cfg["activation"]
    else:
        args.params = os.path.abspath(args.params)
        params = model.load_params(args.params)
        params_source = args.params

    if args.policy_hidden_sizes is None:
        args.policy_hidden_sizes = "256,256,256"
    if args.value_hidden_sizes is None:
        args.value_hidden_sizes = "256,256,256"
    if args.activation is None:
        args.activation = "silu"

    policy_hidden_sizes = _parse_hidden_sizes(args.policy_hidden_sizes)
    value_hidden_sizes = _parse_hidden_sizes(args.value_hidden_sizes)
    activation_map = {
        "silu": jax.nn.silu,
        "relu": jax.nn.relu,
        "tanh": jax.nn.tanh,
    }
    activation_fn = activation_map[args.activation]

    print(f"Eval actor hidden sizes: {policy_hidden_sizes}")
    print(f"Eval critic hidden sizes: {value_hidden_sizes}")
    print(f"Eval activation: {args.activation}")

    network_factory = partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=policy_hidden_sizes,
        value_hidden_layer_sizes=value_hidden_sizes,
        activation=activation_fn,
    )
    nets = network_factory(
        observation_size=env.observation_size,
        action_size=env.action_size,
    )
    make_policy = ppo_networks.make_inference_fn(nets)
    policy = make_policy(params, deterministic=args.deterministic)
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    policy_fn = jax.jit(policy)

    max_steps = args.max_steps if args.max_steps is not None else args.episode_length
    key = jax.random.PRNGKey(args.seed)

    renderer = None
    video_writer = None
    video_path = None
    render_camera = None
    if args.record_video:
        try:
            imageio = importlib.import_module("imageio.v2")
        except Exception as exc:
            raise RuntimeError("Video recording requires imageio. Install with: pip install imageio imageio-ffmpeg") from exc

        mj_model = mujoco.MjModel.from_xml_path(render_xml_path)
        mj_data = mujoco.MjData(mj_model)
        render_width = min(int(args.video_width), int(mj_model.vis.global_.offwidth))
        render_height = min(int(args.video_height), int(mj_model.vis.global_.offheight))
        if render_width != int(args.video_width) or render_height != int(args.video_height):
            print(
                f"video_size_clamped=requested({args.video_width}x{args.video_height}) "
                f"framebuffer({mj_model.vis.global_.offwidth}x{mj_model.vis.global_.offheight}) "
                f"using({render_width}x{render_height})"
            )

        renderer = mujoco.Renderer(mj_model, height=render_height, width=render_width)
        render_camera = _make_tracking_camera(mj_model)
        video_path = os.path.join(eval_out_dir, "evaluation_rollout.mp4")
        video_writer = imageio.get_writer(video_path, fps=args.video_fps)

    episode_returns = []
    episode_lengths = []
    episode_mean_errors = []
    episode_rmse_errors = []
    all_errors = []
    all_actual_xyz = []
    all_target_xyz = []

    try:
        for episode_idx in range(args.episodes):
            key, reset_key = jax.random.split(key)
            state = reset_fn(reset_key)
            total_reward = 0.0
            steps = 0
            step_errors = []
            actual_xyz = []
            target_xyz = []

            while steps < max_steps:
                key, sample_key = jax.random.split(key)
                obs = state.obs
                if hasattr(obs, "ndim") and obs.ndim == 1:
                    obs = jp.expand_dims(obs, axis=0)

                action, _ = policy_fn(obs, sample_key)
                action = _coerce_action(action)
                state = step_fn(state, action)

                pos = _extract_position(state)
                target = _extract_target(state, env)
                traj_error = float(jp.linalg.norm(pos - target))
                if np.isfinite(traj_error):
                    step_errors.append(traj_error)

                actual_xyz.append(np.asarray(pos, dtype=np.float32))
                target_xyz.append(np.asarray(target, dtype=np.float32))

                if video_writer is not None and renderer is not None and episode_idx < args.video_episodes:
                    qpos, qvel = _extract_qpos_qvel(state)
                    if qpos is not None and qvel is not None:
                        mj_data.qpos[: qpos.shape[0]] = qpos
                        mj_data.qvel[: qvel.shape[0]] = qvel
                        mujoco.mj_forward(mj_model, mj_data)
                        if render_camera is not None and qpos.shape[0] >= 3:
                            render_camera.lookat[:] = np.array(qpos[:3], dtype=np.float64)
                        renderer.update_scene(mj_data, camera=render_camera)
                        video_writer.append_data(renderer.render())

                total_reward += float(state.reward)
                steps += 1
                if float(state.done) > 0.5:
                    break

            episode_returns.append(total_reward)
            episode_lengths.append(steps)
            all_errors.append(step_errors)
            all_actual_xyz.append(actual_xyz)
            all_target_xyz.append(target_xyz)

            if step_errors:
                mean_err = float(np.mean(step_errors))
                rmse_err = float(np.sqrt(np.mean(np.square(step_errors))))
            else:
                mean_err = float("nan")
                rmse_err = float("nan")

            episode_mean_errors.append(mean_err)
            episode_rmse_errors.append(rmse_err)

            print(
                f"episode={episode_idx} return={total_reward:.4f} length={steps} "
                f"mean_traj_error={mean_err:.6f} rmse_traj_error={rmse_err:.6f}"
            )
    finally:
        if video_writer is not None:
            video_writer.close()
        if renderer is not None:
            renderer.close()

    mean_return = sum(episode_returns) / len(episode_returns)
    mean_length = sum(episode_lengths) / len(episode_lengths)
    finite_mean_errors = [v for v in episode_mean_errors if np.isfinite(v)]
    finite_rmse_errors = [v for v in episode_rmse_errors if np.isfinite(v)]
    mean_traj_error = float(np.mean(finite_mean_errors)) if finite_mean_errors else float("nan")
    mean_rmse_error = float(np.mean(finite_rmse_errors)) if finite_rmse_errors else float("nan")

    plot_dir = eval_out_dir
    plot_paths = _save_plots(all_errors, all_actual_xyz, all_target_xyz, plot_dir)

    os.makedirs(plot_dir, exist_ok=True)
    csv_path = os.path.join(plot_dir, "trajectory_error_per_episode.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("episode,mean_traj_error,rmse_traj_error,return,length\n")
        for i in range(args.episodes):
            f.write(
                f"{i},{episode_mean_errors[i]},{episode_rmse_errors[i]},"
                f"{episode_returns[i]},{episode_lengths[i]}\n"
            )

    summary_path = os.path.join(plot_dir, "evaluation_summary.json")
    summary = {
        "env": args.env,
        "params": params_source,
        "render_xml": render_xml_path,
        "policy_hidden_sizes": list(policy_hidden_sizes),
        "value_hidden_sizes": list(value_hidden_sizes),
        "activation": args.activation,
        "episodes": args.episodes,
        "deterministic": args.deterministic,
        "mean_return": mean_return,
        "mean_length": mean_length,
        "mean_traj_error": mean_traj_error,
        "mean_rmse_traj_error": mean_rmse_error,
        "plot_paths": plot_paths,
        "csv_path": csv_path,
        "video_path": video_path,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("---")
    print(f"env={args.env} params={params_source}")
    print(f"episodes={args.episodes} deterministic={args.deterministic}")
    print(f"mean_return={mean_return:.4f} mean_length={mean_length:.2f}")
    print(f"mean_traj_error={mean_traj_error:.6f} mean_rmse_traj_error={mean_rmse_error:.6f}")
    print(f"saved_eval_summary={summary_path}")
    if plot_paths is not None:
        print(f"saved_plot_curve={plot_paths['curve']}")
        print(f"saved_plot_hist={plot_paths['hist']}")
        print(f"saved_plot_traj3d={plot_paths['traj3d']}")
    if video_path is not None:
        print(f"saved_video={video_path}")
    print(f"saved_error_csv={csv_path}")


if __name__ == "__main__":
    main()