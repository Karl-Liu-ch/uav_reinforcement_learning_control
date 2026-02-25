import argparse
import json
import os
import sys
import time
from datetime import datetime
from functools import partial

import jax
import mujoco
from brax.envs.base import Env, PipelineEnv, State
from brax.io import mjcf, model
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo_train
from jax import numpy as jp
from mujoco import mjx

# Add parent directory to path for utils imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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


class QuadHoverBraxEnv(PipelineEnv):
    def __init__(
        self,
        xml_path: str,
        backend: str = "mjx",
        n_frames: int = 1,
        target_height: float = 1.0,
        action_min: float = 0.0,
        action_max: float = 1.5,
        pos_limit_xy: float = 3.0,
        pos_limit_z_low: float = 0.02,
        pos_limit_z_high: float = 4.0,
    ):
        sys = mjcf.load(xml_path)
        super().__init__(sys=sys, backend=backend, n_frames=n_frames)

        self._target_pos = jp.array([0.0, 0.0, target_height], dtype=jp.float32)
        self._pos_limit_xy = pos_limit_xy
        self._pos_limit_z_low = pos_limit_z_low
        self._pos_limit_z_high = pos_limit_z_high
        
        # Setup mixing matrix for thrust/torque to motor commands conversion
        self.max_motor_thrust = MAX_MOTOR_THRUST
        k = YAW_TORQUE_COEFF
        l = ARM_LENGTH
        self.max_total_thrust = 4 * self.max_motor_thrust
        self.max_torque = MAX_TORQUE
        
        # Mixing matrix: maps motor forces to [thrust, tau_x, tau_y, tau_z]
        # A @ [F1, F2, F3, F4] = [thrust, tau_x, tau_y, tau_z]
        A = jp.array([
            [ 1,   1,   1,   1],
            [-l,  -l,  +l,  +l],
            [-l,  +l,  +l,  -l],
            [+k,  -k,  +k,  -k],
        ])
        self.A_inv = jp.linalg.inv(A)
        
        # Action bounds for [thrust, tau_x, tau_y, tau_z]
        self._ctrl_min = jp.array([0.0, -self.max_torque, -self.max_torque, -self.max_torque])
        self._ctrl_max = jp.array([self.max_total_thrust, self.max_torque, self.max_torque, self.max_torque])
    
    @property
    def action_size(self):
        # Agent outputs 4 values: [thrust, tau_x, tau_y, tau_z]
        return 4
    
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
        u = jp.array([thrust, tau_x, tau_y, tau_z])
        F = self.A_inv @ u
        return jp.clip(F, 0.0, self.max_motor_thrust)

    def reset(self, rng: jax.Array) -> State:
        rng, rng_q, rng_qd = jax.random.split(rng, 3)

        q = self.sys.init_q + jax.random.uniform(
            rng_q,
            (self.sys.q_size(),),
            minval=-0.01,
            maxval=0.01,
        )
        qd = jax.random.uniform(
            rng_qd,
            (self.sys.qd_size(),),
            minval=-0.01,
            maxval=0.01,
        )

        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)
        reward, done = jp.zeros(2)

        metrics = {
            "pos_error": jp.array(0.0),
            "reward_hover": jp.array(0.0),
            "reward_action": jp.array(0.0),
            "reward": jp.array(0.0),
        }
        info = {"time_out": done}
        return State(pipeline_state, obs, reward, done, metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        # Denormalize action from [-1, 1] to physical units [thrust, tau_x, tau_y, tau_z]
        physical_action = (action + 1.0) * 0.5 * (self._ctrl_max - self._ctrl_min) + self._ctrl_min
        physical_action = jp.clip(physical_action, self._ctrl_min, self._ctrl_max)
        
        # Convert thrust and torques to motor commands through mixing matrix
        thrust, tau_x, tau_y, tau_z = physical_action
        motor_commands = self._mix_to_motors(thrust, tau_x, tau_y, tau_z)

        pipeline_state = self.pipeline_step(state.pipeline_state, motor_commands)
        obs = self._get_obs(pipeline_state)

        pos = pipeline_state.q[:3]
        pos_error = jp.linalg.norm(pos - self._target_pos)

        reward_hover = jp.exp(-2.0 * pos_error * pos_error)
        reward_action = -0.001 * jp.sum(jp.square(action))
        # reward = reward_hover + reward_action
        reward = reward_hover

        out_of_xy = jp.logical_or(
            jp.abs(pos[0]) > self._pos_limit_xy,
            jp.abs(pos[1]) > self._pos_limit_xy,
        )
        out_of_z = jp.logical_or(
            pos[2] < self._pos_limit_z_low,
            pos[2] > self._pos_limit_z_high,
        )
        done = jp.where(jp.logical_or(out_of_xy, out_of_z), 1.0, 0.0)

        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
            metrics={
                "pos_error": pos_error,
                "reward_hover": reward_hover,
                "reward_action": reward_action,
                "reward": reward,
            },
            info={**state.info, "time_out": done},
        )

    def _get_obs(self, pipeline_state):
        return jp.concatenate([pipeline_state.q, pipeline_state.qd], axis=-1)


class JaxMJXQuadBraxEnv(Env):
    def __init__(
        self,
        xml_path: str,
        impl: str = "jax",
        max_episode_steps: int = 500,
        traj_duration_seconds: float = 5.0,
        action_min: float = 0.0,
        action_max: float = 13.0,
        pos_limit_xy: float = 3.0,
        pos_limit_z_low: float = 0.02,
        pos_limit_z_high: float = 4.0,
        vel_limit: float = 20.0,
    ):
        self._xml_path = xml_path
        self._backend = "mjx"
        self._max_episode_steps = int(max_episode_steps)
        self._traj_duration_seconds = float(traj_duration_seconds)
        self._pos_limit_xy = float(pos_limit_xy)
        self._pos_limit_z_low = float(pos_limit_z_low)
        self._pos_limit_z_high = float(pos_limit_z_high)
        self._vel_limit = float(vel_limit)

        self._mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self._mjx_model = mjx.put_model(
            self._mj_model, impl=_normalize_mjx_impl(impl)
        )

        # Agent outputs 4 actions: [thrust, tau_x, tau_y, tau_z]
        self._action_size = 4
        self._observation_size = int(self._mjx_model.nq + self._mjx_model.nv)

        # Setup mixing matrix for thrust/torque to motor commands conversion
        self.max_motor_thrust = MAX_MOTOR_THRUST
        k = YAW_TORQUE_COEFF
        l = ARM_LENGTH
        self.max_total_thrust = 4 * self.max_motor_thrust
        self.max_torque = MAX_TORQUE
        
        # Mixing matrix: maps motor forces to [thrust, tau_x, tau_y, tau_z]
        # A @ [F1, F2, F3, F4] = [thrust, tau_x, tau_y, tau_z]
        A = jp.array([
            [ 1,   1,   1,   1],
            [-l,  -l,  +l,  +l],
            [-l,  +l,  +l,  -l],
            [+k,  -k,  +k,  -k],
        ])
        self.A_inv = jp.linalg.inv(A)
        
        # Action bounds for [thrust, tau_x, tau_y, tau_z]
        self._ctrl_min = jp.array([0.0, -self.max_torque, -self.max_torque, -self.max_torque])
        self._ctrl_max = jp.array([self.max_total_thrust, self.max_torque, self.max_torque, self.max_torque])

    @property
    def observation_size(self):
        return self._observation_size

    @property
    def action_size(self):
        return self._action_size

    @property
    def backend(self):
        return self._backend

    def reset(self, rng: jax.Array) -> State:
        data = mjx.make_data(self._mjx_model)

        qpos = data.qpos
        qvel = jp.zeros_like(data.qvel)

        if qpos.shape[0] >= 3:
            qpos = qpos.at[2].set(1.0)

        if qpos.shape[0] >= 7:
            qpos = qpos.at[3].set(1.0)
            qpos = qpos.at[4].set(0.0)
            qpos = qpos.at[5].set(0.0)
            qpos = qpos.at[6].set(0.0)

        rng, rng_q, rng_qd = jax.random.split(rng, 3)
        q_noise = jax.random.uniform(rng_q, qpos.shape, minval=-0.01, maxval=0.01)
        qd_noise = jax.random.uniform(rng_qd, qvel.shape, minval=-0.01, maxval=0.01)
        qpos = qpos + q_noise
        qvel = qvel + qd_noise

        if qpos.shape[0] >= 7:
            quat = qpos[3:7]
            quat = quat / (jp.linalg.norm(quat) + 1e-8)
            qpos = qpos.at[3:7].set(quat)

        data = data.replace(qpos=qpos, qvel=qvel)

        traj_pos = self._sample_trajectory(
            self._max_episode_steps, self._traj_duration_seconds
        )
        obs = self._get_obs(data)
        reward, done = jp.zeros(2)

        metrics = {
            "pos_error": jp.array(0.0),
            "reward_hover": jp.array(0.0),
            "reward_action": jp.array(0.0),
            "reward": jp.array(0.0),
        }
        info = {
            "time_out": done,
            "step_count": jp.array(0, dtype=jp.int32),
            "traj_pos": traj_pos,
        }
        return State(data, obs, reward, done, metrics, info)

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
        u = jp.array([thrust, tau_x, tau_y, tau_z])
        F = self.A_inv @ u
        return jp.clip(F, 0.0, self.max_motor_thrust)

    def step(self, state: State, action: jax.Array) -> State:
        # Denormalize action from [-1, 1] to physical units [thrust, tau_x, tau_y, tau_z]
        physical_action = (action + 1.0) * 0.5 * (self._ctrl_max - self._ctrl_min) + self._ctrl_min
        physical_action = jp.clip(physical_action, self._ctrl_min, self._ctrl_max)
        
        # Convert thrust and torques to motor commands through mixing matrix
        thrust, tau_x, tau_y, tau_z = physical_action
        motor_commands = self._mix_to_motors(thrust, tau_x, tau_y, tau_z)

        data = state.pipeline_state.replace(ctrl=motor_commands)
        data = mjx.step(self._mjx_model, data)

        step_count = state.info["step_count"] + 1
        idx = jp.minimum(step_count, self._max_episode_steps - 1)
        target = state.info["traj_pos"][idx]
        pos = data.qpos[:3]

        state_is_finite = jp.all(jp.isfinite(data.qpos)) & jp.all(jp.isfinite(data.qvel))
        out_of_xy = (jp.abs(pos[0]) > self._pos_limit_xy) | (jp.abs(pos[1]) > self._pos_limit_xy)
        out_of_z = (pos[2] < self._pos_limit_z_low) | (pos[2] > self._pos_limit_z_high)
        out_of_vel = jp.any(jp.abs(data.qvel[:3]) > self._vel_limit)
        state_is_valid = state_is_finite & (~out_of_xy) & (~out_of_z) & (~out_of_vel)

        pos_error_raw = jp.linalg.norm(pos - target)
        pos_error = jp.where(state_is_valid & jp.isfinite(pos_error_raw), pos_error_raw, 1e3)
        reward_hover = jp.exp(-(pos_error**2))
        reward_action = -0.001 * jp.sum(jp.square(action))
        reward_raw = reward_hover + reward_action
        reward = jp.where(state_is_valid & jp.isfinite(reward_raw), reward_raw, -1.0)
        done = jp.where(state_is_valid, 0.0, 1.0)
        obs = self._get_obs(data)
        obs = jp.where(jp.isfinite(obs), obs, jp.zeros_like(obs))

        return state.replace(
            pipeline_state=data,
            obs=obs,
            reward=reward,
            done=done,
            metrics={
                "pos_error": pos_error,
                "reward_hover": reward_hover,
                "reward_action": reward_action,
                "reward": reward,
            },
            info={
                **state.info,
                "time_out": done,
                "step_count": step_count,
            },
        )

    @staticmethod
    def _sample_trajectory(max_episode_steps: int, traj_duration_seconds: float):
        t = jp.linspace(0.0, traj_duration_seconds, max_episode_steps)
        center = jp.array([0.0, 0.0, 1.0])
        amp = jp.array([0.5, 0.5, 0.2])
        freq = jp.array([0.2, 0.15, 0.1])
        return center + amp * jp.sin(2.0 * jp.pi * freq * t[:, None])

    @staticmethod
    def _get_obs(data):
        return jp.concatenate([data.qpos, data.qvel], axis=-1)


def _default_xml_path() -> str:
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "model",
        "drone",
        "drone.xml",
    )


def _ensure_mesh_assets_resolved(xml_path: str) -> None:
    xml_dir = os.path.dirname(os.path.abspath(xml_path))
    mesh_dir = os.path.join(xml_dir, "assets", "drone")
    if not os.path.isdir(mesh_dir):
        return

    for name in os.listdir(mesh_dir):
        if not name.lower().endswith(".stl"):
            continue

        src = os.path.join(mesh_dir, name)
        dst = os.path.join(xml_dir, name)
        if os.path.exists(dst):
            continue

        try:
            os.symlink(src, dst)
        except OSError:
            with open(src, "rb") as f_src, open(dst, "wb") as f_dst:
                f_dst.write(f_src.read())


def _metric_float(metrics, keys, default=float("nan")):
    for key in keys:
        value = metrics.get(key, None)
        if value is not None:
            try:
                return float(value)
            except Exception:
                continue
    return default


def _is_finite_scalar(x) -> bool:
    try:
        value = float(x)
    except Exception:
        return False
    return value == value and value not in (float("inf"), float("-inf"))


def _parse_hidden_sizes(value: str) -> tuple[int, ...]:
    items = [part.strip() for part in value.split(",") if part.strip()]
    if not items:
        raise ValueError("Hidden layer sizes cannot be empty.")
    sizes = tuple(int(item) for item in items)
    if any(size <= 0 for size in sizes):
        raise ValueError("All hidden layer sizes must be positive integers.")
    return sizes


def main():
    parser = argparse.ArgumentParser(description="Train quadrotor policy with Brax PPO")
    parser.add_argument("--env", type=str, default="hover", choices=["hover", "jax_mjx_quad"])
    parser.add_argument("--xml", type=str, default=_default_xml_path())
    parser.add_argument("--num-timesteps", type=int, default=2_000_000)
    parser.add_argument("--episode-length", type=int, default=500)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--num-evals", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-cost", type=float, default=1e-3)
    parser.add_argument("--discounting", type=float, default=0.99)
    parser.add_argument("--impl", type=str, default="jax")
    parser.add_argument("--traj-duration-seconds", type=float, default=5.0)
    parser.add_argument("--action-min", type=float, default=0.0)
    parser.add_argument("--action-max", type=float, default=13.0)
    parser.add_argument("--unroll-length", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-minibatches", type=int, default=16)
    parser.add_argument("--num-updates-per-batch", type=int, default=4)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--reward-scaling", type=float, default=1.0)
    parser.add_argument("--policy-hidden-sizes", type=str, default="128, 128")
    parser.add_argument("--value-hidden-sizes", type=str, default="128, 128")
    parser.add_argument("--activation", type=str, default="relu", choices=["silu", "relu", "tanh"])
    parser.add_argument("--backend", type=str, default="mjx", choices=["mjx", "generalized", "spring", "positional"])
    parser.add_argument("--checkpoint-interval", type=int, default=200_000)
    parser.add_argument("--restore-checkpoint-path", type=str, default=None)
    parser.add_argument("--restore-value-fn", action="store_true")
    parser.add_argument("--output-dir", type=str, default="models_brax")
    args = parser.parse_args()

    policy_hidden_sizes = _parse_hidden_sizes(args.policy_hidden_sizes)
    value_hidden_sizes = _parse_hidden_sizes(args.value_hidden_sizes)
    activation_map = {
        "silu": jax.nn.silu,
        "relu": jax.nn.relu,
        "tanh": jax.nn.tanh,
    }
    activation_fn = activation_map[args.activation]

    args.output_dir = os.path.abspath(args.output_dir)
    restore_params = None
    if args.restore_checkpoint_path is not None:
        args.restore_checkpoint_path = os.path.abspath(args.restore_checkpoint_path)

        if os.path.isfile(args.restore_checkpoint_path) and args.restore_checkpoint_path.endswith(".msgpack"):
            msgpack_path = args.restore_checkpoint_path
            restore_params = model.load_params(args.restore_checkpoint_path)
            args.restore_checkpoint_path = None
            print(f"Restoring from msgpack params: {msgpack_path}")
        elif os.path.isdir(args.restore_checkpoint_path):
            metadata_path = os.path.join(args.restore_checkpoint_path, "_CHECKPOINT_METADATA")
            if not os.path.exists(metadata_path):
                step_dirs = []
                for name in os.listdir(args.restore_checkpoint_path):
                    if not name.isdigit():
                        continue
                    step_dir = os.path.join(args.restore_checkpoint_path, name)
                    if os.path.isdir(step_dir) and os.path.exists(os.path.join(step_dir, "_CHECKPOINT_METADATA")):
                        step_dirs.append((int(name), step_dir))

                if step_dirs:
                    step_dirs.sort(key=lambda item: item[0])
                    args.restore_checkpoint_path = step_dirs[-1][1]
                    print(f"Resolved latest Orbax checkpoint: {args.restore_checkpoint_path}")

    os.makedirs(args.output_dir, exist_ok=True)
    _ensure_mesh_assets_resolved(args.xml)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.abspath(os.path.join(args.output_dir, run_id))
    os.makedirs(run_dir, exist_ok=True)
    checkpoint_dir = os.path.abspath(os.path.join(run_dir, "checkpoints"))
    os.makedirs(checkpoint_dir, exist_ok=True)

    if args.env == "jax_mjx_quad":
        env = JaxMJXQuadBraxEnv(
            xml_path=args.xml,
            impl=args.impl,
            max_episode_steps=args.episode_length,
            traj_duration_seconds=args.traj_duration_seconds,
            action_min=args.action_min,
            action_max=args.action_max,
        )
        eval_env = JaxMJXQuadBraxEnv(
            xml_path=args.xml,
            impl=args.impl,
            max_episode_steps=args.episode_length,
            traj_duration_seconds=args.traj_duration_seconds,
            action_min=args.action_min,
            action_max=args.action_max,
        )
    else:
        env = QuadHoverBraxEnv(
            xml_path=args.xml,
            backend=args.backend,
            action_min=args.action_min,
            action_max=args.action_max,
        )
        eval_env = QuadHoverBraxEnv(
            xml_path=args.xml,
            backend=args.backend,
            action_min=args.action_min,
            action_max=args.action_max,
        )

    print(f"Env mode: {args.env}")
    print(f"Brax backend: {args.backend}")
    print(f"Devices: {jax.devices()}")
    print(f"Training run dir: {run_dir}")
    print(f"Actor hidden sizes: {policy_hidden_sizes}")
    print(f"Critic hidden sizes: {value_hidden_sizes}")
    print(f"Activation: {args.activation}")

    wall_start = time.time()
    last_checkpoint_step = {"step": 0}
    latest_params = {"value": None}

    def progress(step: int, metrics):
        sps = _metric_float(metrics, ["training/sps"], default=0.0)
        train_reward = _metric_float(
            metrics,
            ["training/episode_reward", "training/sum_reward", "training/reward"],
        )
        eval_reward = _metric_float(
            metrics,
            ["eval/episode_reward", "eval/sum_reward", "eval/reward"],
        )
        eval_valid = _is_finite_scalar(eval_reward)
        train_valid = _is_finite_scalar(train_reward)

        if eval_valid and train_valid:
            print(
                f"step={step:,} train_reward={train_reward:.4f} "
                f"eval_reward={eval_reward:.4f} sps={sps:.1f}"
            )
        elif eval_valid:
            print(f"step={step:,} eval_reward={eval_reward:.4f} sps={sps:.1f}")
        elif train_valid:
            print(f"step={step:,} train_reward={train_reward:.4f} sps={sps:.1f}")
        else:
            print(f"step={step:,} train_reward=invalid eval_reward=invalid sps={sps:.1f}")

        if (
            latest_params["value"] is not None
            and args.checkpoint_interval > 0
            and step - last_checkpoint_step["step"] >= args.checkpoint_interval
        ):
            ckpt_path = os.path.join(checkpoint_dir, f"params_step_{step}.msgpack")
            model.save_params(ckpt_path, latest_params["value"])
            last_checkpoint_step["step"] = step
            print(f"checkpoint_saved={ckpt_path}")

    def policy_params_fn(current_step, make_policy, params):
        del make_policy, current_step
        latest_params["value"] = params

    make_inference_fn, params, metrics = ppo_train.train(
        environment=env,
        eval_env=eval_env,
        num_timesteps=args.num_timesteps,
        episode_length=args.episode_length,
        num_envs=args.num_envs,
        num_evals=args.num_evals,
        learning_rate=args.learning_rate,
        entropy_cost=args.entropy_cost,
        discounting=args.discounting,
        unroll_length=args.unroll_length,
        batch_size=args.batch_size,
        num_minibatches=args.num_minibatches,
        num_updates_per_batch=args.num_updates_per_batch,
        gae_lambda=args.gae_lambda,
        reward_scaling=args.reward_scaling,
        network_factory=partial(
            ppo_networks.make_ppo_networks,
            policy_hidden_layer_sizes=policy_hidden_sizes,
            value_hidden_layer_sizes=value_hidden_sizes,
            activation=activation_fn,
        ),
        normalize_observations=True,
        normalize_advantage=True,
        seed=args.seed,
        progress_fn=progress,
        policy_params_fn=policy_params_fn,
        save_checkpoint_path=checkpoint_dir,
        restore_checkpoint_path=args.restore_checkpoint_path,
        restore_params=restore_params,
        restore_value_fn=args.restore_value_fn,
    )

    del make_inference_fn

    params_path = os.path.join(run_dir, "ppo_params.msgpack")
    model.save_params(params_path, params)

    summary = {
        "run_dir": run_dir,
        "xml": args.xml,
        "env": args.env,
        "backend": args.backend,
        "impl": args.impl,
        "traj_duration_seconds": args.traj_duration_seconds,
        "num_timesteps": args.num_timesteps,
        "episode_length": args.episode_length,
        "num_envs": args.num_envs,
        "num_evals": args.num_evals,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "entropy_cost": args.entropy_cost,
        "discounting": args.discounting,
        "action_min": args.action_min,
        "action_max": args.action_max,
        "unroll_length": args.unroll_length,
        "batch_size": args.batch_size,
        "num_minibatches": args.num_minibatches,
        "num_updates_per_batch": args.num_updates_per_batch,
        "gae_lambda": args.gae_lambda,
        "reward_scaling": args.reward_scaling,
        "policy_hidden_sizes": list(policy_hidden_sizes),
        "value_hidden_sizes": list(value_hidden_sizes),
        "activation": args.activation,
        "checkpoint_interval": args.checkpoint_interval,
        "checkpoint_dir": checkpoint_dir,
        "restore_checkpoint_path": args.restore_checkpoint_path,
        "restore_value_fn": args.restore_value_fn,
        "elapsed_sec": time.time() - wall_start,
        "final_metrics": {k: float(v) if hasattr(v, "__float__") else str(v) for k, v in metrics.items()},
        "params_path": params_path,
    }

    summary_path = os.path.join(run_dir, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved parameters: {params_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()