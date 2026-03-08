#!/usr/bin/env python3
"""Optuna + ASHA tuning for Brax PPO hover training.

Runs multi-fidelity rungs per trial by progressively increasing training timesteps
and optionally restoring checkpoints between rungs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import optuna
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler


def _parse_timesteps_list(text: str) -> list[int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("--rung-timesteps cannot be empty")
    if any(v <= 0 for v in vals):
        raise ValueError("all rung timesteps must be positive")
    return vals


def _find_single_summary(output_root: Path) -> Path:
    summaries = sorted(output_root.glob("*/training_summary.json"))
    if not summaries:
        raise FileNotFoundError(f"No training_summary.json under {output_root}")
    return summaries[-1]


def _metric_float(d: dict[str, Any], keys: list[str], default: float = float("nan")) -> float:
    metrics = d.get("final_metrics", {})
    for key in keys:
        try:
            return float(metrics[key])
        except Exception:
            continue
    return default


@dataclass
class TuneConfig:
    python_bin: str
    project_dir: Path
    output_root: Path
    rung_timesteps: list[int]
    num_envs: int
    episode_length: int
    num_evals: int
    backend: str
    seed: int


def _sample_params(trial: optuna.Trial) -> dict[str, Any]:
    lr = trial.suggest_float("learning_rate", 1e-4, 8e-4, log=True)
    ent = trial.suggest_float("entropy_cost", 3e-4, 3e-3, log=True)
    batch = trial.suggest_categorical("batch_size", [512, 1024, 2048])
    unroll = trial.suggest_categorical("unroll_length", [8, 10, 16, 20])
    gae = trial.suggest_float("gae_lambda", 0.92, 0.98)
    updates = trial.suggest_categorical("num_updates_per_batch", [4, 6, 8])
    discount = trial.suggest_float("discounting", 0.985, 0.995)

    return {
        "learning_rate": lr,
        "entropy_cost": ent,
        "batch_size": batch,
        "unroll_length": unroll,
        "gae_lambda": gae,
        "num_updates_per_batch": updates,
        "discounting": discount,
    }


def _run_training_rung(
    cfg: TuneConfig,
    trial_number: int,
    rung_idx: int,
    timesteps: int,
    params: dict[str, Any],
    restore_ckpt: str | None,
) -> tuple[float, dict[str, Any], str | None]:
    rung_out = cfg.output_root / f"trial_{trial_number:04d}" / f"rung_{rung_idx:02d}"
    rung_out.mkdir(parents=True, exist_ok=True)

    cmd = [
        cfg.python_bin,
        "-u",
        "train_brax_ppo.py",
        "--env",
        "hover",
        "--backend",
        cfg.backend,
        "--num-timesteps",
        str(timesteps),
        "--episode-length",
        str(cfg.episode_length),
        "--num-envs",
        str(cfg.num_envs),
        "--num-evals",
        str(cfg.num_evals),
        "--seed",
        str(cfg.seed),
        "--learning-rate",
        str(params["learning_rate"]),
        "--entropy-cost",
        str(params["entropy_cost"]),
        "--batch-size",
        str(params["batch_size"]),
        "--unroll-length",
        str(params["unroll_length"]),
        "--gae-lambda",
        str(params["gae_lambda"]),
        "--num-updates-per-batch",
        str(params["num_updates_per_batch"]),
        "--discounting",
        str(params["discounting"]),
        "--checkpoint-interval",
        str(max(100000, timesteps // 2)),
        "--output-dir",
        str(rung_out),
    ]

    if restore_ckpt:
        cmd.extend(["--restore-checkpoint-path", restore_ckpt, "--restore-value-fn"])

    env = {**os.environ, "MUJOCO_GL": "egl", "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "TF_FORCE_GPU_ALLOW_GROWTH": "true"}
    proc = subprocess.run(cmd, cwd=str(cfg.project_dir), env=env, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"trial {trial_number} rung {rung_idx} failed with code {proc.returncode}")

    summary_path = _find_single_summary(rung_out)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    eval_reward = _metric_float(
        summary,
        ["eval/episode_reward", "eval/sum_reward", "eval/reward"],
    )

    checkpoint_dir = summary.get("checkpoint_dir")
    return eval_reward, summary, checkpoint_dir


def _build_objective(cfg: TuneConfig):
    def objective(trial: optuna.Trial) -> float:
        params = _sample_params(trial)
        restore_ckpt = None
        last_score = float("-inf")

        for rung_idx, timesteps in enumerate(cfg.rung_timesteps, start=1):
            score, summary, checkpoint_dir = _run_training_rung(
                cfg=cfg,
                trial_number=trial.number,
                rung_idx=rung_idx,
                timesteps=timesteps,
                params=params,
                restore_ckpt=restore_ckpt,
            )

            if not math.isfinite(score):
                score = -1e9

            trial.report(score, step=rung_idx)
            last_score = score
            restore_ckpt = checkpoint_dir

            trial.set_user_attr(f"rung_{rung_idx}_timesteps", timesteps)
            trial.set_user_attr(f"rung_{rung_idx}_eval_reward", score)
            trial.set_user_attr(f"rung_{rung_idx}_run_dir", summary.get("run_dir"))

            if trial.should_prune():
                raise optuna.TrialPruned()

        return last_score

    return objective


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna + ASHA tuning for train_brax_ppo.py")
    parser.add_argument("--python-bin", type=str, default="/work3/s212645/mujoco_playground/.venv/bin/python")
    parser.add_argument("--project-dir", type=str, default="/work3/s212645/PhD_Project/uav_reinforcement_learning_control")
    parser.add_argument("--output-root", type=str, default="/work3/s212645/PhD_Project/uav_reinforcement_learning_control/models_brax_optuna")
    parser.add_argument("--study-name", type=str, default="hover_ppo_optuna_asha")
    parser.add_argument("--storage", type=str, default="sqlite:////work3/s212645/PhD_Project/uav_reinforcement_learning_control/optuna_results/hover_ppo_optuna.db")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=0, help="seconds, 0 means no timeout")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", type=str, default="mjx")
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--episode-length", type=int, default=500)
    parser.add_argument("--num-evals", type=int, default=8)
    parser.add_argument("--rung-timesteps", type=str, default="300000,800000,1500000")
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    optuna_db_path = args.storage.replace("sqlite:///", "")
    if optuna_db_path.startswith("/"):
        Path(optuna_db_path).parent.mkdir(parents=True, exist_ok=True)

    cfg = TuneConfig(
        python_bin=args.python_bin,
        project_dir=project_dir,
        output_root=output_root,
        rung_timesteps=_parse_timesteps_list(args.rung_timesteps),
        num_envs=args.num_envs,
        episode_length=args.episode_length,
        num_evals=args.num_evals,
        backend=args.backend,
        seed=args.seed,
    )

    sampler = TPESampler(seed=args.seed, multivariate=True, group=True)
    pruner = SuccessiveHalvingPruner(min_resource=1, reduction_factor=2, min_early_stopping_rate=0)

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=args.storage,
        load_if_exists=True,
    )

    objective = _build_objective(cfg)
    timeout = None if args.timeout <= 0 else args.timeout
    study.optimize(objective, n_trials=args.n_trials, timeout=timeout)

    best = study.best_trial
    print("\n=== BEST TRIAL ===")
    print(f"number={best.number}")
    print(f"value={best.value}")
    print("params=")
    for key, value in best.params.items():
        print(f"  {key}: {value}")

    print("\nRecommended final 5e6 command:")
    print(
        f"{args.python_bin} -u train_brax_ppo.py "
        f"--env hover --backend {args.backend} --num-timesteps 5000000 "
        f"--episode-length {args.episode_length} --num-envs {args.num_envs} --num-evals 10 "
        f"--learning-rate {best.params['learning_rate']} --entropy-cost {best.params['entropy_cost']} "
        f"--batch-size {best.params['batch_size']} --unroll-length {best.params['unroll_length']} "
        f"--gae-lambda {best.params['gae_lambda']} --num-updates-per-batch {best.params['num_updates_per_batch']} "
        f"--discounting {best.params['discounting']} --checkpoint-interval 500000 --output-dir models_brax"
    )


if __name__ == "__main__":
    main()
