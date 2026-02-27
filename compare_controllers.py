#!/usr/bin/env python3
"""
Comparative evaluation of LQR vs SE(3) Geometric controllers.

This script runs both controllers on the same episodes and generates
performance comparison plots.

Usage:
    python compare_controllers.py --episodes 3 --plot
    python compare_controllers.py --episodes 5 --plot --traj
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Import both controllers
from lqr_controller_world_frame import CascadedPIDController
from se3_geometric_controller import SE3GeometricController
from envs import HoverEnv
from envs.trajectory_follow_env import TrajectoryFollowEnv
from utils.drone_config import DT


def evaluate_controller(controller, env, num_episodes: int = 3, 
                       controller_name: str = "Controller", plot_dir: str = None):
    """Evaluate a controller for multiple episodes without rendering.
    
    Args:
        controller: Controller instance with compute() method
        env: Environment instance
        num_episodes: Number of episodes to run
        controller_name: Name for logging
        plot_dir: Directory for saving data (optional)
    
    Returns:
        metrics: Dict with aggregated performance metrics
        all_episodes: List of episode data dicts
    """
    episode_rewards = []
    episode_lengths = []
    episode_errors = []
    episode_att_errors = []
    all_episodes = []
    
    print(f"\n{'='*60}")
    print(f"Evaluating {controller_name}")
    print(f"{'='*60}")
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        controller.reset()
        
        total_reward = 0.0
        step_count = 0
        done = False
        pos_errors = []
        att_errors = []
        
        ep_data = {
            "times": [],
            "positions": [],
            "targets": [],
            "velocities": [],
            "attitudes": [],
            "angular_velocities": [],
            "actions": [],
            "rewards": [],
            "desired_attitudes": [],
            "attitude_errors_so3": [],  # For SE(3)
        }
        
        state = info["state"]
        tgt_pos = info.get("target", info.get("target_pos", np.zeros(3)))
        tgt_vel = info.get("target_vel", np.zeros(3))
        tgt_acc = info.get("target_acc", np.zeros(3))
        
        print(f"  Episode {ep + 1}/{num_episodes}: ", end="", flush=True)
        
        while not done:
            state = info["state"]
            tgt_pos = info.get("target", info.get("target_pos", np.zeros(3)))
            tgt_vel = info.get("target_vel", np.zeros(3))
            tgt_acc = info.get("target_acc", np.zeros(3))
            traj = (tgt_pos, tgt_vel, tgt_acc)
            
            action, diag = controller.compute(state, traj)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step_count += 1
            
            s = info["state"]
            drone_pos = s[:3]
            pos_err = float(np.linalg.norm(drone_pos - np.asarray(tgt_pos)))
            pos_errors.append(pos_err)
            
            # Collect attitude error
            if "attitude_error" in diag:
                att_errors.append(diag["attitude_error"])
            
            if plot_dir:
                ep_data["times"].append(step_count * DT)
                ep_data["positions"].append(s[:3].copy())
                ep_data["targets"].append(np.asarray(tgt_pos).copy())
                ep_data["attitudes"].append(s[3:6].copy())
                ep_data["velocities"].append(s[6:9].copy())
                ep_data["angular_velocities"].append(s[9:12].copy())
                ep_data["actions"].append(action.copy())
                ep_data["rewards"].append(reward)
                ep_data["desired_attitudes"].append(diag["des_att"])
                if "attitude_error" in diag:
                    ep_data["attitude_errors_so3"].append(diag["attitude_error"])
        
        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)
        episode_errors.append(np.mean(pos_errors))
        if att_errors:
            episode_att_errors.append(np.mean(att_errors))
        
        status = "✓" if terminated else "⚠"
        print(f"{status} Steps: {step_count:4d} | Reward: {total_reward:7.2f} | "
              f"Pos Error: {np.mean(pos_errors):6.3f}m")
        
        if plot_dir:
            all_episodes.append(ep_data)
    
    env.close()
    
    metrics = {
        "name": controller_name,
        "num_episodes": num_episodes,
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "mean_pos_error": np.mean(episode_errors),
        "std_pos_error": np.std(episode_errors),
        "mean_att_error": np.mean(episode_att_errors) if episode_att_errors else 0.0,
        "survival_rate": 100 * sum(1 for l in episode_lengths if l >= 512) / num_episodes,
    }
    
    return metrics, all_episodes


def plot_comparison(metrics_lqr: dict, metrics_se3: dict, save_dir: str = None):
    """Create comparison plots between two controllers.
    
    Args:
        metrics_lqr: Metrics dict from LQR controller
        metrics_se3: Metrics dict from SE(3) controller
        save_dir: Directory to save plots (optional)
    """
    if save_dir is None:
        save_dir = "./plots/comparison"
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("LQR PID vs SE(3) Geometric Controller Comparison", fontsize=14, weight="bold")
    
    controllers = ["LQR PID", "SE(3)"]
    colors = ["C0", "C1"]
    
    # 1. Mean reward
    ax = axes[0, 0]
    means = [metrics_lqr["mean_reward"], metrics_se3["mean_reward"]]
    stds = [metrics_lqr["std_reward"], metrics_se3["std_reward"]]
    bars = ax.bar(controllers, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
    ax.set_ylabel("Reward")
    ax.set_title("Mean Episode Reward")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f"{mean:.1f}", ha="center", va="bottom")
    
    # 2. Episode length
    ax = axes[0, 1]
    means = [metrics_lqr["mean_length"], metrics_se3["mean_length"]]
    stds = [metrics_lqr["std_length"], metrics_se3["std_length"]]
    bars = ax.bar(controllers, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
    ax.set_ylabel("Steps")
    ax.set_title("Mean Episode Length")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f"{int(mean):d}", ha="center", va="bottom")
    
    # 3. Position error
    ax = axes[0, 2]
    means = [metrics_lqr["mean_pos_error"], metrics_se3["mean_pos_error"]]
    stds = [metrics_lqr["std_pos_error"], metrics_se3["std_pos_error"]]
    bars = ax.bar(controllers, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
    ax.set_ylabel("Error (m)")
    ax.set_title("Mean Position Error")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f"{mean:.3f}", ha="center", va="bottom")
    
    # 4. Survival rate
    ax = axes[1, 0]
    means = [metrics_lqr["survival_rate"], metrics_se3["survival_rate"]]
    bars = ax.bar(controllers, means, color=colors, alpha=0.7)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Episode Survival Rate (≥512 steps)")
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3, axis="y")
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f"{mean:.0f}%", ha="center", va="bottom")
    
    # 5. Attitude error (if available)
    ax = axes[1, 1]
    means = [metrics_lqr["mean_att_error"], metrics_se3["mean_att_error"]]
    # Only plot if both have data
    if any(m > 0 for m in means):
        if metrics_lqr["mean_att_error"] == 0:
            means[0] = metrics_se3["mean_att_error"]  # Use SE(3) scale
        bars = ax.bar(controllers, means, color=colors, alpha=0.7)
        ax.set_ylabel("Error magnitude")
        ax.set_title("Mean Attitude Error")
        ax.grid(True, alpha=0.3, axis="y")
        for bar, mean in zip(bars, means):
            if mean > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f"{mean:.4f}", ha="center", va="bottom")
    else:
        ax.text(0.5, 0.5, "Attitude error data\nnot available",
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
    
    # 6. Summary table
    ax = axes[1, 2]
    ax.axis("off")
    
    summary_text = (
        "Evaluation Summary\n"
        f"Episodes: {int(metrics_lqr['num_episodes'])}\n\n"
        "LQR PID (Cascade):\n"
        f"  • Uses Euler angles\n"
        f"  • Local stability\n"
        f"  • Gimbal lock risk\n\n"
        "SE(3) Geometric:\n"
        f"  • Uses SO(3) matrices\n"
        f"  • Global stability\n"
        f"  • No singularities"
    )
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment="top", family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))
    
    fig.tight_layout()
    filepath = os.path.join(save_dir, "controller_comparison.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    print(f"\n✓ Comparison plot saved to {filepath}")
    plt.close(fig)


def plot_episode_comparison(ep_data_lqr: dict, ep_data_se3: dict, 
                           episode_num: int, save_dir: str = None):
    """Create detailed comparison plots for a single episode.
    
    Args:
        ep_data_lqr: Episode data from LQR controller
        ep_data_se3: Episode data from SE(3) controller
        episode_num: Episode number for title
        save_dir: Directory to save plots
    """
    if save_dir is None:
        save_dir = "./plots/comparison"
    os.makedirs(save_dir, exist_ok=True)
    
    t_lqr = np.array(ep_data_lqr["times"])
    t_se3 = np.array(ep_data_se3["times"])
    
    pos_lqr = np.array(ep_data_lqr["positions"])
    pos_se3 = np.array(ep_data_se3["positions"])
    tgt_lqr = np.array(ep_data_lqr["targets"])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Episode {episode_num}: Detailed Comparison", fontsize=12, weight="bold")
    
    # Position X
    ax = axes[0, 0]
    ax.plot(t_lqr, pos_lqr[:, 0], "o-", label="LQR X", alpha=0.7, mfc="none")
    ax.plot(t_se3, pos_se3[:, 0], "s-", label="SE(3) X", alpha=0.7, mfc="none")
    ax.plot(t_lqr, tgt_lqr[:, 0], "k--", label="Target X", alpha=0.5)
    ax.set_ylabel("Position (m)")
    ax.set_title("X Position")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Position Y
    ax = axes[0, 1]
    ax.plot(t_lqr, pos_lqr[:, 1], "o-", label="LQR Y", alpha=0.7, mfc="none")
    ax.plot(t_se3, pos_se3[:, 1], "s-", label="SE(3) Y", alpha=0.7, mfc="none")
    ax.plot(t_lqr, tgt_lqr[:, 1], "k--", label="Target Y", alpha=0.5)
    ax.set_ylabel("Position (m)")
    ax.set_title("Y Position")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Position error
    ax = axes[1, 0]
    err_lqr = np.linalg.norm(pos_lqr - tgt_lqr, axis=1)
    if len(pos_se3) == len(tgt_lqr):
        err_se3 = np.linalg.norm(pos_se3 - tgt_lqr, axis=1)
        ax.plot(t_se3, err_se3, "s-", label="SE(3)", alpha=0.7, mfc="none")
    ax.plot(t_lqr, err_lqr, "o-", label="LQR", alpha=0.7, mfc="none")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error (m)")
    ax.set_title("Euclidean Position Error")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Rewards comparison
    ax = axes[1, 1]
    rewards_lqr = np.array(ep_data_lqr["rewards"])
    rewards_se3 = np.array(ep_data_se3["rewards"])
    cum_lqr = np.cumsum(rewards_lqr)
    cum_se3 = np.cumsum(rewards_se3)
    ax.plot(t_lqr, cum_lqr, "o-", label="LQR cumulative", alpha=0.7, mfc="none")
    ax.plot(t_se3, cum_se3, "s-", label="SE(3) cumulative", alpha=0.7, mfc="none")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Reward Accumulation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    filepath = os.path.join(save_dir, f"episode_{episode_num}_comparison.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    print(f"✓ Episode {episode_num} comparison saved to {filepath}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Comparative evaluation of LQR PID vs SE(3) Geometric controllers"
    )
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of test episodes")
    parser.add_argument("--plot", action="store_true",
                        help="Generate comparison plots")
    parser.add_argument("--gains", type=str, default=None,
                        help="Path to custom gains JSON file")
    parser.add_argument("--traj", action="store_true",
                        help="Use trajectory-following environment")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()
    
    # Set seed for reproducibility
    np.random.seed(args.seed)
    
    # Load gains
    gains = None
    if args.gains and os.path.exists(args.gains):
        with open(args.gains) as f:
            gains = json.load(f)
        print(f"Loaded custom gains from {args.gains}")
    
    # Create comparison output directory
    compare_dir = "./plots/comparison"
    os.makedirs(compare_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("SE(3) Geometric Control vs LQR PID Controller Comparison")
    print("="*70)
    
    # ─────────────────────────────────────────
    # Evaluate LQR Controller
    # ─────────────────────────────────────────
    env_lqr = TrajectoryFollowEnv() if args.traj else HoverEnv()
    controller_lqr = CascadedPIDController(gains)
    
    metrics_lqr, data_lqr = evaluate_controller(
        controller_lqr, env_lqr,
        num_episodes=args.episodes,
        controller_name="LQR PID Controller (Cascaded with Euler Angles)",
        plot_dir=compare_dir if args.plot else None
    )
    
    # ─────────────────────────────────────────
    # Evaluate SE(3) Controller
    # ─────────────────────────────────────────
    env_se3 = TrajectoryFollowEnv() if args.traj else HoverEnv()
    controller_se3 = SE3GeometricController(gains)
    
    metrics_se3, data_se3 = evaluate_controller(
        controller_se3, env_se3,
        num_episodes=args.episodes,
        controller_name="SE(3) Geometric Controller (Rotation Matrices)",
        plot_dir=compare_dir if args.plot else None
    )
    
    # ─────────────────────────────────────────
    # Print Summary
    # ─────────────────────────────────────────
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)
    print(f"\n{'Metric':<30} {'LQR PID':<20} {'SE(3) Geom':<20} {'Difference':<15}")
    print("─" * 85)
    
    # Mean reward
    diff = metrics_se3["mean_reward"] - metrics_lqr["mean_reward"]
    arrow = "↑↑" if diff > 0 else "↓" if diff < 0 else "="
    print(f"{'Mean Reward':<30} {metrics_lqr['mean_reward']:<20.2f} "
          f"{metrics_se3['mean_reward']:<20.2f} {arrow} {diff:+.2f}")
    
    # Mean length
    diff = metrics_se3["mean_length"] - metrics_lqr["mean_length"]
    arrow = "↑↑" if diff > 10 else "↓" if diff < -10 else "="
    print(f"{'Mean Episode Length':<30} {metrics_lqr['mean_length']:<20.1f} "
          f"{metrics_se3['mean_length']:<20.1f} {arrow} {diff:+.1f}")
    
    # Position error
    diff = metrics_lqr["mean_pos_error"] - metrics_se3["mean_pos_error"]
    arrow = "↑↑" if diff > 0.01 else "↓" if diff < -0.01 else "="
    print(f"{'Mean Position Error (m)':<30} {metrics_lqr['mean_pos_error']:<20.4f} "
          f"{metrics_se3['mean_pos_error']:<20.4f} {arrow} {diff:+.4f}")
    
    # Survival rate
    diff = metrics_se3["survival_rate"] - metrics_lqr["survival_rate"]
    arrow = "↑↑" if diff > 5 else "↓" if diff < -5 else "="
    print(f"{'Survival Rate (%)':<30} {metrics_lqr['survival_rate']:<20.1f} "
          f"{metrics_se3['survival_rate']:<20.1f} {arrow} {diff:+.1f}%")
    
    print("─" * 85)
    
    # Generate comparison plots if requested
    if args.plot:
        print("\nGenerating comparison plots...")
        plot_comparison(metrics_lqr, metrics_se3, save_dir=compare_dir)
        
        # Plot individual episode comparisons
        for i in range(min(len(data_lqr), len(data_se3))):
            plot_episode_comparison(data_lqr[i], data_se3[i], i + 1, save_dir=compare_dir)
        
        print(f"\n✓ All comparison plots saved to {compare_dir}/")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
