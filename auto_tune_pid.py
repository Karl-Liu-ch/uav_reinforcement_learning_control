"""Improved auto-tuning script for PID controller using Ziegler-Nichols inspired method.

Strategy:
1. Start with conservative gains
2. Measure oscillation frequency and magnitude
3. Adjust gains using Ziegler-Nichols rules
4. Fine-tune based on different performance metrics
5. Save best configuration

Usage:
    python auto_tune_pid.py --episodes 5 --iterations 15
"""

import argparse
import json
import os
import numpy as np
from collections import deque
import copy

from envs import HoverEnv
from envs.trajectory_follow_env import TrajectoryFollowEnv
from pid_controller_world_frame import CascadedPIDController, angle_diff
from utils.drone_config import MASS, G, DT

_GAINS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pid_gains.json")


def run_episode(controller, env, max_steps=1000):
    """Run one episode and collect detailed metrics."""
    obs, info = env.reset()
    controller.reset()
    
    metrics = {
        "times": [],
        "yaws": [],
        "yaw_rates": [],
        "positions": [],
        "pos_errors": [],
        "roll": [],
        "pitch": [],
        "rewards": [],
    }
    
    state = info["state"]
    tgt_pos = info.get("target", info.get("target_pos", np.zeros(3)))
    tgt_vel = info.get("target_vel", np.zeros(3))
    tgt_acc = info.get("target_acc", np.zeros(3))
    
    done = False
    step_count = 0
    
    while not done and step_count < max_steps:
        state = info["state"]
        tgt_pos = info.get("target", info.get("target_pos", np.zeros(3)))
        tgt_vel = info.get("target_vel", np.zeros(3))
        tgt_acc = info.get("target_acc", np.zeros(3))
        
        traj = (tgt_pos, tgt_vel, tgt_acc)
        action, diag = controller.compute(state, traj)
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1
        
        s = info["state"]
        drone_pos = s[:3]
        roll, pitch, yaw = s[3:6]
        yaw_rate = s[11]
        
        pos_error = float(np.linalg.norm(drone_pos - np.asarray(tgt_pos)))
        
        metrics["times"].append(step_count * DT)
        metrics["yaws"].append(yaw)
        metrics["yaw_rates"].append(yaw_rate)
        metrics["positions"].append(drone_pos.copy())
        metrics["pos_errors"].append(pos_error)
        metrics["roll"].append(roll)
        metrics["pitch"].append(pitch)
        metrics["rewards"].append(reward)
    
    return metrics


def analyze_yaw_oscillation(metrics: dict) -> dict:
    """Analyze yaw oscillation characteristics."""
    yaw_rates = np.array(metrics["yaw_rates"])
    yaws = np.array(metrics["yaws"])
    
    # Rate statistics
    rate_mean = np.mean(np.abs(yaw_rates))
    rate_std = np.std(yaw_rates)
    rate_max = np.max(np.abs(yaw_rates))
    
    # Detect oscillation by counting zero crossings
    rate_sign_changes = np.sum(np.diff(np.sign(yaw_rates)) != 0)
    oscillation_freq = rate_sign_changes / (len(yaw_rates) * DT)  # Hz
    
    # Energy in oscillation (integral of rate variance)
    oscillation_energy = np.sum(np.abs(np.diff(yaw_rates)))
    
    return {
        "rate_mean": rate_mean,
        "rate_std": rate_std,
        "rate_max": rate_max,
        "zero_crossings": rate_sign_changes,
        "oscillation_freq": oscillation_freq,
        "oscillation_energy": oscillation_energy,
    }


def analyze_position_tracking(metrics: dict) -> dict:
    """Analyze position tracking performance."""
    pos_errors = np.array(metrics["pos_errors"])
    rewards = np.array(metrics["rewards"])
    
    return {
        "pos_error_mean": np.mean(pos_errors),
        "pos_error_std": np.std(pos_errors),
        "pos_error_max": np.max(pos_errors),
        "steady_state_error": np.mean(pos_errors[-100:]) if len(pos_errors) > 100 else np.mean(pos_errors),
        "total_reward": np.sum(rewards),
        "episode_length": len(metrics["times"]),
    }


def compute_performance_score(yaw_analysis: dict, pos_analysis: dict) -> dict:
    """Compute detailed performance metrics."""
    # Yaw quality score: low oscillation, reasonable rate
    yaw_score = 0.0
    
    # Penalize high oscillation energy (should be < 0.5)
    osc_penalty = min(1.0, yaw_analysis["oscillation_energy"] / 1.0)
    yaw_score += 0.5 * (1.0 - osc_penalty)
    
    # Penalize high max rate (should be < 2 rad/s)
    rate_penalty = min(1.0, yaw_analysis["rate_max"] / 3.0)
    yaw_score += 0.3 * (1.0 - rate_penalty)
    
    # Reward low frequency oscillation (smooth)
    freq_bonus = max(0, 1.0 - yaw_analysis["oscillation_freq"] / 5.0)
    yaw_score += 0.2 * freq_bonus
    
    # Position score: low error
    pos_score = 1.0 / (1.0 + pos_analysis["pos_error_mean"])
    
    # Combined score
    total_score = 0.6 * yaw_score + 0.4 * pos_score
    
    return {
        "yaw_score": yaw_score,
        "pos_score": pos_score,
        "total_score": total_score,
        "oscillation_energy": yaw_analysis["oscillation_energy"],
        "pos_error_mean": pos_analysis["pos_error_mean"],
        "rate_max": yaw_analysis["rate_max"],
    }


def adjust_gains_smart(gains: dict, analysis: dict, iteration: int) -> dict:
    """Smart gain adjustment based on Ziegler-Nichols principles."""
    new_gains = copy.deepcopy(gains)
    
    osc_energy = analysis["oscillation_energy"]
    rate_max = analysis["rate_max"]
    pos_error = analysis["pos_error_mean"]
    
    print(f"\n  Analysis: osc_energy={osc_energy:.2f}, rate_max={rate_max:.3f}rad/s, pos_err={pos_error:.3f}m")
    
    # Strategy 1: High oscillation energy -> reduce Kp and Kd proportionally
    if osc_energy > 1.0:
        reduction = 0.9
        print(f"  → High oscillation ({osc_energy:.2f}), reducing gains to {reduction:.2%}")
        new_gains["yaw"]["kp"] *= reduction
        new_gains["yaw"]["kd"] *= reduction
        new_gains["rate"]["ki_torque"] *= 0.95
        new_gains["limits"]["yaw_torque_scale"] *= reduction
    
    # Strategy 2: Moderate oscillation -> adjust damping
    elif osc_energy > 0.3:
        print(f"  → Moderate oscillation ({osc_energy:.2f}), increasing damping (Kd)")
        new_gains["yaw"]["kd"] *= 1.15  # Increase damping
        new_gains["yaw"]["kp"] *= 0.95  # Slight reduction in aggressiveness
    
    # Strategy 3: Low oscillation but sluggish response
    elif rate_max < 0.5 and pos_error > 0.15:
        print(f"  → Sluggish response, increasing gains")
        new_gains["yaw"]["kp"] *= 1.1
        new_gains["position_xy"]["kp"] *= 1.05
        new_gains["position_z"]["kp"] *= 1.05
    
    # Strategy 4: Optimize already good performance
    elif osc_energy < 0.2:
        if pos_error > 0.08:
            print(f"  → Good stability but large position error, increasing position gains")
            new_gains["position_xy"]["kp"] *= 1.05
            new_gains["position_z"]["kp"] *= 1.05
        else:
            print(f"  → Good performance, fine-tuning")
            # Iteratively optimize
            if iteration % 3 == 0:
                new_gains["yaw"]["kd"] *= 1.02
            else:
                new_gains["yaw"]["kp"] *= 1.02
    
    # Apply rate controller tuning
    if rate_max > 1.5:
        print(f"  → Rate too high, reducing rate I term")
        new_gains["rate"]["ki_torque"] *= 0.9
    
    # Clamp to prevent instability
    new_gains["yaw"]["kp"] = np.clip(new_gains["yaw"]["kp"], 10.0, 80.0)
    new_gains["yaw"]["kd"] = np.clip(new_gains["yaw"]["kd"], 5.0, 30.0)
    new_gains["limits"]["yaw_torque_scale"] = np.clip(new_gains["limits"]["yaw_torque_scale"], 0.2, 0.8)
    new_gains["rate"]["ki_torque"] = np.clip(new_gains["rate"]["ki_torque"], 0.01, 0.04)
    
    # Also adjust position gains based on error
    new_gains["position_xy"]["kp"] = np.clip(new_gains["position_xy"]["kp"], 1.5, 4.0)
    new_gains["position_z"]["kp"] = np.clip(new_gains["position_z"]["kp"], 2.0, 6.0)
    
    return new_gains


def auto_tune(num_episodes: int = 5, num_iterations: int = 15, use_trajectory_env: bool = False):
    """Auto-tune PID gains with smart iteration."""
    print("="*50)
    print("AUTO-TUNING PID CONTROLLER")
    print("="*50 + "\n")
    
    # Load base gains
    with open(_GAINS_PATH) as f:
        base_gains = json.load(f)
    current_gains = copy.deepcopy(base_gains)
    
    env = TrajectoryFollowEnv() if use_trajectory_env else HoverEnv()
    best_score = -np.inf
    best_gains = None
    best_analysis = None
    score_history = []
    
    for iteration in range(num_iterations):
        print(f"\n{'='*50}")
        print(f"ITERATION {iteration + 1}/{num_iterations}")
        print(f"{'='*50}")
        print(f"Current Yaw: Kp={current_gains['yaw']['kp']:.1f}, "
              f"Kd={current_gains['yaw']['kd']:.1f}, "
              f"τ_scale={current_gains['limits']['yaw_torque_scale']:.2f}")
        print(f"Current Pos XY: Kp={current_gains['position_xy']['kp']:.1f}, "
              f"Kd={current_gains['position_xy']['kd']:.1f}")
        
        # Run episodes
        iteration_scores = []
        all_yaw_analyses = []
        all_pos_analyses = []
        
        for ep in range(num_episodes):
            controller = CascadedPIDController(current_gains)
            metrics = run_episode(controller, env, max_steps=1000)
            
            yaw_analysis = analyze_yaw_oscillation(metrics)
            pos_analysis = analyze_position_tracking(metrics)
            perf = compute_performance_score(yaw_analysis, pos_analysis)
            
            iteration_scores.append(perf["total_score"])
            all_yaw_analyses.append(yaw_analysis)
            all_pos_analyses.append(pos_analysis)
            
            print(f"  Ep {ep+1}: score={perf['total_score']:.3f} | "
                  f"osc={perf['oscillation_energy']:.2f} | "
                  f"pos_err={perf['pos_error_mean']:.3f}m | "
                  f"rate_max={perf['rate_max']:.2f}rad/s")
        
        # Average metrics
        avg_score = np.mean(iteration_scores)
        avg_yaw_analysis = {k: np.mean([a[k] for a in all_yaw_analyses]) 
                           for k in all_yaw_analyses[0].keys()}
        avg_pos_analysis = {k: np.mean([a[k] for a in all_pos_analyses]) 
                           for k in all_pos_analyses[0].keys()}
        
        combined_analysis = {**avg_yaw_analysis, **avg_pos_analysis}
        
        print(f"\n  Iteration Average Score: {avg_score:.3f}")
        print(f"  Avg Osc Energy: {avg_yaw_analysis['oscillation_energy']:.2f}")
        print(f"  Avg Pos Error: {avg_pos_analysis['pos_error_mean']:.3f}m")
        score_history.append(avg_score)
        
        # Update best
        if avg_score > best_score:
            best_score = avg_score
            best_gains = copy.deepcopy(current_gains)
            best_analysis = combined_analysis
            print(f"  ✓✓✓ NEW BEST SCORE! {best_score:.3f} ✓✓✓")
            
            # Save immediately
            best_gains["tuning_notes"]["auto_tuning"] = {
                "iteration": iteration + 1,
                "score": float(best_score),
                "oscillation_energy": float(avg_yaw_analysis["oscillation_energy"]),
                "position_error_mean": float(avg_pos_analysis["pos_error_mean"]),
                "rate_max": float(avg_yaw_analysis["rate_max"]),
            }
            with open(_GAINS_PATH, "w") as f:
                json.dump(best_gains, f, indent=2)
            print(f"  Saved to {_GAINS_PATH}")
        else:
            print(f"  (Best so far: {best_score:.3f})")
        
        # Adjust for next iteration
        current_gains = adjust_gains_smart(current_gains, combined_analysis, iteration)
        
        # Early stopping if converged
        if len(score_history) >= 5:
            recent_scores = score_history[-5:]
            if np.std(recent_scores) < 0.01 and best_score > 0.6:
                print(f"\n  Converged! Stopping early.")
                break
    
    env.close()
    
    print(f"\n{'='*50}")
    print("TUNING COMPLETE")
    print(f"{'='*50}")
    print(f"Best Score: {best_score:.3f}")
    if best_analysis:
        print(f"Best Oscillation Energy: {best_analysis.get('oscillation_energy', 0):.2f}")
        print(f"Best Position Error: {best_analysis.get('pos_error_mean', 0):.3f}m")
        print(f"Best Yaw Rate Max: {best_analysis.get('rate_max', 0):.3f}rad/s")
    print(f"\nBest gains saved to {_GAINS_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Smart auto-tune PID controller")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Episodes per iteration (default: 5)")
    parser.add_argument("--iterations", type=int, default=15,
                        help="Number of tuning iterations (default: 15)")
    parser.add_argument("--traj", action="store_true",
                        help="Use trajectory-following environment")
    args = parser.parse_args()
    
    auto_tune(
        num_episodes=args.episodes,
        num_iterations=args.iterations,
        use_trajectory_env=args.traj,
    )


if __name__ == "__main__":
    main()
