"""Evaluation and visualization script for trained hover policy."""

import argparse
import time
import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO

from envs import HoverEnv


def evaluate(model_path: str, num_episodes: int = 5, render: bool = True):
    """Evaluate a trained policy.

    Args:
        model_path: Path to the trained model (.zip file)
        num_episodes: Number of episodes to run
        render: Whether to render visualization
    """
    # Load model
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path, device="cpu")

    # Create environment
    env = HoverEnv()

    episode_rewards = []
    episode_lengths = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        step_count = 0
        done = False

        # Setup viewer for this episode
        viewer = None
        if render:
            viewer = mujoco.viewer.launch_passive(env.model, env.data)

        print(f"\n--- Episode {ep + 1}/{num_episodes} ---")

        while not done:
            # Get action from policy
            action, _ = model.predict(obs, deterministic=True)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            step_count += 1

            # Render
            if render and viewer is not None and viewer.is_running():
                viewer.sync()
                # Sync with real time
                time.sleep(env.dt * env.frame_skip)
            elif render and viewer is not None and not viewer.is_running():
                break

            # Print status periodically
            if step_count % 100 == 0:
                state = info["state"]
                print(f"  Step {step_count}: pos=[{state[0]:.2f}, {state[1]:.2f}, {state[2]:.2f}], "
                      f"reward={reward:.2f}")

        if viewer is not None:
            viewer.close()

        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)

        status = "TERMINATED" if terminated else "TRUNCATED (max steps)"
        print(f"  {status} after {step_count} steps, total reward: {total_reward:.2f}")

    env.close()

    # Print summary
    print("\n=== Evaluation Summary ===")
    print(f"Episodes: {num_episodes}")
    print(f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"Mean length: {np.mean(episode_lengths):.1f} +/- {np.std(episode_lengths):.1f}")


def interactive_control(render: bool = True):
    """Run environment with hover thrust (no learned policy).

    Useful for testing that the simulation works correctly.
    """
    env = HoverEnv()
    obs, info = env.reset()

    # Calculate hover thrust
    # Total mass = 0.2 (core) + 4*0.025 (arms) + 4*0.025 (thrusters) = 0.4 kg
    # Hover thrust per motor = (mass * g) / (4 * gear_ratio) = (0.4 * 9.81) / (4 * 2) = 0.49
    # In [-1, 1] space: 0.49 -> 2*0.49 - 1 = -0.02
    hover_thrust_normalized = 2 * 0.49 - 1  # Convert [0,1] to [-1,1]

    viewer = None
    if render:
        viewer = mujoco.viewer.launch_passive(env.model, env.data)

    print("Running with constant hover thrust...")
    print(f"Hover thrust (normalized): {hover_thrust_normalized:.3f}")
    print("Press Ctrl+C to stop")

    try:
        step = 0
        while True:
            # Constant hover action: [thrust, roll_rate, pitch_rate, yaw_rate]
            # All in [-1, 1] space
            action = np.array([hover_thrust_normalized, 0.0, 0.0, 0.0], dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action)

            if render and viewer is not None:
                if not viewer.is_running():
                    break
                viewer.sync()
                time.sleep(env.dt * env.frame_skip)

            step += 1
            if step % 100 == 0:
                state = info["state"]
                print(f"Step {step}: z={state[2]:.3f}, reward={reward:.2f}")

            if terminated or truncated:
                print(f"Episode ended after {step} steps")
                obs, info = env.reset()
                step = 0

    except KeyboardInterrupt:
        print("\nStopped by user")

    if viewer is not None:
        viewer.close()
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate or test hover policy")
    parser.add_argument(
        "--model",
        type=str,
        default="./models_trained/20260208_215818/best_model.zip",
        help="Path to trained model"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable rendering"
    )
    parser.add_argument(
        "--test-hover",
        action="store_true",
        help="Test with constant hover thrust (no learned policy)"
    )

    args = parser.parse_args()

    if args.test_hover:
        interactive_control(render=not args.no_render)
    else:
        evaluate(args.model, args.episodes, render=not args.no_render)


if __name__ == "__main__":
    main()
