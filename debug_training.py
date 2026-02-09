"""Debug script to diagnose why episodes are terminating early."""

import numpy as np
from stable_baselines3 import PPO
from envs import HoverEnv


def diagnose_episode(env, model=None, use_random=False, n_steps=20):
    """Run a single episode and diagnose what's happening."""
    obs, info = env.reset()
    print(f"\n{'='*70}")
    print(f"EPISODE DIAGNOSIS ({'Random actions' if use_random else 'Trained policy'})")
    print(f"{'='*70}")
    print(f"Initial state: pos={info['state'][:3]}, vel={info['state'][6:9]}")
    print(f"Target state:  pos={env.target_state.position}")
    print(f"Initial obs range: [{obs.min():.3f}, {obs.max():.3f}]")
    print()

    termination_reasons = []

    for step in range(n_steps):
        if use_random:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        state = info['state']
        pos = state[:3]
        att = state[3:6]
        vel = state[6:9]
        motors = info['motor_commands']

        # Check for issues
        issues = []
        if not np.isfinite(state).all():
            issues.append("NaN/Inf detected")
        if not env._obs_bounds.contains(state):
            issues.append("Out of bounds")
            # Detail which dimension is out of bounds
            for i, (val, low, high) in enumerate(zip(state, env._obs_bounds.low, env._obs_bounds.high)):
                if val < low or val > high:
                    labels = ['x','y','z','r','p','y','vx','vy','vz','wx','wy','wz']
                    issues.append(f"  {labels[i]}={val:.3f} not in [{low:.3f}, {high:.3f}]")

        print(f"Step {step:2d}: reward={reward:7.3f}, pos={pos}, vel={vel}")
        print(f"         motors={motors}, obs=[{obs.min():.2f}, {obs.max():.2f}]")
        if issues:
            print(f"         ⚠️  ISSUES: {', '.join(issues)}")

        if terminated:
            print(f"\n❌ Episode TERMINATED at step {step+1}")
            termination_reasons = issues
            break
        if truncated:
            print(f"\n⏱️  Episode TRUNCATED at step {step+1}")
            break

    return step + 1, termination_reasons


def main():
    print("\n" + "="*70)
    print("TRAINING DIAGNOSTICS")
    print("="*70)

    # Create environment
    env = HoverEnv()

    # Check observation bounds
    print("\nObservation bounds:")
    labels = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz']
    for i, label in enumerate(labels):
        low = env._obs_bounds.low[i]
        high = env._obs_bounds.high[i]
        print(f"  {label:5s}: [{low:7.3f}, {high:7.3f}]")

    # Test with random actions
    print("\n" + "-"*70)
    print("TEST 1: Random Actions")
    print("-"*70)
    steps, reasons = diagnose_episode(env, use_random=True, n_steps=10)
    print(f"Result: Episode lasted {steps} steps")
    if reasons:
        print(f"Termination reasons: {reasons}")

    # Test with trained model
    try:
        model_path = "./models_trained/best_model.zip"
        print(f"\n" + "-"*70)
        print("TEST 2: Trained Policy")
        print("-"*70)
        model = PPO.load(model_path)
        print(f"Loaded model from {model_path}")
        steps, reasons = diagnose_episode(env, model=model, use_random=False, n_steps=20)
        print(f"Result: Episode lasted {steps} steps")
        if reasons:
            print(f"Termination reasons: {reasons}")
    except Exception as e:
        print(f"Could not load trained model: {e}")

    # Run multiple episodes to get statistics
    print("\n" + "-"*70)
    print("TEST 3: Episode Length Statistics (Random Policy)")
    print("-"*70)
    episode_lengths = []
    for i in range(10):
        obs, _ = env.reset()
        for step in range(512):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        episode_lengths.append(step + 1)

    print(f"Episode lengths over 10 episodes: {episode_lengths}")
    print(f"Mean: {np.mean(episode_lengths):.1f}, Std: {np.std(episode_lengths):.1f}")
    print(f"Min: {np.min(episode_lengths)}, Max: {np.max(episode_lengths)}")

    # Check if it's a specific dimension causing early termination
    print("\n" + "-"*70)
    print("TEST 4: Which state dimension goes out of bounds first?")
    print("-"*70)
    obs, info = env.reset()
    for step in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        state = info['state']

        # Check each dimension
        for i, (val, low, high) in enumerate(zip(state, env._obs_bounds.low, env._obs_bounds.high)):
            if val < low or val > high:
                print(f"Step {step}: {labels[i]} = {val:.3f} is out of bounds [{low:.3f}, {high:.3f}]")

        if terminated:
            break

    env.close()
    print("\n" + "="*70)
    print("DIAGNOSTICS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()