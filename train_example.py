#!/usr/bin/env python3
"""
Example training script for JaxMJXQuadBraxEnv with mixing matrix control.

This is a minimal example showing how to train using the new control paradigm:
Agent outputs: [thrust, tau_x, tau_y, tau_z] → Mixing matrix → Motor commands

Usage:
    python train_example.py --num_timesteps 1000000 --episode_length 1000
"""

import os
import argparse
from train_brax_ppo import _default_xml_path

def main():
    parser = argparse.ArgumentParser(
        description="Train PPO agent on quadrotor with mixing matrix control"
    )
    
    # Environment settings
    parser.add_argument("--xml", type=str, default=None, 
                       help="Path to drone.xml (default: model/drone/drone.xml)")
    parser.add_argument("--env", type=str, default="jax_mjx_quad",
                       choices=["hover", "jax_mjx_quad"],
                       help="Environment type")
    parser.add_argument("--impl", type=str, default="jax",
                       choices=["jax", "gpu", "cuda", "cpu"],
                       help="MJX implementation")
    parser.add_argument("--episode_length", type=int, default=1000,
                       help="Maximum episode length")
    parser.add_argument("--traj_duration_seconds", type=float, default=10.0,
                       help="Trajectory duration in seconds")
    
    # Training hyperparameters
    parser.add_argument("--num_timesteps", type=int, default=10_000_000,
                       help="Total number of environment steps")
    parser.add_argument("--num_envs", type=int, default=2048,
                       help="Number of parallel environments")
    parser.add_argument("--num_evals", type=int, default=10,
                       help="Number of evaluation episodes")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed")
    
    # Network architecture
    parser.add_argument("--policy_hidden_sizes", type=str, default="256,256",
                       help="Policy network hidden layer sizes (comma-separated)")
    parser.add_argument("--value_hidden_sizes", type=str, default="256,256",
                       help="Value network hidden layer sizes (comma-separated)")
    parser.add_argument("--activation", type=str, default="swish",
                       choices=["swish", "relu", "tanh", "elu", "silu"],
                       help="Activation function")
    
    # Output settings
    parser.add_argument("--output_dir", type=str, default="./trained_models",
                       help="Output directory for checkpoints and logs")
    parser.add_argument("--checkpoint_interval", type=int, default=1_000_000,
                       help="Save checkpoint every N steps (0 to disable)")
    
    args = parser.parse_args()
    
    # Set default XML path if not provided
    if args.xml is None:
        args.xml = _default_xml_path()
    
    # Additional arguments with defaults
    args.backend = "mjx"
    args.action_min = 0.0
    args.action_max = 13.0
    args.entropy_cost = 1e-2
    args.discounting = 0.97
    args.unroll_length = 10
    args.batch_size = 1024
    args.num_minibatches = 32
    args.num_updates_per_batch = 8
    args.gae_lambda = 0.95
    args.reward_scaling = 1.0
    args.restore_checkpoint_path = None
    args.restore_value_fn = True
    
    print("=" * 70)
    print("Training Configuration")
    print("=" * 70)
    print(f"Environment: {args.env}")
    print(f"Control mode: Agent outputs [thrust, tau_x, tau_y, tau_z]")
    print(f"              → Mixing matrix → 4 motor commands")
    print(f"Episode length: {args.episode_length}")
    print(f"Trajectory duration: {args.traj_duration_seconds}s")
    print(f"Total timesteps: {args.num_timesteps:,}")
    print(f"Parallel envs: {args.num_envs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Policy hidden: {args.policy_hidden_sizes}")
    print(f"Value hidden: {args.value_hidden_sizes}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 70)
    print()
    
    # Import and run training
    from train_brax_ppo import main as train_main
    import sys
    
    # Override sys.argv with our parsed arguments
    sys.argv = ["train_brax_ppo.py"]
    for key, value in vars(args).items():
        sys.argv.extend([f"--{key}", str(value)])
    
    train_main()

if __name__ == "__main__":
    main()
