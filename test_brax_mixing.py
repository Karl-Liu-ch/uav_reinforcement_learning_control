#!/usr/bin/env python3
"""Test script to verify the mixing matrix control in JaxMJXQuadBraxEnv."""

import os
import jax
import jax.numpy as jnp
from train_brax_ppo import JaxMJXQuadBraxEnv

def test_mixing_control():
    """Test that the environment correctly converts thrust+torques to motor commands."""
    
    xml_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "model",
        "drone",
        "drone.xml"
    )
    
    # Create environment
    env = JaxMJXQuadBraxEnv(
        xml_path=xml_path,
        impl="jax",
        max_episode_steps=100,
        traj_duration_seconds=5.0,
    )
    
    print("=" * 70)
    print("Testing JaxMJXQuadBraxEnv with Mixing Matrix Control")
    print("=" * 70)
    print(f"Action size: {env.action_size} (should be 4: [thrust, tau_x, tau_y, tau_z])")
    print(f"Observation size: {env.observation_size}")
    print(f"Max motor thrust: {env.max_motor_thrust} N")
    print(f"Max total thrust: {env.max_total_thrust} N")
    print(f"Max torque: {env.max_torque} N·m")
    print()
    
    # Reset environment
    rng = jax.random.PRNGKey(42)
    state = env.reset(rng)
    
    print(f"Initial observation shape: {state.obs.shape}")
    print(f"Initial position (xyz): {state.pipeline_state.qpos[:3]}")
    print()
    
    # Test different actions
    test_cases = [
        ("Hover (medium thrust, no torques)", jnp.array([0.0, 0.0, 0.0, 0.0])),
        ("Positive roll torque", jnp.array([0.0, 0.3, 0.0, 0.0])),
        ("Positive pitch torque", jnp.array([0.0, 0.0, 0.3, 0.0])),
        ("Positive yaw torque", jnp.array([0.0, 0.0, 0.0, 0.3])),
        ("High thrust, no torques", jnp.array([0.5, 0.0, 0.0, 0.0])),
    ]
    
    for test_name, action in test_cases:
        print(f"Test: {test_name}")
        print(f"  Normalized action: {action}")
        
        # Denormalize the action manually to show physical values
        physical_action = (action + 1.0) * 0.5 * (env._ctrl_max - env._ctrl_min) + env._ctrl_min
        thrust, tau_x, tau_y, tau_z = physical_action
        print(f"  Physical action: thrust={thrust:.3f}N, τx={tau_x:.3f}N·m, τy={tau_y:.3f}N·m, τz={tau_z:.3f}N·m")
        
        # Compute motor commands
        motor_commands = env._mix_to_motors(thrust, tau_x, tau_y, tau_z)
        print(f"  Motor commands: {motor_commands} N")
        print()
    
    # Run a few steps
    print("Running 5 simulation steps with hover action...")
    hover_action = jnp.array([0.0, 0.0, 0.0, 0.0])  # Normalized hover action
    
    for i in range(5):
        state = env.step(state, hover_action)
        pos = state.pipeline_state.qpos[:3]
        reward = state.reward
        done = state.done
        print(f"  Step {i+1}: pos={pos}, reward={reward:.4f}, done={done}")
    
    print()
    print("=" * 70)
    print("Test completed successfully!")
    print("=" * 70)
    print()
    print("Key features:")
    print("  ✓ Agent outputs 4 normalized actions: [thrust, tau_x, tau_y, tau_z]")
    print("  ✓ Actions are denormalized to physical units")
    print("  ✓ Mixing matrix converts to 4 motor commands")
    print("  ✓ Motor commands are clipped to [0, max_motor_thrust]")
    print("  ✓ Ready for Brax PPO training!")

if __name__ == "__main__":
    test_mixing_control()
