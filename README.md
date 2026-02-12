# UAV Reinforcement Learning Control

Quadrotor hovering control using **Proximal Policy Optimization (PPO)** with **MuJoCo** physics simulation. A neural network policy learns to fly a drone to randomized target positions from randomized initial states.

## Repository Structure

```
.
├── train.py                  # Training script (PPO via Stable Baselines3)
├── evaluate.py               # Evaluation, visualization, and plotting
├── optimize.py               # Optuna hyperparameter optimization for PPO
├── debug_training.py         # Diagnostic tool for early episode termination
├── envs/
│   ├── __init__.py
│   └── hover_env.py          # Gymnasium environment (HoverEnv)
├── utils/
│   ├── __init__.py
│   ├── state.py              # QuadState — 12D state with quaternion↔Euler conversion
│   └── normalization.py      # Observation/action normalization utilities
├── model/
│   └── drone/
│       ├── drone.xml         # MuJoCo drone model definition
│       ├── scene.xml         # Scene configuration
│       └── assets/           # STL mesh files
├── models_trained/           # Saved model checkpoints (per training run)
├── logs/                     # TensorBoard training logs
└── plots/                    # Generated evaluation plots
```

## Environment

**`HoverEnv`** defines the quadrotor hovering task:

| Property | Details |
|---|---|
| **Observation** | 12D normalized vector: relative position, attitude (roll/pitch/yaw), linear velocity, angular velocity |
| **Action** | 4D normalized vector: total thrust + 3-axis torques, mapped to 4 motor forces via a mixer matrix |
| **Reward** | `exp(-||position_error||²)` — Gaussian reward peaking at 1.0 when on target |
| **Episode length** | 512 steps (5.12 seconds at 100 Hz) |
| **Termination** | State out of bounds or NaN detected |
| **Randomization** | Both initial drone state and target position are randomized each episode |

The drone has 4 motors each producing 0–13 N of thrust. Motor commands are computed from thrust/torque demands using an inverse mixer matrix.

## Installation

```bash
pip install gymnasium stable-baselines3 mujoco numpy scipy matplotlib
```

## Training

```bash
python train.py
```

This trains a PPO policy with:

- **Policy network**: MLP with 2 hidden layers of 256 units (Tanh activation)
- **16 parallel environments** for experience collection
- **Checkpoints** saved every 50k steps to `models_trained/<timestamp>/`
- **TensorBoard logs** written to `logs/<timestamp>/`
- **Best model** tracked via periodic evaluation (5 episodes every 10k steps)

Key hyperparameters (configured in `train.py`):

| Parameter | Value |
|---|---|
| Total timesteps | 10,000,000 |
| Learning rate | 3e-4 |
| Rollout steps | 800 per env |
| Batch size | 256 |
| PPO epochs | 10 |
| Gamma | 0.99 |
| GAE lambda | 0.95 |
| Clip range | 0.2 |
| Entropy coeff | 0.01 |

Monitor training progress with TensorBoard:

```bash
tensorboard --logdir logs/
```

## Evaluation

```bash
# Evaluate with MuJoCo viewer and save plots
python evaluate.py --model ./models_trained/<run>/best_model.zip --episodes 5 --plot

# Evaluate without rendering
python evaluate.py --model ./models_trained/<run>/best_model.zip --no-render --episodes 10

# Test raw physics with constant hover thrust (no policy)
python evaluate.py --test-hover
```

| Flag | Description |
|---|---|
| `--model PATH` | Path to a trained `.zip` model (default: `best_model.zip`) |
| `--episodes N` | Number of evaluation episodes (default: 5) |
| `--no-render` | Disable MuJoCo viewer |
| `--plot` | Save 6-panel performance plots to `./plots/` |
| `--test-hover` | Apply constant hover thrust instead of a learned policy |

Plots show position tracking, position error, attitude, linear/angular velocity, and motor commands over time.

## Hyperparameter Optimization

Uses [Optuna](https://optuna.readthedocs.io/) to automatically search for the best PPO hyperparameters.

```bash
pip install optuna
```

Each trial trains a PPO model for `--n-timesteps` steps (~1-2 hours per trial at 500k steps). Bad trials are pruned early.

```bash
# Run 50 trials (let it run overnight)
python optimize.py --n-trials 50

# Persistent storage — can stop and resume across sessions
python optimize.py --storage sqlite:///optuna.db --n-trials 100

# Quick smoke test (2 trials, very short)
python optimize.py --n-trials 2 --n-timesteps 5000 --timeout 120
```

| Flag | Description |
|---|---|
| `--n-trials N` | Maximum number of trials (default: 50) |
| `--timeout S` | Total time limit in seconds (default: no limit) |
| `--n-timesteps N` | Training steps per trial (default: 500,000) |
| `--n-envs N` | Parallel training envs per trial (default: 8) |
| `--n-jobs N` | Parallel Optuna workers (default: 1) |
| `--study-name NAME` | Study identifier (default: `ppo_hover`) |
| `--storage URL` | Optuna DB URL for persistence (e.g. `sqlite:///optuna.db`) |

The script tunes learning rate, rollout length, batch size, PPO epochs, discount factor, GAE lambda, clip range, entropy coefficient, network architecture, and activation function. Bad trials are pruned early via Optuna's `MedianPruner`.

After optimization, the best hyperparameters are printed as a ready-to-paste config for `train.py`. Results are saved to `study_results_ppo_hover.csv`.

## Debugging

```bash
python debug_training.py
```

Runs diagnostic episodes to identify which state dimensions cause early termination, useful for tuning environment bounds and reward shaping.
