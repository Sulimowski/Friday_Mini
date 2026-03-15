# FRIDAY Mini — Reinforcement Learning Robotic Grasping in PyBullet

FRIDAY Mini is a reinforcement learning project focused on teaching a custom robot manipulator to approach and grasp a small block in simulation. The project uses **PyBullet** for physics, **Gymnasium** for environment interfaces, and **Stable-Baselines3 (PPO)** for policy optimization.

This repository demonstrates practical RL workflow skills relevant for robotics and applied ML portfolios: environment design, reward shaping, training stabilization, evaluation, and iterative fine-tuning.

## Project Goals

- Train a robotic arm to perform precise contact-based grasping.
- Penalize unstable interactions (wrong contact, excessive object displacement, or stagnation).
- Improve robustness from static scenarios to more dynamic interaction conditions.
- Save reproducible checkpoints, normalization stats, and training logs.

![preview]({37CEDE61-1618-4E05-8D94-9A1F226398AE}.png)  

## Core Features

- **Custom Gymnasium environments** for static and dynamic grasping tasks.
- **Dense reward shaping** based on distance, contact quality, orientation, and control stability.
- **PPO training pipeline** with model checkpointing and early stopping.
- **Observation/Reward normalization** through `VecNormalize`.
- **Evaluation scripts** for visual policy replay in PyBullet GUI mode.
- **Training analytics** via saved reward logs and learning curves.

## Tech Stack

- Python
- PyBullet
- Gymnasium
- Stable-Baselines3 (PPO)
- NumPy, Pandas, Matplotlib

## Repository Structure

```text
.
├── grasp_env.py               # Static grasping environment
├── grasp_env_dynamic.py       # Dynamic/fine-tuning environment
├── train.py                   # PPO training on static environment
├── train_dynamic.py           # Fine-tuning pipeline for dynamic environment
├── test.py                    # Visual test for static policy
├── test_dynamic.py            # Visual test for dynamic policy
├── pretty_mukoco/             # Robot URDF/XML/STL assets
├── models/                    # Saved checkpoints and normalization files
└── logs/                      # Reward logs and training curve plots
```

## Quick Start

### 1) Clone

```bash
git clone https://github.com/Sulimowski/Friday_Mini.git
cd Friday_Mini
```

### 2) Install dependencies

```bash
pip install pybullet gymnasium stable-baselines3 numpy pandas matplotlib
```

### 3) Train base policy

```bash
python train.py
```

### 4) Evaluate (GUI)

```bash
python test.py
```

### 5) Fine-tune on dynamic setup

```bash
python train_dynamic.py
python test_dynamic.py
```

## Notes for Reproducibility

- The environments currently load the robot URDF from a local absolute path in code. Update this path to your local repository path before running training/evaluation.
- Trained checkpoints and normalization statistics are versioned in `models/`.
- Learning curves and reward logs are written to `logs/`.

## Portfolio Value

This project highlights:

- End-to-end RL experimentation workflow.
- Practical reward engineering for robotics behavior.
- Iterative policy improvement from baseline to fine-tuned variants.
- Working with simulation assets and custom robot kinematics in PyBullet.

## Author

**Sulimowski**


