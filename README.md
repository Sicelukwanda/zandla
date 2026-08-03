# Zandla 🦾

**Zandla** is a robot learning repository for the **SO-101 robot arm**. It provides simulation environments built with MuJoCo and Gymnasium, expert dataset collection pipelines, and Action Chunking Transformer (ACT) policy training conditioned on natural language instructions.

---

## 🚀 Installation & Setup

Zandla uses [`uv`](https://github.com/astral-sh/uv) for fast, reproducible Python dependency management.

### Prerequisites
- Python >= 3.12, < 3.13
- `uv` installed (`pip install uv` or via system package manager)

### Environment Setup
Clone the repository and synchronize dependencies:

```bash
# Clone the repository
git clone https://github.com/Akili-African-Robotics-Community/zandla.git
cd zandla

# Install all dependencies into virtual environment
uv sync
```

To install optional dependencies (such as `gym-pusht` for 2D PushT benchmark experiments):
```bash
uv sync --extra pusht
```

---

## 🤖 Simulation Environments

### PushCube with SO-101 Arm (`PushCubeGymEnv`)

The primary simulation environment is `PushCubeGymEnv` ([zandla/envs/push_cube_env.py](zandla/envs/push_cube_env.py)), a Gymnasium-compatible MuJoCo environment featuring:
- **SO-101 Follower Arm**: 6-DoF robot manipulator model with calibrated wrist camera mount.
- **Visual Feedback**: $96 \times 96 \times 3$ RGB wrist camera feed (`wrist_camera`) positioned on the follower wrist.
- **Observations**: Joint positions, joint velocities, end-effector position, green/blue target coordinates, wrist camera images, and natural language task instructions.
- **Task Goal**: Push the target cube into either the green or blue target zone based on natural language instructions (e.g., *"Push the cube to the green target"*).

#### Running the Simulation
Visualize and interact with the PushCube simulation environment:

```bash
# Run camera rendering simulation
uv run python simulate_camera.py

# Run scripted policy rollout test
uv run python scripted_policy_test.py
```

---

## 📊 Dataset Collection & Utilities
This repo assumes data collected for training is stored according to the following directory structure:

```
custom_dataset_name/
├── dataset_info.json                # Precomputed metadata (episode lengths, mean/std, action dims)
├── episode_0000/
│   ├── camera_wrist.mp4             # Video stream of the wrist camera
│   ├── trajectory.npz               # Compressed arrays (qpos, qvel, actions)
│   └── instruction.txt              # Language conditioning prompt
├── episode_0001/
│   ├── camera_wrist.mp4
│   ├── trajectory.npz
│   └── instruction.txt
└── ...
```
### 1. Recording Demonstration Datasets
You can collect expert trajectory datasets for policy training using scripted or manual policies:

```bash
uv run python record_push_cube_dataset.py \
    --num_episodes 100 \
    --dataset_dir data/push_cube
```

This records trajectory data containing joint positions (`joint_positions`), actions, wrist camera images (`wrist_camera`), target positions, and natural language embeddings into `.npz` episode files.

### 2. Data Utilities & Normalization
The [`data_utils.py`](data_utils.py) module provides dataset loaders (`EpisodicDataset`) and automatic normalization statistics (`get_norm_stats`), supporting variable observation keys (`joint_positions`, `qpos`, `state`) and SentenceTransformer instruction embeddings.

---

## 🧠 Training & Policy Evaluation

### 1. PushCube ACT Training & Closed-Loop Evaluation
Train an Action Chunking Transformer (ACT) policy conditioned on natural language instructions for the SO-101 PushCube task:

```bash
uv run python train_push_cube.py \
    --dataset_dir data/push_cube \
    --ckpt_dir checkpoints/push_cube \
    --num_epochs 50 \
    --batch_size 64 \
    --eval \
    --eval_frequency 10 \
    --wandb
```

**Key Features:**
- **Closed-Loop Evaluation**: Periodically evaluates the policy in `PushCubeGymEnv` during training (`--eval`).
- **Video Export**: Automatically renders and saves side-by-side rollout videos (overhead + wrist camera) to `checkpoints/push_cube/videos/`.
- **WandB Logging**: Logs training loss, validation metrics, evaluation success rates, and rollout videos to Weights & Biases.

### 2. PushT Benchmark Example
For training ACT policies on 2D PushT benchmark datasets:

```bash
uv run python train_nl_act.py \
    --dataset_dir data/pusht_simplified \
    --ckpt_dir checkpoints/pusht \
    --num_epochs 100 \
    --batch_size 64
```

---

## 🛠️ Physical Robot Setup

For physical hardware configuration, motor calibration, and leader-follower teleoperation using physical SO-101 robot arms, please refer to the **[Physical Robot Setup Guide](PhysicalRobotSetup.md)**.
