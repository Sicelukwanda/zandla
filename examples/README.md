# PushT Task Example

This directory contains standalone example scripts for converting datasets, training the Natural Language Action Chunking with Transformers (NL-ACT) model by [Kevin Rohling](https://github.com/krohling/nl-act), and running closed-loop evaluations on the **PushT** robotics benchmark.

---

## 1. PushT Task Overview

The **PushT** benchmark requires a 2D planar robot agent (represented as a circular end-effector) to push a T-shaped block into a target zone.

### Environment Specification
* **Agent Position (`qpos` / `state`)**: 2D continuous coordinates $(x, y)$ representing the agent's end-effector position.
* **Environment State (`env_state`)**: 5D state vector containing:
  * Agent position: $(x_a, y_a)$
  * T-block center position: $(x_b, y_b)$
  * T-block rotation angle: $\theta$
* **Action Space (`actions`)**: 2D target position / velocity command $(x, y)$ for the agent end-effector.
* **Image Observations (`camera_wrist` / `observation.image`)**:
  * Raw Environment Rendering: $680 \times 680 \times 3$ RGB image.
  * Policy Input: Resized to $96 \times 96 \times 3$ RGB tensor normalized with standard ImageNet statistics ($\text{mean}=[0.485, 0.456, 0.406]$, $\text{std}=[0.229, 0.224, 0.225]$).

---

## 2. Installation & Dependencies

The PushT gym environment (`gym-pusht`) is defined as an optional dependency extra in `pyproject.toml`.

### Installing PushT Dependencies via `uv`
Run the following command in the repository root:

```bash
uv sync --extra pusht
```

Or using `pip`:

```bash
pip install -e ".[pusht]"
```

---

## 3. Dataset Preparation (`simplify_lerobot_dataset.py`)

Converts a Hugging Face / LeRobot format dataset (such as `lerobot/pusht`) into a simplified flat directory structure.

### Usage Example
```bash
uv run python examples/simplify_lerobot_dataset.py \
    --repo-id lerobot/pusht \
    --output-dir data/pusht_simplified
```

### CLI Arguments Reference

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--repo-id` | `str` | `"lerobot/pusht"` | HuggingFace dataset repository ID |
| `--snapshot-dir` | `str` | Local cache path | Path to local HuggingFace snapshot directory |
| `--output-dir` | `str` | `"data/pusht_simplified"` | Output directory for simplified dataset |
| `--camera-name` | `str` | `"camera_wrist"` | Filename prefix for output video files |
| `--instruction` | `str` | `"Push the T-shaped block to the target area."` | Default task language instruction prompt |
| `--max-episodes` | `int` | `None` | Optional limit on the number of episodes to convert |

---

## 4. Policy Training (`train_pusht.py`)

Trains an Action Chunking with Transformers (ACT) policy conditioned on natural language embeddings, featuring progress bar visual updates, closed-loop evaluation rollouts in `gym_pusht/PushT-v0`, and optional Weights & Biases logging.

### Usage Examples

#### Quick CPU / GPU Test Run (5 Epochs):
```bash
uv run python examples/train_pusht.py --num_epochs 5 --batch_size 16
```

#### Full Training with Closed-Loop Evaluation Rollouts:
```bash
uv run python examples/train_pusht.py \
    --dataset_dir data/pusht_simplified \
    --ckpt_dir checkpoints/pusht \
    --num_epochs 100 \
    --batch_size 64 \
    --eval \
    --eval_frequency 10
```

### Complete CLI Arguments Reference

#### Dataset & Paths
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--dataset_dir` | `str` | `"data/pusht_simplified"` | Directory containing simplified episode subfolders |
| `--ckpt_dir` | `str` | `"checkpoints/pusht"` | Directory to save model `.ckpt` files |
| `--ckpt_frequency` | `int` | `20` | Epoch interval for saving periodic checkpoints |
| `--save_latest` | `flag` | `True` | Automatically save/overwrite `latest.ckpt` each epoch |
| `--resume` | `str` | `None` | Path to existing `.ckpt` file to resume training |

#### Training Parameters
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--num_epochs` | `int` | `100` | Total number of training epochs |
| `--batch_size` | `int` | `64` | Training batch size |
| `--batch_size_val` | `int` | `64` | Validation batch size |
| `--lr` | `float` | `1e-4` | Learning rate for Transformer architecture |
| `--lr_backbone` | `float` | `1e-5` | Learning rate for vision backbone (ResNet18) |
| `--freeze_backbone` | `flag` | `False` | Freeze vision backbone parameters (`requires_grad=False`) |
| `--weight_decay` | `float` | `1e-4` | Weight decay coefficient for AdamW |
| `--seed` | `int` | `42` | Random seed for reproducibility |
| `--device` | `str` | `"cuda"` / `"cpu"` | Target compute device |
| `--num_workers` | `int` | `8` | DataLoader worker processes |
| `--use_amp` | `flag` | Auto | Enable Automatic Mixed Precision (AMP) |

#### ACT Policy Hyperparameters
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--camera_names` | `list` | `["observation.image"]` | Camera key list |
| `--num_queries` | `int` | `100` | Action chunk prediction horizon length |
| `--kl_weight` | `float` | `10.0` | KL divergence loss weight coefficient in CVAE |
| `--hidden_dim` | `int` | `512` | Transformer hidden dimension size |
| `--dim_feedforward` | `int` | `3200` | Feedforward intermediate dimension size |
| `--backbone` | `str` | `"resnet18"` | Vision backbone model architecture |
| `--enc_layers` | `int` | `4` | Number of transformer encoder layers |
| `--dec_layers` | `int` | `7` | Number of transformer decoder layers |
| `--nheads` | `int` | `8` | Multi-head attention head count |
| `--state_dim` | `int` | `2` | Robot state dimension (auto-detected if `None`) |

#### Closed-Loop Evaluation Rollout Arguments
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--eval` | `flag` | `False` | Enable closed-loop evaluation rollouts during training |
| `--eval_frequency` | `int` | `10` | Epoch frequency for running evaluation rollouts |
| `--num_eval_episodes` | `int` | `3` | Number of rollout episodes per evaluation phase |
| `--eval_max_timesteps` | `int` | `300` | Maximum timesteps per rollout episode |
| `--query_frequency` | `int` | `16` | Action chunk execution steps per policy query |
| `--videos_dir` | `str` | `"videos/pusht"` | Output directory to save MP4 evaluation videos |
| `--instruction_text` | `str` | `"Push the T-shaped..."` | Language prompt instruction |

#### Weights & Biases Arguments
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--wandb` | `flag` | `False` | Enable Weights & Biases experiment logging |
| `--wandb_project` | `str` | `"nl-act"` | Wandb project identifier |
| `--wandb_run_name` | `str` | `None` | Custom Wandb run name |

---

## 5. Policy Evaluation (`eval_pusht.py`)

Standalone evaluation script to load a trained ACT policy checkpoint, execute closed-loop rollouts in `gym_pusht/PushT-v0`, and save rollout MP4 videos.

### Usage Example
```bash
uv run python examples/eval_pusht.py \
    --checkpoint checkpoints/pusht/best.ckpt \
    --num_episodes 5 \
    --output_dir videos/pusht_eval
```

### Complete CLI Arguments Reference

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--checkpoint`, `-c` | `str` | `"checkpoints/pusht/policy_epoch_100.ckpt"` | Path to saved model checkpoint (`.ckpt`) |
| `--dataset_dir` | `str` | `"data/pusht_simplified"` | Path to dataset (fallback for normalization statistics) |
| `--output_dir`, `-o` | `str` | `"videos/pusht_eval"` | Directory to save evaluation MP4 videos |
| `--instruction_text` | `str` | `"Push the T-shaped block..."` | Language prompt text |
| `--num_episodes` | `int` | `3` | Number of evaluation rollout episodes |
| `--max_timesteps` | `int` | `300` | Maximum timesteps per rollout episode |
| `--query_frequency` | `int` | `50` | Action chunk execution step frequency |
| `--seed` | `int` | `42` | Random seed for environment initialization |
| `--fps` | `int` | `10` | Output video frame rate (FPS) |
| `--device` | `str` | `"cuda"` / `"cpu"` | Compute device |
