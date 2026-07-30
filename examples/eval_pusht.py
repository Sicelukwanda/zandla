#!/usr/bin/env python3
"""
eval_pusht.py: Standalone evaluation script to load a trained ACT policy checkpoint for PushT,
execute closed-loop evaluation rollouts in gym_pusht/PushT-v0, and save rollout videos.

Usage example:
    uv run python examples/eval_pusht.py --checkpoint checkpoints/pusht/policy_epoch_100.ckpt --num_episodes 3
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import imageio
from tqdm import tqdm

# Resolve repository root directory for imports when executed from examples/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gym_pusht
import gymnasium as gym

from data_utils import get_sentence_transformer_model, load_data
from zandla.policies.ACT import ACTPolicy


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ACT Policy Checkpoint on PushT Task.")

    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        default="checkpoints/pusht/policy_epoch_100.ckpt",
        help="Path to saved model checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="data/pusht_simplified",
        help="Path to simplified dataset (used as fallback for norm_stats if missing in ckpt)",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="videos/pusht_eval",
        help="Directory to save output evaluation videos",
    )
    parser.add_argument(
        "--instruction_text",
        type=str,
        default="Push the T-shaped block to the target area.",
        help="Language instruction prompt",
    )
    parser.add_argument("--num_episodes", type=int, default=3, help="Number of rollout evaluation episodes")
    parser.add_argument("--max_timesteps", type=int, default=300, help="Maximum timesteps per rollout episode")
    parser.add_argument("--query_frequency", type=int, default=50, help="Action chunk execution step frequency")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for environment rollouts")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for output videos")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (cuda or cpu)",
    )

    return parser.parse_args()


def load_policy_and_stats(checkpoint_path, dataset_dir, device):
    """Loads ACT policy and normalization stats from a checkpoint file."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    print(f"[Eval] Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    epoch = checkpoint.get("epoch", "Unknown")
    train_loss = checkpoint.get("train_loss", None)
    val_loss = checkpoint.get("val_loss", None)
    print(f"[Eval] Checkpoint info -> Epoch: {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}")

    policy_config = checkpoint.get("policy_config", None)
    if policy_config is None:
        print("[Eval] Warning: policy_config not found in checkpoint. Using default PushT ACT config.")
        policy_config = {
            "lr": 1e-4,
            "lr_backbone": 1e-5,
            "weight_decay": 1e-4,
            "backbone": "resnet18",
            "kl_weight": 10.0,
            "num_queries": 100,
            "hidden_dim": 512,
            "dim_feedforward": 3200,
            "state_dim": 2,
            "env_state_dim": 3,
            "enc_layers": 4,
            "dec_layers": 7,
            "nheads": 8,
            "camera_names": ["observation.image"],
            "masks": False,
            "dilation": False,
            "position_embedding": "sine",
            "dropout": 0.1,
            "pre_norm": False,
        }

    policy = ACTPolicy(policy_config)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.to(device)
    policy.eval()

    norm_stats = checkpoint.get("norm_stats", None)
    instr_stats = checkpoint.get("instr_stats", None)

    # Fallback to dataset if stats not present in checkpoint
    if norm_stats is None:
        print(f"[Eval] norm_stats not in checkpoint. Loading stats from dataset: {dataset_dir}")
        _, _, norm_stats, instr_stats = load_data(
            dataset_dir=dataset_dir,
            batch_size_train=8,
            batch_size_val=8,
            camera_names=policy_config.get("camera_names", ["observation.image"]),
        )

    return policy, norm_stats, instr_stats, policy_config, epoch


def run_evaluation(args):
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load policy and normalization parameters
    policy, norm_stats, instr_stats, policy_config, epoch = load_policy_and_stats(
        args.checkpoint, args.dataset_dir, device
    )

    # Encode natural language instruction prompt
    print(f"[Eval] Encoding instruction prompt: '{args.instruction_text}'")
    sentence_model = get_sentence_transformer_model("all-mpnet-base-v2")
    raw_instr = sentence_model.encode(args.instruction_text)

    if instr_stats is not None and "mean" in instr_stats and "std" in instr_stats:
        instr_mean = instr_stats["mean"].numpy() if hasattr(instr_stats["mean"], "numpy") else np.array(instr_stats["mean"])
        instr_std = instr_stats["std"].numpy() if hasattr(instr_stats["std"], "numpy") else np.array(instr_stats["std"])
        instr_norm = (raw_instr - instr_mean) / instr_std
    else:
        instr_norm = raw_instr

    instr_tensor = torch.from_numpy(instr_norm).float().unsqueeze(0).to(device)

    # Extract normalization parameters
    action_mean = np.array(norm_stats["action_mean"], dtype=np.float32)
    action_std = np.array(norm_stats["action_std"], dtype=np.float32)
    qpos_mean = np.array(norm_stats["qpos_mean"], dtype=np.float32)
    qpos_std = np.array(norm_stats["qpos_std"], dtype=np.float32)

    # Initialize environment
    print("[Eval] Initializing PushT Gym environment...")
    env = gym.make("gym_pusht/PushT-v0", render_mode="rgb_array")

    successes = []
    max_coverages = []
    total_rewards = []
    saved_videos = []

    print(f"\n[Eval] Running {args.num_episodes} evaluation rollouts...")

    for ep_idx in range(args.num_episodes):
        ep_seed = args.seed + ep_idx * 100
        obs, info = env.reset(seed=ep_seed)
        frames = []

        rendered_img = env.render()
        frames.append(rendered_img)

        ep_reward = 0.0
        max_cov = 0.0
        is_success = False

        t = 0
        pbar = tqdm(total=args.max_timesteps, desc=f"Episode {ep_idx + 1}/{args.num_episodes}", leave=False)

        with torch.no_grad():
            while t < args.max_timesteps:
                # Agent position qpos (2D)
                qpos_raw = obs[:2]
                qpos_norm = (qpos_raw - qpos_mean) / qpos_std
                qpos_tensor = torch.from_numpy(qpos_norm).float().unsqueeze(0).to(device)

                # Preprocess image (680, 680, 3) -> (1, 1, 3, 96, 96) in [0, 1]
                curr_frame = rendered_img
                frame_tensor = torch.from_numpy(curr_frame).permute(2, 0, 1).float() / 255.0
                frame_tensor = F.interpolate(frame_tensor.unsqueeze(0), size=(96, 96), mode="bicubic", align_corners=False)
                image_tensor = frame_tensor.unsqueeze(0).to(device)

                # Query ACT policy
                actions_pred = policy(qpos_tensor, image_tensor, instr_embedding=instr_tensor)
                actions_pred = actions_pred.squeeze(0).cpu().numpy()  # (num_queries, action_dim)

                # Execute action chunk
                steps_to_run = min(args.query_frequency, len(actions_pred), args.max_timesteps - t)
                for step_idx in range(steps_to_run):
                    action_norm = actions_pred[step_idx]
                    action_real = action_norm * action_std + action_mean

                    obs, reward, terminated, truncated, info = env.step(action_real)
                    rendered_img = env.render()
                    frames.append(rendered_img)

                    ep_reward += float(reward)
                    cov = info.get("coverage", 0.0)
                    if cov > max_cov:
                        max_cov = cov
                    if info.get("is_success", False):
                        is_success = True

                    t += 1
                    pbar.update(1)
                    if terminated or truncated:
                        break

                if terminated or truncated:
                    break

        pbar.close()

        successes.append(is_success)
        max_coverages.append(max_cov)
        total_rewards.append(ep_reward)

        # Save video for episode
        ckpt_stem = Path(args.checkpoint).stem
        video_file = output_dir / f"eval_{ckpt_stem}_ep{ep_idx + 1}.mp4"
        imageio.mimwrite(str(video_file), frames, fps=args.fps, codec="libx264", macro_block_size=1)
        saved_videos.append(video_file)

        print(
            f"Episode {ep_idx + 1:02d} (Seed {ep_seed}) | Success: {is_success} | "
            f"Max Coverage: {max_cov * 100:.1f}% | Total Reward: {ep_reward:.2f} | Video: {video_file}"
        )

    env.close()

    primary_video = saved_videos[0] if saved_videos else None

    mean_success = float(np.mean(successes))
    mean_coverage = float(np.mean(max_coverages))
    mean_reward = float(np.mean(total_rewards))

    print("\n" + "=" * 60)
    print("         PushT Policy Evaluation Summary Results")
    print("=" * 60)
    print(f"Checkpoint File:     {args.checkpoint}")
    print(f"Epoch Evaluated:     {epoch}")
    print(f"Episodes Evaluated:  {args.num_episodes}")
    print(f"Success Rate:        {mean_success * 100:.1f}%")
    print(f"Mean Max Coverage:   {mean_coverage * 100:.1f}%")
    print(f"Mean Total Reward:   {mean_reward:.2f}")
    if primary_video:
        print(f"Saved Video:         {primary_video}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)
