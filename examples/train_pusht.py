#!/usr/bin/env python3
"""
train_pusht.py: Script to train an ACT policy on the PushT task and evaluate policy rollouts
in the Gym-PushT environment.

Usage example:
    uv run python examples/train_pusht.py --num_epochs 10 --batch_size 8 --eval --eval_frequency 5

Features:
- Extends train_nl_act.py for PushT environment setup and training
- Closed-loop evaluation rollout in gym_pusht/PushT-v0
- Saves rollout MP4 videos locally and logs rollout metrics/videos to Weights & Biases if enabled
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
import wandb

# Resolve repository root directory for imports when executed from examples/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gym_pusht
import gymnasium as gym

from data_utils import get_sentence_transformer_model
from train_nl_act import get_args_parser, train_nl_act


def parse_pusht_args():
    parser = argparse.ArgumentParser(
        description="Train ACT Policy on PushT Task with Closed-Loop Evaluation.",
        parents=[get_args_parser()],
        conflict_handler="resolve",
    )

    # PushT Specific Defaults (Optimized for Threadripper CPU + RTX 3080 GPU)
    parser.set_defaults(
        dataset_dir="data/pusht_simplified",
        ckpt_dir="checkpoints/pusht",
        state_dim=2,
        env_state_dim=3,
        num_queries=100,
        batch_size=64,
        batch_size_val=64,
        num_workers=8,
        camera_names=["observation.image"],
    )

    # Evaluation Arguments
    parser.add_argument("--eval", action="store_true", default=False, help="Enable PushT closed-loop evaluation rollouts")
    parser.add_argument("--eval_frequency", type=int, default=10, help="Epoch interval between evaluation rollouts")
    parser.add_argument("--num_eval_episodes", type=int, default=3, help="Number of rollout episodes during evaluation")
    parser.add_argument("--eval_max_timesteps", type=int, default=300, help="Maximum timesteps per rollout episode")
    parser.add_argument("--query_frequency", type=int, default=16, help="Number of steps to execute per policy query")
    parser.add_argument("--videos_dir", type=str, default="videos/pusht", help="Directory to save evaluation rollout videos")
    parser.add_argument("--instruction_text", type=str, default="Push the T-shaped block to the target area.", help="Language conditioning prompt")

    return parser.parse_args()


def evaluate_pusht_rollout(epoch, policy, norm_stats, instr_stats, device, args):
    """
    Executes closed-loop evaluation rollouts in gym_pusht/PushT-v0 and records MP4 videos.
    Logs metrics and video to wandb if enabled.
    """
    if not args.eval:
        return

    if epoch % args.eval_frequency != 0 and epoch != args.num_epochs:
        return

    print(f"\n[PushT Eval] Running closed-loop evaluation for Epoch {epoch}...")

    videos_dir = Path(args.videos_dir)
    videos_dir.mkdir(parents=True, exist_ok=True)

    # Compute language embedding for instruction prompt
    sentence_model = get_sentence_transformer_model("all-mpnet-base-v2")
    raw_instr = sentence_model.encode(args.instruction_text)

    # Normalize instruction embedding if instr_stats provided
    if instr_stats is not None and "mean" in instr_stats and "std" in instr_stats:
        instr_mean = instr_stats["mean"].numpy() if hasattr(instr_stats["mean"], "numpy") else np.array(instr_stats["mean"])
        instr_std = instr_stats["std"].numpy() if hasattr(instr_stats["std"], "numpy") else np.array(instr_stats["std"])
        instr_norm = (raw_instr - instr_mean) / instr_std
    else:
        instr_norm = raw_instr

    instr_tensor = torch.from_numpy(instr_norm).float().unsqueeze(0).to(device)

    # Extract normalization statistics for state and action
    action_mean = np.array(norm_stats["action_mean"], dtype=np.float32)
    action_std = np.array(norm_stats["action_std"], dtype=np.float32)
    qpos_mean = np.array(norm_stats["qpos_mean"], dtype=np.float32)
    qpos_std = np.array(norm_stats["qpos_std"], dtype=np.float32)

    # Initialize PushT Gym environment
    env = gym.make("gym_pusht/PushT-v0", render_mode="rgb_array")

    policy.eval()

    successes = []
    max_coverages = []
    total_rewards = []
    saved_video_paths = []

    for ep_idx in range(args.num_eval_episodes):
        obs, info = env.reset(seed=args.seed + ep_idx * 100)
        frames = []

        rendered_img = env.render()
        frames.append(rendered_img)

        ep_reward = 0.0
        max_cov = 0.0
        is_success = False

        t = 0
        with torch.no_grad():
            while t < args.eval_max_timesteps:
                # Extract agent position qpos (2D)
                qpos_raw = obs[:2]
                qpos_norm = (qpos_raw - qpos_mean) / qpos_std
                qpos_tensor = torch.from_numpy(qpos_norm).float().unsqueeze(0).to(device)

                # Preprocess rendered frame (680, 680, 3) -> (1, 1, 3, 96, 96) in [0, 1]
                curr_frame = rendered_img
                frame_tensor = torch.from_numpy(curr_frame).permute(2, 0, 1).float() / 255.0
                frame_tensor = F.interpolate(frame_tensor.unsqueeze(0), size=(96, 96), mode="bicubic", align_corners=False)
                image_tensor = frame_tensor.unsqueeze(0).to(device)  # (1, 1, 3, 96, 96)

                # Query ACT Policy for action chunk prediction
                actions_pred = policy(qpos_tensor, image_tensor, instr_embedding=instr_tensor)
                actions_pred = actions_pred.squeeze(0).cpu().numpy()  # (num_queries, 2)

                # Execute action chunk for query_frequency steps
                steps_to_run = min(args.query_frequency, len(actions_pred), args.eval_max_timesteps - t)
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
                    if terminated or truncated:
                        break

                if terminated or truncated:
                    break

        successes.append(is_success)
        max_coverages.append(max_cov)
        total_rewards.append(ep_reward)

        # Save rollout video as MP4
        video_filename = videos_dir / f"pusht_epoch_{epoch}_ep_{ep_idx + 1}.mp4"
        try:
            imageio.mimwrite(str(video_filename), frames, fps=10, codec="libx264", macro_block_size=1)
            saved_video_paths.append(str(video_filename))
        except Exception as e:
            print(f"[PushT Eval] Error saving video {video_filename}: {e}")

    env.close()

    mean_success = float(np.mean(successes))
    mean_coverage = float(np.mean(max_coverages))
    mean_reward = float(np.mean(total_rewards))

    print(
        f"[PushT Eval Epoch {epoch}] Success Rate: {mean_success * 100:.1f}% | "
        f"Mean Max Coverage: {mean_coverage * 100:.1f}% | "
        f"Mean Reward: {mean_reward:.2f}"
    )

    # Log metrics and rollout video to wandb (minimal logging, no checkpoint artifacts)
    if args.wandb:
        log_payload = {
            "epoch": epoch,
            "eval/success_rate": mean_success,
            "eval/max_coverage": mean_coverage,
            "eval/mean_reward": mean_reward,
        }
        if saved_video_paths:
            primary_video = saved_video_paths[0]
            log_payload["eval/rollout_video"] = wandb.Video(primary_video, fps=10, format="mp4")

        wandb.log(log_payload)


def main():
    args = parse_pusht_args()

    # Pass evaluate_pusht_rollout as evaluation hook into train_nl_act
    eval_hook = evaluate_pusht_rollout if args.eval else None

    print(f"=== PushT ACT Training ===")
    print(f"Dataset Dir:  {args.dataset_dir}")
    print(f"Ckpt Dir:     {args.ckpt_dir}")
    print(f"Epochs:       {args.num_epochs}")
    print(f"Batch Size:   {args.batch_size}")
    print(f"State Dim:    {args.state_dim}")
    print(f"Evaluation:   {args.eval} (freq: {args.eval_frequency} epochs)")
    print(f"Wandb:        {args.wandb}")
    print("==========================")

    train_nl_act(args, eval_fn=eval_hook)


if __name__ == "__main__":
    main()
