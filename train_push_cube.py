#!/usr/bin/env python3
"""
train_push_cube.py: Script to train an ACT policy on the PushCube task and evaluate policy rollouts
in the PushCubeGymEnv environment.

Usage example:
    uv run python train_push_cube.py --dataset_dir data/push_cube --num_epochs 20 --eval --eval_frequency 5

Features:
- Extends train_nl_act.py for PushCube environment setup and training
- Closed-loop evaluation rollout in PushCubeGymEnv using wrist camera and joint state
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
import cv2
from tqdm import tqdm
import wandb

# Resolve repository root directory for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from zandla.envs import PushCubeGymEnv
from data_utils import get_sentence_transformer_model
from train_nl_act import get_args_parser, train_nl_act
from utils.scripted_policy import generate_push_instruction


def parse_push_cube_args():
    parser = argparse.ArgumentParser(
        description="Train ACT Policy on PushCube Task with Closed-Loop Evaluation.",
        parents=[get_args_parser()],
        conflict_handler="resolve",
    )

    # PushCube Specific Defaults
    parser.set_defaults(
        dataset_dir="data/push_cube",
        ckpt_dir="checkpoints/push_cube",
        state_dim=6,
        env_state_dim=6,
        num_queries=100,
        batch_size=64,
        batch_size_val=64,
        num_workers=8,
        camera_names=["camera_wrist"],
    )

    # Evaluation Arguments
    parser.add_argument(
        "--eval",
        action="store_true",
        default=False,
        help="Enable PushCube closed-loop evaluation rollouts",
    )
    parser.add_argument(
        "--eval_frequency",
        type=int,
        default=10,
        help="Epoch interval between evaluation rollouts",
    )
    parser.add_argument(
        "--num_eval_episodes",
        type=int,
        default=4,
        help="Number of rollout episodes during evaluation",
    )
    parser.add_argument(
        "--eval_max_timesteps",
        type=int,
        default=100,
        help="Maximum timesteps per rollout episode",
    )
    parser.add_argument(
        "--query_frequency",
        type=int,
        default=16,
        help="Number of steps to execute per policy query",
    )
    parser.add_argument(
        "--videos_dir",
        type=str,
        default="videos/push_cube",
        help="Directory to save evaluation rollout videos",
    )
    parser.add_argument(
        "--instruction_text",
        type=str,
        default=None,
        help="Specific language prompt override for evaluation rollouts",
    )

    return parser.parse_args()


def create_combined_eval_frame(main_frame, wrist_frame):
    """Combines main perspective camera view (480x640x3) and wrist camera view (96x96x3) side-by-side."""
    h, w, _ = main_frame.shape
    wrist_resized = cv2.resize(
        wrist_frame, (h, h), interpolation=cv2.INTER_NEAREST
    )
    return np.hstack([main_frame, wrist_resized])


def evaluate_push_cube_rollout(
    epoch, policy, norm_stats, instr_stats, device, args
):
    """
    Executes closed-loop evaluation rollouts in PushCubeGymEnv and records MP4 videos.
    Logs metrics and video to wandb if enabled.
    """
    if not args.eval:
        return

    if epoch % args.eval_frequency != 0 and epoch != args.num_epochs:
        return

    print(
        f"\n[PushCube Eval] Running closed-loop evaluation for Epoch {epoch}..."
    )

    videos_dir = Path(args.videos_dir)
    videos_dir.mkdir(parents=True, exist_ok=True)

    sentence_model = get_sentence_transformer_model("all-mpnet-base-v2")

    # Extract normalization statistics for state and action
    action_mean = np.array(norm_stats["action_mean"], dtype=np.float32)
    action_std = np.array(norm_stats["action_std"], dtype=np.float32)
    qpos_mean = np.array(norm_stats["qpos_mean"], dtype=np.float32)
    qpos_std = np.array(norm_stats["qpos_std"], dtype=np.float32)

    # Extract instruction normalization statistics if provided
    instr_mean = None
    instr_std = None
    if instr_stats is not None and "mean" in instr_stats and "std" in instr_stats:
        instr_mean = (
            instr_stats["mean"].numpy()
            if hasattr(instr_stats["mean"], "numpy")
            else np.array(instr_stats["mean"])
        )
        instr_std = (
            instr_stats["std"].numpy()
            if hasattr(instr_stats["std"], "numpy")
            else np.array(instr_stats["std"])
        )

    env = PushCubeGymEnv(render_mode="rgb_array")
    policy.eval()

    successes = []
    total_rewards = []
    final_distances = []
    saved_video_paths = []

    for ep_idx in range(args.num_eval_episodes):
        direction = "left" if ep_idx % 2 == 0 else "right"
        if args.instruction_text is not None:
            instruction_text = args.instruction_text
        else:
            instruction_text = generate_push_instruction(
                direction=direction, swap_targets=False
            )

        # Pre-compute normalized instruction embedding
        raw_instr = sentence_model.encode(instruction_text)
        if instr_mean is not None and instr_std is not None:
            instr_norm = (raw_instr - instr_mean) / instr_std
        else:
            instr_norm = raw_instr
        instr_tensor = (
            torch.from_numpy(instr_norm).float().unsqueeze(0).to(device)
        )

        options = {
            "instruction": instruction_text,
            "init_cube_pos": [0.2, 0.0, 0.015],
            "swap_target_colors": False,
        }

        obs, info = env.reset(seed=args.seed + ep_idx * 100, options=options)
        frames = []

        main_img = env.render()
        wrist_img = obs["wrist_camera"]
        frames.append(create_combined_eval_frame(main_img, wrist_img))

        ep_reward = 0.0
        is_success = False
        dist_cube_target = float("inf")

        t = 0
        with torch.no_grad():
            while t < args.eval_max_timesteps:
                # 1. Normalize 6D joint position qpos
                qpos_raw = obs["joint_positions"]
                qpos_norm = (qpos_raw - qpos_mean) / qpos_std
                qpos_tensor = (
                    torch.from_numpy(qpos_norm).float().unsqueeze(0).to(device)
                )

                # 2. Preprocess wrist camera image (96, 96, 3) -> (1, 1, 3, 96, 96) in [0, 1]
                wrist_frame = obs["wrist_camera"]
                wrist_tensor = (
                    torch.from_numpy(wrist_frame).permute(2, 0, 1).float()
                    / 255.0
                )
                image_tensor = wrist_tensor.unsqueeze(0).unsqueeze(0).to(device)

                # 3. Query ACT policy for action sequence prediction
                actions_pred = policy(
                    qpos_tensor, image_tensor, instr_embedding=instr_tensor
                )
                actions_pred = actions_pred.squeeze(0).cpu().numpy()  # (num_queries, 6)

                # 4. Step environment for query_frequency steps
                steps_to_run = min(
                    args.query_frequency,
                    len(actions_pred),
                    args.eval_max_timesteps - t,
                )
                for step_idx in range(steps_to_run):
                    action_norm = actions_pred[step_idx]
                    action_real = action_norm * action_std + action_mean

                    obs, reward, terminated, truncated, info = env.step(
                        action_real
                    )
                    main_img = env.render()
                    wrist_img = obs["wrist_camera"]
                    frames.append(create_combined_eval_frame(main_img, wrist_img))

                    ep_reward += float(reward)
                    dist_cube_target = info.get("dist_cube_target", dist_cube_target)
                    if info.get("success", False):
                        is_success = True

                    t += 1
                    if terminated or truncated:
                        break

                if terminated or truncated:
                    break

        successes.append(is_success)
        total_rewards.append(ep_reward)
        final_distances.append(dist_cube_target)

        # Save rollout video as MP4
        video_filename = videos_dir / f"push_cube_epoch_{epoch}_ep_{ep_idx + 1}.mp4"
        try:
            imageio.mimwrite(
                str(video_filename),
                frames,
                fps=25,
                codec="libx264",
                quality=8,
                macro_block_size=1,
            )
            saved_video_paths.append(str(video_filename))
        except Exception as e:
            print(f"[PushCube Eval] Error saving video {video_filename}: {e}")

    env.close()

    mean_success = float(np.mean(successes))
    mean_reward = float(np.mean(total_rewards))
    mean_distance = float(np.mean(final_distances))

    print(
        f"[PushCube Eval Epoch {epoch}] Success Rate: {mean_success * 100:.1f}% | "
        f"Mean Reward: {mean_reward:.4f} | "
        f"Mean Final Dist: {mean_distance:.4f} m"
    )

    # Log metrics and rollout video to wandb
    if args.wandb:
        log_payload = {
            "epoch": epoch,
            "eval/success_rate": mean_success,
            "eval/mean_reward": mean_reward,
            "eval/mean_final_distance": mean_distance,
        }
        if saved_video_paths:
            primary_video = saved_video_paths[0]
            log_payload["eval/rollout_video"] = wandb.Video(
                primary_video, fps=25, format="mp4"
            )

        wandb.log(log_payload)


def main():
    args = parse_push_cube_args()

    eval_hook = evaluate_push_cube_rollout if args.eval else None

    print(f"=== PushCube ACT Training ===")
    print(f"Dataset Dir:  {args.dataset_dir}")
    print(f"Ckpt Dir:     {args.ckpt_dir}")
    print(f"Epochs:       {args.num_epochs}")
    print(f"Batch Size:   {args.batch_size}")
    print(f"State Dim:    {args.state_dim}")
    print(f"Evaluation:   {args.eval} (freq: {args.eval_frequency} epochs)")
    print(f"Wandb:        {args.wandb}")
    print("=============================")

    train_nl_act(args, eval_fn=eval_hook)


if __name__ == "__main__":
    main()
