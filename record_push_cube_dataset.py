#!/usr/bin/env python3
"""
record_push_cube_dataset.py: Script to collect demonstration dataset for PushCubeGymEnv
using scripted policy and save in a flat episode directory structure compatible with data_utils.EpisodicDataset.

Usage example:
    uv run python record_push_cube_dataset.py --num_episodes 50 --output_dir data/push_cube
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
import imageio
from tqdm import tqdm

from zandla.envs import PushCubeGymEnv
from utils.scripted_policy import generate_push_instruction, get_next_action


def collect_dataset(
    output_dir="data/push_cube",
    num_episodes=50,
    fps=25,
    seed=42,
    render=False,
    randomize_cube=True,
    randomize_targets=True,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"=== Starting PushCube Dataset Collection ===")
    print(f"Output Directory : {output_path.resolve()}")
    print(f"Num Episodes     : {num_episodes}")
    print(f"FPS              : {fps}")
    print(f"Seed             : {seed}")
    print(f"Randomize Cube   : {randomize_cube}")
    print(f"Randomize Targets: {randomize_targets}")
    print("==========================================")

    render_mode = "human" if render else "rgb_array"
    env = PushCubeGymEnv(render_mode=render_mode)

    all_qpos_list = []
    all_action_list = []
    episode_lengths = []
    total_frames = 0
    saved_episodes = 0
    trial_idx = 0

    pbar = tqdm(total=num_episodes, desc="Collecting Successful Episodes")

    while saved_episodes < num_episodes:
        ep_seed = seed + trial_idx
        trial_idx += 1

        direction = "left" if saved_episodes % 2 == 0 else "right"
        swap_target = (
            bool(np.random.rand() > 0.5) if randomize_targets else False
        )
        instruction = generate_push_instruction(
            direction=direction, swap_targets=swap_target
        )

        options = {
            "instruction": instruction,
            "swap_target_colors": swap_target,
        }

        # If not randomizing cube, specify fixed initial position
        if not randomize_cube:
            options["init_cube_pos"] = [0.2, 0.0, 0.015]

        obs, info = env.reset(seed=ep_seed, options=options)

        ep_qpos = []
        ep_actions = []
        ep_wrist_frames = []

        step_count = 0
        reset_eef = False
        term_delay_counter = 0
        return_step_count = 0
        approach_steps = 45
        delay_steps = 5
        push_steps = 45
        return_steps = 45
        ep_success = False

        # Record initial observation and positions
        ep_qpos.append(obs["joint_positions"].copy())
        ep_wrist_frames.append(obs["wrist_camera"].copy())

        initial_cube_pos = obs["cube_position"].copy()
        target_color = "green" if direction.lower() == "left" else "blue"
        if swap_target:
            target_color = "blue" if direction.lower() == "left" else "green"
        initial_target_pos = obs[f"target_{target_color}_position"].copy()
        initial_dist = np.linalg.norm(initial_cube_pos - initial_target_pos)

        while True:
            # Generate action via scripted policy
            action, is_at_init = get_next_action(
                obs,
                env.unwrapped.physics,
                direction=direction,
                ee_ini_pos=env.unwrapped.ee_rest,
                swap_targets=swap_target,
                step_count=step_count,
                reset_eef_pos=reset_eef,
                return_step_count=return_step_count,
                approach_steps=approach_steps,
                delay_steps=delay_steps,
                push_steps=push_steps,
                return_steps=return_steps,
            )

            # Apply action to environment
            obs, reward, terminated, truncated, info = env.step(action)
            ep_actions.append(action.copy())
            step_count += 1

            if info.get("success", False):
                ep_success = True

            if reset_eef:
                return_step_count += 1

            if terminated or truncated:
                term_delay_counter += 1
                if term_delay_counter >= 2:
                    reset_eef = True

            # Check episode completion condition
            if step_count >= 100 or (reset_eef and is_at_init):
                break

            # Append observation for next step
            ep_qpos.append(obs["joint_positions"].copy())
            ep_wrist_frames.append(obs["wrist_camera"].copy())

        final_cube_pos = obs["cube_position"].copy()
        final_dist = np.linalg.norm(final_cube_pos - initial_target_pos)
        cube_displacement = np.linalg.norm(final_cube_pos - initial_cube_pos)

        # Demonstration Validation Criteria:
        # 1. Cube did not spawn in target area (initial_dist > 0.05m)
        # 2. Final cube position reaches target area (final_dist < 0.05m or ep_success)
        # 3. Cube was actually moved by the arm (displacement > 0.03m)
        # 4. Robot executed minimum required steps (step_count >= 30)
        is_valid_demo = (
            initial_dist > 0.05
            and (ep_success or final_dist < 0.05)
            and cube_displacement > 0.03
            and step_count >= 30
        )

        if not is_valid_demo:
            tqdm.write(
                f"[Discarded] Trial {trial_idx} failed validation: "
                f"initial_dist={initial_dist:.3f}m, final_dist={final_dist:.3f}m, "
                f"disp={cube_displacement:.3f}m, steps={step_count}. Retrying..."
            )
            continue

        # Ensure lengths match
        assert len(ep_qpos) == len(ep_actions) == len(ep_wrist_frames), (
            f"Mismatch in episode {saved_episodes}: qpos={len(ep_qpos)}, actions={len(ep_actions)}, frames={len(ep_wrist_frames)}"
        )

        ep_qpos_np = np.array(ep_qpos, dtype=np.float32)
        ep_actions_np = np.array(ep_actions, dtype=np.float32)

        ep_dir = output_path / f"episode_{saved_episodes:04d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save camera wrist MP4 video
        video_path = ep_dir / "camera_wrist.mp4"
        try:
            imageio.mimwrite(
                str(video_path),
                ep_wrist_frames,
                fps=fps,
                codec="libx264",
                quality=8,
                macro_block_size=1,
            )
        except Exception as e:
            print(f"Warning: Failed to save video with libx264 ({e}), falling back...")
            imageio.mimwrite(str(video_path), ep_wrist_frames, fps=fps)

        # 2. Save trajectory arrays (qpos and action)
        np.savez_compressed(
            str(ep_dir / "trajectory.npz"),
            qpos=ep_qpos_np,
            action=ep_actions_np,
            joint_positions=ep_qpos_np,
        )

        # 3. Save text instruction
        with open(ep_dir / "instruction.txt", "w", encoding="utf-8") as f:
            f.write(instruction + "\n")

        all_qpos_list.append(ep_qpos_np)
        all_action_list.append(ep_actions_np)
        ep_len = len(ep_actions)
        episode_lengths.append(ep_len)
        total_frames += ep_len

        saved_episodes += 1
        pbar.update(1)

    pbar.close()
    env.close()

    # Calculate overall dataset statistics
    cat_qpos = np.concatenate(all_qpos_list, axis=0)
    cat_actions = np.concatenate(all_action_list, axis=0)

    dataset_info = {
        "dataset_name": "zandla/push_cube",
        "total_episodes": num_episodes,
        "total_frames": total_frames,
        "fps": fps,
        "action_dim": int(cat_actions.shape[-1]),
        "state_dim": int(cat_qpos.shape[-1]),
        "episode_lengths": episode_lengths,
        "stats": {
            "qpos": {
                "mean": cat_qpos.mean(axis=0).tolist(),
                "std": cat_qpos.std(axis=0).tolist(),
            },
            "action": {
                "mean": cat_actions.mean(axis=0).tolist(),
                "std": cat_actions.std(axis=0).tolist(),
            },
        },
        "default_instruction": "push the red cube to the target",
    }

    info_path = output_path / "dataset_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2)

    print(f"\n[Success] Dataset collection complete!")
    print(f"Saved {num_episodes} episodes to {output_path.resolve()}")
    print(f"Total frames: {total_frames}")
    print(f"Metadata saved to: {info_path.resolve()}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Record PushCube demonstration dataset for ACT Policy training."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/push_cube",
        help="Directory to save dataset",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=50,
        help="Number of episodes to record",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="Framerate for recorded videos",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Render environment GUI while recording",
    )
    parser.add_argument(
        "--fixed_cube",
        action="store_true",
        default=False,
        help="Disable cube position randomization (fixed at [0.2, 0.0, 0.015])",
    )
    parser.add_argument(
        "--fixed_targets",
        action="store_true",
        default=False,
        help="Disable target color swapping (always green left, blue right)",
    )
    args = parser.parse_args()

    collect_dataset(
        output_dir=args.output_dir,
        num_episodes=args.num_episodes,
        fps=args.fps,
        seed=args.seed,
        render=args.render,
        randomize_cube=not args.fixed_cube,
        randomize_targets=not args.fixed_targets,
    )


if __name__ == "__main__":
    main()
