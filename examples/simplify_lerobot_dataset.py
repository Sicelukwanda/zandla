#!/usr/bin/env python3
"""
Simplify LeRobot Dataset Script

Converts a LeRobot format dataset (such as lerobot/pusht) into a simplified,
flat episode-based directory structure:

custom_dataset/
├── dataset_info.json                # Precomputed metadata (episode lengths, mean/std, action dims)
├── episode_0000/
│   ├── camera_wrist.mp4             # Video stream of camera observation
│   ├── trajectory.npz               # Compressed arrays (qpos, qvel, actions, etc.)
│   └── instruction.txt              # Language conditioning prompt
├── episode_0001/
│   ├── camera_wrist.mp4
│   ├── trajectory.npz
│   └── instruction.txt
└── ...

Usage:
    uv run python examples/simplify_lerobot_dataset.py \
        --repo-id lerobot/pusht \
        --snapshot-dir ~/.cache/huggingface/hub/datasets--lerobot--pusht/snapshots/7628202a2180972f291ba1bc6723834921e72c19 \
        --output-dir data/pusht_simplified

Optional Flags:
    --max-episodes N: Process only the first N episodes for quick testing.
    --camera-name NAME: Set the output video filename (default: camera_wrist).
    --instruction TEXT: Set custom language conditioning text.
"""

import argparse
import json
import logging
from pathlib import Path
import sys

import numpy as np

# Resolve repository root directory for imports when executed from examples/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def write_video_file(frames: np.ndarray, output_path: Path, fps: int = 10):
    """Write an (N, H, W, C) uint8 numpy array of frames to an MP4 file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Try imageio first, fallback to opencv or av
    try:
        import imageio
        writer = imageio.get_writer(str(output_path), fps=fps, codec="libx264", pixelformat="yuv420p")
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        return
    except Exception as e:
        logger.debug(f"imageio write failed ({e}), trying cv2...")

    try:
        import cv2
        h, w, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
        for frame in frames:
            # Convert RGB to BGR for OpenCV
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(bgr_frame)
        writer.release()
        return
    except Exception as e:
        logger.warning(f"Failed to write video to {output_path} with cv2: {e}")


def load_dataset_with_lerobot(repo_id: str, snapshot_dir: Path):
    """Attempt loading via lerobot.common.datasets.lerobot_dataset.LeRobotDataset."""
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        
        # Check if local snapshot exists
        root = snapshot_dir if snapshot_dir and snapshot_dir.exists() else None
        if root:
            logger.info(f"Loading LeRobotDataset from local snapshot: {root}")
            dataset = LeRobotDataset(repo_id=repo_id, root=root)
        else:
            logger.info(f"Loading LeRobotDataset from Hugging Face hub / default cache for repo: {repo_id}")
            dataset = LeRobotDataset(repo_id=repo_id)
        return dataset
    except Exception as e:
        logger.warning(f"Could not load using LeRobotDataset API directly: {e}")
        return None


def convert_from_lerobot_dataset(dataset, output_dir: Path, instruction: str, camera_name: str, max_episodes: int = None):
    """Convert dataset loaded via LeRobotDataset instance."""
    meta = dataset.meta
    info = meta.info if hasattr(meta, "info") else {}
    fps = meta.fps if hasattr(meta, "fps") else 10
    total_episodes = meta.total_episodes if hasattr(meta, "total_episodes") else len(dataset.episode_data_index["from"])
    
    if max_episodes is not None:
        total_episodes = min(total_episodes, max_episodes)

    logger.info(f"Converting {total_episodes} episodes using LeRobotDataset API...")

    episode_lengths = []
    actions_all = []
    states_all = []

    # Map video features
    video_keys = [k for k, v in meta.features.items() if isinstance(v, dict) and v.get("dtype") in ["video", "image"]]
    if not video_keys:
        # Fallback search in features
        video_keys = [k for k in meta.features if "image" in k or "video" in k or "camera" in k]

    logger.info(f"Found image/video features: {video_keys}")

    for ep_idx in range(total_episodes):
        ep_dir = output_dir / f"episode_{ep_idx:04d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        # Get episode frame indices range
        from_idx = dataset.episode_data_index["from"][ep_idx].item()
        to_idx = dataset.episode_data_index["to"][ep_idx].item()
        ep_len = to_idx - from_idx
        episode_lengths.append(ep_len)

        ep_actions = []
        ep_states = []
        ep_timestamps = []
        ep_rewards = []
        ep_dones = []
        ep_frames = {vk: [] for vk in video_keys}

        for i in range(from_idx, to_idx):
            item = dataset[i]

            # Actions
            if "action" in item:
                act = item["action"].cpu().numpy() if hasattr(item["action"], "cpu") else np.array(item["action"])
                ep_actions.append(act)

            # State / observation.state
            if "observation.state" in item:
                st = item["observation.state"].cpu().numpy() if hasattr(item["observation.state"], "cpu") else np.array(item["observation.state"])
                ep_states.append(st)
            elif "state" in item:
                st = item["state"].cpu().numpy() if hasattr(item["state"], "cpu") else np.array(item["state"])
                ep_states.append(st)

            # Timestamp
            if "timestamp" in item:
                ts = item["timestamp"].item() if hasattr(item["timestamp"], "item") else float(item["timestamp"])
                ep_timestamps.append(ts)
            else:
                ep_timestamps.append((i - from_idx) / fps)

            # Reward & Done
            if "next.reward" in item:
                ep_rewards.append(float(item["next.reward"]))
            if "next.done" in item:
                ep_dones.append(bool(item["next.done"]))

            # Video frames
            for vk in video_keys:
                if vk in item:
                    img = item[vk]
                    if hasattr(img, "cpu"):
                        img = img.cpu().numpy()
                    img = np.array(img)
                    # Convert (C, H, W) float [0, 1] to (H, W, C) uint8 if needed
                    if img.ndim == 3 and img.shape[0] in [1, 3]:
                        img = np.transpose(img, (1, 2, 0))
                    if img.dtype != np.uint8:
                        if img.max() <= 1.0:
                            img = (img * 255).astype(np.uint8)
                        else:
                            img = img.astype(np.uint8)
                    ep_frames[vk].append(img)

        # Process numpy trajectory arrays
        actions_arr = np.array(ep_actions, dtype=np.float32) if ep_actions else np.empty((ep_len, 0), dtype=np.float32)
        states_arr = np.array(ep_states, dtype=np.float32) if ep_states else np.empty((ep_len, 0), dtype=np.float32)
        timestamps_arr = np.array(ep_timestamps, dtype=np.float32)
        rewards_arr = np.array(ep_rewards, dtype=np.float32) if ep_rewards else np.zeros((ep_len,), dtype=np.float32)
        dones_arr = np.array(ep_dones, dtype=bool) if ep_dones else np.zeros((ep_len,), dtype=bool)

        # qpos / qvel representation
        qpos_arr = states_arr
        qvel_arr = np.zeros_like(qpos_arr)
        if len(qpos_arr) > 1:
            qvel_arr[1:] = np.diff(qpos_arr, axis=0) * fps

        # Save trajectory.npz
        np.savez_compressed(
            ep_dir / "trajectory.npz",
            actions=actions_arr,
            action=actions_arr,
            qpos=qpos_arr,
            qvel=qvel_arr,
            state=states_arr,
            timestamps=timestamps_arr,
            rewards=rewards_arr,
            dones=dones_arr,
        )

        # Save instruction.txt
        with open(ep_dir / "instruction.txt", "w", encoding="utf-8") as f:
            f.write(instruction.strip() + "\n")

        # Save video files
        for vk, frames_list in ep_frames.items():
            if frames_list:
                frames_np = np.stack(frames_list, axis=0)
                video_filename = f"{camera_name}.mp4" if len(video_keys) == 1 else f"{vk.replace('.', '_')}.mp4"
                write_video_file(frames_np, ep_dir / video_filename, fps=fps)

        if len(actions_arr) > 0:
            actions_all.append(actions_arr)
        if len(states_arr) > 0:
            states_all.append(states_arr)

        if (ep_idx + 1) % 10 == 0 or (ep_idx + 1) == total_episodes:
            logger.info(f"Processed {ep_idx + 1}/{total_episodes} episodes.")

    # Compute precomputed metadata & statistics for dataset_info.json
    all_act = np.concatenate(actions_all, axis=0) if actions_all else np.array([])
    all_st = np.concatenate(states_all, axis=0) if states_all else np.array([])

    dataset_info = {
        "dataset_name": repo_id,
        "total_episodes": total_episodes,
        "total_frames": sum(episode_lengths),
        "fps": fps,
        "action_dim": int(all_act.shape[-1]) if all_act.size > 0 else 0,
        "state_dim": int(all_st.shape[-1]) if all_st.size > 0 else 0,
        "episode_lengths": episode_lengths,
        "stats": {
            "action": {
                "mean": all_act.mean(axis=0).tolist() if all_act.size > 0 else [],
                "std": all_act.std(axis=0).tolist() if all_act.size > 0 else [],
                "min": all_act.min(axis=0).tolist() if all_act.size > 0 else [],
                "max": all_act.max(axis=0).tolist() if all_act.size > 0 else [],
            },
            "state": {
                "mean": all_st.mean(axis=0).tolist() if all_st.size > 0 else [],
                "std": all_st.std(axis=0).tolist() if all_st.size > 0 else [],
                "min": all_st.min(axis=0).tolist() if all_st.size > 0 else [],
                "max": all_st.max(axis=0).tolist() if all_st.size > 0 else [],
            },
        },
        "instruction": instruction,
    }

    with open(output_dir / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2)

    logger.info(f"Successfully converted dataset! Output saved to: {output_dir}")


def convert_from_parquet_files(snapshot_dir: Path, output_dir: Path, repo_id: str, instruction: str, camera_name: str, max_episodes: int = None):
    """Fallback converter reading parquet and info.json directly from snapshot dir."""
    logger.info(f"Attempting direct parquet/json parsing from snapshot: {snapshot_dir}")
    info_path = snapshot_dir / "meta" / "info.json"
    stats_path = snapshot_dir / "meta" / "stats.json"
    
    if not info_path.exists():
        raise FileNotFoundError(f"Cannot find meta/info.json in {snapshot_dir}")

    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    stats = {}
    if stats_path.exists():
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)

    fps = info.get("fps", 10)
    
    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas is required for direct parquet parsing. Please install pandas/pyarrow or lerobot.")
        sys.exit(1)

    # Read data parquets
    data_dir = snapshot_dir / "data"
    parquet_files = sorted(list(data_dir.rglob("*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No data parquet files found in {data_dir}")

    df_list = [pd.read_parquet(pf) for pf in parquet_files]
    full_df = pd.concat(df_list, ignore_index=True)

    if "episode_index" not in full_df.columns:
        raise ValueError("episode_index column missing in parquet dataset.")

    episodes = full_df["episode_index"].unique()
    episodes.sort()

    if max_episodes is not None:
        episodes = episodes[:max_episodes]

    total_episodes = len(episodes)
    logger.info(f"Found {total_episodes} episodes in parquet data.")

    episode_lengths = []
    actions_all = []
    states_all = []

    for ep_idx in episodes:
        ep_df = full_df[full_df["episode_index"] == ep_idx].sort_values("frame_index")
        ep_len = len(ep_df)
        episode_lengths.append(ep_len)

        ep_dir = output_dir / f"episode_{ep_idx:04d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        # Extract actions
        if "action" in ep_df.columns:
            act_raw = ep_df["action"].values
            actions_arr = np.array(act_raw.tolist(), dtype=np.float32)
        else:
            actions_arr = np.empty((ep_len, 0), dtype=np.float32)

        # Extract states
        if "observation.state" in ep_df.columns:
            st_raw = ep_df["observation.state"].values
            states_arr = np.array(st_raw.tolist(), dtype=np.float32)
        elif "state" in ep_df.columns:
            st_raw = ep_df["state"].values
            states_arr = np.array(st_raw.tolist(), dtype=np.float32)
        else:
            states_arr = np.empty((ep_len, 0), dtype=np.float32)

        # Timestamps
        if "timestamp" in ep_df.columns:
            timestamps_arr = np.array(ep_df["timestamp"].values, dtype=np.float32)
        else:
            timestamps_arr = np.linspace(0, ep_len / fps, ep_len, endpoint=False, dtype=np.float32)

        # Rewards / Dones
        rewards_arr = np.array(ep_df["next.reward"].values, dtype=np.float32) if "next.reward" in ep_df.columns else np.zeros((ep_len,), dtype=np.float32)
        dones_arr = np.array(ep_df["next.done"].values, dtype=bool) if "next.done" in ep_df.columns else np.zeros((ep_len,), dtype=bool)

        qpos_arr = states_arr
        qvel_arr = np.zeros_like(qpos_arr)
        if len(qpos_arr) > 1:
            qvel_arr[1:] = np.diff(qpos_arr, axis=0) * fps

        np.savez_compressed(
            ep_dir / "trajectory.npz",
            actions=actions_arr,
            action=actions_arr,
            qpos=qpos_arr,
            qvel=qvel_arr,
            state=states_arr,
            timestamps=timestamps_arr,
            rewards=rewards_arr,
            dones=dones_arr,
        )

        with open(ep_dir / "instruction.txt", "w", encoding="utf-8") as f:
            f.write(instruction.strip() + "\n")

        # Copy/trim video file if available in snapshot
        videos_dir = snapshot_dir / "videos"
        if videos_dir.exists():
            for vid_key_dir in videos_dir.iterdir():
                if vid_key_dir.is_dir():
                    # Check for mp4 files
                    mp4s = list(vid_key_dir.rglob("*.mp4"))
                    if mp4s:
                        # Copy or link video file if it's full episode or single video file
                        src_video = mp4s[0]
                        dst_video = ep_dir / f"{camera_name}.mp4"
                        if src_video.stat().st_size > 1000: # Not LFS pointer
                            import shutil
                            shutil.copy2(src_video, dst_video)

        if len(actions_arr) > 0:
            actions_all.append(actions_arr)
        if len(states_arr) > 0:
            states_all.append(states_arr)

    all_act = np.concatenate(actions_all, axis=0) if actions_all else np.array([])
    all_st = np.concatenate(states_all, axis=0) if states_all else np.array([])

    dataset_info = {
        "dataset_name": repo_id,
        "total_episodes": total_episodes,
        "total_frames": sum(episode_lengths),
        "fps": fps,
        "action_dim": int(all_act.shape[-1]) if all_act.size > 0 else 0,
        "state_dim": int(all_st.shape[-1]) if all_st.size > 0 else 0,
        "episode_lengths": episode_lengths,
        "stats": stats if stats else {
            "action": {
                "mean": all_act.mean(axis=0).tolist() if all_act.size > 0 else [],
                "std": all_act.std(axis=0).tolist() if all_act.size > 0 else [],
                "min": all_act.min(axis=0).tolist() if all_act.size > 0 else [],
                "max": all_act.max(axis=0).tolist() if all_act.size > 0 else [],
            },
            "state": {
                "mean": all_st.mean(axis=0).tolist() if all_st.size > 0 else [],
                "std": all_st.std(axis=0).tolist() if all_st.size > 0 else [],
                "min": all_st.min(axis=0).tolist() if all_st.size > 0 else [],
                "max": all_st.max(axis=0).tolist() if all_st.size > 0 else [],
            },
        },
        "instruction": instruction,
    }

    with open(output_dir / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2)

    logger.info(f"Successfully converted dataset via parquet! Output saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert LeRobot dataset into simplified format.")
    parser.add_argument("--repo-id", type=str, default="lerobot/pusht", help="HuggingFace dataset repository ID")
    parser.add_argument(
        "--snapshot-dir",
        type=str,
        default="/home/sicelukwanda/.cache/huggingface/hub/datasets--lerobot--pusht/snapshots/7628202a2180972f291ba1bc6723834921e72c19",
        help="Path to local LeRobot dataset snapshot directory",
    )
    parser.add_argument("--output-dir", type=str, default="data/pusht_simplified", help="Output directory for simplified dataset")
    parser.add_argument("--camera-name", type=str, default="camera_wrist", help="Camera video filename (default: camera_wrist)")
    parser.add_argument("--instruction", type=str, default="Push the T-shaped block to the target area.", help="Language instruction prompt")
    parser.add_argument("--max-episodes", type=int, default=None, help="Maximum number of episodes to process")

    args = parser.parse_args()

    snapshot_dir = Path(args.snapshot_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Strategy 1: Try using LeRobotDataset API
    dataset = load_dataset_with_lerobot(args.repo_id, snapshot_dir)
    if dataset is not None:
        try:
            convert_from_lerobot_dataset(dataset, output_dir, args.instruction, args.camera_name, args.max_episodes)
            return
        except Exception as e:
            logger.warning(f"LeRobotDataset API conversion encountered error: {e}. Trying parquet fallback...")

    # Strategy 2: Parquet / JSON direct parsing fallback
    convert_from_parquet_files(snapshot_dir, output_dir, args.repo_id, args.instruction, args.camera_name, args.max_episodes)


if __name__ == "__main__":
    main()
