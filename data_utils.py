import os
import glob
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import imageio
from sentence_transformers import SentenceTransformer

# Global cache for SentenceTransformer model to avoid repeated instantiations across workers
_SENTENCE_MODEL_CACHE = None

# Global cache for video frames across workers to avoid repeated disk reads and ffmpeg decoding
_GLOBAL_VIDEO_FRAMES_CACHE = {}
_GLOBAL_VIDEO_SIZE_CACHE = {}

def get_sentence_transformer_model(model_name='all-mpnet-base-v2'):
    """
    Returns a cached CPU instance of SentenceTransformer to avoid re-instantiating
    model weights and CUDA initialization conflicts in DataLoader worker processes.
    """
    global _SENTENCE_MODEL_CACHE
    if _SENTENCE_MODEL_CACHE is None:
        _SENTENCE_MODEL_CACHE = SentenceTransformer(model_name, device='cpu')
    return _SENTENCE_MODEL_CACHE


class EpisodicDataset(Dataset):
    """
    loads robotics trajectory data, camera MP4 videos,
    and language instructions directly from a flat dataset structure:

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
    """

    def __init__(
        self,
        episode_ids,
        dataset_dir,
        camera_names=None,
        norm_stats=None,
        default_instruction="Perform the task.",
        instr_stats=None,
        max_len=300,
    ):
        super(EpisodicDataset, self).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names if camera_names is not None else ["camera_wrist"]
        self.norm_stats = norm_stats
        self.instr_stats = instr_stats
        self.max_len = max_len
        self.is_sim = True

        self._instr_cache = {}

        # Read dataset_info.json if available to extract extract maximum episode length, cumulative lengths and default instruction
        self._default_instr = default_instruction

        self.episode_lengths = None
        self.cum_lens = None
        info_path = os.path.join(dataset_dir, "dataset_info.json")
        if os.path.exists(info_path):
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    info = json.load(f)
                if "episode_lengths" in info and info["episode_lengths"]:
                    self.episode_lengths = info["episode_lengths"]
                    self.cum_lens = np.cumsum([0] + self.episode_lengths)
                    self.max_len = max(max(self.episode_lengths), self.max_len)
                if default_instruction is None:
                    if info.get("default_instruction"):
                        self._default_instr = info["default_instruction"]
                    elif info.get("instruction"):
                        self._default_instr = info["instruction"]
            except Exception:
                pass
        # Pre-compute default instruction embedding
        self._default_instr_emb = torch.from_numpy(get_sentence_transformer_model('all-mpnet-base-v2').encode(self._default_instr)).float()

    def _get_cached_video_frames(self, video_path):
        global _GLOBAL_VIDEO_FRAMES_CACHE, _GLOBAL_VIDEO_SIZE_CACHE
        if video_path not in _GLOBAL_VIDEO_FRAMES_CACHE:
            try:
                sz = os.path.getsize(video_path)
                cache_key = (sz, os.path.basename(video_path))
                if cache_key in _GLOBAL_VIDEO_SIZE_CACHE:
                    _GLOBAL_VIDEO_FRAMES_CACHE[video_path] = _GLOBAL_VIDEO_SIZE_CACHE[cache_key]
                else:
                    frames = self.load_mp4_to_frames(video_path)
                    _GLOBAL_VIDEO_SIZE_CACHE[cache_key] = frames
                    _GLOBAL_VIDEO_FRAMES_CACHE[video_path] = frames
            except Exception:
                _GLOBAL_VIDEO_FRAMES_CACHE[video_path] = self.load_mp4_to_frames(video_path)
        return _GLOBAL_VIDEO_FRAMES_CACHE[video_path]

    def _get_cached_instr_embedding(self, instruction):
        if instruction == self._default_instr:
            return self._default_instr_emb.clone()
        if instruction not in self._instr_cache:
            model = get_sentence_transformer_model('all-mpnet-base-v2')
            raw_embedding = model.encode(instruction)
            self._instr_cache[instruction] = torch.from_numpy(raw_embedding).float()
        return self._instr_cache[instruction].clone()

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False  # hardcode full episode sampling flag

        raw_ep_id = self.episode_ids[index]

        # Handle episode folder indexing (support 0, '178', episode_0000, or episode_0)
        if isinstance(raw_ep_id, (int, np.integer)) or (isinstance(raw_ep_id, str) and raw_ep_id.isdigit()):
            ep_num = int(raw_ep_id)
            ep_folder = f"episode_{ep_num:04d}"
            if not os.path.exists(os.path.join(self.dataset_dir, ep_folder)):
                ep_folder = f"episode_{ep_num}"
        else:
            ep_folder = str(raw_ep_id)
            if ep_folder.startswith("episode_"):
                try:
                    ep_num = int(ep_folder.split("_")[1])
                except ValueError:
                    ep_num = index
            else:
                ep_num = index
                ep_folder = f"episode_{ep_folder}"

        ep_path = os.path.join(self.dataset_dir, ep_folder)
        if not os.path.exists(ep_path):
            raise FileNotFoundError(f"Episode directory not found: {ep_path}")

        # 1. Load compressed trajectory npz file
        traj_path = os.path.join(ep_path, "trajectory.npz")
        traj = np.load(traj_path)

        # Flexible key extraction (actions vs action, qpos vs state vs joint_positions)
        action_all = traj['action'] if 'action' in traj else traj['actions']
        if 'qpos' in traj:
            qpos_all = traj['qpos']
        elif 'state' in traj:
            qpos_all = traj['state']
        else:
            qpos_all = traj['joint_positions']

        episode_len = action_all.shape[0]
        action_dim = action_all.shape[-1] if action_all.ndim > 1 else 1

        if sample_full_episode:
            start_ts = 0
        else:
            start_ts = np.random.choice(episode_len)

        # Get state observation at start_ts
        qpos = qpos_all[start_ts]

        # Get all actions after and including start_ts
        action = action_all[start_ts:]
        action_len = len(action)

        # Pad action sequence to maximum sequence length (self.max_len) for batch uniform shape
        target_len = max(self.max_len, episode_len)
        padded_action = np.zeros((target_len, action_dim), dtype=np.float32)
        valid_len = min(action_len, target_len)
        padded_action[:valid_len] = action[:valid_len]
        
        is_pad = np.zeros(target_len, dtype=bool)
        is_pad[valid_len:] = True

        # 2. Load observation video frame for specified camera_names
        all_cam_tensors = []
        for cam_name in self.camera_names:
            # Check for camera video file (cam_name.mp4, camera_wrist.mp4, or first available .mp4)
            video_path = os.path.join(ep_path, f"{cam_name}.mp4")
            if not os.path.exists(video_path):
                video_path = os.path.join(ep_path, f"{cam_name.replace('.', '_')}.mp4")
            if not os.path.exists(video_path):
                mp4_files = glob.glob(os.path.join(ep_path, "*.mp4"))
                if mp4_files:
                    video_path = mp4_files[0]

            frames = self._get_cached_video_frames(video_path)
            if frames:
                # Check if video contains full dataset stream (concatenated across episodes)
                if self.cum_lens is not None and len(frames) >= self.cum_lens[-1]:
                    # Map to global frame offset for ep_num
                    global_frame_idx = self.cum_lens[ep_num] + start_ts
                    frame_idx = min(global_frame_idx, len(frames) - 1)
                else:
                    frame_idx = min(start_ts, len(frames) - 1)
                frame = frames[frame_idx]
            else:
                # Fallback zero frame if video loading fails
                frame = np.zeros((96, 96, 3), dtype=np.uint8)

            # Convert single frame to PyTorch tensor (C, H, W) in [0, 1] range
            cam_tensor = self.frames_to_torch_tensor([frame])[0]
            all_cam_tensors.append(cam_tensor)

        # Stack camera observation tensors -> (num_cams, C, H, W)
        image_data = torch.stack(all_cam_tensors, dim=0)

        # Convert numpy arrays to Torch Tensors
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad_data = torch.from_numpy(is_pad).bool()

        # Apply state and action normalization if norm_stats provided
        if self.norm_stats is not None:
            action_mean = torch.as_tensor(self.norm_stats["action_mean"], dtype=torch.float32)
            action_std = torch.as_tensor(self.norm_stats["action_std"], dtype=torch.float32)
            qpos_mean = torch.as_tensor(self.norm_stats["qpos_mean"], dtype=torch.float32)
            qpos_std = torch.as_tensor(self.norm_stats["qpos_std"], dtype=torch.float32)

            action_data = (action_data - action_mean) / action_std
            qpos_data = (qpos_data - qpos_mean) / qpos_std

        # 3. Load text prompt instruction from instruction.txt (fallback to self._default_instr)
        instruction_path = os.path.join(ep_path, "instruction.txt")
        instruction = self._default_instr
        if os.path.exists(instruction_path):
            with open(instruction_path, "r", encoding="utf-8") as f:
                txt = f.read().strip()
                if txt:
                    instruction = txt

        # 4. Generate cached sentence embedding using SentenceTransformer
        instr_embedding = self._get_cached_instr_embedding(instruction)

        if self.instr_stats is not None and "mean" in self.instr_stats and "std" in self.instr_stats:
            instr_mean = torch.as_tensor(self.instr_stats["mean"], dtype=torch.float32)
            instr_std = torch.as_tensor(self.instr_stats["std"], dtype=torch.float32)
            instr_embedding = (instr_embedding - instr_mean) / instr_std

        # task_id = 0 is arbitrary. For multi-task datasets, this would vary depending on task variant.
        task_id = 0
        return image_data, qpos_data, action_data, is_pad_data, task_id, instr_embedding

    @staticmethod
    def _save_frames_to_mp4(frames, output_path, fps):
        """Helper to save a list of numpy frames to an MP4 video."""
        if not frames:
            return

        # Ensure frames are uint8 for video encoding
        frames_uint8 = [f.astype(np.uint8) for f in frames]

        # Use imageio to write video
        try:
            # Using codec='libx264' and quality for good compression and compatibility
            imageio.mimwrite(output_path, frames_uint8, fps=fps, codec='libx264', quality=8, macro_block_size=1)
            print(f"Video saved to {output_path}")
        except Exception as e:
            print(f"Error saving video with imageio: {e}")
            print("Please ensure ffmpeg is installed and discoverable in your PATH.")

    @staticmethod
    def load_mp4_to_frames(mp4_path):
        """Loads an MP4 video and returns a list of numpy frames."""
        try:
            try:
                import imageio.v2 as iio
                frames = list(iio.mimread(mp4_path, memtest=False))
            except Exception:
                frames = list(imageio.imiter(mp4_path, plugin="ffmpeg"))
            return [np.array(frame) for frame in frames]
        except Exception as e:
            print(f"Error loading video frames from {mp4_path}: {e}")
            return []

    @staticmethod
    def frames_to_torch_tensor(frames, resize=None):
        """
        Converts a list of numpy frames (H, W, C) to a PyTorch tensor (B, C, H, W).
        Optionally resizes frames using bicubic interpolation.
        """
        if not frames:
            return torch.empty(0)

        # Convert to numpy array (B, H, W, C)
        frames_np = np.stack(frames)

        # Convert to PyTorch tensor
        # (B, H, W, C) -> (B, C, H, W)
        frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float() / 255.0

        if resize:
            frames_tensor = F.interpolate(frames_tensor, size=resize, mode='bicubic', align_corners=False)

        return frames_tensor


def get_norm_stats(dataset_dir, num_episodes=None):
    """
    Extracts or computes normalization statistics (mean and std for actions and qpos)
    from dataset_info.json or directly by iterating through trajectory files in dataset_dir.
    """
    info_path = os.path.join(dataset_dir, "dataset_info.json")
    if os.path.exists(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        if "stats" in info and "action" in info["stats"]:
            st_key = None
            for key in ["observation.state", "state", "qpos", "joint_positions"]:
                if key in info["stats"]:
                    st_key = key
                    break
            if st_key is not None:
                action_mean = np.array(info["stats"]["action"]["mean"], dtype=np.float32)
                action_std = np.array(info["stats"]["action"]["std"], dtype=np.float32)
                qpos_mean = np.array(info["stats"][st_key]["mean"], dtype=np.float32)
                qpos_std = np.array(info["stats"][st_key]["std"], dtype=np.float32)

                action_std = np.clip(action_std, 1e-2, np.inf)
                qpos_std = np.clip(qpos_std, 1e-2, np.inf)

                return {
                    "action_mean": action_mean,
                    "action_std": action_std,
                    "qpos_mean": qpos_mean,
                    "qpos_std": qpos_std,
                }

    # Fallback: compute statistics manually from trajectory.npz files
    ep_dirs = sorted([d for d in os.listdir(dataset_dir) if d.startswith("episode_") and os.path.isdir(os.path.join(dataset_dir, d))])
    if num_episodes is not None:
        ep_dirs = ep_dirs[:num_episodes]

    all_qpos_data = []
    all_action_data = []
    for ep_folder in ep_dirs:
        traj_path = os.path.join(dataset_dir, ep_folder, "trajectory.npz")
        if os.path.exists(traj_path):
            traj = np.load(traj_path)
            if 'qpos' in traj:
                qpos = traj['qpos']
            elif 'state' in traj:
                qpos = traj['state']
            else:
                qpos = traj['joint_positions']
            action = traj['action'] if 'action' in traj else traj['actions']
            all_qpos_data.append(torch.from_numpy(qpos))
            all_action_data.append(torch.from_numpy(action))

    if not all_qpos_data:
        raise ValueError(f"No valid trajectory files found in dataset directory: {dataset_dir}")

    all_qpos_cat = torch.cat(all_qpos_data, dim=0)
    all_action_cat = torch.cat(all_action_data, dim=0)

    action_mean = all_action_cat.mean(dim=0)
    action_std = torch.clip(all_action_cat.std(dim=0), 1e-2, np.inf)
    qpos_mean = all_qpos_cat.mean(dim=0)
    qpos_std = torch.clip(all_qpos_cat.std(dim=0), 1e-2, np.inf)

    return {
        "action_mean": action_mean.numpy(),
        "action_std": action_std.numpy(),
        "qpos_mean": qpos_mean.numpy(),
        "qpos_std": qpos_std.numpy(),
    }


def load_data(
    dataset_dir, 
    batch_size_train, 
    batch_size_val,
    train_instr_path=None,
    val_instr_path=None,
    camera_names=None,
    default_instruction=None,
    num_workers=8,
):
    """
    Constructs train and validation DataLoaders for the flat episode dataset structure.
    """
    if camera_names is None:
        camera_names = ["camera_wrist"]

    print(f"\nLoading Dataset from: {dataset_dir}\n")

    ep_dirs = sorted([d for d in os.listdir(dataset_dir) if d.startswith("episode_") and os.path.isdir(os.path.join(dataset_dir, d))])
    num_episodes = len(ep_dirs)

    if num_episodes == 0:
        raise FileNotFoundError(f"No episode directories found in {dataset_dir}")

    # Train / Val split (80% / 20%)
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # Obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)
    instr_stats = {
        "mean": torch.zeros(768, dtype=torch.float32),
        "std": torch.ones(768, dtype=torch.float32),
    }

    # Pre-load video frames into global memory cache
    unique_mp4_files = set()
    for ep_dir in ep_dirs:
        ep_path = os.path.join(dataset_dir, ep_dir)
        for cam_name in camera_names:
            video_path = os.path.join(ep_path, f"{cam_name}.mp4")
            if not os.path.exists(video_path):
                video_path = os.path.join(ep_path, f"{cam_name.replace('.', '_')}.mp4")
            if not os.path.exists(video_path):
                mp4_files = glob.glob(os.path.join(ep_path, "*.mp4"))
                if mp4_files:
                    video_path = mp4_files[0]
            if os.path.exists(video_path):
                unique_mp4_files.add(video_path)

    if unique_mp4_files:
        print(f"[Dataset] Pre-loading video streams into system RAM cache...")
        dummy_ds = EpisodicDataset(
            train_indices, dataset_dir, camera_names, norm_stats, default_instruction=default_instruction, instr_stats=instr_stats
        )
        for video_path in sorted(list(unique_mp4_files)):
            dummy_ds._get_cached_video_frames(video_path)
        print(f"[Dataset] Video preloading complete! Cached {len(_GLOBAL_VIDEO_SIZE_CACHE)} unique video stream(s).")

    train_dataset = EpisodicDataset(
        train_indices, dataset_dir, camera_names, norm_stats, default_instruction=default_instruction, instr_stats=instr_stats
    )
    val_dataset = EpisodicDataset(
        val_indices, dataset_dir, camera_names, norm_stats, default_instruction=default_instruction, instr_stats=instr_stats
    )

    persistent_workers = num_workers > 0
    prefetch_factor = 2 if num_workers > 0 else None

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size_train, 
        shuffle=True, 
        pin_memory=True, 
        num_workers=num_workers, 
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size_val, 
        shuffle=True, 
        pin_memory=True, 
        num_workers=num_workers, 
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor
    )

    return train_dataloader, val_dataloader, norm_stats, instr_stats


### Helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
