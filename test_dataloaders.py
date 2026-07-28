#!/usr/bin/env python3
"""
Test script for EpisodicDataset and DataLoader on PushT flat dataset structure.
Verifies dataset initialization, video frame decoding, sentence embedding,
and DataLoader batch tensor sizes and shapes.
"""

import os
import torch
from data_utils import EpisodicDataset, load_data, get_norm_stats


def main():
    dataset_dir = "data/pusht_simplified"
    print(f"=== Testing EpisodicDataset and DataLoader on: {dataset_dir} ===")

    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory {dataset_dir} does not exist!")
        return

    camera_names = ["camera_wrist"]

    # 1. Test get_norm_stats
    print("\n--- 1. Testing get_norm_stats ---")
    norm_stats = get_norm_stats(dataset_dir)
    print(f"Action Mean shape: {norm_stats['action_mean'].shape}, values: {norm_stats['action_mean']}")
    print(f"Action Std shape:  {norm_stats['action_std'].shape}, values: {norm_stats['action_std']}")
    print(f"Qpos Mean shape:   {norm_stats['qpos_mean'].shape}, values: {norm_stats['qpos_mean']}")
    print(f"Qpos Std shape:    {norm_stats['qpos_std'].shape}, values: {norm_stats['qpos_std']}")

    # 2. Test single sample output from EpisodicDataset
    print("\n--- 2. Testing EpisodicDataset __getitem__ ---")
    episode_ids = [0, 1, 2, 3, 4]
    dataset = EpisodicDataset(
        episode_ids=episode_ids,
        dataset_dir=dataset_dir,
        camera_names=camera_names,
        norm_stats=norm_stats,
    )
    print(f"Dataset length: {len(dataset)}")

    image_data, qpos_data, action_data, is_pad, task_id, instr_embedding = dataset[0]

    print("\nSingle Item Shapes and Dtypes:")
    print(f"  image_data:      {image_data.shape} \t (type: {image_data.dtype}, min: {image_data.min():.3f}, max: {image_data.max():.3f})")
    print(f"  qpos_data:       {qpos_data.shape} \t (type: {qpos_data.dtype})")
    print(f"  action_data:     {action_data.shape} \t (type: {action_data.dtype})")
    print(f"  is_pad:          {is_pad.shape} \t (type: {is_pad.dtype})")
    print(f"  task_id:         {task_id}")
    print(f"  instr_embedding: {instr_embedding.shape} \t (type: {instr_embedding.dtype})")

    # 3. Test DataLoader batch shapes
    print("\n--- 3. Testing DataLoader Batch Output ---")
    batch_size = 2
    train_dataloader, val_dataloader, norm_stats, instr_stats = load_data(
        dataset_dir=dataset_dir,
        batch_size_train=batch_size,
        batch_size_val=batch_size,
        camera_names=camera_names,
    )

    print(f"Train Dataloader batches count: {len(train_dataloader)}")
    print(f"Val Dataloader batches count:   {len(val_dataloader)}")

    for batch_idx, batch in enumerate(train_dataloader):
        images, qpos, actions, is_pad_batch, task_ids, instr_embs = batch
        print(f"\nBatch {batch_idx} Shapes:")
        print(f"  Images batch:     {images.shape}   \t (B, Cam, C, H, W)")
        print(f"  Qpos batch:       {qpos.shape}     \t (B, Qpos_dim)")
        print(f"  Actions batch:    {actions.shape}  \t (B, Seq_len, Action_dim)")
        print(f"  Is_pad batch:     {is_pad_batch.shape} \t (B, Seq_len)")
        print(f"  Instr embs batch: {instr_embs.shape} \t (B, Embed_dim)")

        # Verify expected dimensions
        assert images.ndim == 5, f"Expected 5D tensor for images (B, Cam, C, H, W), got {images.ndim}D"
        assert qpos.ndim == 2, f"Expected 2D tensor for qpos (B, Qpos_dim), got {qpos.ndim}D"
        assert actions.ndim == 3, f"Expected 3D tensor for actions (B, Seq_len, Action_dim), got {actions.ndim}D"
        assert instr_embs.shape[1] == 768, f"Expected sentence embedding dim 768, got {instr_embs.shape[1]}"
        break

    print("\n=======================================================")
    print("[SUCCESS] All dataset & dataloader shapes and sizes verified!")
    print("=======================================================")


if __name__ == "__main__":
    main()
