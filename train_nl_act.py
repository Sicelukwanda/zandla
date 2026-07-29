#!/usr/bin/env python3
"""
train_nl_act.py: Training script for Action Chunking with Transformers (ACT) policy
conditioned on natural language embeddings.

Features:
- Dataset loading from flat episode directory structure (via data_utils.py)
- Command-line arguments with reasonable defaults for training and checkpoint saving
- Progress visualization with tqdm displaying loss, component metrics, and ETA time estimates
- Minimal wandb logging (gradients, loss metrics, rollout videos) without checkpoint artifacts
- Full GPU/CPU device management and checkpoint saving/resuming
"""

import os
import time
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from data_utils import load_data, set_seed
from zandla.policies.ACT import ACTPolicy


def get_args_parser():
    parser = argparse.ArgumentParser(description="Train Natural Language Conditioned ACT Policy.")

    # Dataset & Checkpoint Paths
    parser.add_argument("--dataset_dir", type=str, default="data/pusht_simplified", help="Path to simplified dataset directory")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints/pusht", help="Directory to save model checkpoints")
    parser.add_argument("--ckpt_frequency", type=int, default=20, help="Epoch frequency for saving periodic checkpoints")
    parser.add_argument("--save_latest", action="store_true", default=True, help="Always save/overwrite latest.ckpt at each epoch")
    parser.add_argument("--resume", type=str, default=None, help="Path to existing .ckpt checkpoint to resume training")

    # Training Parameters
    parser.add_argument("--num_epochs", type=int, default=100, help="Total number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--batch_size_val", type=int, default=8, help="Batch size for validation")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for transformer")
    parser.add_argument("--lr_backbone", type=float, default=1e-5, help="Learning rate for CNN backbone")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW optimizer")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for training reproducibility")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")

    # ACT Policy Hyperparameters
    parser.add_argument("--camera_names", nargs="+", default=["observation.image"], help="List of camera observation names")
    parser.add_argument("--num_queries", type=int, default=100, help="Action chunk sequence length (num_queries)")
    parser.add_argument("--kl_weight", type=float, default=10.0, help="Weight for KL divergence loss in VAE")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Transformer hidden dimension size")
    parser.add_argument("--dim_feedforward", type=int, default=3200, help="Transformer feedforward dimension size")
    parser.add_argument("--backbone", type=str, default="resnet18", help="Vision backbone architecture")
    parser.add_argument("--enc_layers", type=int, default=4, help="Number of transformer encoder layers")
    parser.add_argument("--dec_layers", type=int, default=7, help="Number of transformer decoder layers")
    parser.add_argument("--nheads", type=int, default=8, help="Number of multi-head attention heads")
    parser.add_argument("--state_dim", type=int, default=None, help="Robot state/qpos dimension (auto-detected if None)")
    parser.add_argument("--env_state_dim", type=int, default=7, help="Environment ground-truth state dimension (for state-only mode)")

    # Weights & Biases Logging
    parser.add_argument("--wandb", action="store_true", default=False, help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="nl-act", help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity name")

    return parser


def parse_args():
    return get_args_parser().parse_args()


def compute_gradient_norm(model):
    """Computes total 2-norm of model gradients across all trainable parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def save_checkpoint(ckpt_path, policy, optimizer, epoch, train_loss, val_loss, norm_stats, instr_stats, policy_config):
    """Saves model weights and training metadata locally (not to wandb)."""
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    state = {
        "model_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "norm_stats": norm_stats,
        "instr_stats": instr_stats,
        "policy_config": policy_config,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    torch.save(state, ckpt_path)


def load_checkpoint(ckpt_path, policy, optimizer=None):
    """Loads state dict and metadata from checkpoint path."""
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    policy.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"[Checkpoint] Resumed from {ckpt_path} (epoch {checkpoint.get('epoch', 0)})")
    return checkpoint


def train_nl_act(args, eval_fn=None):
    """Main training routine for language-conditioned ACT policy."""
    set_seed(args.seed)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb if enabled (minimal logging, no checkpoint artifacts)
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            config=vars(args),
        )

    # Load dataset using data_utils.py
    print(f"\n[Dataset] Loading dataset from: {args.dataset_dir}")
    train_dataloader, val_dataloader, norm_stats, instr_stats = load_data(
        dataset_dir=args.dataset_dir,
        batch_size_train=args.batch_size,
        batch_size_val=args.batch_size_val,
        camera_names=args.camera_names,
    )

    # Infer state_dim if not explicitly specified
    sample_batch = next(iter(train_dataloader))
    sample_images, sample_qpos, sample_actions, sample_is_pad, _, sample_instr = sample_batch
    inferred_state_dim = sample_qpos.shape[-1]
    if args.state_dim is None:
        args.state_dim = inferred_state_dim
    print(f"[Dataset] State dimension: {args.state_dim}, Action sequence length: {sample_actions.shape[1]}")

    # Build model configuration dictionary for ACTPolicy
    policy_config = {
        "lr": args.lr,
        "lr_backbone": args.lr_backbone,
        "weight_decay": args.weight_decay,
        "backbone": args.backbone,
        "kl_weight": args.kl_weight,
        "num_queries": args.num_queries,
        "hidden_dim": args.hidden_dim,
        "dim_feedforward": args.dim_feedforward,
        "state_dim": args.state_dim,
        "env_state_dim": args.env_state_dim,
        "enc_layers": args.enc_layers,
        "dec_layers": args.dec_layers,
        "nheads": args.nheads,
        "camera_names": args.camera_names,
        "masks": False,
        "dilation": False,
        "position_embedding": "sine",
        "dropout": 0.1,
        "pre_norm": False,
    }

    # Initialize ACT Policy and device configuration
    print("\n[Model] Initializing ACT Policy...")
    policy = ACTPolicy(policy_config)
    device = torch.device(args.device)
    policy.to(device)

    optimizer = policy.configure_optimizers()

    start_epoch = 0
    best_val_loss = float("inf")

    # Resume from checkpoint if specified
    if args.resume and os.path.exists(args.resume):
        checkpoint = load_checkpoint(args.resume, policy, optimizer)
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_loss = checkpoint.get("val_loss", float("inf"))

    print(f"[Training] Starting training on {device} for {args.num_epochs} epochs...\n")

    start_time = time.time()
    epoch_times = []

    # Main epoch progress bar
    epoch_pbar = tqdm(range(start_epoch, args.num_epochs), desc="Overall Progress", unit="epoch", dynamic_ncols=True)

    for epoch in epoch_pbar:
        epoch_start_time = time.time()

        # ----------------------------------------------------
        # Training Phase
        # ----------------------------------------------------
        policy.train()
        train_loss_sum = 0.0
        train_l1_sum = 0.0
        train_kl_sum = 0.0
        train_steps = 0
        latest_grad_norm = 0.0

        train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{args.num_epochs} [Train]", leave=False, dynamic_ncols=True)
        for images, qpos, actions, is_pad, _, instr_emb in train_pbar:
            images = images.to(device)
            qpos = qpos.to(device)
            actions = actions.to(device)
            is_pad = is_pad.to(device)
            instr_emb = instr_emb.to(device)

            optimizer.zero_grad()
            loss_dict = policy(qpos, images, actions, is_pad, instr_embedding=instr_emb)
            loss = loss_dict["loss"]

            loss.backward()
            latest_grad_norm = compute_gradient_norm(policy)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
            optimizer.step()

            batch_loss = loss.item()
            batch_l1 = loss_dict["l1"].item()
            batch_kl = loss_dict["kl"].item()

            train_loss_sum += batch_loss
            train_l1_sum += batch_l1
            train_kl_sum += batch_kl
            train_steps += 1

            train_pbar.set_postfix({
                "loss": f"{batch_loss:.4f}",
                "l1": f"{batch_l1:.4f}",
                "kl": f"{batch_kl:.4f}",
                "grad_norm": f"{latest_grad_norm:.2f}",
            })

        avg_train_loss = train_loss_sum / train_steps
        avg_train_l1 = train_l1_sum / train_steps
        avg_train_kl = train_kl_sum / train_steps

        # ----------------------------------------------------
        # Validation Phase
        # ----------------------------------------------------
        policy.eval()
        val_loss_sum = 0.0
        val_l1_sum = 0.0
        val_kl_sum = 0.0
        val_steps = 0

        with torch.no_grad():
            val_pbar = tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{args.num_epochs} [Val]", leave=False, dynamic_ncols=True)
            for images, qpos, actions, is_pad, _, instr_emb in val_pbar:
                images = images.to(device)
                qpos = qpos.to(device)
                actions = actions.to(device)
                is_pad = is_pad.to(device)
                instr_emb = instr_emb.to(device)

                loss_dict = policy(qpos, images, actions, is_pad, instr_embedding=instr_emb)
                val_loss_sum += loss_dict["loss"].item()
                val_l1_sum += loss_dict["l1"].item()
                val_kl_sum += loss_dict["kl"].item()
                val_steps += 1

        avg_val_loss = val_loss_sum / val_steps
        avg_val_l1 = val_l1_sum / val_steps
        avg_val_kl = val_kl_sum / val_steps

        epoch_duration = time.time() - epoch_start_time
        epoch_times.append(epoch_duration)
        avg_epoch_time = np.mean(epoch_times)
        remaining_epochs = args.num_epochs - (epoch + 1)
        estimated_time_remaining = remaining_epochs * avg_epoch_time

        # Format ETA string for tqdm
        eta_str = time.strftime("%H:%M:%S", time.gmtime(estimated_time_remaining))

        epoch_pbar.set_postfix({
            "train_loss": f"{avg_train_loss:.4f}",
            "val_loss": f"{avg_val_loss:.4f}",
            "sec/epoch": f"{epoch_duration:.1f}s",
            "ETA": eta_str,
        })

        tqdm.write(
            f"Epoch {epoch + 1:03d}/{args.num_epochs:03d} | "
            f"Train Loss: {avg_train_loss:.4f} (L1: {avg_train_l1:.4f}, KL: {avg_train_kl:.4f}) | "
            f"Val Loss: {avg_val_loss:.4f} (L1: {avg_val_l1:.4f}, KL: {avg_val_kl:.4f}) | "
            f"Grad Norm: {latest_grad_norm:.2f} | "
            f"Time: {epoch_duration:.1f}s | ETA: {eta_str}"
        )

        # Minimal wandb logging (loss metrics, gradient norm, epoch timing)
        if args.wandb:
            log_payload = {
                "epoch": epoch + 1,
                "train/loss": avg_train_loss,
                "train/l1": avg_train_l1,
                "train/kl": avg_train_kl,
                "val/loss": avg_val_loss,
                "val/l1": avg_val_l1,
                "val/kl": avg_val_kl,
                "train/grad_norm": latest_grad_norm,
                "time/epoch_seconds": epoch_duration,
            }
            wandb.log(log_payload)

        # ----------------------------------------------------
        # Checkpoint Saving (Saved locally only, NOT to wandb)
        # ----------------------------------------------------
        if args.save_latest:
            latest_path = ckpt_dir / "latest.ckpt"
            save_checkpoint(latest_path, policy, optimizer, epoch, avg_train_loss, avg_val_loss, norm_stats, instr_stats, policy_config)

        if (epoch + 1) % args.ckpt_frequency == 0 or (epoch + 1) == args.num_epochs:
            periodic_path = ckpt_dir / f"policy_epoch_{epoch + 1}.ckpt"
            save_checkpoint(periodic_path, policy, optimizer, epoch, avg_train_loss, avg_val_loss, norm_stats, instr_stats, policy_config)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = ckpt_dir / "best.ckpt"
            save_checkpoint(best_path, policy, optimizer, epoch, avg_train_loss, avg_val_loss, norm_stats, instr_stats, policy_config)

        # Optional rollout evaluation hook
        if eval_fn is not None:
            eval_fn(epoch + 1, policy, norm_stats, instr_stats, device, args)

    total_time = time.time() - start_time
    total_time_str = time.strftime("%H:%M:%S", time.gmtime(total_time))
    print(f"\n[Training Complete] Total elapsed time: {total_time_str}. Best Val Loss: {best_val_loss:.4f}\n")


if __name__ == "__main__":
    args = parse_args()
    train_nl_act(args)
