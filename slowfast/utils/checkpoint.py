"""
Checkpoint utilities for M²MVT.
"""

import os
import torch
import slowfast.utils.logging as logging

logger = logging.get_logger(__name__)


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_val_acc):
    """Save a full training checkpoint."""
    state = {
        "epoch": epoch,
        "best_val_acc": best_val_acc,
        "model_state_dict": (
            model.module.state_dict()
            if hasattr(model, "module")
            else model.state_dict()
        ),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict":    scaler.state_dict(),
    }
    torch.save(state, path)
    logger.info(f"Checkpoint saved: {path}")


def load_checkpoint(path, model, optimizer, scheduler, scaler):
    """
    Load a full training checkpoint.
    Returns (start_epoch, best_val_acc).
    """
    ckpt = torch.load(path, map_location="cpu")

    model_state = ckpt.get("model_state_dict", ckpt)
    if hasattr(model, "module"):
        model.module.load_state_dict(model_state, strict=False)
    else:
        model.load_state_dict(model_state, strict=False)

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    epoch        = ckpt.get("epoch", 0)
    best_val_acc = ckpt.get("best_val_acc", 0.0)
    logger.info(f"Checkpoint loaded: epoch={epoch}, best_val_acc={best_val_acc:.4f}")
    return epoch, best_val_acc


def load_checkpoint_weights_only(path, model):
    """Load only model weights from a checkpoint (no optimizer state)."""
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    if hasattr(model, "module"):
        model.module.load_state_dict(state, strict=False)
    else:
        model.load_state_dict(state, strict=False)
    logger.info(f"Model weights loaded from: {path}")
