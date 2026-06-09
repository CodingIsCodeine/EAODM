#!/usr/bin/env python3
"""
train.py – M²MVT training on DAAD-X
=====================================
Usage:
    # Fresh training
    python tools/train.py --cfg configs/DAAD/M2MVT_s_16x4.yaml

    # Resume from checkpoint
    python tools/train.py --cfg configs/DAAD/M2MVT_s_16x4.yaml \
        TRAIN.AUTO_RESUME True

    # Override data path
    python tools/train.py --cfg configs/DAAD/M2MVT_s_16x4.yaml \
        DATA.PATH_TO_DATA_DIR /path/to/DAAD-X
"""

import argparse
import os
import sys
import time
import math
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

# Add repo root to path so slowfast imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slowfast.config.defaults import get_cfg
from slowfast.datasets.daad import Daad
from slowfast.models.m2mvt import M2MVT
from slowfast.utils.checkpoint import save_checkpoint, load_checkpoint
import slowfast.utils.logging as sf_logging
import slowfast.utils.metrics as metrics

logger = sf_logging.get_logger(__name__)


# ─────────────────────────── Config ──────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="M²MVT training on DAAD-X")
    parser.add_argument("--cfg", required=True, help="Path to config YAML")
    parser.add_argument(
        "opts",
        help="Key=value pairs to override config",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()


def build_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.cfg)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


# ─────────────────────────── Data ────────────────────────────────────────────

def build_loader(cfg, mode):
    dataset = Daad(cfg, mode)
    shuffle = mode == "train"
    loader = DataLoader(
        dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=mode == "train",
        collate_fn=m2mvt_collate,
    )
    return loader


def m2mvt_collate(batch):
    """
    batch: list of (mv_frames[5], ego_frames[1], label, index)
    Returns:
        mv_list  : list of 5 tensors  [B, C, T, H, W]
        ego_list : list of 1 tensor   [B, C, T, H, W]
        labels   : [B]
        indices  : [B]
    """
    mv_list  = [torch.stack([item[0][v] for item in batch]) for v in range(5)]
    ego_list = [torch.stack([item[1][v] for item in batch]) for v in range(1)]
    labels   = torch.tensor([item[2] for item in batch], dtype=torch.long)
    indices  = torch.tensor([item[3] for item in batch], dtype=torch.long)
    return mv_list, ego_list, labels, indices


# ─────────────────────────── Model ───────────────────────────────────────────

def build_model(cfg):
    model = M2MVT(cfg)
    if cfg.NUM_GPUS > 1:
        model = nn.DataParallel(model)
    return model.cuda()


# ─────────────────────────── Optimiser ───────────────────────────────────────

def build_optimizer(cfg, model):
    # Separate parameters with and without weight decay (ZERO_WD_1D_PARAM)
    decay_params   = []
    no_decay_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or "bias" in name or "norm" in name.lower():
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    param_groups = [
        {"params": decay_params,    "weight_decay": cfg.SOLVER.WEIGHT_DECAY},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    optimizer = optim.AdamW(
        param_groups,
        lr=cfg.SOLVER.BASE_LR,
        betas=(cfg.SOLVER.MOMENTUM, 0.999),
    )
    return optimizer


def build_scheduler(cfg, optimizer, steps_per_epoch):
    warmup_steps = int(cfg.SOLVER.WARMUP_EPOCHS * steps_per_epoch)
    total_steps  = int(cfg.SOLVER.MAX_EPOCH * steps_per_epoch)

    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warm-up
            return cfg.SOLVER.WARMUP_START_LR / cfg.SOLVER.BASE_LR + (
                1.0 - cfg.SOLVER.WARMUP_START_LR / cfg.SOLVER.BASE_LR
            ) * step / max(1, warmup_steps)
        # Cosine decay
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        end_ratio = cfg.SOLVER.COSINE_END_LR / cfg.SOLVER.BASE_LR
        return end_ratio + (1.0 - end_ratio) * cosine

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────── Loss ────────────────────────────────────────────

class SoftCrossEntropyLoss(nn.Module):
    """Cross entropy that works with both hard (long) and soft (float) targets."""
    def __init__(self, label_smoothing=0.0):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        if targets.dtype == torch.long:
            return F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing)
        # Soft targets
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(targets * log_probs).sum(dim=-1).mean()
        return loss


import torch.nn.functional as F


# ─────────────────────────── Train loop ──────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, scaler, criterion, epoch, cfg):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for step, (mv_list, ego_list, labels, _) in enumerate(loader):
        mv_list  = [v.cuda(non_blocking=True) for v in mv_list]
        ego_list = [v.cuda(non_blocking=True) for v in ego_list]
        labels   = labels.cuda(non_blocking=True)

        optimizer.zero_grad()

        with autocast():
            logits = model(mv_list, ego_list)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()

        # Gradient clip
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(
            model.parameters(), cfg.SOLVER.CLIP_GRAD_L2NORM
        )

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        preds = logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.shape[0]
        total_loss    += loss.item() * labels.shape[0]

        if step % 50 == 0:
            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch [{epoch}] Step [{step}/{len(loader)}] "
                f"Loss: {loss.item():.4f}  "
                f"Acc: {total_correct/total_samples:.4f}  "
                f"LR: {lr:.6f}"
            )

    avg_loss = total_loss / total_samples
    avg_acc  = total_correct / total_samples
    return avg_loss, avg_acc


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    for mv_list, ego_list, labels, _ in loader:
        mv_list  = [v.cuda(non_blocking=True) for v in mv_list]
        ego_list = [v.cuda(non_blocking=True) for v in ego_list]
        labels   = labels.cuda(non_blocking=True)

        logits = model(mv_list, ego_list)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * labels.shape[0]
        all_preds.append(logits.argmax(dim=-1).cpu())
        all_labels.append(labels.cpu())

    all_preds  = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    avg_loss = total_loss / len(all_labels)
    acc      = (all_preds == all_labels).float().mean().item()

    # Per-class accuracy
    n_cls = all_labels.max().item() + 1
    per_cls = []
    for c in range(n_cls):
        mask = all_labels == c
        if mask.sum() > 0:
            per_cls.append((all_preds[mask] == c).float().mean().item())
    f1 = sum(per_cls) / len(per_cls) if per_cls else 0.0  # macro avg recall ≈ balanced acc

    return avg_loss, acc, f1


# ─────────────────────────── Main ────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = build_cfg(args)

    sf_logging.setup_logging(cfg.OUTPUT_DIR)
    logger.info("M²MVT training started.")
    logger.info(f"Config:\n{cfg}")

    torch.manual_seed(cfg.RNG_SEED)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader = build_loader(cfg, "train")
    val_loader   = build_loader(cfg, "val")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg)
    logger.info(
        f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M"
    )

    # ── Optimiser & scheduler ─────────────────────────────────────────────────
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer, len(train_loader))
    scaler    = GradScaler()
    criterion = SoftCrossEntropyLoss(
        label_smoothing=getattr(cfg.MIXUP, "LABEL_SMOOTH_VALUE", 0.1)
    )

    # ── Checkpoint resume ─────────────────────────────────────────────────────
    start_epoch = 0
    best_val_acc = 0.0
    ckpt_dir = os.path.join(cfg.OUTPUT_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    if cfg.TRAIN.AUTO_RESUME:
        last_ckpt = os.path.join(ckpt_dir, "last.pth")
        if os.path.isfile(last_ckpt):
            logger.info(f"Resuming from {last_ckpt}")
            start_epoch, best_val_acc = load_checkpoint(
                last_ckpt, model, optimizer, scheduler, scaler
            )
            logger.info(f"Resumed at epoch {start_epoch}")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, criterion, epoch, cfg
        )
        elapsed = time.time() - t0
        logger.info(
            f"Epoch [{epoch}] Train Loss: {tr_loss:.4f}  Train Acc: {tr_acc:.4f}  "
            f"Time: {elapsed:.1f}s"
        )

        # ── Validation ────────────────────────────────────────────────────────
        if (epoch + 1) % cfg.TRAIN.EVAL_PERIOD == 0:
            val_loss, val_acc, val_f1 = eval_epoch(model, val_loader, criterion)
            logger.info(
                f"Epoch [{epoch}] Val Loss: {val_loss:.4f}  "
                f"Val Acc: {val_acc:.4f}  Val F1: {val_f1:.4f}"
            )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(
                    os.path.join(ckpt_dir, "best.pth"),
                    model, optimizer, scheduler, scaler, epoch, best_val_acc
                )
                logger.info(f"  ↳ Best checkpoint saved (acc={best_val_acc:.4f})")

        # ── Periodic save ─────────────────────────────────────────────────────
        if (epoch + 1) % cfg.TRAIN.CHECKPOINT_PERIOD == 0:
            save_checkpoint(
                os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pth"),
                model, optimizer, scheduler, scaler, epoch, best_val_acc
            )

        # Always save last
        save_checkpoint(
            os.path.join(ckpt_dir, "last.pth"),
            model, optimizer, scheduler, scaler, epoch + 1, best_val_acc
        )

    logger.info(f"Training complete. Best Val Acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
