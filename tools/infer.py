#!/usr/bin/env python3
"""
infer.py – M²MVT inference on DAAD-X
======================================
Full multi-view inference:
    python tools/infer.py --cfg configs/DAAD/M2MVT_s_16x4.yaml \
        --checkpoint checkpoints/best.pth \
        --split test

Single-view inference (any one of the 6 streams):
    python tools/infer.py --cfg configs/DAAD/M2MVT_s_16x4.yaml \
        --checkpoint checkpoints/best.pth \
        --split test \
        --active-views front

    python tools/infer.py ... --active-views aria_gaze

    python tools/infer.py ... --active-views rear

Multiple views (subset):
    python tools/infer.py ... --active-views front rear

All views (default):
    python tools/infer.py ... --active-views front rear left right driver aria_gaze

The trained model is loaded as-is; no retraining is required.
"""

import argparse
import os
import sys
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slowfast.config.defaults import get_cfg
from slowfast.datasets.daad import Daad, ALL_VIEWS, MV_VIEWS, EGO_VIEWS
from slowfast.models.m2mvt import M2MVT, build_active_view_masks
from slowfast.utils.checkpoint import load_checkpoint_weights_only
import slowfast.utils.logging as sf_logging

logger = sf_logging.get_logger(__name__)

MANEUVER_NAMES = {
    0: "GoStraight",
    1: "LeftTurn",
    2: "RightTurn",
    3: "LeftLaneChange",
    4: "RightLaneChange",
    5: "SlowStop",
    6: "UTurn",
}


def parse_args():
    parser = argparse.ArgumentParser(description="M²MVT inference")
    parser.add_argument("--cfg",        required=True, help="Config YAML path")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--split",      default="test",
                        choices=["train", "val", "test"])
    parser.add_argument(
        "--active-views",
        nargs="+",
        default=ALL_VIEWS,
        help=(
            "Views to use. Options: front rear left right driver aria_gaze. "
            "Default: all 6 views."
        ),
    )
    parser.add_argument("--output-json", default=None,
                        help="If set, dump per-sample predictions to this JSON file.")
    parser.add_argument(
        "opts", default=None, nargs=argparse.REMAINDER,
        help="Extra config overrides (key=value)"
    )
    return parser.parse_args()


def build_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.cfg)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def m2mvt_collate(batch):
    mv_list  = [torch.stack([item[0][v] for item in batch]) for v in range(5)]
    ego_list = [torch.stack([item[1][v] for item in batch]) for v in range(1)]
    labels   = torch.tensor([item[2] for item in batch], dtype=torch.long)
    indices  = torch.tensor([item[3] for item in batch], dtype=torch.long)
    return mv_list, ego_list, labels, indices


@torch.no_grad()
def run_inference(model, loader, active_views, device):
    """
    Run inference and return (all_preds, all_labels, all_probs).
    active_views: list of view name strings
    """
    model.eval()

    active_mv, active_ego = build_active_view_masks(active_views)
    logger.info(f"Active MV views  : {active_mv}  (front/rear/left/right/driver)")
    logger.info(f"Active EGO views : {active_ego} (aria_gaze)")

    all_preds  = []
    all_labels = []
    all_probs  = []

    for mv_list, ego_list, labels, _ in loader:
        mv_list  = [v.to(device) for v in mv_list]
        ego_list = [v.to(device) for v in ego_list]

        logits = model(mv_list, ego_list,
                       active_mv_views=active_mv,
                       active_ego_views=active_ego)
        probs  = F.softmax(logits, dim=-1)
        preds  = logits.argmax(dim=-1)

        all_preds.append(preds.cpu())
        all_labels.append(labels)
        all_probs.append(probs.cpu())

    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs  = torch.cat(all_probs).numpy()

    return all_preds, all_labels, all_probs


def print_results(all_preds, all_labels, active_views):
    names = [MANEUVER_NAMES[i] for i in range(len(MANEUVER_NAMES))]
    acc   = (all_preds == all_labels).mean()

    print("\n" + "=" * 60)
    print(f"Active views : {active_views}")
    print(f"Overall Acc  : {acc:.4f} ({acc*100:.2f}%)")
    print("-" * 60)
    print(classification_report(
        all_labels, all_preds,
        target_names=names,
        digits=4,
        zero_division=0,
    ))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("=" * 60 + "\n")


def main():
    args   = parse_args()
    cfg    = build_cfg(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sf_logging.setup_logging(cfg.OUTPUT_DIR)

    # ── Validate active views ─────────────────────────────────────────────────
    for v in args.active_views:
        if v not in ALL_VIEWS:
            raise ValueError(
                f"Unknown view '{v}'. Choose from: {ALL_VIEWS}"
            )

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = Daad(cfg, args.split)
    loader  = DataLoader(
        dataset,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        collate_fn=m2mvt_collate,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = M2MVT(cfg).to(device)
    load_checkpoint_weights_only(args.checkpoint, model)
    logger.info(f"Checkpoint loaded: {args.checkpoint}")

    # ── Inference ─────────────────────────────────────────────────────────────
    all_preds, all_labels, all_probs = run_inference(
        model, loader, args.active_views, device
    )

    print_results(all_preds, all_labels, args.active_views)

    # ── Optional JSON dump ────────────────────────────────────────────────────
    if args.output_json:
        results = {
            "active_views": args.active_views,
            "split": args.split,
            "accuracy": float((all_preds == all_labels).mean()),
            "samples": [
                {
                    "index": int(i),
                    "label": int(all_labels[i]),
                    "pred": int(all_preds[i]),
                    "probs": all_probs[i].tolist(),
                }
                for i in range(len(all_preds))
            ],
        }
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
