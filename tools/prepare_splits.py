#!/usr/bin/env python3
"""
prepare_splits.py – Convert DAAD-X annotation CSV to train/val/test splits.

The original DAAD-X annotation format (single CSV):
    filename.mp4, maneuver_id, start_frame, [17-dim explanation vector]

This script:
  1. Reads the master annotation CSV.
  2. Uses stratified sampling to split 70/15/15 (or the user-specified ratio).
  3. Writes train.csv, val.csv, test.csv to the data directory.

Usage:
    python tools/prepare_splits.py \
        --ann-csv /path/to/DAAD-X/annotations.csv \
        --data-dir /path/to/DAAD-X \
        --train-ratio 0.70 \
        --val-ratio   0.15 \
        --seed 0
"""

import argparse
import csv
import os
import random
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare DAAD-X splits")
    parser.add_argument("--ann-csv",      required=True, help="Master annotation CSV")
    parser.add_argument("--data-dir",     required=True, help="Output directory for splits")
    parser.add_argument("--train-ratio",  type=float, default=0.70)
    parser.add_argument("--val-ratio",    type=float, default=0.15)
    parser.add_argument("--seed",         type=int,   default=0)
    return parser.parse_args()


def read_annotations(csv_path):
    """
    Returns list of rows: [filename, maneuver_id, rest_of_row...]
    """
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            rows.append(row)
    return rows


def stratified_split(rows, train_ratio, val_ratio, seed):
    """
    Split rows by maneuver class using stratified sampling.
    """
    random.seed(seed)

    # Group by maneuver_id (column 1)
    class_to_rows = defaultdict(list)
    for row in rows:
        cls = int(row[1].strip())
        class_to_rows[cls].append(row)

    train, val, test = [], [], []
    for cls, cls_rows in class_to_rows.items():
        random.shuffle(cls_rows)
        n = len(cls_rows)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        train += cls_rows[:n_train]
        val   += cls_rows[n_train:n_train + n_val]
        test  += cls_rows[n_train + n_val:]

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


def write_split(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"Written {len(rows)} rows → {path}")


def main():
    args = parse_args()
    rows = read_annotations(args.ann_csv)
    print(f"Loaded {len(rows)} annotations from {args.ann_csv}")

    # Print class distribution
    from collections import Counter
    dist = Counter(int(r[1]) for r in rows)
    print("Class distribution:", dict(sorted(dist.items())))

    train, val, test = stratified_split(
        rows, args.train_ratio, args.val_ratio, args.seed
    )
    print(f"Splits: train={len(train)}, val={len(val)}, test={len(test)}")

    write_split(train, os.path.join(args.data_dir, "train.csv"))
    write_split(val,   os.path.join(args.data_dir, "val.csv"))
    write_split(test,  os.path.join(args.data_dir, "test.csv"))


if __name__ == "__main__":
    main()
