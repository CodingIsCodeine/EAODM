"""
DAAD-X Dataset for M²MVT.

Each sample has six synchronized video streams:
    front/, rear/, left/, right/, driver/, aria_gaze/

The annotation CSV columns are:
    0: filename (e.g. abc.mp4)
    1: maneuver_id  (0-6)
    2: start_frame
    3: 17-dim explanation vector  ← IGNORED

Only maneuver_id (column 1) is used as the target label.
"""

import os
import csv
import ast
import random
from pathlib import Path

import numpy as np
import torch
import torch.utils.data

from slowfast.datasets import utils as dataset_utils
from slowfast.datasets.decoder import decode
from slowfast.datasets.video_container import get_video_container
from slowfast.datasets.transform import VideoDataAugmentationDAAD
import slowfast.utils.logging as logging

logger = logging.get_logger(__name__)

# ── view names ────────────────────────────────────────────────────────────────
MV_VIEWS  = ["front", "rear", "left", "right", "driver"]   # 5-view branch
EGO_VIEWS = ["aria_gaze"]                                   # ego branch
ALL_VIEWS  = MV_VIEWS + EGO_VIEWS

NUM_CLASSES = 7


class Daad(torch.utils.data.Dataset):
    """
    DAAD-X video dataset for M²MVT.

    Returns:
        mv_frames  : list of 5 tensors  [C, T, H, W]  (front/rear/left/right/driver)
        ego_frames : list of 1 tensor   [C, T, H, W]  (aria_gaze)
        label      : int  (maneuver class 0-6)
        index      : int
    """

    def __init__(self, cfg, mode, num_retries=10):
        assert mode in ("train", "val", "test"), f"Unknown mode: {mode}"
        self.cfg       = cfg
        self.mode      = mode
        self._num_retries = num_retries

        # Temporal sampling from cfg
        self._num_frames    = cfg.DATA.NUM_FRAMES       # 16
        self._sampling_rate = cfg.DATA.SAMPLING_RATE   # 4
        self._video_fps     = 30

        # Spatial
        if mode == "train":
            self._crop_size   = cfg.DATA.TRAIN_CROP_SIZE
            self._jitter_min  = cfg.DATA.TRAIN_JITTER_SCALES[0]
            self._jitter_max  = cfg.DATA.TRAIN_JITTER_SCALES[1]
        else:
            self._crop_size   = cfg.DATA.TEST_CROP_SIZE

        self._data_root = cfg.DATA.PATH_TO_DATA_DIR
        self._aug       = VideoDataAugmentationDAAD(cfg, mode)

        self._load_annotations()
        logger.info(f"DAAD-X [{mode}]: {len(self._path_to_videos)} samples loaded.")

    # ── annotation loading ────────────────────────────────────────────────────
    def _load_annotations(self):
        split_file = {
            "train": "train.csv",
            "val":   "val.csv",
            "test":  "test.csv",
        }[self.mode]

        ann_path = os.path.join(self._data_root, split_file)
        self._path_to_videos = []
        self._labels         = []

        with open(ann_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or row[0].startswith("#"):
                    continue
                filename    = row[0].strip()
                maneuver_id = int(row[1].strip())
                # row[2] = start_frame, row[3] = explanation vector – both ignored

                # Build per-view paths
                view_paths = {
                    v: os.path.join(self._data_root, v, filename)
                    for v in ALL_VIEWS
                }
                self._path_to_videos.append(view_paths)
                self._labels.append(maneuver_id)

    # ── __len__ ───────────────────────────────────────────────────────────────
    def __len__(self):
        return len(self._path_to_videos)

    # ── clip sampling ─────────────────────────────────────────────────────────
    def _sample_clip_indices(self, total_frames):
        """
        Identical clip indices used for ALL views so temporal alignment is preserved.
        Mirrors the SlowFast offset / uniform sampling logic.
        """
        clip_len = self._num_frames * self._sampling_rate  # 64 frames span

        if self.cfg.DATA.USE_OFFSET_SAMPLING and self.mode == "train":
            # Random start within the video
            start = random.randint(0, max(0, total_frames - clip_len))
        else:
            # Centered clip
            start = max(0, (total_frames - clip_len) // 2)

        # Build frame index list with stride
        indices = [start + i * self._sampling_rate for i in range(self._num_frames)]
        # Clamp to valid range
        indices = [min(idx, total_frames - 1) for idx in indices]
        return indices

    # ── load one view ─────────────────────────────────────────────────────────
    def _load_view(self, path, indices):
        """
        Decode a single video file and extract frames at `indices`.
        Returns tensor [C, T, H, W] float32 in [0, 1].
        """
        container = get_video_container(path, self.cfg.DATA.DECODING_BACKEND)
        frames = decode(
            container,
            self._sampling_rate,
            self._num_frames,
            clip_idx=0,
            num_clips=1,
            video_meta={},
            target_fps=self._video_fps,
            backend=self.cfg.DATA.DECODING_BACKEND,
            max_spatial_scale=(
                self._jitter_max if self.mode == "train" else self._crop_size
            ),
            use_offset=self.cfg.DATA.USE_OFFSET_SAMPLING,
            rigid_decode_all_video=True,     # decode full video then index
        )
        # frames: [T, H, W, C]  torch.uint8
        if frames is None or frames.shape[0] == 0:
            return None

        # Select exactly the desired indices (clamped)
        n = frames.shape[0]
        sel = [min(i, n - 1) for i in indices]
        frames = frames[sel, ...]                   # [T, H, W, C]
        frames = frames.permute(3, 0, 1, 2).float() / 255.0  # [C, T, H, W]
        return frames

    # ── __getitem__ ───────────────────────────────────────────────────────────
    def __getitem__(self, index):
        for attempt in range(self._num_retries):
            try:
                return self._get_item_impl(index)
            except Exception as e:
                logger.warning(
                    f"Error loading sample {index} (attempt {attempt}): {e}"
                )
                index = random.randint(0, len(self) - 1)
        raise RuntimeError(f"Failed to load a valid sample after {self._num_retries} retries.")

    def _get_item_impl(self, index):
        view_paths = self._path_to_videos[index]
        label      = self._labels[index]

        # ── determine shared clip indices from the front view ─────────────────
        container_front = get_video_container(
            view_paths["front"], self.cfg.DATA.DECODING_BACKEND
        )
        total_frames = container_front.streams.video[0].frames or 300
        indices = self._sample_clip_indices(total_frames)

        # ── load all views ────────────────────────────────────────────────────
        view_tensors = {}
        for vname in ALL_VIEWS:
            t = self._load_view(view_paths[vname], indices)
            if t is None:
                raise ValueError(f"Failed decoding {vname}: {view_paths[vname]}")
            view_tensors[vname] = t   # [C, T, H, W]

        # ── augment (same crop / flip applied to all views) ───────────────────
        all_tensors = [view_tensors[v] for v in ALL_VIEWS]
        all_tensors = self._aug(all_tensors)   # returns list of [C, T, H, W]

        # ── split back into mv / ego ──────────────────────────────────────────
        mv_frames  = all_tensors[:5]           # front, rear, left, right, driver
        ego_frames = all_tensors[5:]           # aria_gaze

        return mv_frames, ego_frames, label, index
