"""
Augmentation utilities for DAAD-X.

The key requirement: the SAME spatial crop / horizontal flip must be applied
to every view of the same sample so that multi-view spatial alignment is preserved.

This is achieved by:
  1. Computing crop parameters once from the first (front) view.
  2. Applying the same parameters to all views.
"""

import random
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class VideoDataAugmentationDAAD:
    """
    Spatial augmentation applied identically across all views of one sample.

    Input:  list of N tensors, each [C, T, H, W], float32 in [0, 1]
    Output: list of N tensors, each [C, T, crop_size, crop_size]
    """

    def __init__(self, cfg, mode):
        self.mode = mode
        self.crop_size = (
            cfg.DATA.TRAIN_CROP_SIZE if mode == "train" else cfg.DATA.TEST_CROP_SIZE
        )
        if mode == "train":
            self.jitter_min = cfg.DATA.TRAIN_JITTER_SCALES[0]
            self.jitter_max = cfg.DATA.TRAIN_JITTER_SCALES[1]
        self.mean = torch.tensor([0.45, 0.45, 0.45])
        self.std  = torch.tensor([0.225, 0.225, 0.225])

    def __call__(self, tensors):
        """
        tensors: list of [C, T, H, W] float tensors
        Returns: list of [C, T, H, W] augmented tensors
        """
        assert len(tensors) > 0

        C, T, H, W = tensors[0].shape

        # ── 1. Compute shared spatial transform parameters ─────────────────────
        if self.mode == "train":
            # Random short-side scale jitter
            scale = random.uniform(self.jitter_min, self.jitter_max)
            new_short = int(scale)
            if H <= W:
                new_h = new_short
                new_w = int(W * new_short / H)
            else:
                new_w = new_short
                new_h = int(H * new_short / W)

            # Random crop location
            i = random.randint(0, max(0, new_h - self.crop_size))
            j = random.randint(0, max(0, new_w - self.crop_size))

            # Random horizontal flip
            do_flip = random.random() < 0.5
        else:
            # Center crop at test time
            new_short = self.crop_size
            if H <= W:
                new_h = new_short
                new_w = int(W * new_short / H)
            else:
                new_w = new_short
                new_h = int(H * new_short / W)
            i = (new_h - self.crop_size) // 2
            j = (new_w - self.crop_size) // 2
            do_flip = False

        # ── 2. Apply to every view ─────────────────────────────────────────────
        out = []
        for t in tensors:
            # [C, T, H, W] → [T, C, H, W] for torchvision ops
            t = t.permute(1, 0, 2, 3)

            # Resize (short-side scale)
            t = F.interpolate(
                t, size=(new_h, new_w), mode="bilinear", align_corners=False
            )

            # Crop
            t = t[:, :, i:i + self.crop_size, j:j + self.crop_size]

            # Flip
            if do_flip:
                t = torch.flip(t, dims=[3])

            # [T, C, H, W] → [C, T, H, W]
            t = t.permute(1, 0, 2, 3)

            # Normalise
            mean = self.mean.view(3, 1, 1, 1)
            std  = self.std.view(3, 1, 1, 1)
            t = (t - mean) / std

            out.append(t)

        return out
