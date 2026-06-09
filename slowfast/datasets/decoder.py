"""
decoder.py – Video frame decoder for DAAD-X.

Wraps PyAV to decode and return the full video frame sequence as a tensor.
The dataset then indexes into this tensor with the pre-computed clip indices.
"""

import math
import numpy as np
import torch
import av


def decode(
    container,
    sampling_rate,
    num_frames,
    clip_idx=0,
    num_clips=1,
    video_meta=None,
    target_fps=30,
    backend="torchvision",
    max_spatial_scale=0,
    use_offset=False,
    rigid_decode_all_video=False,
):
    """
    Decode video frames from a PyAV container.

    Args:
        container        : av.container.InputContainer
        sampling_rate    : temporal stride between sampled frames
        num_frames       : number of frames to sample
        rigid_decode_all_video : if True, decode all frames (DAAD mode)
        max_spatial_scale: if > 0, resize short side to this value

    Returns:
        frames: torch.Tensor [T, H, W, C] uint8, or None on failure
    """
    try:
        video_stream = container.streams.video[0]
        total_frames = video_stream.frames

        # Collect all frames
        frame_list = []
        container.seek(0)
        for frame in container.decode(video=0):
            img = frame.to_ndarray(format="rgb24")   # [H, W, 3]
            frame_list.append(img)

        if len(frame_list) == 0:
            return None

        frames = np.stack(frame_list, axis=0)         # [T, H, W, 3]

        # Optional spatial resize (short-side)
        if max_spatial_scale > 0:
            T, H, W, C = frames.shape
            if H < W:
                new_H = max_spatial_scale
                new_W = int(W * max_spatial_scale / H)
            else:
                new_W = max_spatial_scale
                new_H = int(H * max_spatial_scale / W)

            import cv2
            resized = []
            for i in range(T):
                r = cv2.resize(
                    frames[i], (new_W, new_H), interpolation=cv2.INTER_LINEAR
                )
                resized.append(r)
            frames = np.stack(resized, axis=0)

        return torch.from_numpy(frames)   # [T, H, W, C]  uint8

    except Exception as e:
        return None
