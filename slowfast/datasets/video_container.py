"""
video_container.py – thin wrapper around torchvision / pyav video containers.

Provides: get_video_container(path, backend)
"""

import av
import torch


def get_video_container(path, backend="torchvision"):
    """
    Open a video file and return a container.
    Supports 'torchvision' (uses pyav internally) and 'pyav'.
    """
    try:
        container = av.open(path)
        return container
    except Exception as e:
        raise IOError(f"Cannot open video '{path}': {e}")
