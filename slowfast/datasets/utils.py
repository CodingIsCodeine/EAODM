"""
slowfast/datasets/utils.py – common dataset utilities.
"""

import torch
import numpy as np


def pack_pathway_output(cfg, frames):
    """Identity for non-SlowFast models (M2MVT uses its own pathway logic)."""
    return [frames]
