"""
Minimal logging setup for M²MVT.
Compatible with the slowfast.utils.logging interface.
"""

import logging
import os
import sys


def setup_logging(output_dir):
    """Set up logging to stdout and to output_dir/train.log."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "train.log")

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="a"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def get_logger(name):
    return logging.getLogger(name)
