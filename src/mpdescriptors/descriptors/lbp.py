"""Local Binary Patterns (Ojala, Pietikainen & Harwood, 1996), 3x3 neighborhood."""

from __future__ import annotations

import numpy as np

# Clockwise from the top-left neighbor; neighbor k sets bit k.
NEIGHBOR_OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))


def lbp_codes(tile: np.ndarray) -> np.ndarray:
    """LBP code of every interior pixel: bit k is set when neighbor k >= center."""
    h, w = tile.shape
    center = tile[1:-1, 1:-1]
    codes = np.zeros(center.shape, dtype=np.uint8)
    for bit, (dy, dx) in enumerate(NEIGHBOR_OFFSETS):
        neighbor = tile[1 + dy : h - 1 + dy, 1 + dx : w - 1 + dx]
        codes |= (neighbor >= center).astype(np.uint8) << bit
    return codes


def lbp(tile: np.ndarray) -> np.ndarray:
    """256-bin LBP histogram, normalized to sum to 1."""
    codes = lbp_codes(tile)
    return np.bincount(codes.ravel(), minlength=256) / codes.size
