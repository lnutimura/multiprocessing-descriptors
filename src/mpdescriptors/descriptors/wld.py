"""Weber Local Descriptor (Chen et al., "WLD: A Robust Local Image Descriptor", TPAMI 2010)."""

from __future__ import annotations

import numpy as np

M = 6  # excitation segments
T = 8  # dominant orientations
S = 20  # bins per excitation segment


def differential_excitation(tile: np.ndarray) -> np.ndarray:
    """xi = arctan(sum(x_i - x_c) / x_c) for every interior pixel, in [-pi/2, pi/2]."""
    x = tile.astype(np.int32)
    h, w = x.shape
    center = x[1:-1, 1:-1]
    neighbors = sum(
        x[1 + dy : h - 1 + dy, 1 + dx : w - 1 + dx]
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dy, dx) != (0, 0)
    )
    # The center is never negative, so arctan2 equals arctan(v00 / v01) and handles x_c == 0.
    return np.arctan2(neighbors - 8 * center, center)


def gradient_orientation(tile: np.ndarray) -> np.ndarray:
    """theta' = arctan2(v11, v10) + pi in [0, 2*pi].

    v10 = bottom - top and v11 = left - right neighbor of every interior pixel.
    """
    x = tile.astype(np.int32)
    v10 = x[2:, 1:-1] - x[:-2, 1:-1]
    v11 = x[1:-1, :-2] - x[1:-1, 2:]
    return np.arctan2(v11, v10) + np.pi


def dominant_orientation(theta: np.ndarray, t: int = T) -> np.ndarray:
    """Quantize theta' to the nearest of `t` dominant orientations 2*k*pi/t (k = 0 .. t-1)."""
    return np.mod(np.floor(theta / (2 * np.pi / t) + 0.5), t).astype(np.intp)


def wld(tile: np.ndarray, m: int = M, t: int = T, s: int = S) -> np.ndarray:
    """Joint excitation/orientation histogram laid out as [segment][orientation][sub-bin]."""
    xi = differential_excitation(tile).ravel()
    phi = dominant_orientation(gradient_orientation(tile), t).ravel()

    excitation_bin = np.clip(np.floor((xi + np.pi / 2) / np.pi * (m * s)), 0, m * s - 1)
    segment, sub_bin = np.divmod(excitation_bin.astype(np.intp), s)

    index = (segment * t + phi) * s + sub_bin
    return np.bincount(index, minlength=m * t * s) / index.size
