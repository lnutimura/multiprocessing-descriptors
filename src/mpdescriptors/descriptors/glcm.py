"""Gray-Level Co-occurrence Matrix features (Haralick, Shanmugam & Dinstein, 1973)."""

from __future__ import annotations

import numpy as np

LEVELS = 256

# (row, col) offsets at distance 1; the matrices are symmetric, so each covers both directions.
OFFSETS = {0: (0, 1), 45: (-1, 1), 90: (-1, 0), 135: (-1, -1)}

FEATURES = ("energy", "entropy", "contrast", "homogeneity", "correlation")

_I, _J = np.indices((LEVELS, LEVELS))


def cooccurrence(tile: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Symmetric co-occurrence matrix for offset (dy, dx), normalized to sum to 1."""
    h, w = tile.shape
    r0, r1 = max(0, -dy), h - max(0, dy)
    c0, c1 = max(0, -dx), w - max(0, dx)
    a = tile[r0:r1, c0:c1].astype(np.intp)
    b = tile[r0 + dy : r1 + dy, c0 + dx : c1 + dx].astype(np.intp)

    counts = np.bincount((a * LEVELS + b).ravel(), minlength=LEVELS * LEVELS)
    counts = counts.reshape(LEVELS, LEVELS)
    matrix = (counts + counts.T).astype(np.float64)
    return matrix / matrix.sum()


def haralick_features(p: np.ndarray) -> np.ndarray:
    """Energy (ASM), entropy, contrast, homogeneity (IDM) and correlation of a normalized GLCM."""
    diff2 = (_I - _J) ** 2
    nonzero = p[p > 0]

    energy = np.sum(p**2)
    entropy = -np.sum(nonzero * np.log(nonzero))
    contrast = np.sum(p * diff2)
    homogeneity = np.sum(p / (1.0 + diff2))

    mu_i, mu_j = np.sum(_I * p), np.sum(_J * p)
    sigma_i = np.sqrt(np.sum((_I - mu_i) ** 2 * p))
    sigma_j = np.sqrt(np.sum((_J - mu_j) ** 2 * p))
    if sigma_i < 1e-15 or sigma_j < 1e-15:
        # Constant tile: perfectly correlated by convention (same as scikit-image).
        correlation = 1.0
    else:
        correlation = np.sum((_I - mu_i) * (_J - mu_j) * p) / (sigma_i * sigma_j)

    return np.array([energy, entropy, contrast, homogeneity, correlation])


def glcm(tile: np.ndarray) -> np.ndarray:
    """Five Haralick features at 0, 45, 90 and 135 degrees (20 values, grouped by angle)."""
    return np.concatenate(
        [haralick_features(cooccurrence(tile, dy, dx)) for dy, dx in OFFSETS.values()]
    )
