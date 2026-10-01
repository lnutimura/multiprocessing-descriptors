"""Compute a descriptor for every tile, spread over a pool of worker processes."""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tqdm import tqdm

from mpdescriptors.descriptors import Descriptor


def standardize(features: np.ndarray) -> np.ndarray:
    """Z-score each column; constant columns become zero."""
    std = features.std(axis=0)
    std[std == 0] = 1.0
    return (features - features.mean(axis=0)) / std


def extract(descriptor: Descriptor, tiles: np.ndarray, workers: int | None = None) -> np.ndarray:
    """Return an (N, D) feature matrix; `workers=1` runs in-process, `None` uses every CPU."""
    workers = workers or os.cpu_count() or 1
    progress = {"total": len(tiles), "desc": descriptor.name, "unit": "tile", "leave": False}

    if workers == 1:
        features = [descriptor.extract(t) for t in tqdm(tiles, **progress)]
    else:
        chunksize = max(1, len(tiles) // (workers * 8))
        # "spawn": forking is unsafe once tqdm's monitor thread is running.
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            results = pool.map(descriptor.extract, tiles, chunksize=chunksize)
            features = list(tqdm(results, **progress))

    matrix = np.vstack(features)
    return standardize(matrix) if descriptor.standardize else matrix
