"""Texture descriptors: each maps a uint8 tile to a 1-D feature vector."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from mpdescriptors.descriptors.glcm import glcm
from mpdescriptors.descriptors.lbp import lbp
from mpdescriptors.descriptors.wld import wld


@dataclass(frozen=True)
class Descriptor:
    name: str
    extract: Callable[[np.ndarray], np.ndarray]
    standardize: bool  # z-score columns before ranking (features on different scales)


DESCRIPTORS = {
    "lbp": Descriptor("LBP", lbp, standardize=False),
    "glcm": Descriptor("GLCM", glcm, standardize=True),
    "wld": Descriptor("WLD", wld, standardize=False),
}
