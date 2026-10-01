"""Load Brodatz textures and split them into non-overlapping tiles (one class per texture)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class TileSet:
    tiles: np.ndarray  # (N, window, window) uint8
    labels: np.ndarray  # (N,) class index of each tile
    class_names: list[str]


def find_images(data_dir: Path) -> list[Path]:
    """Return the `D<n>.gif` files in `data_dir`, sorted by plate number."""
    paths = [p for p in data_dir.glob("D*.gif") if re.fullmatch(r"D\d+\.gif", p.name)]
    return sorted(paths, key=lambda p: int(p.stem[1:]))


def load_grayscale(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def tile(image: np.ndarray, window: int) -> np.ndarray:
    """Split `image` into non-overlapping `window x window` tiles, row by row; edges are cropped."""
    rows, cols = image.shape[0] // window, image.shape[1] // window
    cropped = image[: rows * window, : cols * window]
    return cropped.reshape(rows, window, cols, window).swapaxes(1, 2).reshape(-1, window, window)


def load_tiles(data_dir: Path, window: int) -> TileSet:
    paths = find_images(data_dir)
    if not paths:
        raise FileNotFoundError(f"No D<n>.gif images found in {data_dir}; run `download` first.")

    images = [load_grayscale(p) for p in paths]
    # Crop every texture to the smallest one so all classes have the same number of tiles.
    height = min(img.shape[0] for img in images)
    width = min(img.shape[1] for img in images)

    per_image = [tile(img[:height, :width], window) for img in images]
    tiles_per_class = len(per_image[0])
    if tiles_per_class < 2:
        raise ValueError(
            f"Window {window} yields {tiles_per_class} tile(s) per {height}x{width} texture; "
            "retrieval needs at least 2."
        )

    return TileSet(
        tiles=np.concatenate(per_image),
        labels=np.repeat(np.arange(len(paths)), tiles_per_class),
        class_names=[p.stem for p in paths],
    )
