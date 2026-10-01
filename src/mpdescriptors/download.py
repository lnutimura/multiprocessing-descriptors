"""Download the Brodatz texture album (D1.gif ... D112.gif)."""

from __future__ import annotations

import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

BASE_URL = "https://www.ux.uis.no/~tranden/brodatz"
NUM_TEXTURES = 112


def _fetch(name: str, data_dir: Path) -> str:
    target = data_dir / name
    if target.exists():
        return "skipped"
    try:
        with urllib.request.urlopen(f"{BASE_URL}/{name}", timeout=60) as response:
            payload = response.read()
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return "missing"
        raise
    tmp = target.with_suffix(".part")
    tmp.write_bytes(payload)
    tmp.rename(target)
    return "downloaded"


def download(data_dir: Path, workers: int = 8) -> dict[str, list[str]]:
    """Fetch every texture not already in `data_dir`; returns file names grouped by outcome."""
    data_dir.mkdir(parents=True, exist_ok=True)
    names = [f"D{i}.gif" for i in range(1, NUM_TEXTURES + 1)]

    outcome: dict[str, list[str]] = {"downloaded": [], "skipped": [], "missing": []}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = pool.map(lambda name: _fetch(name, data_dir), names)
        for name, status in tqdm(
            zip(names, results, strict=True), total=len(names), desc="Downloading"
        ):
            outcome[status].append(name)
    return outcome
