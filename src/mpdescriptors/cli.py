"""Command-line interface: `download` the Brodatz album, then `run` the retrieval benchmark."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from mpdescriptors.dataset import load_tiles
from mpdescriptors.descriptors import DESCRIPTORS
from mpdescriptors.download import download
from mpdescriptors.evaluation import evaluate
from mpdescriptors.extraction import extract
from mpdescriptors.plotting import plot_pr_curves

DEFAULT_DATA_DIR = Path("data/brodatz")


def _download(args: argparse.Namespace) -> int:
    outcome = download(args.data_dir)
    print(
        f"{len(outcome['downloaded'])} downloaded, {len(outcome['skipped'])} already present, "
        f"{len(outcome['missing'])} unavailable {outcome['missing'] or ''}".rstrip()
    )
    return 0


def _run(args: argparse.Namespace) -> int:
    dataset = load_tiles(args.data_dir, args.window)
    tiles_per_class = len(dataset.labels) // len(dataset.class_names)
    print(
        f"Brodatz: {len(dataset.class_names)} textures x {tiles_per_class} tiles of "
        f"{args.window}x{args.window} = {len(dataset.labels)} queries\n"
    )

    print(f"{'Descriptor':<11}{'Dims':>6}{'mAP':>8}{'AUC-PR':>8}{'Time (s)':>10}", flush=True)
    results = {}
    for key in args.descriptor:
        descriptor = DESCRIPTORS[key]
        start = time.perf_counter()
        features = extract(descriptor, dataset.tiles, args.workers)
        elapsed = time.perf_counter() - start

        result = evaluate(features, dataset.labels)
        results[descriptor.name] = result
        print(
            f"{descriptor.name:<11}{features.shape[1]:>6}{result.mean_average_precision:>8.3f}"
            f"{result.auc_pr:>8.3f}{elapsed:>10.1f}",
            flush=True,
        )

    if args.plot:
        title = f"Brodatz retrieval, {args.window}x{args.window} tiles"
        plot_pr_curves(results, args.plot, title)
        print(f"\nSaved precision-recall curves to {args.plot}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mpdescriptors",
        description="Texture retrieval on the Brodatz album with LBP, GLCM and WLD descriptors.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    fetch = commands.add_parser("download", help="Download the Brodatz textures.")
    fetch.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    fetch.set_defaults(handler=_download)

    run = commands.add_parser("run", help="Extract descriptors and evaluate retrieval.")
    run.add_argument(
        "--descriptor",
        nargs="+",
        choices=list(DESCRIPTORS),
        default=list(DESCRIPTORS),
        help="Descriptors to compare (default: all).",
    )
    run.add_argument("--window", type=int, default=160, help="Tile size in pixels (default: 160).")
    run.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    run.add_argument(
        "--workers", type=int, default=None, help="Worker processes (default: all CPUs)."
    )
    run.add_argument("--plot", type=Path, help="Save precision-recall curves to this PNG.")
    run.set_defaults(handler=_run)

    args = parser.parse_args(argv)
    return args.handler(args)
