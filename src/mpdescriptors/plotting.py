"""Precision-recall plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mpdescriptors.evaluation import RetrievalResult  # noqa: E402


def plot_pr_curves(results: dict[str, RetrievalResult], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, result in results.items():
        label = f"{name} (mAP {result.mean_average_precision:.3f})"
        ax.plot(result.recall, result.mean_curve, marker="o", markersize=3, label=label)

    ax.set(xlabel="Recall", ylabel="Precision", title=title, xlim=(0, 1.02), ylim=(0, 1.02))
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
