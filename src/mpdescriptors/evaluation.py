"""Query-by-example retrieval: every tile queries all others, ranked by Euclidean distance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist


@dataclass
class RetrievalResult:
    precision: np.ndarray  # (queries, R): precision when the k-th relevant tile is retrieved

    @property
    def recall(self) -> np.ndarray:
        r = self.precision.shape[1]
        return np.arange(1, r + 1) / r

    @property
    def mean_curve(self) -> np.ndarray:
        return self.precision.mean(axis=0)

    @property
    def mean_average_precision(self) -> float:
        return float(self.precision.mean(axis=1).mean())

    @property
    def auc_pr(self) -> float:
        """Trapezoidal area under the mean curve, held flat from recall 0 to the first point."""
        recall = np.concatenate([[0.0], self.recall])
        curve = np.concatenate([self.mean_curve[:1], self.mean_curve])
        return float(np.trapezoid(curve, recall))


def evaluate(features: np.ndarray, labels: np.ndarray, chunk_size: int = 256) -> RetrievalResult:
    """Rank the whole collection for each query; classes must all have the same size."""
    counts = np.unique(labels, return_counts=True)[1]
    if counts.min() != counts.max() or counts.min() < 2:
        raise ValueError("Every class needs the same number (>= 2) of samples.")
    relevant_per_query = counts[0] - 1

    n = len(labels)
    ranks = np.arange(1, n)
    precision = np.empty((n, relevant_per_query))

    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        distances = cdist(features[start:stop], features)
        rows = np.arange(stop - start)
        distances[rows, rows + start] = np.inf  # exclude the query itself

        # The query sorts last; ties keep collection order.
        order = np.argsort(distances, axis=1, kind="stable")[:, :-1]
        relevant = labels[order] == labels[start:stop, None]
        hits = np.cumsum(relevant, axis=1)
        precision[start:stop] = (hits / ranks)[relevant].reshape(-1, relevant_per_query)

    return RetrievalResult(precision)
