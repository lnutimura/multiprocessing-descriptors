import numpy as np

from mpdescriptors.descriptors import DESCRIPTORS
from mpdescriptors.extraction import extract, standardize


def test_process_pool_matches_serial():
    tiles = np.random.default_rng(0).integers(0, 256, (6, 24, 24), dtype=np.uint8)
    for descriptor in DESCRIPTORS.values():
        np.testing.assert_allclose(
            extract(descriptor, tiles, workers=2), extract(descriptor, tiles, workers=1)
        )


def test_standardize():
    features = np.array([[1.0, 5.0], [3.0, 5.0]])
    np.testing.assert_allclose(standardize(features), [[-1.0, 0.0], [1.0, 0.0]])
