import numpy as np
import pytest

from mpdescriptors.descriptors.lbp import NEIGHBOR_OFFSETS, lbp, lbp_codes


def test_hand_computed_code():
    tile = np.array([[5, 9, 1], [4, 6, 7], [2, 6, 3]], dtype=np.uint8)
    # Neighbors >= 6, clockwise from top-left: 9 (bit 1), 7 (bit 3), 6 (bit 5).
    assert lbp_codes(tile).tolist() == [[0b101010]]


def test_matches_naive_loop():
    tile = np.random.default_rng(1).integers(0, 256, (16, 12), dtype=np.uint8)
    expected = np.zeros((14, 10), dtype=np.uint8)
    for y in range(1, 15):
        for x in range(1, 11):
            for bit, (dy, dx) in enumerate(NEIGHBOR_OFFSETS):
                if tile[y + dy, x + dx] >= tile[y, x]:
                    expected[y - 1, x - 1] |= 1 << bit
    np.testing.assert_array_equal(lbp_codes(tile), expected)


def test_constant_tile_is_all_ones_pattern():
    hist = lbp(np.full((5, 5), 42, dtype=np.uint8))
    assert hist.shape == (256,)
    assert hist[255] == 1.0


def test_histogram_is_normalized():
    tile = np.random.default_rng(2).integers(0, 256, (20, 20), dtype=np.uint8)
    assert lbp(tile).sum() == pytest.approx(1.0)
