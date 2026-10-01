import numpy as np
import pytest

from mpdescriptors.descriptors.wld import (
    M,
    S,
    T,
    differential_excitation,
    dominant_orientation,
    gradient_orientation,
    wld,
)


def cross(top, bottom, left, right):
    return np.array([[0, top, 0], [left, 0, right], [0, bottom, 0]], dtype=np.uint8)


@pytest.mark.parametrize(
    ("tile", "expected"),
    [
        (cross(top=0, bottom=1, left=1, right=0), 5 * np.pi / 4),  # v10 > 0, v11 > 0
        (cross(top=1, bottom=0, left=1, right=0), 7 * np.pi / 4),  # v10 < 0, v11 > 0
        (cross(top=1, bottom=0, left=0, right=1), np.pi / 4),  # v10 < 0, v11 < 0
        (cross(top=0, bottom=1, left=0, right=1), 3 * np.pi / 4),  # v10 > 0, v11 < 0
    ],
)
def test_orientation_quadrants(tile, expected):
    assert gradient_orientation(tile)[0, 0] == pytest.approx(expected)


def test_dominant_orientation_rounds_to_nearest():
    step = 2 * np.pi / T
    theta = np.array([0, step / 2 - 1e-9, step / 2, np.pi, 2 * np.pi - 1e-9, 2 * np.pi])
    assert dominant_orientation(theta).tolist() == [0, 0, 1, 4, 0, 0]


def test_differential_excitation():
    tile = np.full((3, 3), 20, dtype=np.uint8)
    tile[1, 1] = 10
    assert differential_excitation(tile)[0, 0] == pytest.approx(np.arctan(8.0))

    tile[1, 1] = 0  # x_c == 0 saturates at +pi/2 instead of dividing by zero
    assert differential_excitation(tile)[0, 0] == pytest.approx(np.pi / 2)


def test_constant_tile_lands_in_zero_excitation_bin():
    hist = wld(np.full((10, 10), 100, dtype=np.uint8))
    # xi = 0 is the first sub-bin of the middle segment; flat gradients give theta' = pi.
    segment, orientation, sub_bin = M // 2, T // 2, 0
    assert hist[(segment * T + orientation) * S + sub_bin] == 1.0


def test_histogram_shape_and_normalization():
    tile = np.random.default_rng(3).integers(0, 256, (32, 32), dtype=np.uint8)
    hist = wld(tile)
    assert hist.shape == (M * T * S,) == (960,)
    assert hist.sum() == pytest.approx(1.0)
