import numpy as np
import pytest
from skimage.feature import graycomatrix, graycoprops

from mpdescriptors.descriptors.glcm import OFFSETS, cooccurrence, glcm, haralick_features

# scikit-image measures angles with rows pointing down; symmetric matrices make the sign irrelevant.
SKIMAGE_ANGLES = {0: 0.0, 45: 3 * np.pi / 4, 90: np.pi / 2, 135: np.pi / 4}


@pytest.fixture
def image():
    return np.random.default_rng(0).integers(0, 256, (48, 37), dtype=np.uint8)


@pytest.mark.parametrize("angle", OFFSETS)
def test_matches_scikit_image(image, angle):
    reference = graycomatrix(
        image, [1], [SKIMAGE_ANGLES[angle]], levels=256, symmetric=True, normed=True
    )
    p = cooccurrence(image, *OFFSETS[angle])
    np.testing.assert_allclose(p, reference[:, :, 0, 0])

    energy, _, contrast, homogeneity, correlation = haralick_features(p)
    expected = {
        "ASM": energy,
        "contrast": contrast,
        "homogeneity": homogeneity,
        "correlation": correlation,
    }
    for prop, value in expected.items():
        np.testing.assert_allclose(value, graycoprops(reference, prop)[0, 0], err_msg=prop)


def test_entropy_of_uniform_pairs():
    # Two equally likely gray levels in a checkerboard: horizontal pairs are (0,1) and (1,0).
    board = np.indices((8, 8)).sum(axis=0) % 2
    _, entropy, *_ = haralick_features(cooccurrence(board.astype(np.uint8), 0, 1))
    assert entropy == pytest.approx(np.log(2))


def test_constant_tile():
    features = glcm(np.full((10, 10), 7, dtype=np.uint8)).reshape(4, 5)
    np.testing.assert_allclose(features, [[1.0, 0.0, 0.0, 1.0, 1.0]] * 4)
