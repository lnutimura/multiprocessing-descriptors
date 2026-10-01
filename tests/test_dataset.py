import numpy as np
import pytest
from PIL import Image

from mpdescriptors.dataset import find_images, load_tiles, tile


def test_tile_row_by_row_and_crops_edges():
    image = np.random.default_rng(0).integers(0, 256, (643, 643), dtype=np.uint8)
    tiles = tile(image, 160)
    assert tiles.shape == (16, 160, 160)
    np.testing.assert_array_equal(tiles[0], image[:160, :160])
    np.testing.assert_array_equal(tiles[1], image[:160, 160:320])
    np.testing.assert_array_equal(tiles[4], image[160:320, :160])


@pytest.fixture
def data_dir(tmp_path):
    rng = np.random.default_rng(1)
    for name, size in [("D10.gif", 640), ("D2.gif", 643), ("D1.gif", 640)]:
        pixels = rng.integers(0, 256, (size, size), dtype=np.uint8)
        Image.fromarray(pixels).save(tmp_path / name)
    (tmp_path / "notes.gif").touch()
    return tmp_path


def test_find_images_sorts_numerically(data_dir):
    assert [p.name for p in find_images(data_dir)] == ["D1.gif", "D2.gif", "D10.gif"]


def test_load_tiles(data_dir):
    dataset = load_tiles(data_dir, 320)
    assert dataset.class_names == ["D1", "D2", "D10"]
    assert dataset.tiles.shape == (12, 320, 320)
    assert dataset.labels.tolist() == [0] * 4 + [1] * 4 + [2] * 4


def test_window_too_large(data_dir):
    with pytest.raises(ValueError, match="at least 2"):
        load_tiles(data_dir, 400)


def test_missing_dataset(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_tiles(tmp_path, 160)
