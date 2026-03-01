"""Tests for image utilities."""

import numpy as np
import pytest
from PIL import Image

from shieldshot.utils.image import load_image, save_image, to_tensor, to_pil


@pytest.fixture
def sample_image(tmp_path):
    """Create a sample 256x256 RGB image."""
    img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    path = tmp_path / "sample.jpg"
    img.save(path)
    return path


def test_load_image_returns_pil(sample_image):
    img = load_image(str(sample_image))
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"


def test_load_image_nonexistent_raises():
    with pytest.raises(FileNotFoundError):
        load_image("/nonexistent/path.jpg")


def test_save_image_jpeg(tmp_path, sample_image):
    img = load_image(str(sample_image))
    out = tmp_path / "out.jpg"
    save_image(img, str(out), quality=85)
    assert out.exists()
    reloaded = load_image(str(out))
    assert reloaded.size == img.size


def test_save_image_png(tmp_path, sample_image):
    img = load_image(str(sample_image))
    out = tmp_path / "out.png"
    save_image(img, str(out))
    assert out.exists()


def test_to_tensor_shape(sample_image):
    img = load_image(str(sample_image))
    tensor = to_tensor(img)
    assert tensor.shape == (1, 3, 256, 256)
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0


def test_to_pil_roundtrip(sample_image):
    img = load_image(str(sample_image))
    tensor = to_tensor(img)
    result = to_pil(tensor)
    assert isinstance(result, Image.Image)
    assert result.size == img.size
