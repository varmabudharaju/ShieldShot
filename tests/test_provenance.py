"""Tests for C2PA provenance signing and verification."""

import pytest
import numpy as np
from pathlib import Path
from PIL import Image

from shieldshot.provenance.c2pa import init_keys, sign_image, verify_image


@pytest.fixture
def keys_dir(tmp_path):
    return init_keys(keys_dir=str(tmp_path / "keys"))


@pytest.fixture
def sample_jpeg(tmp_path):
    img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    path = tmp_path / "sample.jpg"
    img.save(path, quality=95)
    return str(path)


def test_init_keys_creates_cert(keys_dir):
    keys_path = Path(keys_dir)
    assert (keys_path / "cert.pem").exists()
    assert (keys_path / "key.pem").exists()


def test_sign_image_creates_output(keys_dir, sample_jpeg, tmp_path):
    output = str(tmp_path / "signed.jpg")
    sign_image(sample_jpeg, output, keys_dir=keys_dir)
    assert Path(output).exists()


def test_verify_signed_image(keys_dir, sample_jpeg, tmp_path):
    output = str(tmp_path / "signed.jpg")
    sign_image(sample_jpeg, output, keys_dir=keys_dir)
    result = verify_image(output)
    assert result["valid"] is True
    assert "shieldshot" in result.get("software", "").lower()


def test_verify_unsigned_image(sample_jpeg):
    result = verify_image(sample_jpeg)
    assert result["valid"] is False
