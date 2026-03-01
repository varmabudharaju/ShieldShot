"""Tests for quality gate utilities."""

import numpy as np
import pytest
import torch
from PIL import Image

from shieldshot.utils.image import to_tensor
from shieldshot.utils.quality import compute_ssim, check_quality


@pytest.fixture
def identical_pair():
    """Two identical tensors."""
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    pil = Image.fromarray(img)
    t = to_tensor(pil)
    return t, t.clone()


@pytest.fixture
def different_pair():
    """Two very different tensors."""
    img1 = np.zeros((256, 256, 3), dtype=np.uint8)
    img2 = np.full((256, 256, 3), 255, dtype=np.uint8)
    t1 = to_tensor(Image.fromarray(img1))
    t2 = to_tensor(Image.fromarray(img2))
    return t1, t2


def test_ssim_identical(identical_pair):
    t1, t2 = identical_pair
    ssim = compute_ssim(t1, t2)
    assert ssim > 0.99


def test_ssim_different(different_pair):
    t1, t2 = different_pair
    ssim = compute_ssim(t1, t2)
    assert ssim < 0.1


def test_check_quality_passes_identical(identical_pair):
    t1, t2 = identical_pair
    passed, metrics = check_quality(t1, t2, ssim_threshold=0.95)
    assert passed is True
    assert metrics["ssim"] > 0.95


def test_check_quality_fails_different(different_pair):
    t1, t2 = different_pair
    passed, metrics = check_quality(t1, t2, ssim_threshold=0.95)
    assert passed is False
