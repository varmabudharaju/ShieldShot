"""Tests for training augmentations."""

import pytest
import torch
from train.augmentations import (
    jpeg_compress, screenshot_simulate, random_crop_resize, apply_random_augmentation,
)

@pytest.fixture
def sample_tensor():
    return torch.randn(1, 3, 256, 256).clamp(0, 1)

def test_jpeg_compress_shape(sample_tensor):
    result = jpeg_compress(sample_tensor, quality=70)
    assert result.shape == sample_tensor.shape

def test_jpeg_compress_modifies(sample_tensor):
    result = jpeg_compress(sample_tensor, quality=50)
    assert not torch.allclose(result, sample_tensor, atol=1e-3)

def test_screenshot_simulate_shape(sample_tensor):
    result = screenshot_simulate(sample_tensor)
    assert result.shape == sample_tensor.shape

def test_random_crop_resize_shape(sample_tensor):
    result = random_crop_resize(sample_tensor, min_crop=0.7)
    assert result.shape == sample_tensor.shape

def test_apply_random_augmentation_shape(sample_tensor):
    result = apply_random_augmentation(sample_tensor)
    assert result.shape == sample_tensor.shape
