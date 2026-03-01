"""Tests for generator training utilities."""

import torch
import pytest
from train.train_generator import differentiable_jpeg_approx


def test_differentiable_jpeg_approx_shape():
    img = torch.randn(2, 3, 64, 64).clamp(0, 1)
    result = differentiable_jpeg_approx(img)
    assert result.shape == img.shape


def test_differentiable_jpeg_approx_range():
    img = torch.randn(2, 3, 64, 64).clamp(0, 1)
    result = differentiable_jpeg_approx(img)
    assert result.min() >= -0.01
    assert result.max() <= 1.01


def test_differentiable_jpeg_approx_modifies():
    img = torch.randn(2, 3, 64, 64).clamp(0, 1)
    result = differentiable_jpeg_approx(img, quality=30)
    assert not torch.allclose(result, img, atol=1e-3)


def test_differentiable_jpeg_approx_preserves_grad():
    img = torch.randn(1, 3, 32, 32).clamp(0, 1).detach().requires_grad_(True)
    result = differentiable_jpeg_approx(img)
    result.sum().backward()
    assert img.grad is not None
