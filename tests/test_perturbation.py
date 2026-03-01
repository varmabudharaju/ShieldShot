"""Tests for PGD adversarial perturbation."""

import pytest
import torch

from shieldshot.perturb.pgd import pgd_attack


@pytest.fixture
def dummy_face():
    """Random 112x112 face tensor."""
    return torch.randn(1, 3, 112, 112).clamp(0, 1)


def test_pgd_returns_tensor(dummy_face):
    perturbed = pgd_attack(dummy_face, num_steps=5, epsilon=8 / 255)
    assert isinstance(perturbed, torch.Tensor)
    assert perturbed.shape == dummy_face.shape


def test_pgd_output_within_epsilon(dummy_face):
    epsilon = 8 / 255
    perturbed = pgd_attack(dummy_face, num_steps=5, epsilon=epsilon)
    diff = (perturbed - dummy_face).abs()
    assert diff.max().item() <= epsilon + 1e-5


def test_pgd_output_in_valid_range(dummy_face):
    perturbed = pgd_attack(dummy_face, num_steps=5, epsilon=8 / 255)
    assert perturbed.min() >= 0.0
    assert perturbed.max() <= 1.0


def test_pgd_modifies_image(dummy_face):
    perturbed = pgd_attack(dummy_face, num_steps=10, epsilon=8 / 255)
    assert not torch.allclose(perturbed, dummy_face, atol=1e-6)
