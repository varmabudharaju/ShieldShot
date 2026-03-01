"""Tests for perturbation generator network."""

import pytest
import torch
from shieldshot.perturb.generator import PerturbationGenerator


@pytest.fixture
def generator():
    return PerturbationGenerator()

@pytest.fixture
def sample_face():
    return torch.randn(1, 3, 112, 112).clamp(0, 1)


def test_generator_output_shape(generator, sample_face):
    perturbed = generator(sample_face)
    assert perturbed.shape == sample_face.shape

def test_generator_output_in_valid_range(generator, sample_face):
    perturbed = generator(sample_face)
    assert perturbed.min() >= 0.0
    assert perturbed.max() <= 1.0

def test_generator_perturbation_bounded(generator, sample_face):
    perturbed = generator(sample_face)
    diff = (perturbed - sample_face).abs().max().item()
    assert diff <= generator.epsilon + 1e-5
