"""Tests for adversarial loss functions."""

import pytest
import torch

from shieldshot.perturb.losses import cosine_distance_loss, multi_model_loss


def test_cosine_distance_identical():
    emb = torch.randn(1, 512)
    loss = cosine_distance_loss(emb, emb.clone())
    assert loss.item() == pytest.approx(0.0, abs=0.01)


def test_cosine_distance_orthogonal():
    emb1 = torch.zeros(1, 512)
    emb1[0, 0] = 1.0
    emb2 = torch.zeros(1, 512)
    emb2[0, 1] = 1.0
    loss = cosine_distance_loss(emb1, emb2)
    assert loss.item() > 0.5


def test_multi_model_loss_returns_scalar():
    clean = {"arcface": torch.randn(1, 512), "facenet": torch.randn(1, 512)}
    perturbed = {"arcface": torch.randn(1, 512), "facenet": torch.randn(1, 512)}
    loss = multi_model_loss(clean, perturbed)
    assert loss.dim() == 0


def test_multi_model_loss_with_weights():
    clean = {"arcface": torch.randn(1, 512), "facenet": torch.randn(1, 512)}
    perturbed = {"arcface": torch.randn(1, 512), "facenet": torch.randn(1, 512)}
    weights = {"arcface": 2.0, "facenet": 0.5}
    loss = multi_model_loss(clean, perturbed, weights=weights)
    assert loss.dim() == 0
