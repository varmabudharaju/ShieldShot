"""Tests for watermark training utilities."""

import torch
import pytest
from train.train_watermark import high_freq_penalty, compute_bit_accuracy


def test_high_freq_penalty_shape():
    residual = torch.randn(2, 3, 64, 64)
    loss = high_freq_penalty(residual)
    assert loss.dim() == 0  # scalar


def test_high_freq_penalty_zero_for_constant():
    residual = torch.ones(1, 3, 64, 64) * 0.01
    loss = high_freq_penalty(residual)
    assert loss.item() < 0.01


def test_high_freq_penalty_higher_for_noise():
    constant = torch.ones(1, 3, 64, 64) * 0.01
    noisy = torch.randn(1, 3, 64, 64) * 0.1
    loss_const = high_freq_penalty(constant)
    loss_noisy = high_freq_penalty(noisy)
    assert loss_noisy > loss_const


def test_bit_accuracy_perfect():
    logits = torch.tensor([[10.0, -10.0, 10.0]])
    payload = torch.tensor([[1.0, 0.0, 1.0]])
    acc = compute_bit_accuracy(logits, payload)
    assert acc == pytest.approx(1.0)


def test_bit_accuracy_half():
    logits = torch.tensor([[10.0, 10.0, -10.0, -10.0]])
    payload = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    acc = compute_bit_accuracy(logits, payload)
    assert acc == pytest.approx(0.5)
