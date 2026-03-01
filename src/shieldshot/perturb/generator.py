"""Perturbation generator — single-pass adversarial noise prediction."""

import torch
import torch.nn as nn


class PerturbationGenerator(nn.Module):
    def __init__(self, epsilon: float = 8 / 255, hidden_dim: int = 64):
        super().__init__()
        self.epsilon = epsilon
        self.enc1 = self._conv_block(3, hidden_dim)
        self.enc2 = self._conv_block(hidden_dim, hidden_dim * 2)
        self.enc3 = self._conv_block(hidden_dim * 2, hidden_dim * 4)
        self.pool = nn.AvgPool2d(2)
        self.bottleneck = self._conv_block(hidden_dim * 4, hidden_dim * 4)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = self._conv_block(hidden_dim * 8, hidden_dim * 2)
        self.dec2 = self._conv_block(hidden_dim * 4, hidden_dim)
        self.dec1 = self._conv_block(hidden_dim * 2, hidden_dim)
        self.out = nn.Sequential(nn.Conv2d(hidden_dim, 3, 1), nn.Tanh())

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(),
        )

    def forward(self, image):
        e1 = self.enc1(image)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
        delta = self.out(d1) * self.epsilon
        perturbed = torch.clamp(image + delta, 0, 1)
        return perturbed
