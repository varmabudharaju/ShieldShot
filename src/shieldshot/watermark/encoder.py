"""Watermark encoder — embeds payload bits into an image."""

import torch
import torch.nn as nn


class WatermarkEncoder(nn.Module):
    def __init__(self, payload_bits: int = 96, hidden_dim: int = 64):
        super().__init__()
        self.payload_bits = payload_bits
        self.payload_proj = nn.Sequential(
            nn.Linear(payload_bits, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.enc1 = self._conv_block(3 + hidden_dim, hidden_dim)
        self.enc2 = self._conv_block(hidden_dim, hidden_dim * 2)
        self.pool = nn.AvgPool2d(2)
        self.bottleneck = self._conv_block(hidden_dim * 2, hidden_dim * 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = self._conv_block(hidden_dim * 4, hidden_dim)
        self.dec1 = self._conv_block(hidden_dim * 2, hidden_dim)
        self.out = nn.Sequential(nn.Conv2d(hidden_dim, 3, 1), nn.Tanh())
        self.strength = 0.1

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(),
        )

    def forward(self, image, payload):
        B, _, H, W = image.shape
        p = self.payload_proj(payload).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        x = torch.cat([image, p], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
        residual = self.out(d1) * self.strength
        return image + residual
