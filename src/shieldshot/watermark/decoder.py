"""Watermark decoder — extracts payload bits from a watermarked image."""

import torch
import torch.nn as nn


class WatermarkDecoder(nn.Module):
    def __init__(self, payload_bits: int = 96):
        super().__init__()
        self.payload_bits = payload_bits
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, payload_bits),
        )

    def forward(self, image):
        x = self.features(image)
        x = x.flatten(1)
        return self.classifier(x)
