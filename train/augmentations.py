"""Training augmentations simulating real-world image degradation."""

import io
import random
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from shieldshot.utils.image import to_pil, to_tensor


def jpeg_compress(tensor, quality=70):
    pil_img = to_pil(tensor)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    compressed = Image.open(buf).convert("RGB")
    return to_tensor(compressed).to(tensor.device)


def screenshot_simulate(tensor):
    gamma = random.uniform(0.8, 1.2)
    result = tensor.clamp(min=1e-6).pow(gamma)
    k = 3
    padding = k // 2
    channels = result.shape[1]
    kernel = torch.ones(channels, 1, k, k, device=tensor.device) / (k * k)
    result = F.conv2d(result, kernel, padding=padding, groups=channels)
    _, _, H, W = result.shape
    scale = random.uniform(0.7, 0.9)
    small = F.interpolate(result, scale_factor=scale, mode="bilinear", align_corners=False)
    result = F.interpolate(small, size=(H, W), mode="bilinear", align_corners=False)
    return result.clamp(0, 1)


def random_crop_resize(tensor, min_crop=0.5):
    _, _, H, W = tensor.shape
    crop_ratio = random.uniform(min_crop, 0.95)
    crop_h, crop_w = int(H * crop_ratio), int(W * crop_ratio)
    top = random.randint(0, H - crop_h)
    left = random.randint(0, W - crop_w)
    cropped = tensor[:, :, top:top + crop_h, left:left + crop_w]
    return F.interpolate(cropped, size=(H, W), mode="bilinear", align_corners=False)


def apply_random_augmentation(tensor):
    aug = random.choice([
        lambda t: jpeg_compress(t, quality=random.randint(50, 95)),
        screenshot_simulate,
        lambda t: random_crop_resize(t, min_crop=0.6),
        lambda t: t + torch.randn_like(t) * random.uniform(0.01, 0.05),
        lambda t: t * random.uniform(0.7, 1.3),
    ])
    return aug(tensor).clamp(0, 1)
