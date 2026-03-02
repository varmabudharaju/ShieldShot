"""Image loading, saving, and tensor conversion utilities."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps


def load_image(path: str) -> Image.Image:
    """Load an image and convert to RGB, applying EXIF rotation."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img = Image.open(p)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def save_image(img: Image.Image, path: str, quality: int = 95) -> None:
    """Save an image. JPEG quality is configurable."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() in (".jpg", ".jpeg"):
        img.save(p, quality=quality)
    else:
        img.save(p)


def to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL Image to float tensor [1, 3, H, W] in [0, 1]."""
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


def to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert float tensor [1, 3, H, W] in [0, 1] to PIL Image."""
    arr = tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy()
    return Image.fromarray((arr * 255).astype(np.uint8))
