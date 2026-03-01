# Training Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix all training code gaps before RunPod training run — add LPIPS loss, adversarial compatibility loss, 3-term generator loss, output clamping, validation split, checkpoint resume, and quality gate improvements.

**Architecture:** Upgrade both training scripts with proper loss functions matching the design doc. Watermark training gets LPIPS + adversarial compat loss. Generator training switches from MSE distillation to end-to-end 3-term adversarial loss (no longer needs precomputed PGD deltas). Infrastructure improvements: clamping, validation, resume, better metrics.

**Tech Stack:** PyTorch, lpips (perceptual loss), pytorch-msssim, existing shieldshot modules

---

### Task 1: Add LPIPS dependency

**Files:**
- Modify: `pyproject.toml:12-29`

**Step 1: Add lpips to dependencies**

In `pyproject.toml`, add `"lpips>=0.1.4"` to the `dependencies` list after `"pytorch-msssim>=1.0"`.

**Step 2: Install and verify**

Run: `pip install -e ".[dev]" && python -c "import lpips; print('OK')"`
Expected: OK

**Step 3: Commit**

```
feat: add lpips perceptual loss dependency
```

---

### Task 2: Add LPIPS to quality gate

**Files:**
- Modify: `src/shieldshot/utils/quality.py`
- Modify: `tests/test_quality.py`

**Step 1: Write failing tests**

Add to `tests/test_quality.py`:

```python
def test_compute_lpips_identical(identical_pair):
    t1, t2 = identical_pair
    from shieldshot.utils.quality import compute_lpips
    lpips_val = compute_lpips(t1, t2)
    assert lpips_val < 0.01


def test_compute_lpips_different(different_pair):
    t1, t2 = different_pair
    from shieldshot.utils.quality import compute_lpips
    lpips_val = compute_lpips(t1, t2)
    assert lpips_val > 0.1


def test_check_quality_returns_lpips(identical_pair):
    t1, t2 = identical_pair
    passed, metrics = check_quality(t1, t2)
    assert "lpips" in metrics
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_quality.py -v`
Expected: 3 FAIL (compute_lpips not found, lpips not in metrics)

**Step 3: Implement compute_lpips and update check_quality**

Update `src/shieldshot/utils/quality.py`:

```python
"""Quality gate -- SSIM and LPIPS checks for perturbation visibility."""

import torch
from pytorch_msssim import ssim
import lpips as _lpips

_lpips_model = None


def _get_lpips_model():
    global _lpips_model
    if _lpips_model is None:
        _lpips_model = _lpips.LPIPS(net="squeeze", verbose=False)
    return _lpips_model


def compute_ssim(original: torch.Tensor, modified: torch.Tensor) -> float:
    """Compute SSIM between two image tensors [1, 3, H, W] in [0, 1].

    Returns float in [0, 1] where 1 = identical.
    """
    with torch.no_grad():
        return ssim(original, modified, data_range=1.0, size_average=True).item()


def compute_lpips(original: torch.Tensor, modified: torch.Tensor) -> float:
    """Compute LPIPS perceptual distance between two image tensors [1, 3, H, W] in [0, 1].

    Returns float where 0 = identical, higher = more different.
    """
    model = _get_lpips_model()
    device = original.device
    model = model.to(device)
    with torch.no_grad():
        # LPIPS expects [-1, 1] range
        orig_scaled = original * 2 - 1
        mod_scaled = modified * 2 - 1
        return model(orig_scaled, mod_scaled).item()


def check_quality(
    original: torch.Tensor,
    modified: torch.Tensor,
    ssim_threshold: float = 0.95,
    lpips_threshold: float = 0.05,
) -> tuple[bool, dict]:
    """Check if modified image passes quality gate.

    Returns (passed, metrics_dict).
    """
    ssim_val = compute_ssim(original, modified)
    lpips_val = compute_lpips(original, modified)
    passed = ssim_val >= ssim_threshold and lpips_val <= lpips_threshold
    return passed, {"ssim": ssim_val, "lpips": lpips_val}
```

Note: use `net="squeeze"` for LPIPS — it's the smallest/fastest model (SqueezeNet backbone). VGG is more accurate but ~10x larger; squeeze is fine for a quality gate.

**Step 4: Run tests**

Run: `pytest tests/test_quality.py -v`
Expected: ALL PASS

**Step 5: Commit**

```
feat: add LPIPS perceptual distance to quality gate
```

---

### Task 3: Upgrade watermark training — LPIPS + compat loss + clamping + validation + resume

**Files:**
- Modify: `train/train_watermark.py`
- Create: `tests/test_train_watermark.py`

**Step 1: Write tests for the new training components**

Create `tests/test_train_watermark.py`:

```python
"""Tests for watermark training utilities."""

import torch
import pytest
from train.train_watermark import high_freq_penalty, compute_bit_accuracy


def test_high_freq_penalty_shape():
    residual = torch.randn(2, 3, 64, 64)
    loss = high_freq_penalty(residual)
    assert loss.dim() == 0  # scalar


def test_high_freq_penalty_zero_for_constant():
    # A constant residual has no high-frequency content
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
```

**Step 2: Run to verify failure**

Run: `pytest tests/test_train_watermark.py -v`
Expected: FAIL (functions don't exist yet)

**Step 3: Rewrite train_watermark.py**

Full rewrite of `train/train_watermark.py`:

```python
"""Watermark training script — trains encoder & decoder to embed/extract payloads.

Usage:
    python3 -m train.train_watermark --data-dir /path/to/face/images

Trains WatermarkEncoder + WatermarkDecoder with a noise layer so watermarks
survive JPEG compression, screenshots, and cropping. Saves encoder.pt and
decoder.pt to the output directory.

Loss: BCE (payload) + SSIM (pixel quality) + LPIPS (perceptual quality)
      + high-freq penalty (adversarial compatibility)
"""

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from pytorch_msssim import ssim
import lpips as _lpips

from shieldshot.utils.image import to_tensor
from shieldshot.watermark.encoder import WatermarkEncoder
from shieldshot.watermark.decoder import WatermarkDecoder
from shieldshot.watermark.payload import PAYLOAD_BITS
from train.augmentations import apply_random_augmentation


# Laplacian kernel for high-frequency detection
_LAPLACIAN = torch.tensor(
    [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32
).reshape(1, 1, 3, 3)


def high_freq_penalty(residual: torch.Tensor) -> torch.Tensor:
    """Penalize high-frequency energy in the watermark residual.

    Applies a Laplacian filter per channel and returns the mean L2 norm.
    This steers the watermark away from high-frequency bands used by
    adversarial perturbations.
    """
    B, C, H, W = residual.shape
    lap = _LAPLACIAN.to(residual.device).expand(C, 1, 3, 3)
    filtered = torch.nn.functional.conv2d(residual, lap, padding=1, groups=C)
    return (filtered ** 2).mean()


def compute_bit_accuracy(logits: torch.Tensor, payload: torch.Tensor) -> float:
    """Compute bit-level accuracy between predicted logits and target payload."""
    predicted = (logits > 0).float()
    return (predicted == payload).float().mean().item()


class FaceImageDataset(Dataset):
    """Load face images from a directory."""

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def __init__(self, data_dir: str, image_size: int = 256):
        self.image_size = image_size
        self.paths = [
            p for p in Path(data_dir).iterdir()
            if p.suffix.lower() in self.EXTENSIONS
        ]
        if not self.paths:
            raise ValueError(f"No images found in {data_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
        tensor = to_tensor(img).squeeze(0)  # [3, H, W]
        return tensor


def random_payload(batch_size: int, bits: int = PAYLOAD_BITS) -> torch.Tensor:
    """Generate random binary payload tensor."""
    return torch.randint(0, 2, (batch_size, bits), dtype=torch.float32)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Models
    encoder = WatermarkEncoder(payload_bits=PAYLOAD_BITS).to(device)
    decoder = WatermarkDecoder(payload_bits=PAYLOAD_BITS).to(device)

    # Dataset with validation split
    full_dataset = FaceImageDataset(args.data_dir, image_size=args.image_size)
    val_size = max(1, int(len(full_dataset) * 0.1))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=False,
    )
    print(f"Dataset: {train_size} train, {val_size} val images")

    # Optimizer + scheduler
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Losses
    bce_loss = nn.BCEWithLogitsLoss()
    lpips_model = _lpips.LPIPS(net="squeeze", verbose=False).to(device)

    # Resume from checkpoint
    start_epoch = 1
    if args.resume:
        ckpt_path = Path(args.output_dir) / "checkpoint.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, weights_only=False)
            encoder.load_state_dict(ckpt["encoder"])
            decoder.load_state_dict(ckpt["decoder"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"] + 1
            print(f"Resumed from epoch {ckpt['epoch']}")

    for epoch in range(start_epoch, args.epochs + 1):
        # --- Training ---
        encoder.train()
        decoder.train()
        train_loss = 0.0
        train_bce = 0.0
        train_ssim = 0.0
        train_lpips = 0.0
        train_compat = 0.0
        train_acc = 0.0
        n_train = 0

        for images in train_loader:
            images = images.to(device)
            B = images.shape[0]
            payload = random_payload(B).to(device)

            # Encode watermark
            watermarked = encoder(images, payload).clamp(0, 1)

            # Apply random augmentation (noise layer)
            augmented = torch.stack([
                apply_random_augmentation(watermarked[i:i+1]).squeeze(0)
                for i in range(B)
            ])

            # Decode payload from augmented image
            logits = decoder(augmented)

            # Losses
            loss_bce = bce_loss(logits, payload)

            loss_ssim = 1.0 - ssim(
                watermarked, images,
                data_range=1.0, size_average=True,
            )

            # LPIPS expects [-1, 1]
            loss_lpips = lpips_model(
                watermarked * 2 - 1, images * 2 - 1,
            ).mean()

            # Adversarial compatibility — penalize high-freq watermark energy
            residual = watermarked - images
            loss_compat = high_freq_penalty(residual)

            loss = (
                args.w_bce * loss_bce
                + args.w_ssim * loss_ssim
                + args.w_lpips * loss_lpips
                + args.w_compat * loss_compat
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_bce += loss_bce.item()
            train_ssim += loss_ssim.item()
            train_lpips += loss_lpips.item()
            train_compat += loss_compat.item()
            train_acc += compute_bit_accuracy(logits.detach(), payload)
            n_train += 1

        scheduler.step()

        # --- Validation ---
        encoder.eval()
        decoder.eval()
        val_loss = 0.0
        val_acc = 0.0
        n_val = 0

        with torch.no_grad():
            for images in val_loader:
                images = images.to(device)
                B = images.shape[0]
                payload = random_payload(B).to(device)

                watermarked = encoder(images, payload).clamp(0, 1)

                augmented = torch.stack([
                    apply_random_augmentation(watermarked[i:i+1]).squeeze(0)
                    for i in range(B)
                ])

                logits = decoder(augmented)
                loss_bce = bce_loss(logits, payload)
                val_loss += loss_bce.item()
                val_acc += compute_bit_accuracy(logits, payload)
                n_val += 1

        # Logging
        t_loss = train_loss / max(n_train, 1)
        t_bce = train_bce / max(n_train, 1)
        t_ssim = train_ssim / max(n_train, 1)
        t_lpips = train_lpips / max(n_train, 1)
        t_compat = train_compat / max(n_train, 1)
        t_acc = train_acc / max(n_train, 1)
        v_loss = val_loss / max(n_val, 1)
        v_acc = val_acc / max(n_val, 1)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Loss: {t_loss:.4f} | "
            f"BCE: {t_bce:.4f} | "
            f"SSIM: {1 - t_ssim:.4f} | "
            f"LPIPS: {t_lpips:.4f} | "
            f"Compat: {t_compat:.4f} | "
            f"BitAcc: {t_acc:.2%} | "
            f"ValBCE: {v_loss:.4f} | "
            f"ValAcc: {v_acc:.2%} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        # Save checkpoints
        if epoch % args.save_every == 0 or epoch == args.epochs:
            out = Path(args.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            torch.save(encoder.state_dict(), out / "encoder.pt")
            torch.save(decoder.state_dict(), out / "decoder.pt")
            torch.save({
                "epoch": epoch,
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, out / "checkpoint.pt")
            print(f"  Saved checkpoints to {out}")

    print("Training complete.")


def main():
    parser = argparse.ArgumentParser(description="Train watermark encoder/decoder")
    parser.add_argument("--data-dir", required=True, help="Directory of face images")
    parser.add_argument("--output-dir", default="checkpoints/watermark",
                        help="Where to save encoder.pt and decoder.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--w-bce", type=float, default=1.0,
                        help="Weight for payload BCE loss")
    parser.add_argument("--w-ssim", type=float, default=0.3,
                        help="Weight for SSIM image quality loss")
    parser.add_argument("--w-lpips", type=float, default=0.7,
                        help="Weight for LPIPS perceptual quality loss")
    parser.add_argument("--w-compat", type=float, default=0.1,
                        help="Weight for adversarial compatibility loss")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=10,
                        help="Save checkpoints every N epochs")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
```

**Step 4: Run tests**

Run: `pytest tests/test_train_watermark.py -v`
Expected: ALL PASS

**Step 5: Commit**

```
feat: upgrade watermark training with LPIPS, compat loss, validation, resume
```

---

### Task 4: Rewrite generator training with 3-term adversarial loss

**Files:**
- Modify: `train/train_generator.py`
- Create: `tests/test_train_generator.py`

**Step 1: Write tests for generator training components**

Create `tests/test_train_generator.py`:

```python
"""Tests for generator training utilities."""

import torch
import pytest
from unittest.mock import patch
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
    # Low quality should modify the image noticeably
    assert not torch.allclose(result, img, atol=1e-3)


def test_differentiable_jpeg_approx_preserves_grad():
    img = torch.randn(1, 3, 32, 32, requires_grad=True).clamp(0, 1)
    # Need to detach then re-require grad since clamp breaks the graph
    img2 = img.detach().requires_grad_(True)
    result = differentiable_jpeg_approx(img2)
    result.sum().backward()
    assert img2.grad is not None
```

**Step 2: Run to verify failure**

Run: `pytest tests/test_train_generator.py -v`
Expected: FAIL (function doesn't exist)

**Step 3: Rewrite train_generator.py with 3-term loss**

Full rewrite of `train/train_generator.py`:

```python
"""Train the perturbation generator with end-to-end adversarial loss.

Instead of MSE distillation from PGD deltas, the generator trains directly
against target models with a 3-term loss:
  L = λ₁·L_distortion + λ₂·L_quality + λ₃·L_compression

Usage:
    python -m train.train_generator \
        --data-dir faces/ --epochs 50 --lr 1e-4
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import lpips as _lpips

from shieldshot.perturb.generator import PerturbationGenerator
from shieldshot.perturb.losses import multi_model_loss
from shieldshot.perturb.models import ALL_MODELS, MODEL_LOADERS, _resize_for_model, _run_model
from shieldshot.utils.image import load_image, to_tensor


def differentiable_jpeg_approx(tensor: torch.Tensor, quality: int = 75) -> torch.Tensor:
    """Differentiable JPEG approximation using average-pool blur + noise.

    Real JPEG is non-differentiable (DCT + quantization). This simulates
    its effect with operations that allow gradient flow:
    - Average pooling (simulates block-level DCT quantization loss)
    - Additive noise scaled by quality (simulates quantization error)

    Good enough to teach the generator JPEG-robust features without
    breaking the computation graph.
    """
    # Lower quality = more aggressive pooling and noise
    # Map quality 30-95 to block size 4-2
    block = max(2, min(4, int(4 - (quality - 30) / 65 * 2)))
    noise_scale = max(0.001, (100 - quality) / 100 * 0.05)

    B, C, H, W = tensor.shape
    # Pad to make divisible by block
    pad_h = (block - H % block) % block
    pad_w = (block - W % block) % block
    if pad_h > 0 or pad_w > 0:
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")

    # Average pool then upsample = lossy block compression
    small = torch.nn.functional.avg_pool2d(tensor, block)
    restored = torch.nn.functional.interpolate(
        small, size=tensor.shape[-2:], mode="nearest"
    )

    # Blend original with block-compressed (keep some detail)
    alpha = quality / 100.0
    blended = alpha * tensor + (1 - alpha) * restored

    # Add quantization-like noise
    noise = torch.randn_like(blended) * noise_scale
    result = (blended + noise).clamp(0, 1)

    # Remove padding
    if pad_h > 0 or pad_w > 0:
        result = result[:, :, :H, :W]

    return result


class FaceImageDataset(Dataset):
    """Load face images from a directory."""

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def __init__(self, data_dir: Path, image_size: int = 256) -> None:
        self.image_size = image_size
        self.paths = [
            p for p in Path(data_dir).iterdir()
            if p.suffix.lower() in self.EXTENSIONS
        ]
        if not self.paths:
            raise ValueError(f"No images found in {data_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
        return to_tensor(img).squeeze(0)  # [3, H, W]


def _get_embeddings(
    tensor: torch.Tensor,
    models: list[str],
    no_grad: bool = False,
) -> dict[str, torch.Tensor]:
    """Get embeddings from all target models."""
    embeddings = {}
    for name in models:
        model = MODEL_LOADERS[name]()
        model.eval()
        inp = _resize_for_model(tensor, name)
        if no_grad:
            with torch.no_grad():
                embeddings[name] = _run_model(name, model, inp)
        else:
            embeddings[name] = _run_model(name, model, inp)
    return embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="Train perturbation generator")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory of face images")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--image-size", type=int, default=256, help="Resize images to this size")
    parser.add_argument("--w-distortion", type=float, default=1.0,
                        help="Weight for adversarial distortion loss")
    parser.add_argument("--w-quality", type=float, default=0.5,
                        help="Weight for LPIPS quality loss")
    parser.add_argument("--w-compression", type=float, default=0.3,
                        help="Weight for JPEG compression robustness loss")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("generator.pt"), help="Output weights path")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--save-every", type=int, default=10, help="Save checkpoint every N epochs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset with validation split
    full_dataset = FaceImageDataset(args.data_dir, image_size=args.image_size)
    val_size = max(1, int(len(full_dataset) * 0.1))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=False,
    )
    print(f"Dataset: {train_size} train, {val_size} val images")
    print(f"Target models: {ALL_MODELS}")

    # Models
    generator = PerturbationGenerator().to(device)
    lpips_model = _lpips.LPIPS(net="squeeze", verbose=False).to(device)

    # Pre-load all target models (cached)
    print("Loading target models...")
    for name in ALL_MODELS:
        model = MODEL_LOADERS[name]()
        model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
    print("Target models loaded.")

    optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Resume
    start_epoch = 1
    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "generator_checkpoint.pt"
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, weights_only=False)
        generator.load_state_dict(ckpt["generator"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {ckpt['epoch']}")

    for epoch in range(start_epoch, args.epochs + 1):
        # --- Training ---
        generator.train()
        t_total = 0.0
        t_dist = 0.0
        t_qual = 0.0
        t_comp = 0.0
        n_train = 0

        for images in train_loader:
            images = images.to(device)

            # Clean embeddings (no grad — these are the reference)
            clean_emb = _get_embeddings(images, ALL_MODELS, no_grad=True)

            # Generator forward
            perturbed = generator(images)

            # L_distortion: maximize feature divergence
            perturbed_emb = _get_embeddings(perturbed, ALL_MODELS)
            loss_distortion = multi_model_loss(clean_emb, perturbed_emb)

            # L_quality: LPIPS between original and perturbed
            loss_quality = lpips_model(
                perturbed * 2 - 1, images * 2 - 1,
            ).mean()

            # L_compression: distortion after JPEG compression
            jpeg_quality = random.randint(60, 95)
            compressed = differentiable_jpeg_approx(perturbed, quality=jpeg_quality)
            compressed_emb = _get_embeddings(compressed, ALL_MODELS)
            loss_compression = multi_model_loss(clean_emb, compressed_emb)

            loss = (
                args.w_distortion * loss_distortion
                + args.w_quality * loss_quality
                + args.w_compression * loss_compression
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t_total += loss.item()
            t_dist += loss_distortion.item()
            t_qual += loss_quality.item()
            t_comp += loss_compression.item()
            n_train += 1

        scheduler.step()

        # --- Validation ---
        generator.eval()
        v_dist = 0.0
        v_qual = 0.0
        n_val = 0

        with torch.no_grad():
            for images in val_loader:
                images = images.to(device)
                clean_emb = _get_embeddings(images, ALL_MODELS, no_grad=True)
                perturbed = generator(images)
                perturbed_emb = _get_embeddings(perturbed, ALL_MODELS, no_grad=True)

                v_dist += multi_model_loss(clean_emb, perturbed_emb).item()
                v_qual += lpips_model(
                    perturbed * 2 - 1, images * 2 - 1,
                ).mean().item()
                n_val += 1

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Loss: {t_total / max(n_train, 1):.4f} | "
            f"Distort: {t_dist / max(n_train, 1):.4f} | "
            f"LPIPS: {t_qual / max(n_train, 1):.4f} | "
            f"Compress: {t_comp / max(n_train, 1):.4f} | "
            f"ValDist: {v_dist / max(n_val, 1):.4f} | "
            f"ValLPIPS: {v_qual / max(n_val, 1):.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        # Save checkpoints
        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save(generator.state_dict(), args.output)
            torch.save({
                "epoch": epoch,
                "generator": generator.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, ckpt_path)
            print(f"  Saved checkpoint (epoch {epoch})")

    print(f"Training complete. Saved to {args.output}")


if __name__ == "__main__":
    main()
```

**Step 4: Run tests**

Run: `pytest tests/test_train_generator.py -v`
Expected: ALL PASS

**Step 5: Commit**

```
feat: rewrite generator training with 3-term adversarial loss
```

---

### Task 5: Update RunPod training script

**Files:**
- Modify: `scripts/runpod_train.sh`

**Step 1: Update the RunPod script**

Key changes:
- Remove Step 3 (PGD target generation) — no longer needed
- Update Step 3 (was Step 4) to use new generator training args (no --image-dir/--delta-dir, just --data-dir)
- Add `lpips` to pip install
- Update time estimates (now ~4-5 hours)
- Update step numbering to 4 steps total
- Add `--resume` flag to training commands for crash recovery
- Use a subset for generator training too (target models are expensive)

Rewrite of `scripts/runpod_train.sh`:

```bash
#!/bin/bash
# =============================================================================
# ShieldShot — Full Training Pipeline for RunPod
# =============================================================================
#
# Usage:
#   1. Spin up a RunPod pod with:
#      - Template: RunPod PyTorch 2.x
#      - GPU: RTX 4090 (or A100 for faster)
#      - Disk: 100GB (for dataset + models)
#
#   2. Open terminal and run:
#      wget -O train.sh <url>/scripts/runpod_train.sh
#      chmod +x train.sh
#      ./train.sh
#
#   3. Come back in ~4-5 hours. Trained models will be in /workspace/shieldshot/trained_models/
#
# Estimated time (RTX 4090):
#   Step 1 (setup):           ~10 min
#   Step 2 (watermark):       ~2 hrs
#   Step 3 (generator):       ~2-3 hrs
#   Step 4 (validation):      ~5 min
#   Total:                    ~4-5 hrs, ~$4-5
#
# =============================================================================

set -e  # Exit on any error

WORKSPACE="/workspace"
PROJECT_DIR="$WORKSPACE/shieldshot"
DATA_DIR="$WORKSPACE/data"
FFHQ_DIR="$DATA_DIR/ffhq"
MODELS_DIR="$PROJECT_DIR/trained_models"
LOG_DIR="$PROJECT_DIR/training_logs"

# Number of images for generator training (loads all 5 target models per batch)
GENERATOR_SUBSET_SIZE=5000

echo "============================================="
echo " ShieldShot Training Pipeline"
echo " $(date)"
echo "============================================="
echo ""

# Check GPU
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || {
    echo "ERROR: No GPU detected. This script requires a GPU."
    exit 1
}
echo ""

# =============================================================================
# STEP 1: Setup — Clone repo, install deps, download dataset
# =============================================================================
echo "============================================="
echo " Step 1/4: Setup"
echo "============================================="

# Clone repo (or pull if already exists)
if [ -d "$PROJECT_DIR" ]; then
    echo "Project directory exists, pulling latest..."
    cd "$PROJECT_DIR" && git pull
else
    echo "Cloning ShieldShot..."
    cd "$WORKSPACE"
    # Upload via scp if not on GitHub:
    # scp -r /Users/varma/shieldshot/ runpod:/workspace/shieldshot/
    git clone https://github.com/<YOUR_USERNAME>/shieldshot.git
fi

cd "$PROJECT_DIR"

# Install dependencies
echo "Installing dependencies..."
pip install -e ".[dev]" --quiet
pip install gdown kaggle lpips --quiet

# Create directories
mkdir -p "$DATA_DIR" "$MODELS_DIR" "$LOG_DIR"

# Download FFHQ dataset
if [ -d "$FFHQ_DIR" ] && [ "$(ls -1 "$FFHQ_DIR"/*.png 2>/dev/null | wc -l)" -gt 1000 ]; then
    echo "FFHQ dataset already exists ($(ls -1 "$FFHQ_DIR"/*.png | wc -l) images)"
else
    echo "Downloading FFHQ dataset..."
    mkdir -p "$FFHQ_DIR"

    gdown --fuzzy "https://drive.google.com/uc?id=1SH5a4M5IpEGZINmE_4FnmMKjcRmYEbak" \
        -O "$DATA_DIR/ffhq_thumbs.zip" 2>/dev/null || {
        echo ""
        echo "Auto-download failed. Please download FFHQ manually:"
        echo "  1. Go to: https://github.com/NVlabs/ffhq-dataset"
        echo "  2. Download thumbnails128x128.zip"
        echo "  3. Upload to: $DATA_DIR/ffhq_thumbs.zip"
        echo ""
        echo "Then re-run this script."
        exit 1
    }

    echo "Extracting FFHQ..."
    cd "$DATA_DIR"
    unzip -q ffhq_thumbs.zip -d ffhq_temp
    find ffhq_temp -name "*.png" -exec mv {} "$FFHQ_DIR/" \;
    rm -rf ffhq_temp ffhq_thumbs.zip
    echo "FFHQ ready: $(ls -1 "$FFHQ_DIR"/*.png | wc -l) images"
fi

cd "$PROJECT_DIR"

echo ""
echo "Setup complete."
echo ""

# =============================================================================
# STEP 2: Train Watermark Encoder/Decoder
# =============================================================================
echo "============================================="
echo " Step 2/4: Training Watermark Model"
echo " Started: $(date)"
echo "============================================="

python3 -m train.train_watermark \
    --data-dir "$FFHQ_DIR" \
    --output-dir "$MODELS_DIR" \
    --epochs 100 \
    --batch-size 32 \
    --image-size 256 \
    --lr 1e-3 \
    --w-bce 1.0 \
    --w-ssim 0.3 \
    --w-lpips 0.7 \
    --w-compat 0.1 \
    --num-workers 4 \
    --save-every 10 \
    --resume \
    2>&1 | tee "$LOG_DIR/watermark_training.log"

echo ""
echo "Watermark training complete: $(date)"
echo "Models saved to $MODELS_DIR/encoder.pt and $MODELS_DIR/decoder.pt"
echo ""

# =============================================================================
# STEP 3: Train Perturbation Generator (end-to-end, no PGD targets needed)
# =============================================================================
echo "============================================="
echo " Step 3/4: Training Perturbation Generator"
echo " Using $GENERATOR_SUBSET_SIZE images with 5 target models"
echo " Started: $(date)"
echo "============================================="

# Create a subset for generator training (5 models per batch is expensive)
GEN_SUBSET_DIR="$DATA_DIR/ffhq_gen_subset"
if [ ! -d "$GEN_SUBSET_DIR" ] || [ "$(ls -1 "$GEN_SUBSET_DIR" 2>/dev/null | wc -l)" -lt "$GENERATOR_SUBSET_SIZE" ]; then
    echo "Creating subset of $GENERATOR_SUBSET_SIZE images for generator training..."
    mkdir -p "$GEN_SUBSET_DIR"
    ls "$FFHQ_DIR"/*.png | shuf -n "$GENERATOR_SUBSET_SIZE" | while read f; do
        ln -sf "$f" "$GEN_SUBSET_DIR/$(basename "$f")"
    done
    echo "Subset ready: $(ls -1 "$GEN_SUBSET_DIR" | wc -l) images"
fi

python3 -m train.train_generator \
    --data-dir "$GEN_SUBSET_DIR" \
    --epochs 50 \
    --batch-size 8 \
    --image-size 256 \
    --lr 1e-4 \
    --w-distortion 1.0 \
    --w-quality 0.5 \
    --w-compression 0.3 \
    --num-workers 4 \
    --save-every 10 \
    --resume \
    --output "$MODELS_DIR/generator.pt" \
    2>&1 | tee "$LOG_DIR/generator_training.log"

echo ""
echo "Generator training complete: $(date)"
echo "Model saved to $MODELS_DIR/generator.pt"
echo ""

# =============================================================================
# STEP 4: Validation — Quick sanity check
# =============================================================================
echo "============================================="
echo " Step 4/4: Validation"
echo "============================================="

python3 -c "
import torch
from pathlib import Path

models_dir = Path('$MODELS_DIR')

# Check all model files exist
files = ['encoder.pt', 'decoder.pt', 'generator.pt']
for f in files:
    path = models_dir / f
    if path.exists():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f'  OK: {f} ({size_mb:.1f} MB)')
    else:
        print(f'  MISSING: {f}')

# Quick inference test
print()
print('Running quick inference test...')

from shieldshot.watermark.encoder import WatermarkEncoder
from shieldshot.watermark.decoder import WatermarkDecoder
from shieldshot.watermark.payload import encode_payload, decode_payload, PAYLOAD_BITS
from shieldshot.perturb.generator import PerturbationGenerator
from shieldshot.utils.quality import compute_ssim, compute_lpips

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Test watermark round-trip
encoder = WatermarkEncoder(payload_bits=PAYLOAD_BITS).to(device)
decoder = WatermarkDecoder(payload_bits=PAYLOAD_BITS).to(device)
encoder.load_state_dict(torch.load(models_dir / 'encoder.pt', weights_only=True))
decoder.load_state_dict(torch.load(models_dir / 'decoder.pt', weights_only=True))
encoder.eval()
decoder.eval()

test_img = torch.randn(1, 3, 256, 256).clamp(0, 1).to(device)
payload = encode_payload(user_id='test', timestamp=1234567890)
payload_tensor = torch.tensor(payload, dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    watermarked = encoder(test_img, payload_tensor).clamp(0, 1)
    logits = decoder(watermarked)
    predicted_bits = (logits > 0).int().squeeze(0).tolist()

result = decode_payload(predicted_bits)
ssim_val = compute_ssim(test_img, watermarked)
lpips_val = compute_lpips(test_img, watermarked)

print(f'  Watermark SSIM: {ssim_val:.4f} (target: >0.95)')
print(f'  Watermark LPIPS: {lpips_val:.4f} (target: <0.05)')
print(f'  Watermark payload valid: {result[\"valid\"]}')
print(f'  Watermark bit accuracy: {sum(a == b for a, b in zip(payload, predicted_bits)) / len(payload):.2%}')

# Test generator
gen = PerturbationGenerator().to(device)
gen.load_state_dict(torch.load(models_dir / 'generator.pt', weights_only=True))
gen.eval()

face = torch.randn(1, 3, 256, 256).clamp(0, 1).to(device)
with torch.no_grad():
    perturbed = gen(face)

diff = (perturbed - face).abs().max().item()
gen_ssim = compute_ssim(face, perturbed)
gen_lpips = compute_lpips(face, perturbed)
print(f'  Generator max perturbation: {diff:.4f} (budget: {8/255:.4f})')
print(f'  Generator SSIM: {gen_ssim:.4f} (target: >0.95)')
print(f'  Generator LPIPS: {gen_lpips:.4f} (target: <0.05)')

print()
print('Validation complete.')
"

echo ""
echo "============================================="
echo " ALL DONE"
echo " $(date)"
echo "============================================="
echo ""
echo "Trained models are in: $MODELS_DIR/"
echo ""
echo "To use them locally:"
echo "  1. Download the models:"
echo "     scp runpod:$MODELS_DIR/*.pt ~/.shieldshot/models/"
echo ""
echo "  2. Then run:"
echo "     shieldshot protect photo.jpg -o protected.jpg"
echo ""
echo "Training logs are in: $LOG_DIR/"
echo ""
```

**Step 2: Commit**

```
feat: update RunPod script for new training pipeline (no PGD step)
```

---

### Task 6: Update protect.py to use LPIPS in quality gate

**Files:**
- Modify: `src/shieldshot/protect.py:104-111`

**Step 1: Update quality gate usage in protect.py**

The `check_quality` function now returns LPIPS in metrics. Update the quality gate failure message to include both SSIM and LPIPS:

```python
    # Quality check
    original_tensor = to_tensor(pil_image).to(device)
    passed, metrics = check_quality(original_tensor, watermarked)
    result["ssim"] = metrics["ssim"]
    result["lpips"] = metrics.get("lpips")

    if not passed:
        result["success"] = False
        result["reason"] = (
            f"Quality gate failed: SSIM={metrics['ssim']:.4f}, "
            f"LPIPS={metrics.get('lpips', 'N/A')}"
        )
        return result
```

**Step 2: Run existing tests**

Run: `pytest tests/test_integration.py tests/test_full_pipeline.py -v -m "not slow"`
Expected: ALL PASS

**Step 3: Commit**

```
feat: include LPIPS in quality gate output
```

---

### Task 7: Run full test suite and fix any breakage

**Step 1: Run all tests**

Run: `pytest tests/ -v -m "not slow"`
Expected: ALL PASS

Fix any failures that arise from the changes.

**Step 2: Commit any fixes**

```
fix: resolve test failures from training improvements
```
