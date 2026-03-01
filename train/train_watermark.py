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

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
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
