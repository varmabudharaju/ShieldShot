"""Watermark training script — trains encoder & decoder to embed/extract payloads.

Usage:
    python3 -m train.train_watermark --data-dir /path/to/face/images

Trains WatermarkEncoder + WatermarkDecoder with a noise layer so watermarks
survive JPEG compression, screenshots, and cropping. Saves encoder.pt and
decoder.pt to the output directory.
"""

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pytorch_msssim import ssim

from shieldshot.utils.image import to_tensor
from shieldshot.watermark.encoder import WatermarkEncoder
from shieldshot.watermark.decoder import WatermarkDecoder
from shieldshot.watermark.payload import PAYLOAD_BITS
from train.augmentations import apply_random_augmentation


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

    # Dataset
    dataset = FaceImageDataset(args.data_dir, image_size=args.image_size)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
    )
    print(f"Dataset: {len(dataset)} images")

    # Optimizers
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Loss weights
    bce_loss = nn.BCEWithLogitsLoss()

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        decoder.train()
        epoch_loss = 0.0
        epoch_bce = 0.0
        epoch_ssim = 0.0
        n_batches = 0

        for images in loader:
            images = images.to(device)
            B = images.shape[0]
            payload = random_payload(B).to(device)

            # Encode watermark
            watermarked = encoder(images, payload)

            # Apply random augmentation (noise layer)
            augmented = torch.stack([
                apply_random_augmentation(watermarked[i:i+1]).squeeze(0)
                for i in range(B)
            ])

            # Decode payload from augmented image
            logits = decoder(augmented)

            # Payload recovery loss (BCE)
            loss_bce = bce_loss(logits, payload)

            # Image quality loss (1 - SSIM)
            loss_ssim = 1.0 - ssim(
                watermarked, images,
                data_range=1.0, size_average=True,
            )

            # Combined loss
            loss = args.w_bce * loss_bce + args.w_ssim * loss_ssim

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_bce += loss_bce.item()
            epoch_ssim += loss_ssim.item()
            n_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_bce = epoch_bce / max(n_batches, 1)
        avg_ssim = epoch_ssim / max(n_batches, 1)
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"BCE: {avg_bce:.4f} | "
            f"SSIM-loss: {avg_ssim:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        # Save checkpoints periodically
        if epoch % args.save_every == 0 or epoch == args.epochs:
            out = Path(args.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            torch.save(encoder.state_dict(), out / "encoder.pt")
            torch.save(decoder.state_dict(), out / "decoder.pt")
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
    parser.add_argument("--w-ssim", type=float, default=0.5,
                        help="Weight for SSIM image quality loss")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=10,
                        help="Save checkpoints every N epochs")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
