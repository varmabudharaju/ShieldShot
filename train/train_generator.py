"""Train the perturbation generator to predict PGD deltas in one forward pass.

Loads face images from --image-dir and corresponding PGD deltas from
--delta-dir (produced by generate_pgd_targets.py), then trains a U-Net
generator to predict those deltas directly.

Usage:
    python -m train.train_generator \
        --image-dir faces/ --delta-dir deltas/ --epochs 50 --lr 1e-4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from shieldshot.perturb.generator import PerturbationGenerator
from shieldshot.utils.image import load_image


class DeltaDataset(Dataset):
    """Pairs face images with precomputed PGD deltas."""

    def __init__(self, image_dir: Path, delta_dir: Path) -> None:
        self.pairs: list[tuple[Path, Path]] = []
        for delta_path in sorted(delta_dir.glob("*.pt")):
            stem = delta_path.stem
            # Try common image extensions
            for ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
                img_path = image_dir / f"{stem}{ext}"
                if img_path.exists():
                    self.pairs.append((img_path, delta_path))
                    break

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, delta_path = self.pairs[idx]
        image = load_image(str(img_path)).squeeze(0)  # [3, H, W]
        delta = torch.load(delta_path, weights_only=True).squeeze(0)  # [3, H, W]
        return image, delta


def main() -> None:
    parser = argparse.ArgumentParser(description="Train perturbation generator")
    parser.add_argument("--image-dir", type=Path, required=True, help="Directory of face images")
    parser.add_argument("--delta-dir", type=Path, required=True, help="Directory of .pt PGD deltas")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output", type=Path, default=Path("generator.pt"), help="Output weights path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DeltaDataset(args.image_dir, args.delta_dir)
    print(f"Dataset: {len(dataset)} image-delta pairs")
    if len(dataset) == 0:
        print("No matching pairs found. Check that image stems match delta stems.")
        return

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    generator = PerturbationGenerator().to(device)
    optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    for epoch in range(args.epochs):
        total_loss = 0.0
        for images, deltas in loader:
            images, deltas = images.to(device), deltas.to(device)

            predicted_perturbed = generator(images)
            predicted_delta = predicted_perturbed - images
            loss = criterion(predicted_delta, deltas)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1}/{args.epochs}  loss={avg_loss:.6f}")

    torch.save(generator.state_dict(), args.output)
    print(f"Saved generator weights to {args.output}")


if __name__ == "__main__":
    main()
