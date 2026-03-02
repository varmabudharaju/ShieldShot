"""Train the perturbation generator with end-to-end adversarial loss.

Instead of MSE distillation from PGD deltas, the generator trains directly
against target models with a 3-term loss:
  L = lambda1 * L_distortion + lambda2 * L_quality + lambda3 * L_compression

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
from shieldshot.utils.image import to_tensor


def differentiable_jpeg_approx(tensor: torch.Tensor, quality: int = 75) -> torch.Tensor:
    """Differentiable JPEG approximation using average-pool blur + noise.

    Real JPEG is non-differentiable (DCT + quantization). This simulates
    its effect with operations that allow gradient flow:
    - Average pooling (simulates block-level DCT quantization loss)
    - Additive noise scaled by quality (simulates quantization error)
    """
    block = max(2, min(4, int(4 - (quality - 30) / 65 * 2)))
    noise_scale = max(0.001, (100 - quality) / 100 * 0.05)

    B, C, H, W = tensor.shape
    pad_h = (block - H % block) % block
    pad_w = (block - W % block) % block
    if pad_h > 0 or pad_w > 0:
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")

    small = torch.nn.functional.avg_pool2d(tensor, block)
    restored = torch.nn.functional.interpolate(
        small, size=tensor.shape[-2:], mode="nearest"
    )

    alpha = quality / 100.0
    blended = alpha * tensor + (1 - alpha) * restored

    noise = torch.randn_like(blended) * noise_scale
    result = (blended + noise).clamp(0, 1)

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
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated list of target models (default: all)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Select target models
    if args.models:
        target_models = [m.strip() for m in args.models.split(",")]
    else:
        target_models = ALL_MODELS

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
    print(f"Target models: {target_models}")

    # Models
    generator = PerturbationGenerator().to(device)
    lpips_model = _lpips.LPIPS(net="squeeze", verbose=False).to(device)

    # Pre-load target models (cached)
    print("Loading target models...")
    for name in target_models:
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
            clean_emb = _get_embeddings(images, target_models, no_grad=True)

            # Generator forward
            perturbed = generator(images)

            # L_distortion: maximize feature divergence
            perturbed_emb = _get_embeddings(perturbed, target_models)
            # multi_model_loss returns -total (for PGD minimization)
            # For generator: minimize -total = maximize distortion
            loss_distortion = multi_model_loss(clean_emb, perturbed_emb)

            # L_quality: LPIPS between original and perturbed
            loss_quality = lpips_model(
                perturbed * 2 - 1, images * 2 - 1,
            ).mean()

            # L_compression: distortion after JPEG compression
            jpeg_quality = random.randint(60, 95)
            compressed = differentiable_jpeg_approx(perturbed, quality=jpeg_quality)
            compressed_emb = _get_embeddings(compressed, target_models)
            loss_compression = multi_model_loss(clean_emb, compressed_emb)

            loss = (
                args.w_distortion * loss_distortion
                + args.w_quality * loss_quality
                + args.w_compression * loss_compression
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
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
                clean_emb = _get_embeddings(images, target_models, no_grad=True)
                perturbed = generator(images)
                perturbed_emb = _get_embeddings(perturbed, target_models, no_grad=True)

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
