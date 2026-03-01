"""Generate PGD target deltas for training the perturbation generator.

For each face image in --data-dir, runs the iterative PGD attack and saves
the resulting perturbation delta as a .pt file in --output-dir.  These deltas
serve as ground-truth supervision when training the single-pass generator.

Usage:
    python -m train.generate_pgd_targets --data-dir faces/ --output-dir deltas/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from shieldshot.perturb.pgd import pgd_attack
from shieldshot.utils.image import load_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PGD deltas for generator training")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory of face images")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save .pt deltas")
    parser.add_argument("--epsilon", type=float, default=8 / 255, help="L-inf perturbation budget")
    parser.add_argument("--num-steps", type=int, default=100, help="PGD iteration count")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in args.data_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    print(f"Found {len(image_paths)} images in {args.data_dir}")

    for i, img_path in enumerate(image_paths):
        face_tensor = load_image(str(img_path))  # [1, 3, H, W] in [0, 1]
        perturbed = pgd_attack(
            face_tensor,
            num_steps=args.num_steps,
            epsilon=args.epsilon,
        )
        delta = perturbed - face_tensor  # [1, 3, H, W]
        out_path = args.output_dir / f"{img_path.stem}.pt"
        torch.save(delta, out_path)
        if (i + 1) % 10 == 0 or (i + 1) == len(image_paths):
            print(f"  [{i + 1}/{len(image_paths)}] saved {out_path.name}")

    print("Done.")


if __name__ == "__main__":
    main()
