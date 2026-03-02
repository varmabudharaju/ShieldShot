"""Test if ShieldShot protection disrupts face recognition models.

Usage:
    python3 scripts/test_protection.py original.jpg protected.jpg
"""

import sys
import torch
import torch.nn.functional as F
from PIL import Image
from shieldshot.utils.image import to_tensor
from shieldshot.perturb.models import load_arcface, load_facenet, _resize_for_model


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 scripts/test_protection.py <original> <protected>")
        sys.exit(1)

    orig_path = sys.argv[1]
    prot_path = sys.argv[2]

    device = "cpu"

    # Load images
    orig_full = Image.open(orig_path).convert("RGB")
    prot_full = Image.open(prot_path).convert("RGB")
    # Resize to same size for quality comparison
    prot_full = prot_full.resize(orig_full.size, Image.LANCZOS)
    orig_t_full = to_tensor(orig_full).to(device)
    prot_t_full = to_tensor(prot_full).to(device)
    # Small version for face models
    orig = orig_full.resize((256, 256), Image.LANCZOS)
    prot = prot_full.resize((256, 256), Image.LANCZOS)
    orig_t = to_tensor(orig).to(device)
    prot_t = to_tensor(prot).to(device)

    print()
    print("=" * 50)
    print("  ShieldShot Protection Test")
    print("=" * 50)
    print(f"  Original:  {orig_path}")
    print(f"  Protected: {prot_path}")
    print("=" * 50)

    # ArcFace
    print()
    print("  ArcFace (used by face-swap tools)")
    print("  " + "-" * 40)
    arcface = load_arcface().to(device).eval()
    with torch.no_grad():
        orig_emb = arcface(_resize_for_model(orig_t, "arcface"))
        prot_emb = arcface(_resize_for_model(prot_t, "arcface"))
    cos_sim = F.cosine_similarity(orig_emb, prot_emb).item()
    match = cos_sim > 0.5
    print(f"  Cosine similarity: {cos_sim:.4f}")
    print(f"  Match as same person: {'YES (not broken)' if match else 'NO (broken!)'}")

    # FaceNet
    print()
    print("  FaceNet (used by face-swap tools)")
    print("  " + "-" * 40)
    facenet = load_facenet().to(device).eval()
    with torch.no_grad():
        orig_emb = facenet(_resize_for_model(orig_t, "facenet"))
        prot_emb = facenet(_resize_for_model(prot_t, "facenet"))
    cos_sim = F.cosine_similarity(orig_emb, prot_emb).item()
    match = cos_sim > 0.5
    print(f"  Cosine similarity: {cos_sim:.4f}")
    print(f"  Match as same person: {'YES (not broken)' if match else 'NO (broken!)'}")

    # Visual quality
    print()
    print("  Visual Quality")
    print("  " + "-" * 40)
    from shieldshot.utils.quality import compute_ssim, compute_lpips
    ssim = compute_ssim(orig_t_full, prot_t_full)
    lpips = compute_lpips(orig_t_full, prot_t_full)
    print(f"  SSIM:  {ssim:.4f}  (1.0 = identical, >0.95 = invisible)")
    print(f"  LPIPS: {lpips:.4f}  (0.0 = identical, <0.10 = invisible)")

    print()
    print("=" * 50)
    print()


if __name__ == "__main__":
    main()
