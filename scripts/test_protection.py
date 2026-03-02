"""Test if ShieldShot protection disrupts face recognition models.

Usage:
    python3 scripts/test_protection.py original.jpg protected.jpg
"""

import sys
import torch
import torch.nn.functional as F
from PIL import Image
from shieldshot.utils.image import load_image, to_tensor
from shieldshot.perturb.models import load_arcface, load_facenet, _resize_for_model


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 scripts/test_protection.py <original> <protected>")
        sys.exit(1)

    orig_path = sys.argv[1]
    prot_path = sys.argv[2]

    device = "cpu"

    # Load images (handles EXIF rotation)
    orig_full = load_image(orig_path)
    prot_full = load_image(prot_path)
    # Ensure same size for quality comparison
    if prot_full.size != orig_full.size:
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

    # Detect and crop faces for model testing
    from shieldshot.detect.face_detector import FaceDetector
    detector = FaceDetector()

    orig_faces = detector.detect(orig_full)
    prot_faces = detector.detect(prot_full)

    if not orig_faces or not prot_faces:
        print("  Could not detect faces. Using full image.")
        orig_face_t = orig_t
        prot_face_t = prot_t
    else:
        # Crop face regions
        ob = orig_faces[0]["bbox"]
        orig_face = orig_full.crop((ob[0], ob[1], ob[2], ob[3])).resize((256, 256), Image.LANCZOS)
        pb = prot_faces[0]["bbox"]
        prot_face = prot_full.crop((pb[0], pb[1], pb[2], pb[3])).resize((256, 256), Image.LANCZOS)
        orig_face_t = to_tensor(orig_face).to(device)
        prot_face_t = to_tensor(prot_face).to(device)
        print(f"  Face detected in both images")

    # ArcFace
    print()
    print("  ArcFace (used by face-swap tools)")
    print("  " + "-" * 40)
    arcface = load_arcface().to(device).eval()
    with torch.no_grad():
        orig_emb = arcface(_resize_for_model(orig_face_t, "arcface"))
        prot_emb = arcface(_resize_for_model(prot_face_t, "arcface"))
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
        orig_emb = facenet(_resize_for_model(orig_face_t, "facenet"))
        prot_emb = facenet(_resize_for_model(prot_face_t, "facenet"))
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
