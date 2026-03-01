"""Main protection pipeline — orchestrates all three layers."""

import time
from pathlib import Path

import torch

from shieldshot.detect.face_detector import FaceDetector
from shieldshot.perturb.pgd import pgd_attack
from shieldshot.perturb.models import FACE_MODELS, ALL_MODELS
from shieldshot.utils.image import load_image, save_image, to_tensor, to_pil
from shieldshot.utils.quality import check_quality
from shieldshot.watermark.encoder import WatermarkEncoder
from shieldshot.watermark.decoder import WatermarkDecoder
from shieldshot.watermark.payload import encode_payload, PAYLOAD_BITS


def protect_image(
    input_path: str,
    output_path: str,
    mode: str = "fast",
    user_id: str = "default",
    skip_no_face: bool = False,
    sign_c2pa: bool = False,
    keys_dir: str | None = None,
    target_models: list[str] | None = None,
) -> dict:
    start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pil_image = load_image(input_path)
    tensor = to_tensor(pil_image).to(device)

    detector = FaceDetector()
    faces = detector.detect(pil_image)

    result = {
        "success": True,
        "faces_found": len(faces),
        "perturbation_applied": False,
        "watermark_embedded": False,
        "c2pa_signed": False,
        "ssim": 1.0,
    }

    # Adversarial perturbation (face regions)
    if len(faces) > 0:
        for face in faces:
            x1, y1, x2, y2 = face["bbox"]
            _, _, H, W = tensor.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            face_crop = tensor[:, :, y1:y2, x1:x2]

            # Select target models based on mode
            models = target_models
            if models is None:
                models = ALL_MODELS if mode == "thorough" else FACE_MODELS

            if mode == "thorough":
                perturbed_crop = pgd_attack(face_crop, num_steps=100, target_models=models)
            else:
                try:
                    from shieldshot.perturb.generator import PerturbationGenerator

                    gen = PerturbationGenerator().to(device)
                    weights_path = (
                        Path.home() / ".shieldshot" / "models" / "generator.pt"
                    )
                    if weights_path.exists():
                        gen.load_state_dict(
                            torch.load(weights_path, weights_only=True)
                        )
                    gen.eval()
                    with torch.no_grad():
                        perturbed_crop = gen(face_crop)
                except Exception:
                    perturbed_crop = pgd_attack(face_crop, num_steps=20, target_models=models)

            tensor[:, :, y1:y2, x1:x2] = perturbed_crop
        result["perturbation_applied"] = True
    elif not skip_no_face:
        result["success"] = False
        result["reason"] = "No faces detected"
        return result

    # Watermark
    encoder = WatermarkEncoder(payload_bits=PAYLOAD_BITS).to(device)
    wm_weights = Path.home() / ".shieldshot" / "models" / "watermark_encoder.pt"
    if wm_weights.exists():
        encoder.load_state_dict(torch.load(wm_weights, weights_only=True))
    encoder.eval()

    payload_bits = encode_payload(user_id=user_id, timestamp=int(time.time()))
    payload_tensor = (
        torch.tensor(payload_bits, dtype=torch.float32).unsqueeze(0).to(device)
    )

    with torch.no_grad():
        watermarked = encoder(tensor, payload_tensor).clamp(0, 1)
    result["watermark_embedded"] = True

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

    output_pil = to_pil(watermarked)
    save_image(output_pil, output_path)

    # C2PA signing
    if sign_c2pa:
        try:
            from shieldshot.provenance.c2pa import sign_image

            signed_output = output_path + ".tmp"
            sign_image(output_path, signed_output, keys_dir=keys_dir)
            Path(signed_output).rename(output_path)
            result["c2pa_signed"] = True
        except Exception as e:
            result["c2pa_warning"] = f"C2PA signing failed: {e}"

    result["time_seconds"] = round(time.time() - start, 2)
    return result
