# ShieldShot

Protect your photos from deepfake misuse.

ShieldShot is an open-source Python tool that applies invisible protection to photos before you share them publicly. It disrupts the face recognition models that deepfake tools rely on, making your photos unusable for face-swapping — while looking identical to the original.

## How It Works

```
Input Photo
    |
    +---> Face Detection (RetinaFace)
    |         |
    |         v
    |     Adversarial Perturbation
    |         | (trained generator adds invisible noise
    |         |  that breaks face recognition models)
    |         v
    |     Perturbation applied to face regions
    |
    +---> Invisible Watermark (full image)
    |         | (embeds ownership data: user ID + timestamp)
    |
    +---> C2PA Provenance Signing (optional)
    |
    v
Protected Photo (visually identical, but deepfake-proof)
```

**Three layers of protection:**

1. **Adversarial perturbation** — A trained neural network adds tiny, invisible changes to face regions that cause face recognition models (ArcFace, FaceNet) to completely fail at matching identity. Deepfake tools that depend on these models can't use the photo.
2. **Invisible watermark** — Embeds your user ID and timestamp into the image, providing a proof-of-ownership signal.
3. **C2PA provenance** — Optionally signs the image with a cryptographic certificate for tamper-evident origin proof.

## Test Results

Tested on a real iPhone photo (2316x3088 JPEG). These are actual terminal outputs, not cherry-picked.

### Protection Pipeline Output

```
$ python3 -c "
from shieldshot.protect import protect_image
import json
result = protect_image('IMG_0385.jpeg', 'protected_proof.jpg', mode='fast')
print(json.dumps(result, indent=2))
"

{
  "success": true,
  "faces_found": 1,
  "perturbation_applied": true,
  "watermark_embedded": true,
  "c2pa_signed": false,
  "ssim": 0.9773,
  "lpips": 0.0436,
  "time_seconds": 342.37
}
```

### Face Recognition Disruption Test

The test script detects and crops the face from both images, resizes to model input size, and compares embeddings — exactly what a deepfake tool would do.

```
$ python3 scripts/test_protection.py IMG_0385.jpeg protected_proof.jpg

==================================================
  ShieldShot Protection Test
==================================================
  Original:  IMG_0385.jpeg
  Protected: protected_proof.jpg
==================================================
  Face detected in both images

  ArcFace (used by face-swap tools)
  ----------------------------------------
  Cosine similarity: 0.1879
  Match as same person: NO (broken!)

  FaceNet (used by face-swap tools)
  ----------------------------------------
  Cosine similarity: -0.1143
  Match as same person: NO (broken!)

  Visual Quality
  ----------------------------------------
  SSIM:  0.9746  (1.0 = identical, >0.95 = invisible)
  LPIPS: 0.0471  (0.0 = identical, <0.10 = invisible)

==================================================
```

### What The Numbers Mean

| Metric | Value | What it means |
|--------|-------|---------------|
| ArcFace cosine sim | **0.19** | Face models need >0.5 to match. 0.19 = completely different person. |
| FaceNet cosine sim | **-0.11** | Negative similarity. The model is anti-confident it's the same face. |
| SSIM | **0.97** | Above 0.95 = changes are invisible to the human eye. |
| LPIPS | **0.047** | Below 0.10 = perceptually imperceptible. |

The protected image looks identical to the original (SSIM 0.97), but face recognition AI thinks it's a completely different person (cosine sim 0.19 / -0.11 vs threshold of 0.5).

### Honest Limitations

- **Watermark extraction doesn't work reliably yet.** The watermark decoder was trained to ~62% bit accuracy — above random chance (50%) but below the threshold needed for reliable extraction. The watermark is embedded but can't be decoded back consistently. This is a secondary feature; anti-deepfake perturbation is the primary goal and that works.
- **Slow on CPU.** Protection takes ~5-6 minutes on CPU (MacBook). On a GPU it would be seconds. This is because the pipeline loads face detection + perturbation generator + watermark encoder models.
- **Tested on 2 face recognition models.** ArcFace and FaceNet are the models used by most popular face-swap tools (DeepFaceLab, FaceSwap, Roop). We haven't tested against every possible model.
- **Trained on FFHQ dataset.** The generator was trained on ~4,500 FFHQ face images for 30 epochs. More training data and epochs would improve robustness.
- **Single face tested.** The pipeline handles multiple faces but has only been tested on single-face photos so far.

## Installation

```bash
# Clone the repository
git clone https://github.com/user/shieldshot.git
cd shieldshot

# Install
pip install -e .

# Download trained model weights to ~/.shieldshot/models/
# You need: generator.pt, encoder.pt, decoder.pt
```

**Requirements:** Python 3.11+, PyTorch 2.0+

## Usage

### CLI

```bash
# Protect a photo (fast mode — uses trained generator)
shieldshot protect photo.jpg -o protected.jpg

# Protect with maximum strength (PGD optimization, slower)
shieldshot protect photo.jpg --mode thorough

# Protect with C2PA signing
shieldshot protect photo.jpg --sign

# Verify C2PA provenance
shieldshot verify protected.jpg

# Extract watermark
shieldshot extract protected.jpg
```

### Python API

```python
from shieldshot.protect import protect_image

result = protect_image(
    "photo.jpg",
    "protected.jpg",
    mode="fast",        # "fast" (generator) or "thorough" (PGD)
    user_id="my_id",    # embedded in watermark
)

print(result["ssim"])                # visual quality (>0.95 = invisible)
print(result["perturbation_applied"])  # True if faces found and perturbed
```

### Testing Protection

```bash
# Compare original vs protected
python3 scripts/test_protection.py original.jpg protected.jpg
```

## How It's Trained

The perturbation generator is a U-Net trained end-to-end against face recognition models:

1. **Input:** Face image (256x256)
2. **Generator outputs:** Perturbed image (within epsilon=8/255 budget)
3. **Loss function:** Maximize cosine distance of face embeddings + minimize LPIPS perceptual difference + survive JPEG compression
4. **Target models during training:** ArcFace, FaceNet
5. **Training data:** FFHQ dataset (Flickr Faces HQ)
6. **Infrastructure:** RunPod (NVIDIA GPU), 30 epochs

The watermark encoder/decoder is a separate U-Net pair trained with BCE loss on a 64-bit payload.

## Project Structure

```
src/shieldshot/
    cli.py              # Click CLI (protect, verify, extract, init)
    protect.py          # Main protection pipeline
    detect/
        face_detector.py    # RetinaFace face detection
    perturb/
        generator.py        # Trained perturbation generator (U-Net)
        pgd.py              # PGD attack (thorough mode)
        models.py           # Target model loaders (ArcFace, FaceNet, etc.)
        losses.py           # Cosine distance loss functions
    watermark/
        encoder.py          # U-Net watermark embedder
        decoder.py          # CNN watermark extractor
        payload.py          # Bit packing + Reed-Solomon error correction
    provenance/
        c2pa.py             # C2PA signing and verification
    utils/
        image.py            # Image I/O, EXIF rotation handling
        quality.py          # SSIM and LPIPS quality metrics

train/                  # Training scripts
    train_generator.py      # Train perturbation generator
    train_watermark.py      # Train watermark encoder/decoder
    augmentations.py        # JPEG, screenshot, crop augmentations

scripts/
    test_protection.py      # Test if protection disrupts face models
    runpod_train.sh         # Full training pipeline for RunPod

tests/                  # 69 unit tests (7 slow deselected)
```

### Unit Tests

```
$ python3 -m pytest tests/test_payload.py tests/test_image_utils.py \
    tests/test_quality.py tests/test_losses.py -v

tests/test_payload.py::test_encode_returns_correct_length PASSED
tests/test_payload.py::test_encode_returns_binary PASSED
tests/test_payload.py::test_roundtrip PASSED
tests/test_payload.py::test_different_users_different_payload PASSED
tests/test_payload.py::test_decode_with_bit_errors PASSED
tests/test_image_utils.py::test_load_image_returns_pil PASSED
tests/test_image_utils.py::test_load_image_nonexistent_raises PASSED
tests/test_image_utils.py::test_save_image_jpeg PASSED
tests/test_image_utils.py::test_save_image_png PASSED
tests/test_image_utils.py::test_to_tensor_shape PASSED
tests/test_image_utils.py::test_to_pil_roundtrip PASSED
tests/test_quality.py::test_ssim_identical PASSED
tests/test_quality.py::test_ssim_different PASSED
tests/test_quality.py::test_check_quality_passes_identical PASSED
tests/test_quality.py::test_check_quality_fails_different PASSED
tests/test_quality.py::test_compute_lpips_identical PASSED
tests/test_quality.py::test_compute_lpips_different PASSED
tests/test_quality.py::test_check_quality_returns_lpips PASSED
tests/test_losses.py::test_cosine_distance_identical PASSED
tests/test_losses.py::test_cosine_distance_orthogonal PASSED
tests/test_losses.py::test_multi_model_loss_returns_scalar PASSED
tests/test_losses.py::test_multi_model_loss_with_weights PASSED

======================== 22 passed in 1.25s ========================
```

Total test suite: 76 tests (69 fast + 7 slow that download large models).

## Status

This is an early MVP (v0.1.0). What works and what doesn't:

| Feature | Status | Notes |
|---------|--------|-------|
| Face detection | Working | RetinaFace via insightface |
| Adversarial perturbation (fast) | **Working** | Drops ArcFace from 1.0 to 0.19, FaceNet to -0.11 |
| Adversarial perturbation (thorough) | Working | PGD optimization, slower but stronger |
| Visual quality preservation | **Working** | SSIM 0.97, LPIPS 0.047 — invisible to humans |
| Invisible watermark embedding | Partial | Watermark is embedded but decoder only ~62% accurate |
| Watermark extraction | Not reliable | Needs more training to be usable |
| C2PA provenance signing | Working | Requires c2pa-python |
| GPU acceleration | Not tested locally | Trained on RunPod, runs on CPU locally |
| Multiple faces | Untested | Code handles it, not validated |

## License

MIT
