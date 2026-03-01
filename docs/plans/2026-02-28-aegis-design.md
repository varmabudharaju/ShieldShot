# Aegis — Deepfake Protection Tool Design

## Problem

Photos posted on public platforms can be downloaded and used to create deepfakes for blackmail or impersonation. No usable, maintained, open-source tool exists that protects photos from this threat. Existing research tools (Fawkes, PhotoGuard, FaceShield) are academic prototypes — hard to install, unmaintained, and target only single model architectures.

## Solution

Aegis is a Python library + CLI that applies three layers of invisible protection to photos before they're shared publicly:

1. **Adversarial perturbation** — invisible noise that disrupts deepfake models
2. **Invisible watermark** — proves ownership, survives compression and screenshots
3. **C2PA provenance** — cryptographic proof of origin

## High-Level Flow

```
Input Photo
    │
    ├─► Face Detection (RetinaFace)
    │       │
    │       ▼
    │   Adversarial Perturbation (Generator or PGD)
    │       │ (targets ArcFace, FaceNet, SD VAE, e4e)
    │       ▼
    │   Perturbation applied to face regions
    │
    ├─► Watermark Encoder (full image)
    │       │ (trained encoder-decoder network)
    │       ▼
    │   Invisible watermark embedded
    │
    ├─► C2PA Metadata Signing
    │
    ▼
Protected Photo (visually identical)
```

## CLI Interface

```bash
# First-time setup — generate signing key, download models
aegis init

# Protect a photo (fast mode — generator network, default)
aegis protect photo.jpg -o protected.jpg

# Protect with max strength (PGD optimization)
aegis protect photo.jpg -o protected.jpg --mode thorough

# Batch protect a directory
aegis protect ./photos/ -o ./protected/

# Verify provenance
aegis verify protected.jpg

# Extract watermark (prove ownership)
aegis extract protected.jpg
```

## Layer 1: Adversarial Perturbation

### Target Models

| Category | Models | Reason |
|----------|--------|--------|
| Face encoders | ArcFace, FaceNet | Used by face-swap tools (DeepFaceLab, FaceSwap) |
| Diffusion VAE | Stable Diffusion VAE | Used for Dreambooth/LoRA fine-tuning on faces |
| GAN encoders | e4e, pSp | Used by GAN-based face-swap pipelines |

### Two Modes of Operation

**Fast mode (default) — Perturbation Generator Network:**

A trained U-Net that predicts adversarial perturbation in a single forward pass.

- Speed: ~200-500ms on GPU, ~2-5s on CPU
- Protection: ~80-90% as effective as PGD
- Mobile-viable for future app

Training process:
1. Compute PGD perturbations for thousands of face images (offline, one-time)
2. Train a U-Net to predict those perturbations from input images
3. The generator learns generalized adversarial patterns

Generator training loss:
```
L = λ₁·L_distortion (maximize feature divergence across all target models)
  + λ₂·L_quality (LPIPS/SSIM — output must be visually identical)
  + λ₃·L_compression (survive JPEG — apply JPEG augmentation in training)
```

**Thorough mode — Per-Image PGD Optimization:**

Iterative Projected Gradient Descent customized per image.

- Speed: ~10-25s on GPU, ~3-6 min on CPU
- Protection: Maximum strength, custom-tailored per image

Process:
1. Detect faces using RetinaFace
2. For each face, run PGD (100-200 iterations):
   - Forward pass through all target models
   - Compute loss that maximizes feature distortion across all models
   - Backpropagate to update perturbation
   - Project back within imperceptibility budget (L∞ ε ≈ 8/255)
3. Quality gate — reject if SSIM < 0.95 or LPIPS exceeds threshold

Compression resilience: During PGD optimization, randomly apply JPEG compression (quality 60-95) and Gaussian blur before computing loss. Forces perturbation into frequencies that survive compression.

### Multi-Model Loss Function

```
L_total = α·L_arcface + β·L_facenet + γ·L_sd_vae + δ·L_gan_encoder

Each L = maximize cosine distance between clean and perturbed features
```

Weights (α, β, γ, δ) are tunable to prioritize specific threats.

### Design Decisions

- **Face region only** — perturbation targets face + surrounding area, not entire image
- **Both modes ship in v1** — fast (default) for everyday use, thorough for high-risk situations
- **Generator enables mobile path** — small enough to run on-device in future app

## Layer 2: Invisible Watermark

### Architecture

Trained encoder-decoder network:

```
Encoder: Input Image + Payload ──► U-Net ──► Watermarked Image (visually identical)
Decoder: Watermarked Image ──► Small CNN ──► Payload bits
```

### Payload (64-128 bits)

- User ID hash (32 bits)
- Timestamp (24 bits)
- Image perceptual hash (32 bits)
- Reed-Solomon error correction (remaining bits)

### Training

The encoder-decoder is trained with a noise/attack layer between them:

```
Encoder ──► Noise Layer (random per batch) ──► Decoder
                │
                ├── JPEG compression (quality 50-95)
                ├── Screenshot simulation (gamma, blur, rescale)
                ├── Cropping (random 50-90%)
                ├── Gaussian noise
                ├── Brightness/contrast shifts
                └── Resolution changes
```

Training loss:
```
L = λ₁·L_image (LPIPS — watermarked looks like original)
  + λ₂·L_payload (BCE — decoded bits match embedded bits)
  + λ₃·L_adversarial_compat (don't interfere with perturbation layer)
```

The compatibility term ensures the watermark avoids frequency bands used by the adversarial perturbation.

Training data: CelebA-HQ or FFHQ. We train the network to hide/recover bits, not to learn identities.

### Robustness Targets

| Attack | Bit accuracy target |
|--------|-------------------|
| Clean (no attack) | >99% |
| JPEG quality 70 | >95% |
| Screenshot simulation | >90% |
| 50% crop (face visible) | >85% |
| Combined (JPEG + resize) | >85% |

## Layer 3: C2PA Provenance

### Implementation

Uses the official `c2pa-python` SDK. Embeds a signed manifest:

```
Photo file
  └── C2PA Manifest
        ├── Creator identity (public key)
        ├── Timestamp (signed)
        ├── Software ("aegis v1.0")
        ├── Thumbnail of original
        ├── Hash of original pixels
        └── Signature
```

### Key Setup

`aegis init` generates a self-signed certificate stored at `~/.aegis/keys/`. Users can import existing certificates.

### Limitations

- Most platforms strip C2PA metadata today (Instagram, Twitter, WhatsApp)
- Self-signed keys prove "same author" not "real identity"
- Useful for legal disputes where you have the original file
- Included because it's nearly zero cost and adoption is growing (Google, Meta, Adobe, BBC)

## Screenshot Resilience

| Layer | Survives screenshot? | Notes |
|-------|---------------------|-------|
| Adversarial perturbation | Mostly yes | Pixel noise baked into image, captured in screenshot |
| Invisible watermark | Yes (trained for it) | Encoder-decoder trained with screenshot augmentation |
| C2PA metadata | No | Metadata stripped — watermark is the fallback |

## Project Structure

```
aegis/
├── src/
│   └── aegis/
│       ├── __init__.py
│       ├── cli.py                  # Click CLI
│       ├── protect.py              # Orchestrates all three layers
│       ├── detect/
│       │   └── face_detector.py    # RetinaFace wrapper
│       ├── perturb/
│       │   ├── pgd.py              # PGD optimization loop
│       │   ├── generator.py        # Trained perturbation generator
│       │   ├── losses.py           # Multi-model loss functions
│       │   └── models.py           # Target model loaders
│       ├── watermark/
│       │   ├── encoder.py          # U-Net watermark embedder
│       │   ├── decoder.py          # CNN watermark extractor
│       │   └── payload.py          # Bit packing + Reed-Solomon
│       ├── provenance/
│       │   └── c2pa.py             # C2PA signing/verification
│       └── utils/
│           ├── image.py            # Image I/O, format handling
│           └── quality.py          # SSIM/LPIPS quality checks
├── train/
│   ├── train_watermark.py          # Watermark encoder-decoder training
│   ├── train_generator.py          # Perturbation generator training
│   ├── generate_pgd_targets.py     # Generate PGD targets for generator training
│   ├── datasets.py                 # Dataset loaders
│   └── augmentations.py           # Screenshot/JPEG/crop simulations
├── models/
│   └── download_models.py          # Download pretrained weights
├── tests/
│   ├── test_face_detection.py
│   ├── test_perturbation.py
│   ├── test_generator.py
│   ├── test_watermark.py
│   ├── test_provenance.py
│   ├── test_quality.py
│   └── test_integration.py
├── docs/
│   └── plans/
├── pyproject.toml
└── README.md
```

## Tech Stack

- Python 3.11+
- PyTorch (training and inference)
- Click (CLI)
- Pillow / OpenCV (image processing)
- c2pa-python (provenance)
- facenet-pytorch / insightface (face detection + target models)

## Testing Strategy

| Layer | Tests | Method |
|-------|-------|--------|
| Face detection | Finds faces, handles no-face | Sample images with known face locations |
| Perturbation (PGD) | Breaks target models invisibly | Assert cosine distance increased AND SSIM > 0.95 |
| Perturbation (generator) | Single-pass output is effective | Compare generator output vs PGD baseline |
| Watermark | Payload survives attacks | Embed → attack → extract → assert bit accuracy |
| Provenance | Sign/verify round-trip | Sign → verify → assert valid |
| Quality gate | Rejects bad output | Intentionally strong perturbation → assert rejection |
| Integration | Full pipeline end-to-end | Input → protect → verify all layers |

## Performance Targets

| Metric | Target |
|--------|--------|
| Fast mode (GPU) | <1s per photo |
| Fast mode (CPU) | <5s per photo |
| Thorough mode (GPU) | <30s per photo |
| Visual quality (SSIM) | >0.95 |
| Visual quality (LPIPS) | <0.05 |
| Feature distortion on target models | >60% accuracy drop |
| Watermark bit accuracy post-JPEG | >95% |
| Watermark bit accuracy post-screenshot | >90% |

## Success Criteria for v1

1. Protected photo is visually indistinguishable from original
2. Face feature extraction accuracy drops >60% on target models
3. Watermark survives JPEG Q70 with >95% bit accuracy
4. Watermark survives simulated screenshot with >90% bit accuracy
5. Fast mode runs in <1s on consumer GPU
6. `pip install aegis` and it works

## Out of Scope (Future)

- Mobile app (needs generator model optimization)
- Video protection
- Real-time camera filter
- CA-signed certificates
- Platform API integrations
- Adversarial purification detection
- Web UI
- Multi-face prioritization ("protect only my face")
