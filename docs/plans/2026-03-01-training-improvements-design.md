# Training Improvements — Pre-Training Fixes

Goal: Fix all training code gaps before the ~10-hour RunPod run so we don't need to retrain.

## 1. Watermark Training (`train_watermark.py`)

### Loss function: BCE + SSIM + LPIPS + adversarial compatibility

Current: `L = w_bce * BCE + w_ssim * (1 - SSIM)`

New:
```
L = w_bce * BCE(logits, payload)
  + w_ssim * (1 - SSIM(watermarked, original))
  + w_lpips * LPIPS(watermarked, original)
  + w_compat * HighFreqPenalty(watermark_residual)
```

Weights: `w_bce=1.0, w_ssim=0.3, w_lpips=0.7, w_compat=0.1`

**LPIPS** — perceptual loss using pretrained VGG features. Better than SSIM alone at detecting visual artifacts. Add `lpips` pip dependency.

**Adversarial compatibility** — penalize watermark energy in high-frequency bands where adversarial perturbations live. Implementation: compute residual (`watermarked - original`), apply Laplacian filter, penalize L2 norm of the result. Light weight (0.1) so it steers without dominating.

### Output clamping

Clamp encoder output to [0, 1] before computing SSIM/LPIPS. Currently unclamped values leak into quality metrics.

### Validation split + better metrics

- 90/10 train/val split (deterministic seed for reproducibility)
- Log **bit accuracy** each epoch (the real metric, not just BCE loss)
- Log val loss + val bit accuracy to detect overfitting
- Print SSIM value (not just 1-SSIM loss) for readability

### Checkpoint resume

Save full training state each checkpoint:
- encoder + decoder state dicts
- optimizer state dict
- epoch number
- scheduler state dict

Add `--resume` CLI flag to reload and continue from last checkpoint.

## 2. Generator Training (`train_generator.py`)

### Replace MSE with 3-term adversarial loss

Current: `L = MSE(predicted_delta, pgd_delta)` — passive mimicry

New:
```
L = λ₁ * L_distortion (maximize feature divergence on target models)
  + λ₂ * L_quality (LPIPS between perturbed and original)
  + λ₃ * L_compression (apply JPEG, then recompute distortion)
```

Weights: `λ₁=1.0, λ₂=0.5, λ₃=0.3`

**L_distortion** — feed perturbed image through all target models, compute negative cosine similarity vs clean embeddings. Reuses `multi_model_loss` from `losses.py`. This directly optimizes what we care about.

**L_quality** — LPIPS between `generator(image)` and `image`. Keeps perturbations invisible.

**L_compression** — apply random JPEG compression (quality 60-95) to the perturbed output, then recompute L_distortion on the compressed version. Forces the generator to produce perturbations that survive JPEG.

Note: this approach no longer needs precomputed PGD deltas for training. The generator trains end-to-end against the target models directly. PGD deltas can optionally be used for MSE warmup but are not required.

### Training changes

- Load all 5 target models at start (cached in memory)
- Compute clean embeddings once per image (no grad)
- Cosine LR scheduler (currently missing)
- Checkpoint resume support (same pattern as watermark)
- Validation split with metrics: cosine distance per model, LPIPS, SSIM

## 3. Infrastructure

### `generate_pgd_targets.py` — no longer strictly needed

With the 3-term loss, the generator trains directly against target models. We can remove Step 3 from the RunPod pipeline entirely, saving ~5-6 hours. The generator training will be slower per epoch (loads target models) but produces better results.

Optional: keep `generate_pgd_targets.py` as a standalone tool but remove it from the automated pipeline.

### RunPod script updates

- Remove Step 3 (PGD target generation)
- Update Step 4 to use new generator training with target models
- Add LPIPS dependency to install step
- Fix placeholder GitHub URL (leave as scp instruction)
- Update time estimates (~4-5 hours total instead of ~10)

### Dependencies

Add to `pyproject.toml`:
- `lpips>=0.1.4` — perceptual loss

## 4. Quality gate (`quality.py`)

Add LPIPS to the quality gate alongside SSIM:
- `check_quality()` returns both SSIM and LPIPS metrics
- Default thresholds: SSIM > 0.95, LPIPS < 0.05

## Not included

- e4e/pSp GAN encoder targets — skip for v1
- `train/datasets.py` shared module — not worth refactoring, inline datasets work fine
- Batch-vectorized augmentations — JPEG requires PIL round-trip, can't vectorize
