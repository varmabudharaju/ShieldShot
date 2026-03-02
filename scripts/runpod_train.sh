#!/bin/bash
# =============================================================================
# ShieldShot — Full Training Pipeline for RunPod
# =============================================================================
#
# Usage:
#   1. Spin up a RunPod pod with:
#      - Template: RunPod PyTorch 2.x
#      - GPU: RTX 4090 (or A100 for faster)
#      - Disk: 100GB (for dataset + models)
#
#   2. Open terminal and run:
#      wget -O train.sh <url>/scripts/runpod_train.sh
#      chmod +x train.sh
#      ./train.sh
#
#   3. Come back in ~1-2 hours. Trained models will be in /workspace/shieldshot/trained_models/
#
# Estimated time (RTX 4090):
#   Step 1 (setup):           ~5 min
#   Step 2 (watermark):       ~30-45 min (15k images, 60 epochs)
#   Step 3 (generator):       ~30-45 min (5k images, 30 epochs)
#   Step 4 (validation):      ~5 min
#   Total:                    ~1-1.5 hrs
#
# =============================================================================

set -e  # Exit on any error

WORKSPACE="/workspace"
PROJECT_DIR="$WORKSPACE/shieldshot"
DATA_DIR="$WORKSPACE/data"
FFHQ_DIR="$DATA_DIR/ffhq"
MODELS_DIR="$PROJECT_DIR/trained_models"
LOG_DIR="$PROJECT_DIR/training_logs"

# Training data sizes
WATERMARK_SUBSET_SIZE=15000
GENERATOR_SUBSET_SIZE=5000

echo "============================================="
echo " ShieldShot Training Pipeline"
echo " $(date)"
echo "============================================="
echo ""

# Check GPU
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || {
    echo "ERROR: No GPU detected. This script requires a GPU."
    exit 1
}
echo ""

# =============================================================================
# STEP 1: Setup — Clone repo, install deps, download dataset
# =============================================================================
echo "============================================="
echo " Step 1/4: Setup"
echo "============================================="

# Clone repo (or pull if already exists)
if [ -d "$PROJECT_DIR" ]; then
    echo "Project directory exists, pulling latest..."
    cd "$PROJECT_DIR" && git pull
else
    echo "Cloning ShieldShot..."
    cd "$WORKSPACE"
    git clone https://github.com/varmabudharaju/ShieldShot.git shieldshot
fi

cd "$PROJECT_DIR"

# Install dependencies
echo "Installing dependencies..."
pip install -e ".[dev]" --quiet
pip install gdown kaggle lpips --quiet

# Create directories
mkdir -p "$DATA_DIR" "$MODELS_DIR" "$LOG_DIR"

# Download FFHQ dataset
if [ -d "$FFHQ_DIR" ] && [ "$(ls -1 "$FFHQ_DIR"/*.png 2>/dev/null | wc -l)" -gt 1000 ]; then
    echo "FFHQ dataset already exists ($(ls -1 "$FFHQ_DIR"/*.png | wc -l) images)"
else
    echo "Downloading FFHQ dataset..."
    mkdir -p "$FFHQ_DIR"

    gdown --fuzzy "https://drive.google.com/uc?id=1SH5a4M5IpEGZINmE_4FnmMKjcRmYEbak" \
        -O "$DATA_DIR/ffhq_thumbs.zip" 2>/dev/null || {
        echo ""
        echo "Auto-download failed. Please download FFHQ manually:"
        echo "  1. Go to: https://github.com/NVlabs/ffhq-dataset"
        echo "  2. Download thumbnails128x128.zip"
        echo "  3. Upload to: $DATA_DIR/ffhq_thumbs.zip"
        echo ""
        echo "Then re-run this script."
        exit 1
    }

    echo "Extracting FFHQ..."
    cd "$DATA_DIR"
    unzip -q ffhq_thumbs.zip -d ffhq_temp
    find ffhq_temp -name "*.png" -exec mv {} "$FFHQ_DIR/" \;
    rm -rf ffhq_temp ffhq_thumbs.zip
    echo "FFHQ ready: $(ls -1 "$FFHQ_DIR"/*.png | wc -l) images"
fi

cd "$PROJECT_DIR"

echo ""
echo "Setup complete."
echo ""

# =============================================================================
# STEP 2: Train Watermark Encoder/Decoder
# =============================================================================
echo "============================================="
echo " Step 2/4: Training Watermark Model"
echo " Started: $(date)"
echo "============================================="

# Create a subset for watermark training
WM_SUBSET_DIR="$DATA_DIR/ffhq_wm_subset"
if [ ! -d "$WM_SUBSET_DIR" ] || [ "$(ls -1 "$WM_SUBSET_DIR" 2>/dev/null | wc -l)" -lt "$WATERMARK_SUBSET_SIZE" ]; then
    echo "Creating subset of $WATERMARK_SUBSET_SIZE images for watermark training..."
    mkdir -p "$WM_SUBSET_DIR"
    ls "$FFHQ_DIR"/*.png | shuf -n "$WATERMARK_SUBSET_SIZE" | while read f; do
        ln -sf "$f" "$WM_SUBSET_DIR/$(basename "$f")"
    done
    echo "Subset ready: $(ls -1 "$WM_SUBSET_DIR" | wc -l) images"
fi

python3 -m train.train_watermark \
    --data-dir "$WM_SUBSET_DIR" \
    --output-dir "$MODELS_DIR" \
    --epochs 60 \
    --batch-size 32 \
    --image-size 256 \
    --lr 1e-3 \
    --w-bce 5.0 \
    --w-ssim 0.1 \
    --w-lpips 0.3 \
    --w-compat 0.05 \
    --num-workers 4 \
    --save-every 10 \
    --resume \
    2>&1 | tee "$LOG_DIR/watermark_training.log"

echo ""
echo "Watermark training complete: $(date)"
echo "Models saved to $MODELS_DIR/encoder.pt and $MODELS_DIR/decoder.pt"
echo ""

# =============================================================================
# STEP 3: Train Perturbation Generator (end-to-end, no PGD targets needed)
# =============================================================================
echo "============================================="
echo " Step 3/4: Training Perturbation Generator"
echo " Using $GENERATOR_SUBSET_SIZE images with 5 target models"
echo " Started: $(date)"
echo "============================================="

# Create a subset for generator training (5 models per batch is expensive)
GEN_SUBSET_DIR="$DATA_DIR/ffhq_gen_subset"
if [ ! -d "$GEN_SUBSET_DIR" ] || [ "$(ls -1 "$GEN_SUBSET_DIR" 2>/dev/null | wc -l)" -lt "$GENERATOR_SUBSET_SIZE" ]; then
    echo "Creating subset of $GENERATOR_SUBSET_SIZE images for generator training..."
    mkdir -p "$GEN_SUBSET_DIR"
    ls "$FFHQ_DIR"/*.png | shuf -n "$GENERATOR_SUBSET_SIZE" | while read f; do
        ln -sf "$f" "$GEN_SUBSET_DIR/$(basename "$f")"
    done
    echo "Subset ready: $(ls -1 "$GEN_SUBSET_DIR" | wc -l) images"
fi

python3 -m train.train_generator \
    --data-dir "$GEN_SUBSET_DIR" \
    --epochs 30 \
    --batch-size 8 \
    --image-size 256 \
    --lr 1e-4 \
    --w-distortion 1.0 \
    --w-quality 0.5 \
    --w-compression 0.3 \
    --num-workers 4 \
    --save-every 10 \
    --resume \
    --output "$MODELS_DIR/generator.pt" \
    2>&1 | tee "$LOG_DIR/generator_training.log"

echo ""
echo "Generator training complete: $(date)"
echo "Model saved to $MODELS_DIR/generator.pt"
echo ""

# =============================================================================
# STEP 4: Validation — Quick sanity check
# =============================================================================
echo "============================================="
echo " Step 4/4: Validation"
echo "============================================="

python3 -c "
import torch
from pathlib import Path

models_dir = Path('$MODELS_DIR')

# Check all model files exist
files = ['encoder.pt', 'decoder.pt', 'generator.pt']
for f in files:
    path = models_dir / f
    if path.exists():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f'  OK: {f} ({size_mb:.1f} MB)')
    else:
        print(f'  MISSING: {f}')

# Quick inference test
print()
print('Running quick inference test...')

from shieldshot.watermark.encoder import WatermarkEncoder
from shieldshot.watermark.decoder import WatermarkDecoder
from shieldshot.watermark.payload import encode_payload, decode_payload, PAYLOAD_BITS
from shieldshot.perturb.generator import PerturbationGenerator
from shieldshot.utils.quality import compute_ssim, compute_lpips

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Test watermark round-trip
encoder = WatermarkEncoder(payload_bits=PAYLOAD_BITS).to(device)
decoder = WatermarkDecoder(payload_bits=PAYLOAD_BITS).to(device)
encoder.load_state_dict(torch.load(models_dir / 'encoder.pt', weights_only=True))
decoder.load_state_dict(torch.load(models_dir / 'decoder.pt', weights_only=True))
encoder.eval()
decoder.eval()

test_img = torch.randn(1, 3, 256, 256).clamp(0, 1).to(device)
payload = encode_payload(user_id='test', timestamp=1234567890)
payload_tensor = torch.tensor(payload, dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    watermarked = encoder(test_img, payload_tensor).clamp(0, 1)
    logits = decoder(watermarked)
    predicted_bits = (logits > 0).int().squeeze(0).tolist()

result = decode_payload(predicted_bits)
ssim_val = compute_ssim(test_img, watermarked)
lpips_val = compute_lpips(test_img, watermarked)

print(f'  Watermark SSIM: {ssim_val:.4f} (target: >0.95)')
print(f'  Watermark LPIPS: {lpips_val:.4f} (target: <0.05)')
print(f'  Watermark payload valid: {result[\"valid\"]}')
print(f'  Watermark bit accuracy: {sum(a == b for a, b in zip(payload, predicted_bits)) / len(payload):.2%}')

# Test generator
gen = PerturbationGenerator().to(device)
gen.load_state_dict(torch.load(models_dir / 'generator.pt', weights_only=True))
gen.eval()

face = torch.randn(1, 3, 256, 256).clamp(0, 1).to(device)
with torch.no_grad():
    perturbed = gen(face)

diff = (perturbed - face).abs().max().item()
gen_ssim = compute_ssim(face, perturbed)
gen_lpips = compute_lpips(face, perturbed)
print(f'  Generator max perturbation: {diff:.4f} (budget: {8/255:.4f})')
print(f'  Generator SSIM: {gen_ssim:.4f} (target: >0.95)')
print(f'  Generator LPIPS: {gen_lpips:.4f} (target: <0.05)')

print()
print('Validation complete.')
"

echo ""
echo "============================================="
echo " ALL DONE"
echo " $(date)"
echo "============================================="
echo ""
echo "Trained models are in: $MODELS_DIR/"
echo ""
echo "To use them locally:"
echo "  1. Download the models:"
echo "     scp runpod:$MODELS_DIR/*.pt ~/.shieldshot/models/"
echo ""
echo "  2. Then run:"
echo "     shieldshot protect photo.jpg -o protected.jpg"
echo ""
echo "Training logs are in: $LOG_DIR/"
echo ""
