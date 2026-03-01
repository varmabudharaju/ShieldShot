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
#      wget -O train.sh https://raw.githubusercontent.com/<your-repo>/main/scripts/runpod_train.sh
#      chmod +x train.sh
#      ./train.sh
#
#   3. Come back in ~8-10 hours. Trained models will be in /workspace/shieldshot/trained_models/
#
# Estimated time (RTX 4090):
#   Step 1 (setup):           ~10 min
#   Step 2 (watermark):       ~2 hrs
#   Step 3 (PGD targets):     ~5-6 hrs
#   Step 4 (generator):       ~1-2 hrs
#   Step 5 (validation):      ~5 min
#   Total:                    ~8-10 hrs, ~$7-10
#
# =============================================================================

set -e  # Exit on any error

WORKSPACE="/workspace"
PROJECT_DIR="$WORKSPACE/shieldshot"
DATA_DIR="$WORKSPACE/data"
FFHQ_DIR="$DATA_DIR/ffhq"
MODELS_DIR="$PROJECT_DIR/trained_models"
DELTAS_DIR="$DATA_DIR/pgd_deltas"
LOG_DIR="$PROJECT_DIR/training_logs"

# Number of images to use for PGD target generation (subset of FFHQ)
# Full FFHQ is 70k images — PGD on all of them would take days.
# 5000 is a good balance of quality vs time.
PGD_SUBSET_SIZE=5000

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
echo " Step 1/5: Setup"
echo "============================================="

# Clone repo (or pull if already exists)
if [ -d "$PROJECT_DIR" ]; then
    echo "Project directory exists, pulling latest..."
    cd "$PROJECT_DIR" && git pull
else
    echo "Cloning ShieldShot..."
    cd "$WORKSPACE"
    git clone https://github.com/<YOUR_USERNAME>/shieldshot.git
    # If not on GitHub yet, upload the code manually:
    # scp -r /Users/varma/shieldshot/ runpod:/workspace/shieldshot/
fi

cd "$PROJECT_DIR"

# Install dependencies
echo "Installing dependencies..."
pip install -e ".[dev]" --quiet
pip install gdown kaggle --quiet  # For dataset download

# Create directories
mkdir -p "$DATA_DIR" "$MODELS_DIR" "$DELTAS_DIR" "$LOG_DIR"

# Download FFHQ dataset
if [ -d "$FFHQ_DIR" ] && [ "$(ls -1 "$FFHQ_DIR"/*.png 2>/dev/null | wc -l)" -gt 1000 ]; then
    echo "FFHQ dataset already exists ($(ls -1 "$FFHQ_DIR"/*.png | wc -l) images)"
else
    echo "Downloading FFHQ dataset..."
    echo ""
    echo "Option A: If you have Kaggle credentials (~/.kaggle/kaggle.json):"
    echo "  kaggle datasets download -d arnaud58/flickrfaceshq-dataset-ffhq -p $DATA_DIR"
    echo ""
    echo "Option B: Download FFHQ thumbnails (smaller, 128x128, faster):"
    echo "  Using Google Drive mirror..."

    mkdir -p "$FFHQ_DIR"

    # Try downloading FFHQ thumbnails via gdown (Google Drive)
    # This is the official FFHQ thumbnails128x128 dataset (~1.8GB)
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
    # Flatten nested directories
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
echo " Step 2/5: Training Watermark Model"
echo " Started: $(date)"
echo "============================================="

python3 -m train.train_watermark \
    --data-dir "$FFHQ_DIR" \
    --output-dir "$MODELS_DIR" \
    --epochs 100 \
    --batch-size 32 \
    --image-size 256 \
    --lr 1e-3 \
    --w-bce 1.0 \
    --w-ssim 0.5 \
    --num-workers 4 \
    --save-every 10 \
    2>&1 | tee "$LOG_DIR/watermark_training.log"

echo ""
echo "Watermark training complete: $(date)"
echo "Models saved to $MODELS_DIR/encoder.pt and $MODELS_DIR/decoder.pt"
echo ""

# =============================================================================
# STEP 3: Generate PGD Targets (using all 5 target models)
# =============================================================================
echo "============================================="
echo " Step 3/5: Generating PGD Targets"
echo " Using $PGD_SUBSET_SIZE images with 5 target models"
echo " Started: $(date)"
echo "============================================="

# Create a subset directory for PGD (full FFHQ is too many)
PGD_SUBSET_DIR="$DATA_DIR/ffhq_subset"
if [ ! -d "$PGD_SUBSET_DIR" ] || [ "$(ls -1 "$PGD_SUBSET_DIR" 2>/dev/null | wc -l)" -lt "$PGD_SUBSET_SIZE" ]; then
    echo "Creating subset of $PGD_SUBSET_SIZE images for PGD..."
    mkdir -p "$PGD_SUBSET_DIR"
    ls "$FFHQ_DIR"/*.png | shuf -n "$PGD_SUBSET_SIZE" | while read f; do
        ln -sf "$f" "$PGD_SUBSET_DIR/$(basename "$f")"
    done
    echo "Subset ready: $(ls -1 "$PGD_SUBSET_DIR" | wc -l) images"
fi

# Generate PGD deltas with all 5 target models
# The pgd_attack function uses target_models parameter
python3 -c "
import sys
sys.path.insert(0, '.')
from pathlib import Path
import torch
from shieldshot.utils.image import load_image, to_tensor
from shieldshot.perturb.pgd import pgd_attack
from shieldshot.perturb.models import ALL_MODELS

data_dir = Path('$PGD_SUBSET_DIR')
output_dir = Path('$DELTAS_DIR')
output_dir.mkdir(parents=True, exist_ok=True)

image_paths = sorted(p for p in data_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
print(f'Generating PGD targets for {len(image_paths)} images using models: {ALL_MODELS}')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

for i, img_path in enumerate(image_paths):
    out_path = output_dir / f'{img_path.stem}.pt'
    if out_path.exists():
        if (i + 1) % 100 == 0:
            print(f'  [{i+1}/{len(image_paths)}] skipping (already exists)')
        continue

    img = load_image(str(img_path))
    tensor = to_tensor(img).to(device)

    # Resize to 112x112 for face models (PGD will resize internally for CLIP/VAE)
    face_tensor = torch.nn.functional.interpolate(tensor, size=(112, 112), mode='bilinear', align_corners=False)

    perturbed = pgd_attack(
        face_tensor,
        num_steps=100,
        epsilon=8/255,
        target_models=ALL_MODELS,
    )

    delta = (perturbed - face_tensor).cpu()
    torch.save(delta, out_path)

    if (i + 1) % 50 == 0 or (i + 1) == len(image_paths):
        print(f'  [{i+1}/{len(image_paths)}] saved {out_path.name}')

print('PGD target generation complete.')
" 2>&1 | tee "$LOG_DIR/pgd_targets.log"

echo ""
echo "PGD targets complete: $(date)"
echo "Deltas saved to $DELTAS_DIR ($(ls -1 "$DELTAS_DIR"/*.pt | wc -l) files)"
echo ""

# =============================================================================
# STEP 4: Train Perturbation Generator
# =============================================================================
echo "============================================="
echo " Step 4/5: Training Perturbation Generator"
echo " Started: $(date)"
echo "============================================="

python3 -m train.train_generator \
    --image-dir "$PGD_SUBSET_DIR" \
    --delta-dir "$DELTAS_DIR" \
    --epochs 50 \
    --batch-size 32 \
    --lr 1e-4 \
    --output "$MODELS_DIR/generator.pt" \
    2>&1 | tee "$LOG_DIR/generator_training.log"

echo ""
echo "Generator training complete: $(date)"
echo "Model saved to $MODELS_DIR/generator.pt"
echo ""

# =============================================================================
# STEP 5: Validation — Quick sanity check
# =============================================================================
echo "============================================="
echo " Step 5/5: Validation"
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
from shieldshot.utils.quality import compute_ssim

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

print(f'  Watermark SSIM: {ssim_val:.4f} (target: >0.95)')
print(f'  Watermark payload valid: {result[\"valid\"]}')
print(f'  Watermark bit accuracy: {sum(a == b for a, b in zip(payload, predicted_bits)) / len(payload):.2%}')

# Test generator
gen = PerturbationGenerator().to(device)
gen.load_state_dict(torch.load(models_dir / 'generator.pt', weights_only=True))
gen.eval()

face = torch.randn(1, 3, 112, 112).clamp(0, 1).to(device)
with torch.no_grad():
    perturbed = gen(face)

diff = (perturbed - face).abs().max().item()
gen_ssim = compute_ssim(face, perturbed)
print(f'  Generator max perturbation: {diff:.4f} (budget: {8/255:.4f})')
print(f'  Generator SSIM: {gen_ssim:.4f} (target: >0.95)')

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
echo "  2. Or copy to your machine:"
echo "     rsync -av runpod:$MODELS_DIR/ ~/.shieldshot/models/"
echo ""
echo "  3. Then run:"
echo "     shieldshot protect photo.jpg -o protected.jpg"
echo "     shieldshot protect photo.jpg --mode thorough  # All 5 models"
echo ""
echo "Training logs are in: $LOG_DIR/"
echo ""
