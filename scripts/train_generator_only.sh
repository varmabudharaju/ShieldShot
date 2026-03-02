#!/bin/bash
# Run generator training only (after watermark training is done)
# Usage: ./scripts/train_generator_only.sh

set -e

cd /workspace/shieldshot

echo "Upgrading PyTorch to 2.4..."
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121 --quiet

echo "Starting generator training..."
python3 -m train.train_generator \
    --data-dir /workspace/data/ffhq_gen_subset \
    --epochs 30 \
    --batch-size 8 \
    --image-size 256 \
    --lr 1e-4 \
    --w-distortion 1.0 \
    --w-quality 0.5 \
    --w-compression 0.3 \
    --num-workers 4 \
    --save-every 10 \
    --output /workspace/shieldshot/trained_models/generator.pt \
    2>&1 | tee /workspace/generator_training.log

echo ""
echo "Generator training complete!"
echo "Model saved to /workspace/shieldshot/trained_models/generator.pt"
