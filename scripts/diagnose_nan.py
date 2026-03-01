"""Diagnose NaN issues in watermark training."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import glob
from PIL import Image

from shieldshot.watermark.encoder import WatermarkEncoder
from shieldshot.watermark.decoder import WatermarkDecoder
from shieldshot.watermark.payload import PAYLOAD_BITS
from pytorch_msssim import ssim
import lpips


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    encoder = WatermarkEncoder(payload_bits=PAYLOAD_BITS).to(device)
    decoder = WatermarkDecoder(payload_bits=PAYLOAD_BITS).to(device)

    img = torch.rand(2, 3, 256, 256, device=device)
    payload = torch.randint(0, 2, (2, PAYLOAD_BITS), dtype=torch.float32, device=device)

    # 1. Encoder
    wm = encoder(img, payload)
    print(f"1. Encoder output: min={wm.min():.4f} max={wm.max():.4f} nan={torch.isnan(wm).any()}")
    wm = wm.clamp(0, 1)

    # 2. Decoder
    logits = decoder(wm)
    print(f"2. Decoder logits: min={logits.min():.4f} max={logits.max():.4f} nan={torch.isnan(logits).any()}")

    # 3. BCE
    bce = torch.nn.BCEWithLogitsLoss()(logits, payload)
    print(f"3. BCE loss: {bce.item()} nan={torch.isnan(bce)}")

    # 4. SSIM
    s = ssim(wm, img, data_range=1.0, size_average=True)
    print(f"4. SSIM: {s.item()} nan={torch.isnan(s)}")

    # 5. LPIPS
    lp = lpips.LPIPS(net="squeeze", verbose=False).to(device)
    l = lp(wm * 2 - 1, img * 2 - 1).mean()
    print(f"5. LPIPS: {l.item()} nan={torch.isnan(l)}")

    # 6. Augmentations
    from train.augmentations import apply_random_augmentation
    for i in range(20):
        aug = apply_random_augmentation(wm[:1])
        if torch.isnan(aug).any():
            print(f"6. Augmentation produced NaN on try {i}!")
            break
    else:
        print("6. Augmentations OK (20 tries)")

    # 7. Full training step
    print("7. Testing full training step...")
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3
    )
    encoder.train()
    decoder.train()

    wm = encoder(img, payload).clamp(0, 1)
    augmented = torch.stack([
        apply_random_augmentation(wm[i:i+1]).squeeze(0)
        for i in range(2)
    ])
    logits = decoder(augmented)

    loss_bce = torch.nn.BCEWithLogitsLoss()(logits, payload)
    loss_ssim = 1.0 - ssim(wm, img, data_range=1.0, size_average=True)
    loss_lpips = lp(wm * 2 - 1, img * 2 - 1).mean()
    residual = wm - img
    from train.train_watermark import high_freq_penalty
    loss_compat = high_freq_penalty(residual)

    loss = loss_bce + 0.3 * loss_ssim + 0.7 * loss_lpips + 0.1 * loss_compat

    print(f"   BCE={loss_bce.item():.4f} SSIM_loss={loss_ssim.item():.4f} LPIPS={loss_lpips.item():.4f} Compat={loss_compat.item():.4f}")
    print(f"   Total loss: {loss.item():.4f} nan={torch.isnan(loss)}")

    optimizer.zero_grad()
    loss.backward()

    for name, p in list(encoder.named_parameters())[:5]:
        if p.grad is not None:
            print(f"   Grad {name}: min={p.grad.min():.4f} max={p.grad.max():.4f} nan={torch.isnan(p.grad).any()}")

    # 8. Check real images
    imgs = sorted(glob.glob("/workspace/data/ffhq/*"))[:5]
    print(f"8. Found {len(imgs)} sample images in /workspace/data/ffhq/")
    for p in imgs:
        im = Image.open(p).convert("RGB")
        print(f"   {p}: size={im.size} mode={im.mode}")


if __name__ == "__main__":
    main()
