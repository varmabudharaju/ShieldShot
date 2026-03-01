"""Quality gate -- SSIM and LPIPS checks for perturbation visibility."""

import torch
from pytorch_msssim import ssim
import lpips as _lpips

_lpips_model = None


def _get_lpips_model():
    global _lpips_model
    if _lpips_model is None:
        _lpips_model = _lpips.LPIPS(net="squeeze", verbose=False)
    return _lpips_model


def compute_ssim(original: torch.Tensor, modified: torch.Tensor) -> float:
    """Compute SSIM between two image tensors [1, 3, H, W] in [0, 1].

    Returns float in [0, 1] where 1 = identical.
    """
    with torch.no_grad():
        return ssim(original, modified, data_range=1.0, size_average=True).item()


def compute_lpips(original: torch.Tensor, modified: torch.Tensor) -> float:
    """Compute LPIPS perceptual distance between two image tensors [1, 3, H, W] in [0, 1].

    Returns float where 0 = identical, higher = more different.
    """
    model = _get_lpips_model()
    device = original.device
    model = model.to(device)
    with torch.no_grad():
        orig_scaled = original * 2 - 1
        mod_scaled = modified * 2 - 1
        return model(orig_scaled, mod_scaled).item()


def check_quality(
    original: torch.Tensor,
    modified: torch.Tensor,
    ssim_threshold: float = 0.95,
    lpips_threshold: float = 0.05,
) -> tuple[bool, dict]:
    """Check if modified image passes quality gate.

    Returns (passed, metrics_dict).
    """
    ssim_val = compute_ssim(original, modified)
    lpips_val = compute_lpips(original, modified)
    passed = ssim_val >= ssim_threshold and lpips_val <= lpips_threshold
    return passed, {"ssim": ssim_val, "lpips": lpips_val}
