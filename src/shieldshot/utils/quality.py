"""Quality gate -- SSIM checks for perturbation visibility."""

import torch
from pytorch_msssim import ssim


def compute_ssim(original: torch.Tensor, modified: torch.Tensor) -> float:
    """Compute SSIM between two image tensors [1, 3, H, W] in [0, 1].

    Returns float in [0, 1] where 1 = identical.
    """
    with torch.no_grad():
        return ssim(original, modified, data_range=1.0, size_average=True).item()


def check_quality(
    original: torch.Tensor,
    modified: torch.Tensor,
    ssim_threshold: float = 0.95,
) -> tuple[bool, dict]:
    """Check if modified image passes quality gate.

    Returns (passed, metrics_dict).
    """
    ssim_val = compute_ssim(original, modified)
    passed = ssim_val >= ssim_threshold
    return passed, {"ssim": ssim_val}
