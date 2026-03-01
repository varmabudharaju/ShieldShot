"""PGD (Projected Gradient Descent) adversarial perturbation."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from shieldshot.perturb.losses import multi_model_loss
from shieldshot.perturb.models import load_arcface, load_facenet


_MODEL_LOADERS = {
    "arcface": load_arcface,
    "facenet": load_facenet,
}


def _get_embeddings_with_grad(
    face_tensor: torch.Tensor,
    models: list[str],
) -> dict[str, torch.Tensor]:
    """Get face embeddings without disabling gradients.

    Unlike ``get_face_embedding``, this does **not** wrap calls in
    ``torch.no_grad()``, so gradients flow back through the models --
    which is required for PGD.
    """
    embeddings: dict[str, torch.Tensor] = {}
    for name in models:
        model = _MODEL_LOADERS[name]()
        model.eval()
        if name == "facenet" and face_tensor.shape[-2:] != (160, 160):
            inp = F.interpolate(
                face_tensor, size=(160, 160), mode="bilinear", align_corners=False
            )
        else:
            inp = face_tensor
        embeddings[name] = model(inp)
    return embeddings


def pgd_attack(
    face_tensor: torch.Tensor,
    num_steps: int = 100,
    epsilon: float = 8 / 255,
    step_size: float | None = None,
    target_models: list[str] | None = None,
    weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Apply PGD adversarial perturbation to a face tensor.

    Args:
        face_tensor: [1, 3, H, W] in [0, 1].
        num_steps: Number of PGD iterations.
        epsilon: L-infinity perturbation budget.
        step_size: Per-step size. Default: epsilon / (num_steps / 4).
        target_models: Which models to attack. Default: ["arcface", "facenet"].
        weights: Per-model loss weights.

    Returns:
        Perturbed tensor [1, 3, H, W] in [0, 1].
    """
    if step_size is None:
        step_size = epsilon / max(num_steps / 4, 1)
    if target_models is None:
        target_models = ["arcface", "facenet"]

    # Get clean embeddings (no grad needed for the reference)
    with torch.no_grad():
        clean_embeddings = _get_embeddings_with_grad(face_tensor, models=target_models)

    # Initialize perturbation with random start inside epsilon ball
    delta = torch.zeros_like(face_tensor)
    delta.uniform_(-epsilon, epsilon)
    delta = torch.clamp(face_tensor + delta, 0, 1) - face_tensor

    for _ in range(num_steps):
        delta.requires_grad_(True)
        perturbed = face_tensor + delta

        perturbed_embeddings = _get_embeddings_with_grad(perturbed, models=target_models)

        loss = multi_model_loss(clean_embeddings, perturbed_embeddings, weights=weights)
        loss.backward()

        grad = delta.grad.detach()
        delta = delta.detach() - step_size * grad.sign()

        # Project back into epsilon ball and valid pixel range
        delta = torch.clamp(delta, -epsilon, epsilon)
        delta = torch.clamp(face_tensor + delta, 0, 1) - face_tensor

    return (face_tensor + delta).detach()
