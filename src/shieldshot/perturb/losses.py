"""Multi-model adversarial loss functions."""

import torch
import torch.nn.functional as F


def cosine_distance_loss(clean_emb: torch.Tensor, perturbed_emb: torch.Tensor) -> torch.Tensor:
    """Compute loss that maximizes cosine distance between embeddings.

    Returns 1 - cosine_similarity (0 = identical, 2 = opposite).
    """
    cos_sim = F.cosine_similarity(clean_emb, perturbed_emb, dim=1)
    return (1.0 - cos_sim).mean()


def multi_model_loss(
    clean_embeddings: dict[str, torch.Tensor],
    perturbed_embeddings: dict[str, torch.Tensor],
    weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Combine cosine distance losses across multiple models.

    Negated because PGD minimizes loss, and we want to maximize feature distortion.
    """
    if weights is None:
        weights = {k: 1.0 for k in clean_embeddings}

    total = torch.tensor(0.0, device=next(iter(clean_embeddings.values())).device)
    for name in clean_embeddings:
        dist = cosine_distance_loss(clean_embeddings[name], perturbed_embeddings[name])
        total = total + weights[name] * dist

    return -total
