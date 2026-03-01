"""Target model loaders for adversarial perturbation.

Loads pretrained face recognition models for use as targets in
adversarial perturbation generation. Each model takes a face crop
tensor and returns a 512-dimensional embedding.

Models:
    - ArcFace: InceptionResnetV1 trained on CASIA-Webface, accepts [B, 3, 112, 112]
    - FaceNet: InceptionResnetV1 trained on VGGFace2, accepts [B, 3, 160, 160]
"""

from __future__ import annotations

import ssl

import torch
import torch.nn as nn
import torch.nn.functional as F


_model_cache: dict[str, nn.Module] = {}


class _ArcFaceWrapper(nn.Module):
    """Wraps InceptionResnetV1 to accept 112x112 input (ArcFace standard size).

    Internally resizes to 160x160 for the InceptionResnetV1 backbone.
    """

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Resize from 112x112 to 160x160
        if x.shape[-2:] != (160, 160):
            x = F.interpolate(x, size=(160, 160), mode="bilinear", align_corners=False)
        return self.backbone(x)


def _patch_ssl() -> None:
    """Work around SSL certificate verification issues on macOS."""
    ssl._create_default_https_context = ssl._create_unverified_context


def load_arcface() -> nn.Module:
    """Load pretrained ArcFace-style model.

    Returns an nn.Module that accepts [B, 3, 112, 112] tensors and
    returns [B, 512] embeddings.

    Uses InceptionResnetV1 trained on CASIA-Webface as the backbone,
    wrapped to accept the standard ArcFace input size of 112x112.
    """
    if "arcface" not in _model_cache:
        _patch_ssl()
        from facenet_pytorch import InceptionResnetV1

        backbone = InceptionResnetV1(pretrained="casia-webface").eval()
        model = _ArcFaceWrapper(backbone).eval()
        _model_cache["arcface"] = model
    return _model_cache["arcface"]


def load_facenet() -> nn.Module:
    """Load pretrained FaceNet (InceptionResnetV1) model.

    Returns an nn.Module that accepts [B, 3, 160, 160] tensors and
    returns [B, 512] embeddings.
    """
    if "facenet" not in _model_cache:
        _patch_ssl()
        from facenet_pytorch import InceptionResnetV1

        model = InceptionResnetV1(pretrained="vggface2").eval()
        _model_cache["facenet"] = model
    return _model_cache["facenet"]


def get_face_embedding(
    face_tensor: torch.Tensor,
    models: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Get embeddings from multiple face recognition models.

    Args:
        face_tensor: Input face tensor. Shape depends on the models used.
            For ArcFace: [B, 3, 112, 112]. For FaceNet: [B, 3, 160, 160].
            When using both, pass [B, 3, 112, 112] -- ArcFace will use it
            directly and FaceNet will receive a resized version.
        models: List of model names to use. Defaults to ["arcface", "facenet"].

    Returns:
        Dictionary mapping model name to embedding tensor of shape [B, 512].
    """
    if models is None:
        models = ["arcface", "facenet"]

    loaders = {
        "arcface": load_arcface,
        "facenet": load_facenet,
    }

    embeddings = {}
    for name in models:
        loader = loaders[name]
        model = loader()
        model.eval()
        with torch.no_grad():
            if name == "facenet" and face_tensor.shape[-2:] != (160, 160):
                # Resize for FaceNet if input is not 160x160
                inp = F.interpolate(
                    face_tensor, size=(160, 160), mode="bilinear", align_corners=False
                )
            else:
                inp = face_tensor
            emb = model(inp)
        embeddings[name] = emb

    return embeddings
