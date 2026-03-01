"""Target model loaders for adversarial perturbation.

Loads pretrained models for use as targets in adversarial perturbation.

Face encoders (disrupts face-swap and identity-based generation):
    - ArcFace: InceptionResnetV1, [B, 3, 112, 112] → [B, 512]
    - FaceNet: InceptionResnetV1, [B, 3, 160, 160] → [B, 512]

Vision encoders (disrupts image generation models):
    - CLIP ViT-L/14: Used by SD 1.5, [B, 3, 224, 224] → [B, 768]
    - OpenCLIP ViT-H/14: Used by SDXL, IP-Adapter, PhotoMaker, Flux, PuLID, [B, 3, 224, 224] → [B, 1280]

Latent encoders (disrupts Dreambooth/LoRA fine-tuning):
    - SD VAE: Stable Diffusion VAE encoder, [B, 3, 512, 512] → [B, 4, 64, 64] (flattened for loss)
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


def load_clip() -> nn.Module:
    """Load CLIP ViT-L/14 image encoder.

    Used by Stable Diffusion 1.5 for image understanding.
    Returns an nn.Module: [B, 3, 224, 224] → [B, 768].
    """
    if "clip" not in _model_cache:
        from transformers import CLIPVisionModelWithProjection

        model = CLIPVisionModelWithProjection.from_pretrained(
            "openai/clip-vit-large-patch14"
        ).eval()
        _model_cache["clip"] = model
    return _model_cache["clip"]


def load_openclip() -> nn.Module:
    """Load OpenCLIP ViT-H/14 image encoder.

    Used by SDXL, IP-Adapter, PhotoMaker, Flux, PuLID.
    Returns an nn.Module: [B, 3, 224, 224] → [B, 1280].
    """
    if "openclip" not in _model_cache:
        import open_clip

        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-H-14", pretrained="laion2b_s32b_b79k"
        )
        model = model.visual.eval()
        _model_cache["openclip"] = model
    return _model_cache["openclip"]


def load_sd_vae() -> nn.Module:
    """Load Stable Diffusion VAE encoder.

    Disrupts Dreambooth/LoRA fine-tuning by corrupting latent representations.
    Returns an nn.Module: [B, 3, H, W] → [B, 4, H/8, W/8].
    """
    if "sd_vae" not in _model_cache:
        from diffusers import AutoencoderKL

        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse"
        ).eval()
        _model_cache["sd_vae"] = vae
    return _model_cache["sd_vae"]


# Input size requirements per model
MODEL_INPUT_SIZES = {
    "arcface": (112, 112),
    "facenet": (160, 160),
    "clip": (224, 224),
    "openclip": (224, 224),
    "sd_vae": (512, 512),
}

# All available model loaders
MODEL_LOADERS = {
    "arcface": load_arcface,
    "facenet": load_facenet,
    "clip": load_clip,
    "openclip": load_openclip,
    "sd_vae": load_sd_vae,
}

# Default models for face-region perturbation
FACE_MODELS = ["arcface", "facenet"]

# All models for maximum protection
ALL_MODELS = ["arcface", "facenet", "clip", "openclip", "sd_vae"]


def _resize_for_model(tensor: torch.Tensor, model_name: str) -> torch.Tensor:
    """Resize tensor to the required input size for a given model."""
    target_size = MODEL_INPUT_SIZES.get(model_name)
    if target_size and tensor.shape[-2:] != target_size:
        return F.interpolate(tensor, size=target_size, mode="bilinear", align_corners=False)
    return tensor


def _run_model(model_name: str, model: nn.Module, tensor: torch.Tensor) -> torch.Tensor:
    """Run a model and return a flat embedding vector."""
    inp = _resize_for_model(tensor, model_name)

    if model_name == "clip":
        out = model(pixel_values=inp)
        return out.image_embeds  # [B, 768]
    elif model_name == "sd_vae":
        latent = model.encode(inp).latent_dist.mean  # [B, 4, H/8, W/8]
        return latent.flatten(1)  # Flatten for cosine distance loss
    else:
        return model(inp)


def get_face_embedding(
    face_tensor: torch.Tensor,
    models: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Get embeddings from multiple models.

    Args:
        face_tensor: Input tensor [B, 3, H, W] in [0, 1].
        models: List of model names. Defaults to FACE_MODELS (arcface + facenet).

    Returns:
        Dictionary mapping model name to embedding tensor.
    """
    if models is None:
        models = FACE_MODELS

    embeddings = {}
    for name in models:
        loader = MODEL_LOADERS[name]
        model = loader()
        model.eval()
        with torch.no_grad():
            emb = _run_model(name, model, face_tensor)
        embeddings[name] = emb

    return embeddings
