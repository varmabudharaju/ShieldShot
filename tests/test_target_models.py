"""Tests for target model loaders."""

import pytest
import torch

from shieldshot.perturb.models import (
    load_arcface, load_facenet, load_clip, load_openclip, load_sd_vae,
    get_face_embedding, _run_model, FACE_MODELS, ALL_MODELS,
)


@pytest.fixture
def dummy_face_tensor():
    """A random 112x112 face tensor (ArcFace input size)."""
    return torch.randn(1, 3, 112, 112)


def test_load_arcface():
    model = load_arcface()
    assert model is not None


def test_load_facenet():
    model = load_facenet()
    assert model is not None


def test_arcface_output_shape(dummy_face_tensor):
    model = load_arcface()
    model.eval()
    with torch.no_grad():
        emb = model(dummy_face_tensor)
    assert emb.shape[0] == 1
    assert emb.shape[1] == 512


def test_facenet_output_shape():
    model = load_facenet()
    model.eval()
    face_tensor = torch.randn(1, 3, 160, 160)
    with torch.no_grad():
        emb = model(face_tensor)
    assert emb.shape[0] == 1
    assert emb.shape[1] == 512


def test_get_face_embedding_returns_dict(dummy_face_tensor):
    embeddings = get_face_embedding(dummy_face_tensor)
    assert "arcface" in embeddings
    assert isinstance(embeddings["arcface"], torch.Tensor)


# --- New image generation model targets ---

@pytest.mark.slow
def test_load_clip():
    model = load_clip()
    assert model is not None


@pytest.mark.slow
def test_clip_output_shape():
    model = load_clip()
    model.eval()
    tensor = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        emb = _run_model("clip", model, tensor)
    assert emb.shape[0] == 1
    assert emb.shape[1] == 768


@pytest.mark.slow
def test_load_openclip():
    model = load_openclip()
    assert model is not None


@pytest.mark.slow
def test_openclip_output_shape():
    model = load_openclip()
    model.eval()
    tensor = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        emb = _run_model("openclip", model, tensor)
    assert emb.shape[0] == 1
    assert emb.shape[1] == 1280


@pytest.mark.slow
def test_load_sd_vae():
    model = load_sd_vae()
    assert model is not None


@pytest.mark.slow
def test_sd_vae_output_shape():
    model = load_sd_vae()
    tensor = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        emb = _run_model("sd_vae", model, tensor)
    assert emb.shape[0] == 1
    # 4 * 64 * 64 = 16384
    assert emb.shape[1] == 16384


@pytest.mark.slow
def test_get_embedding_all_models():
    tensor = torch.randn(1, 3, 224, 224)
    embeddings = get_face_embedding(tensor, models=ALL_MODELS)
    assert len(embeddings) == 5
    assert "clip" in embeddings
    assert "openclip" in embeddings
    assert "sd_vae" in embeddings
