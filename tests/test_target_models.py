"""Tests for target model loaders."""

import pytest
import torch

from shieldshot.perturb.models import load_arcface, load_facenet, get_face_embedding


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
