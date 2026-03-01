"""Tests for watermark encoder/decoder networks."""

import pytest
import torch
from shieldshot.watermark.encoder import WatermarkEncoder
from shieldshot.watermark.decoder import WatermarkDecoder
from shieldshot.watermark.payload import PAYLOAD_BITS


@pytest.fixture
def encoder():
    return WatermarkEncoder(payload_bits=PAYLOAD_BITS)

@pytest.fixture
def decoder():
    return WatermarkDecoder(payload_bits=PAYLOAD_BITS)

@pytest.fixture
def sample_image():
    return torch.randn(1, 3, 256, 256).clamp(0, 1)

@pytest.fixture
def sample_payload():
    return torch.randint(0, 2, (1, PAYLOAD_BITS)).float()


def test_encoder_output_shape(encoder, sample_image, sample_payload):
    encoded = encoder(sample_image, sample_payload)
    assert encoded.shape == sample_image.shape

def test_encoder_output_range(encoder, sample_image, sample_payload):
    encoded = encoder(sample_image, sample_payload)
    assert encoded.min() >= -0.5
    assert encoded.max() <= 1.5

def test_decoder_output_shape(decoder, sample_image):
    decoded = decoder(sample_image)
    assert decoded.shape == (1, PAYLOAD_BITS)

def test_encoder_decoder_untrained_roundtrip(encoder, decoder, sample_image, sample_payload):
    encoded = encoder(sample_image, sample_payload)
    clamped = encoded.clamp(0, 1)
    decoded = decoder(clamped)
    assert decoded.shape == (1, PAYLOAD_BITS)
