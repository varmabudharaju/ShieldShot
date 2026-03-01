"""Tests for watermark payload encoding/decoding."""

import time
import pytest
from shieldshot.watermark.payload import encode_payload, decode_payload, PAYLOAD_BITS


def test_encode_returns_correct_length():
    bits = encode_payload(user_id="testuser", timestamp=int(time.time()))
    assert len(bits) == PAYLOAD_BITS


def test_encode_returns_binary():
    bits = encode_payload(user_id="testuser", timestamp=int(time.time()))
    assert all(b in (0, 1) for b in bits)


def test_roundtrip():
    ts = int(time.time())
    bits = encode_payload(user_id="testuser", timestamp=ts)
    result = decode_payload(bits)
    assert result["valid"]
    assert result["timestamp"] == (ts & 0xFFFFFFFF)


def test_different_users_different_payload():
    ts = int(time.time())
    bits1 = encode_payload(user_id="alice", timestamp=ts)
    bits2 = encode_payload(user_id="bob", timestamp=ts)
    assert bits1 != bits2


def test_decode_with_bit_errors():
    ts = int(time.time())
    bits = encode_payload(user_id="testuser", timestamp=ts)
    corrupted = list(bits)
    corrupted[0] = 1 - corrupted[0]
    corrupted[5] = 1 - corrupted[5]
    result = decode_payload(corrupted)
    assert result is not None
    assert result["valid"]
