"""Watermark payload encoding/decoding with Reed-Solomon error correction."""

import hashlib
import struct
from reedsolo import RSCodec

PAYLOAD_BITS = 96
DATA_BYTES = 8
ECC_SYMBOLS = 4

_rs = RSCodec(ECC_SYMBOLS)


def _hash_user_id(user_id: str) -> int:
    h = hashlib.sha256(user_id.encode()).digest()
    return struct.unpack(">I", h[:4])[0]


def encode_payload(user_id: str, timestamp: int) -> list[int]:
    uid_hash = _hash_user_id(user_id)
    ts_32 = timestamp & 0xFFFFFFFF
    data_bytes = struct.pack(">II", uid_hash, ts_32)
    encoded = bytes(_rs.encode(data_bytes))
    bits = []
    for byte in encoded:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits[:PAYLOAD_BITS]


def decode_payload(bits: list[int]) -> dict:
    byte_list = []
    for i in range(0, len(bits), 8):
        byte_val = 0
        for j in range(8):
            if i + j < len(bits):
                byte_val = (byte_val << 1) | bits[i + j]
            else:
                byte_val <<= 1
        byte_list.append(byte_val)
    try:
        decoded = bytes(_rs.decode(bytes(byte_list))[0])
        uid_hash, ts_32 = struct.unpack(">II", bytes(decoded[:DATA_BYTES]))
        return {"user_id_hash": uid_hash, "timestamp": ts_32, "valid": True}
    except Exception:
        return {"user_id_hash": None, "timestamp": None, "valid": False}
