from __future__ import annotations

import json
import math
import struct
import zlib
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

from mono_dfcgs.gaussian_codec import flatten_static_anchor, uniform_dequantize, uniform_quantize, unflatten_static_anchor


MAGIC = b"MDFCGS1\n"


def _bitpacked_length(value_count: int, bits: int) -> int:
    return (int(value_count) * int(bits) + 7) // 8


def _pack_quantized_bits(q_np: np.ndarray, bits: int) -> bytes:
    flat = np.asarray(q_np, dtype=np.uint32).reshape(-1)
    if flat.size == 0:
        return b""
    qmax = (1 << bits) - 1
    if int(flat.max()) > qmax:
        raise ValueError(f"Quantized value exceeds q{bits} range")
    if bits == 8:
        return flat.astype(np.uint8).tobytes(order="C")
    if bits == 16:
        return flat.astype("<u2").tobytes(order="C")
    bit_positions = np.arange(bits, dtype=np.uint32)
    bit_matrix = ((flat[:, None] >> bit_positions) & 1).astype(np.uint8, copy=False)
    packed = np.packbits(bit_matrix.reshape(-1), bitorder="little")
    return packed[:_bitpacked_length(flat.size, bits)].tobytes()


def _unpack_quantized_bits(payload: bytes, bits: int, shape: Tuple[int, ...]) -> np.ndarray:
    value_count = math.prod(shape)
    expected_length = _bitpacked_length(value_count, bits)
    if len(payload) != expected_length:
        raise ValueError(f"Bitpacked payload length mismatch: expected {expected_length}, got {len(payload)}")
    if value_count == 0:
        return np.empty(shape, dtype=np.uint32)
    if bits == 8:
        return np.frombuffer(payload, dtype=np.uint8).astype(np.uint32).reshape(shape)
    if bits == 16:
        return np.frombuffer(payload, dtype="<u2").astype(np.uint32).reshape(shape)
    packed = np.frombuffer(payload, dtype=np.uint8)
    bits_np = np.unpackbits(packed, bitorder="little")[: value_count * bits]
    if bits_np.size != value_count * bits:
        raise ValueError("Bitpacked payload ended before all values were decoded")
    powers = (1 << np.arange(bits, dtype=np.uint32)).astype(np.uint32)
    values = (bits_np.reshape(value_count, bits).astype(np.uint32, copy=False) * powers).sum(axis=1, dtype=np.uint32)
    return values.reshape(shape)


def _legacy_payload_dtype(bits: int):
    if bits <= 8:
        return "uint8", np.uint8
    return "uint16", np.uint16


def _compact_json(data: dict) -> bytes:
    return json.dumps(data, separators=(",", ":")).encode("utf-8")


def _split_container(blob: bytes) -> Tuple[dict, bytes]:
    if len(blob) < len(MAGIC) + 4 or blob[: len(MAGIC)] != MAGIC:
        raise ValueError("Invalid Mono-DFCGS anchor bitstream magic")
    header_len = struct.unpack("<I", blob[len(MAGIC): len(MAGIC) + 4])[0]
    start = len(MAGIC) + 4
    end = start + header_len
    header = json.loads(blob[start:end].decode("utf-8"))
    return header, blob[end:]


def encode_anchor_bitstream(
    anchors: Iterable[Dict[str, torch.Tensor]],
    frame_indices: Iterable[int],
    timestamps: Iterable[int] | None = None,
    bits: int = 8,
    compression: str = "none",
    payload_encoding: str = "bitpack",
) -> bytes:
    if bits <= 0 or bits > 16:
        raise ValueError(f"bits should be in [1, 16], got {bits}")
    if payload_encoding not in {"bitpack", "dtype"}:
        raise ValueError(f"Unsupported payload encoding: {payload_encoding}")
    anchors = list(anchors)
    frame_indices = [int(idx) for idx in frame_indices]
    if timestamps is None:
        timestamps = frame_indices
    timestamps = [int(ts) for ts in timestamps]
    if len(anchors) != len(frame_indices) or len(anchors) != len(timestamps):
        raise ValueError("anchors, frame_indices and timestamps must have the same length")

    payload_parts: List[bytes] = []
    records = []
    payload_offset = 0
    for anchor, frame_index, timestamp in zip(anchors, frame_indices, timestamps):
        attrs = flatten_static_anchor(anchor).detach().float().cpu()
        q, mins, scales = uniform_quantize(attrs, bits=bits)
        q_np = q.squeeze(0).contiguous().numpy()
        if payload_encoding == "bitpack":
            payload_dtype = "bitpacked"
            q_bytes = _pack_quantized_bits(q_np, bits)
        else:
            payload_dtype, dtype = _legacy_payload_dtype(bits)
            q_bytes = q_np.astype(dtype).tobytes(order="C")
        payload_parts.append(q_bytes)
        records.append({
            "frame_index": frame_index,
            "timestamp": timestamp,
            "shape": list(q_np.shape),
            "payload_encoding": payload_encoding,
            "payload_dtype": payload_dtype,
            "payload_offset": payload_offset,
            "payload_length": len(q_bytes),
            "mins": [float(v) for v in mins.reshape(-1).tolist()],
            "scales": [float(v) for v in scales.reshape(-1).tolist()],
        })
        payload_offset += len(q_bytes)

    payload = b"".join(payload_parts)
    if compression == "none":
        encoded_payload = payload
    elif compression == "zlib":
        encoded_payload = zlib.compress(payload, level=9)
    else:
        raise ValueError(f"Unsupported compression: {compression}")

    header = {
        "version": 2,
        "bits": bits,
        "compression": compression,
        "payload_encoding": payload_encoding,
        "anchor_count": len(anchors),
        "uncompressed_payload_length": len(payload),
        "records": records,
    }
    header_bytes = _compact_json(header)
    return MAGIC + struct.pack("<I", len(header_bytes)) + header_bytes + encoded_payload


def decode_anchor_bitstream(blob: bytes) -> Tuple[List[Dict[str, torch.Tensor]], dict]:
    header, encoded_payload = _split_container(blob)
    bits = int(header["bits"])
    if bits <= 0 or bits > 16:
        raise ValueError(f"Unsupported quantization bits: {bits}")
    compression = header["compression"]
    if compression == "none":
        payload = encoded_payload
    elif compression == "zlib":
        payload = zlib.decompress(encoded_payload)
    else:
        raise ValueError(f"Unsupported compression: {compression}")
    if len(payload) != int(header["uncompressed_payload_length"]):
        raise ValueError("Decoded payload length mismatch")

    anchors = []
    for record in header["records"]:
        offset = int(record["payload_offset"])
        length = int(record["payload_length"])
        shape = tuple(int(v) for v in record["shape"])
        payload_slice = payload[offset: offset + length]
        payload_dtype = record.get("payload_dtype", "uint8")
        payload_encoding = record.get("payload_encoding")
        if payload_encoding is None:
            payload_encoding = "bitpack" if payload_dtype == "bitpacked" else "dtype"
        if payload_encoding == "bitpack":
            q_np = _unpack_quantized_bits(payload_slice, bits, shape)
        elif payload_encoding == "dtype":
            if payload_dtype == "uint8":
                dtype = np.uint8
            elif payload_dtype == "uint16":
                dtype = np.uint16
            else:
                raise ValueError(f"Unsupported payload dtype: {payload_dtype}")
            q_np = np.frombuffer(payload_slice, dtype=dtype).reshape(shape)
        else:
            raise ValueError(f"Unsupported payload encoding: {payload_encoding}")
        q = torch.from_numpy(q_np.astype(np.int32, copy=False)).unsqueeze(0)
        mins = torch.tensor(record["mins"], dtype=torch.float32).reshape(1, 1, -1)
        scales = torch.tensor(record["scales"], dtype=torch.float32).reshape(1, 1, -1)
        anchors.append(unflatten_static_anchor(uniform_dequantize(q, mins, scales)))
    return anchors, header
