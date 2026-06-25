from __future__ import annotations

import json
import struct
import zlib
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

from mono_dfcgs.gaussian_codec import flatten_static_anchor, uniform_dequantize, uniform_quantize, unflatten_static_anchor


MAGIC = b"MDFCGS1\n"


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
) -> bytes:
    if bits != 8:
        raise ValueError("Stage31 prototype currently supports q8 payloads only")
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
        q_np = q.squeeze(0).to(torch.uint8).contiguous().numpy()
        q_bytes = q_np.tobytes(order="C")
        payload_parts.append(q_bytes)
        records.append({
            "frame_index": frame_index,
            "timestamp": timestamp,
            "shape": list(q_np.shape),
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
        "version": 1,
        "bits": bits,
        "compression": compression,
        "anchor_count": len(anchors),
        "uncompressed_payload_length": len(payload),
        "records": records,
    }
    header_bytes = _compact_json(header)
    return MAGIC + struct.pack("<I", len(header_bytes)) + header_bytes + encoded_payload


def decode_anchor_bitstream(blob: bytes) -> Tuple[List[Dict[str, torch.Tensor]], dict]:
    header, encoded_payload = _split_container(blob)
    if int(header["bits"]) != 8:
        raise ValueError("Stage31 prototype currently supports q8 payloads only")
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
        q_np = np.frombuffer(payload[offset: offset + length], dtype=np.uint8).reshape(shape)
        q = torch.from_numpy(q_np.astype(np.int32)).unsqueeze(0)
        mins = torch.tensor(record["mins"], dtype=torch.float32).reshape(1, 1, -1)
        scales = torch.tensor(record["scales"], dtype=torch.float32).reshape(1, 1, -1)
        anchors.append(unflatten_static_anchor(uniform_dequantize(q, mins, scales)))
    return anchors, header
