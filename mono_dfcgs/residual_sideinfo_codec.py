import math
import struct

import numpy as np
import torch


MAGIC = b"RSI1"
VERSION = 1
HEADER_STRUCT = struct.Struct("<4sBBBBHII")
FLOAT16_BYTES = 2


def _pack_ints(values, bits):
    if bits <= 0:
        raise ValueError("bits must be positive")
    max_value = (1 << bits) - 1
    out = bytearray()
    acc = 0
    acc_bits = 0
    for value in values:
        value = int(value)
        if value < 0 or value > max_value:
            raise ValueError(f"value {value} cannot be represented with {bits} bits")
        acc |= value << acc_bits
        acc_bits += bits
        while acc_bits >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            acc_bits -= 8
    if acc_bits:
        out.append(acc & 0xFF)
    return bytes(out)


def _unpack_ints(data, count, bits):
    if bits <= 0:
        raise ValueError("bits must be positive")
    values = []
    mask = (1 << bits) - 1
    acc = 0
    acc_bits = 0
    offset = 0
    for _ in range(count):
        while acc_bits < bits:
            if offset >= len(data):
                raise ValueError("not enough packed integer data")
            acc |= int(data[offset]) << acc_bits
            offset += 1
            acc_bits += 8
        values.append(acc & mask)
        acc >>= bits
        acc_bits -= bits
    return values


def sideinfo_bits_without_header(gaussian_count, keep_count, attr_dim, side_bits, metadata_bits_per_attr_value=16):
    if keep_count <= 0:
        return 0
    index_bits = math.ceil(math.log2(max(int(gaussian_count), 2)))
    payload_bits = int(keep_count) * (index_bits + int(attr_dim) * int(side_bits))
    metadata_bits = int(attr_dim) * 2 * int(metadata_bits_per_attr_value)
    return payload_bits + metadata_bits


def sideinfo_mib_without_header(gaussian_count, keep_count, attr_dim, side_bits, metadata_bits_per_attr_value=16):
    return sideinfo_bits_without_header(
        gaussian_count,
        keep_count,
        attr_dim,
        side_bits,
        metadata_bits_per_attr_value,
    ) / 8.0 / (1024.0 * 1024.0)


def encode_topk_residual_sideinfo(base_attrs, target_attrs, keep_fraction, side_bits, eps=1e-8):
    """Encode top-k residual side-info into a fixed-length byte payload.

    The packet stores indices and q residual values with bit packing. Min/max
    metadata is stored as float16 to match the 16-bit metadata accounting used
    in Stage87-90.
    """
    base_cpu = base_attrs.detach().cpu().float()
    target_cpu = target_attrs.detach().cpu().float()
    if base_cpu.shape != target_cpu.shape or base_cpu.ndim != 3 or base_cpu.shape[0] != 1:
        raise ValueError(f"expected matching [1, N, D] attrs, got {base_cpu.shape} and {target_cpu.shape}")
    gaussian_count = int(base_cpu.shape[1])
    attr_dim = int(base_cpu.shape[2])
    keep_count = min(max(int(round(gaussian_count * float(keep_fraction))), 0), gaussian_count)
    index_bits = math.ceil(math.log2(max(gaussian_count, 2)))

    residual = target_cpu - base_cpu
    if keep_count > 0:
        energy = torch.sum(residual[0] ** 2, dim=-1)
        keep_idx = torch.topk(energy, k=keep_count, largest=True).indices.to(torch.int64)
        kept = residual[0, keep_idx, :]
        mins = kept.amin(dim=0)
        maxs = kept.amax(dim=0)
    else:
        keep_idx = torch.empty((0,), dtype=torch.int64)
        kept = torch.empty((0, attr_dim), dtype=torch.float32)
        mins = torch.zeros((attr_dim,), dtype=torch.float32)
        maxs = torch.zeros((attr_dim,), dtype=torch.float32)

    mins_half = mins.numpy().astype("<f2")
    maxs_half = maxs.numpy().astype("<f2")
    mins_codec = torch.from_numpy(mins_half.astype("<f4")).float()
    maxs_codec = torch.from_numpy(maxs_half.astype("<f4")).float()

    if keep_count > 0:
        qmax = (1 << int(side_bits)) - 1
        scales = (maxs_codec - mins_codec).clamp_min(eps) / qmax
        q_values = torch.round((kept - mins_codec) / scales).clamp(0, qmax).to(torch.int64)
        q_flat = q_values.reshape(-1).tolist()
    else:
        q_flat = []

    header = HEADER_STRUCT.pack(
        MAGIC,
        VERSION,
        int(side_bits),
        int(index_bits),
        0,
        int(attr_dim),
        int(gaussian_count),
        int(keep_count),
    )
    metadata = mins_half.tobytes() + maxs_half.tobytes()
    index_payload = _pack_ints(keep_idx.tolist(), index_bits)
    residual_payload = _pack_ints(q_flat, int(side_bits))
    payload = header + metadata + index_payload + residual_payload

    theoretical_bits = sideinfo_bits_without_header(gaussian_count, keep_count, attr_dim, side_bits)
    info = {
        "gaussian_count": gaussian_count,
        "attr_dim": attr_dim,
        "keep_count": keep_count,
        "side_bits": int(side_bits),
        "index_bits": int(index_bits),
        "header_bytes": HEADER_STRUCT.size,
        "metadata_bytes": len(metadata),
        "index_bytes": len(index_payload),
        "residual_bytes": len(residual_payload),
        "payload_bytes": len(payload),
        "theoretical_bits_without_header": theoretical_bits,
        "theoretical_bytes_without_header": theoretical_bits / 8.0,
        "theoretical_mib_without_header": theoretical_bits / 8.0 / (1024.0 * 1024.0),
    }
    return payload, info


def decode_residual_sideinfo(base_attrs, payload, eps=1e-8):
    base_cpu = base_attrs.detach().cpu().float()
    if base_cpu.ndim != 3 or base_cpu.shape[0] != 1:
        raise ValueError(f"expected [1, N, D] attrs, got {base_cpu.shape}")
    header_size = HEADER_STRUCT.size
    magic, version, side_bits, index_bits, _flags, attr_dim, gaussian_count, keep_count = HEADER_STRUCT.unpack(payload[:header_size])
    if magic != MAGIC:
        raise ValueError("invalid residual side-info magic")
    if version != VERSION:
        raise ValueError(f"unsupported residual side-info version: {version}")
    if int(base_cpu.shape[1]) != gaussian_count or int(base_cpu.shape[2]) != attr_dim:
        raise ValueError("payload shape does not match base attrs")

    offset = header_size
    metadata_bytes = int(attr_dim) * 2 * FLOAT16_BYTES
    mins = np.frombuffer(payload[offset:offset + int(attr_dim) * FLOAT16_BYTES], dtype="<f2").astype("<f4").copy()
    offset += int(attr_dim) * FLOAT16_BYTES
    maxs = np.frombuffer(payload[offset:offset + int(attr_dim) * FLOAT16_BYTES], dtype="<f2").astype("<f4").copy()
    offset = header_size + metadata_bytes

    index_bytes = (int(keep_count) * int(index_bits) + 7) // 8
    index_payload = payload[offset:offset + index_bytes]
    offset += index_bytes
    residual_values = int(keep_count) * int(attr_dim)
    residual_bytes = (residual_values * int(side_bits) + 7) // 8
    residual_payload = payload[offset:offset + residual_bytes]

    keep_idx = _unpack_ints(index_payload, int(keep_count), int(index_bits))
    q_flat = _unpack_ints(residual_payload, residual_values, int(side_bits))
    decoded = base_cpu.clone()
    if keep_count > 0:
        q = torch.tensor(q_flat, dtype=torch.float32).reshape(int(keep_count), int(attr_dim))
        mins_t = torch.from_numpy(mins).float()
        maxs_t = torch.from_numpy(maxs).float()
        qmax = (1 << int(side_bits)) - 1
        scales = (maxs_t - mins_t).clamp_min(eps) / qmax
        residual = q * scales + mins_t
        decoded[0, torch.tensor(keep_idx, dtype=torch.long), :] += residual
    return decoded.to(device=base_attrs.device, dtype=base_attrs.dtype)
