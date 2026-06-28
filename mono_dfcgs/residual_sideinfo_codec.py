import math
import struct
import zlib

import numpy as np
import torch


MAGIC = b"RSI1"
ENTROPY_MAGIC = b"RSE1"
DETERMINISTIC_MAGIC = b"RSD1"
DETERMINISTIC_ENTROPY_MAGIC = b"RSDZ"
VERSION = 1
HEADER_STRUCT = struct.Struct("<4sBBBBHII")
ENTROPY_HEADER_STRUCT = struct.Struct("<4sBBBBHIIIII")
DETERMINISTIC_HEADER_STRUCT = struct.Struct("<4sBBBBHII")
DETERMINISTIC_ENTROPY_HEADER_STRUCT = struct.Struct("<4sBBBBHIIII")
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


def deterministic_sideinfo_bits_without_header(keep_count, attr_dim, side_bits, metadata_bits_per_attr_value=16):
    if keep_count <= 0:
        return 0
    payload_bits = int(keep_count) * int(attr_dim) * int(side_bits)
    metadata_bits = int(attr_dim) * 2 * int(metadata_bits_per_attr_value)
    return payload_bits + metadata_bits


def deterministic_sideinfo_mib_without_header(keep_count, attr_dim, side_bits, metadata_bits_per_attr_value=16):
    return deterministic_sideinfo_bits_without_header(
        keep_count,
        attr_dim,
        side_bits,
        metadata_bits_per_attr_value,
    ) / 8.0 / (1024.0 * 1024.0)


def _prepare_selected_residual_quantization(base_attrs, target_attrs, selected_indices, side_bits, eps=1e-8):
    base_cpu = base_attrs.detach().cpu().float()
    target_cpu = target_attrs.detach().cpu().float()
    if base_cpu.shape != target_cpu.shape or base_cpu.ndim != 3 or base_cpu.shape[0] != 1:
        raise ValueError(f"expected matching [1, N, D] attrs, got {base_cpu.shape} and {target_cpu.shape}")
    gaussian_count = int(base_cpu.shape[1])
    attr_dim = int(base_cpu.shape[2])
    keep_idx = torch.as_tensor(selected_indices, dtype=torch.int64).detach().cpu().reshape(-1)
    if keep_idx.numel() > 0:
        if int(keep_idx.min().item()) < 0 or int(keep_idx.max().item()) >= gaussian_count:
            raise ValueError("selected index out of range")
        if int(torch.unique(keep_idx).numel()) != int(keep_idx.numel()):
            raise ValueError("selected indices must be unique")
    keep_count = int(keep_idx.numel())
    residual = target_cpu - base_cpu
    if keep_count > 0:
        kept = residual[0, keep_idx, :]
        mins = kept.amin(dim=0)
        maxs = kept.amax(dim=0)
    else:
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
    else:
        q_values = torch.empty((0, attr_dim), dtype=torch.int64)
    return {
        "gaussian_count": gaussian_count,
        "attr_dim": attr_dim,
        "keep_count": keep_count,
        "keep_idx": keep_idx,
        "q_values": q_values,
        "mins_half": mins_half,
        "maxs_half": maxs_half,
    }


def _prepare_topk_residual_quantization(base_attrs, target_attrs, keep_fraction, side_bits, eps=1e-8, sort_indices=False):
    base_cpu = base_attrs.detach().cpu().float()
    target_cpu = target_attrs.detach().cpu().float()
    if base_cpu.shape != target_cpu.shape or base_cpu.ndim != 3 or base_cpu.shape[0] != 1:
        raise ValueError(f"expected matching [1, N, D] attrs, got {base_cpu.shape} and {target_cpu.shape}")
    gaussian_count = int(base_cpu.shape[1])
    attr_dim = int(base_cpu.shape[2])
    keep_count = min(max(int(round(gaussian_count * float(keep_fraction))), 0), gaussian_count)
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
        if sort_indices:
            order = torch.argsort(keep_idx)
            keep_idx = keep_idx[order]
            q_values = q_values[order]
    else:
        q_values = torch.empty((0, attr_dim), dtype=torch.int64)

    return {
        "gaussian_count": gaussian_count,
        "attr_dim": attr_dim,
        "keep_count": keep_count,
        "keep_idx": keep_idx,
        "q_values": q_values,
        "mins_half": mins_half,
        "maxs_half": maxs_half,
    }


def encode_topk_residual_sideinfo(base_attrs, target_attrs, keep_fraction, side_bits, eps=1e-8):
    """Encode top-k residual side-info into a fixed-length byte payload.

    The packet stores indices and q residual values with bit packing. Min/max
    metadata is stored as float16 to match the 16-bit metadata accounting used
    in Stage87-90.
    """
    prepared = _prepare_topk_residual_quantization(base_attrs, target_attrs, keep_fraction, side_bits, eps=eps, sort_indices=False)
    gaussian_count = prepared["gaussian_count"]
    attr_dim = prepared["attr_dim"]
    keep_count = prepared["keep_count"]
    index_bits = math.ceil(math.log2(max(gaussian_count, 2)))
    keep_idx = prepared["keep_idx"]
    q_flat = prepared["q_values"].reshape(-1).tolist()

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
    metadata = prepared["mins_half"].tobytes() + prepared["maxs_half"].tobytes()
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


def encode_selected_residual_sideinfo(base_attrs, target_attrs, selected_indices, side_bits, eps=1e-8):
    """Encode residual values at caller-supplied indices with indices included."""
    prepared = _prepare_selected_residual_quantization(base_attrs, target_attrs, selected_indices, side_bits, eps=eps)
    gaussian_count = prepared["gaussian_count"]
    attr_dim = prepared["attr_dim"]
    keep_count = prepared["keep_count"]
    index_bits = math.ceil(math.log2(max(gaussian_count, 2)))
    metadata = prepared["mins_half"].tobytes() + prepared["maxs_half"].tobytes()
    index_payload = _pack_ints(prepared["keep_idx"].tolist(), index_bits)
    residual_payload = _pack_ints(prepared["q_values"].reshape(-1).tolist(), int(side_bits))
    header = HEADER_STRUCT.pack(
        MAGIC,
        VERSION,
        int(side_bits),
        int(index_bits),
        1,
        int(attr_dim),
        int(gaussian_count),
        int(keep_count),
    )
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


def encode_selected_residual_values_sideinfo(base_attrs, target_attrs, selected_indices, side_bits, eps=1e-8):
    """Encode residual values at deterministic decoder-known indices without storing indices."""
    prepared = _prepare_selected_residual_quantization(base_attrs, target_attrs, selected_indices, side_bits, eps=eps)
    gaussian_count = prepared["gaussian_count"]
    attr_dim = prepared["attr_dim"]
    keep_count = prepared["keep_count"]
    metadata = prepared["mins_half"].tobytes() + prepared["maxs_half"].tobytes()
    residual_payload = _pack_ints(prepared["q_values"].reshape(-1).tolist(), int(side_bits))
    header = DETERMINISTIC_HEADER_STRUCT.pack(
        DETERMINISTIC_MAGIC,
        VERSION,
        int(side_bits),
        0,
        0,
        int(attr_dim),
        int(gaussian_count),
        int(keep_count),
    )
    payload = header + metadata + residual_payload
    theoretical_bits = deterministic_sideinfo_bits_without_header(keep_count, attr_dim, side_bits)
    fixed_index_bits = int(keep_count) * math.ceil(math.log2(max(gaussian_count, 2)))
    info = {
        "gaussian_count": gaussian_count,
        "attr_dim": attr_dim,
        "keep_count": keep_count,
        "side_bits": int(side_bits),
        "header_bytes": DETERMINISTIC_HEADER_STRUCT.size,
        "metadata_bytes": len(metadata),
        "index_bytes": 0,
        "residual_bytes": len(residual_payload),
        "payload_bytes": len(payload),
        "omitted_index_bits_vs_fixed": fixed_index_bits,
        "omitted_index_bytes_vs_fixed_aligned": (fixed_index_bits + 7) // 8,
        "theoretical_bits_without_header": theoretical_bits,
        "theoretical_bytes_without_header": theoretical_bits / 8.0,
        "theoretical_mib_without_header": theoretical_bits / 8.0 / (1024.0 * 1024.0),
    }
    return payload, info


def encode_selected_residual_values_sideinfo_entropy(base_attrs, target_attrs, selected_indices, side_bits, zlib_level=9, eps=1e-8):
    """Encode deterministic-index residual values with zlib-compressed components."""
    prepared = _prepare_selected_residual_quantization(base_attrs, target_attrs, selected_indices, side_bits, eps=eps)
    gaussian_count = prepared["gaussian_count"]
    attr_dim = prepared["attr_dim"]
    keep_count = prepared["keep_count"]
    metadata = prepared["mins_half"].tobytes() + prepared["maxs_half"].tobytes()
    residual_payload = _pack_ints(prepared["q_values"].reshape(-1).tolist(), int(side_bits))
    metadata_z = zlib.compress(metadata, int(zlib_level))
    residual_z = zlib.compress(residual_payload, int(zlib_level))
    header = DETERMINISTIC_ENTROPY_HEADER_STRUCT.pack(
        DETERMINISTIC_ENTROPY_MAGIC,
        VERSION,
        int(side_bits),
        0,
        0,
        int(attr_dim),
        int(gaussian_count),
        int(keep_count),
        len(metadata_z),
        len(residual_z),
    )
    payload = header + metadata_z + residual_z
    theoretical_bits = deterministic_sideinfo_bits_without_header(keep_count, attr_dim, side_bits)
    fixed_index_bits = int(keep_count) * math.ceil(math.log2(max(gaussian_count, 2)))
    raw_payload_bytes = DETERMINISTIC_HEADER_STRUCT.size + len(metadata) + len(residual_payload)
    info = {
        "gaussian_count": gaussian_count,
        "attr_dim": attr_dim,
        "keep_count": keep_count,
        "side_bits": int(side_bits),
        "zlib_level": int(zlib_level),
        "header_bytes": DETERMINISTIC_ENTROPY_HEADER_STRUCT.size,
        "metadata_bytes": len(metadata),
        "residual_bytes": len(residual_payload),
        "metadata_zlib_bytes": len(metadata_z),
        "residual_zlib_bytes": len(residual_z),
        "index_bytes": 0,
        "payload_bytes": len(payload),
        "raw_deterministic_payload_bytes": raw_payload_bytes,
        "compressed_ratio_vs_raw_deterministic": len(payload) / raw_payload_bytes if raw_payload_bytes > 0 else 0.0,
        "omitted_index_bits_vs_fixed": fixed_index_bits,
        "omitted_index_bytes_vs_fixed_aligned": (fixed_index_bits + 7) // 8,
        "theoretical_bits_without_header": theoretical_bits,
        "theoretical_bytes_without_header": theoretical_bits / 8.0,
        "theoretical_mib_without_header": theoretical_bits / 8.0 / (1024.0 * 1024.0),
    }
    return payload, info


def _index_deltas(sorted_indices):
    out = []
    prev = -1
    for index in sorted_indices:
        index = int(index)
        out.append(index - prev)
        prev = index
    return out


def _indices_from_deltas(deltas):
    out = []
    prev = -1
    for delta in deltas:
        value = prev + int(delta)
        out.append(value)
        prev = value
    return out


def encode_topk_residual_sideinfo_entropy(base_attrs, target_attrs, keep_fraction, side_bits, zlib_level=9, eps=1e-8):
    """Encode top-k residual side-info with sorted-index deltas and zlib.

    This is a decode-capable version of the best Stage92 preflight candidate.
    Indices are sorted before delta coding, and q residual rows are reordered to
    match sorted indices.
    """
    prepared = _prepare_topk_residual_quantization(base_attrs, target_attrs, keep_fraction, side_bits, eps=eps, sort_indices=True)
    gaussian_count = prepared["gaussian_count"]
    attr_dim = prepared["attr_dim"]
    keep_count = prepared["keep_count"]
    delta_bits = math.ceil(math.log2(max(gaussian_count + 1, 2)))
    metadata = prepared["mins_half"].tobytes() + prepared["maxs_half"].tobytes()
    delta_payload = _pack_ints(_index_deltas(prepared["keep_idx"].tolist()), delta_bits)
    residual_payload = _pack_ints(prepared["q_values"].reshape(-1).tolist(), int(side_bits))
    metadata_z = zlib.compress(metadata, int(zlib_level))
    delta_z = zlib.compress(delta_payload, int(zlib_level))
    residual_z = zlib.compress(residual_payload, int(zlib_level))
    header = ENTROPY_HEADER_STRUCT.pack(
        ENTROPY_MAGIC,
        VERSION,
        int(side_bits),
        int(delta_bits),
        0,
        int(attr_dim),
        int(gaussian_count),
        int(keep_count),
        len(metadata_z),
        len(delta_z),
        len(residual_z),
    )
    payload = header + metadata_z + delta_z + residual_z
    theoretical_bits = sideinfo_bits_without_header(gaussian_count, keep_count, attr_dim, side_bits)
    fixed_bytes = HEADER_STRUCT.size + len(metadata) + ((keep_count * math.ceil(math.log2(max(gaussian_count, 2))) + 7) // 8) + len(residual_payload)
    info = {
        "gaussian_count": gaussian_count,
        "attr_dim": attr_dim,
        "keep_count": keep_count,
        "side_bits": int(side_bits),
        "delta_bits": int(delta_bits),
        "header_bytes": ENTROPY_HEADER_STRUCT.size,
        "metadata_bytes": len(metadata),
        "delta_bytes": len(delta_payload),
        "residual_bytes": len(residual_payload),
        "metadata_zlib_bytes": len(metadata_z),
        "delta_zlib_bytes": len(delta_z),
        "residual_zlib_bytes": len(residual_z),
        "payload_bytes": len(payload),
        "fixed_payload_bytes": fixed_bytes,
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


def decode_selected_residual_values_sideinfo(base_attrs, payload, selected_indices, eps=1e-8):
    base_cpu = base_attrs.detach().cpu().float()
    if base_cpu.ndim != 3 or base_cpu.shape[0] != 1:
        raise ValueError(f"expected [1, N, D] attrs, got {base_cpu.shape}")
    header_size = DETERMINISTIC_HEADER_STRUCT.size
    magic, version, side_bits, _unused, _flags, attr_dim, gaussian_count, keep_count = DETERMINISTIC_HEADER_STRUCT.unpack(payload[:header_size])
    if magic != DETERMINISTIC_MAGIC:
        raise ValueError("invalid deterministic residual side-info magic")
    if version != VERSION:
        raise ValueError(f"unsupported deterministic residual side-info version: {version}")
    if int(base_cpu.shape[1]) != gaussian_count or int(base_cpu.shape[2]) != attr_dim:
        raise ValueError("deterministic payload shape does not match base attrs")
    keep_idx = torch.as_tensor(selected_indices, dtype=torch.int64).detach().cpu().reshape(-1)
    if int(keep_idx.numel()) != int(keep_count):
        raise ValueError("selected index count does not match deterministic payload")
    if keep_idx.numel() > 0:
        if int(keep_idx.min().item()) < 0 or int(keep_idx.max().item()) >= int(gaussian_count):
            raise ValueError("selected index out of range")

    offset = header_size
    metadata_bytes = int(attr_dim) * 2 * FLOAT16_BYTES
    mins = np.frombuffer(payload[offset:offset + int(attr_dim) * FLOAT16_BYTES], dtype="<f2").astype("<f4").copy()
    offset += int(attr_dim) * FLOAT16_BYTES
    maxs = np.frombuffer(payload[offset:offset + int(attr_dim) * FLOAT16_BYTES], dtype="<f2").astype("<f4").copy()
    offset = header_size + metadata_bytes
    residual_values = int(keep_count) * int(attr_dim)
    residual_bytes = (residual_values * int(side_bits) + 7) // 8
    residual_payload = payload[offset:offset + residual_bytes]
    q_flat = _unpack_ints(residual_payload, residual_values, int(side_bits))
    decoded = base_cpu.clone()
    if keep_count > 0:
        q = torch.tensor(q_flat, dtype=torch.float32).reshape(int(keep_count), int(attr_dim))
        mins_t = torch.from_numpy(mins).float()
        maxs_t = torch.from_numpy(maxs).float()
        qmax = (1 << int(side_bits)) - 1
        scales = (maxs_t - mins_t).clamp_min(eps) / qmax
        residual = q * scales + mins_t
        decoded[0, keep_idx, :] += residual
    return decoded.to(device=base_attrs.device, dtype=base_attrs.dtype)


def decode_selected_residual_values_sideinfo_entropy(base_attrs, payload, selected_indices, eps=1e-8):
    base_cpu = base_attrs.detach().cpu().float()
    if base_cpu.ndim != 3 or base_cpu.shape[0] != 1:
        raise ValueError(f"expected [1, N, D] attrs, got {base_cpu.shape}")
    header_size = DETERMINISTIC_ENTROPY_HEADER_STRUCT.size
    (
        magic,
        version,
        side_bits,
        _unused,
        _flags,
        attr_dim,
        gaussian_count,
        keep_count,
        metadata_z_bytes,
        residual_z_bytes,
    ) = DETERMINISTIC_ENTROPY_HEADER_STRUCT.unpack(payload[:header_size])
    if magic != DETERMINISTIC_ENTROPY_MAGIC:
        raise ValueError("invalid compressed deterministic residual side-info magic")
    if version != VERSION:
        raise ValueError(f"unsupported compressed deterministic residual side-info version: {version}")
    if int(base_cpu.shape[1]) != gaussian_count or int(base_cpu.shape[2]) != attr_dim:
        raise ValueError("compressed deterministic payload shape does not match base attrs")
    keep_idx = torch.as_tensor(selected_indices, dtype=torch.int64).detach().cpu().reshape(-1)
    if int(keep_idx.numel()) != int(keep_count):
        raise ValueError("selected index count does not match compressed deterministic payload")
    if keep_idx.numel() > 0:
        if int(keep_idx.min().item()) < 0 or int(keep_idx.max().item()) >= int(gaussian_count):
            raise ValueError("selected index out of range")

    offset = header_size
    metadata_z = payload[offset:offset + int(metadata_z_bytes)]
    offset += int(metadata_z_bytes)
    residual_z = payload[offset:offset + int(residual_z_bytes)]
    metadata = zlib.decompress(metadata_z)
    expected_metadata_bytes = int(attr_dim) * 2 * FLOAT16_BYTES
    if len(metadata) != expected_metadata_bytes:
        raise ValueError("invalid compressed deterministic metadata length")
    residual_payload = zlib.decompress(residual_z)
    residual_values = int(keep_count) * int(attr_dim)
    expected_residual_bytes = (residual_values * int(side_bits) + 7) // 8
    if len(residual_payload) != expected_residual_bytes:
        raise ValueError("invalid compressed deterministic residual length")

    mins = np.frombuffer(metadata[:int(attr_dim) * FLOAT16_BYTES], dtype="<f2").astype("<f4").copy()
    maxs = np.frombuffer(metadata[int(attr_dim) * FLOAT16_BYTES:], dtype="<f2").astype("<f4").copy()
    q_flat = _unpack_ints(residual_payload, residual_values, int(side_bits))
    decoded = base_cpu.clone()
    if keep_count > 0:
        q = torch.tensor(q_flat, dtype=torch.float32).reshape(int(keep_count), int(attr_dim))
        mins_t = torch.from_numpy(mins).float()
        maxs_t = torch.from_numpy(maxs).float()
        qmax = (1 << int(side_bits)) - 1
        scales = (maxs_t - mins_t).clamp_min(eps) / qmax
        residual = q * scales + mins_t
        decoded[0, keep_idx, :] += residual
    return decoded.to(device=base_attrs.device, dtype=base_attrs.dtype)


def decode_residual_sideinfo_entropy(base_attrs, payload, eps=1e-8):
    base_cpu = base_attrs.detach().cpu().float()
    if base_cpu.ndim != 3 or base_cpu.shape[0] != 1:
        raise ValueError(f"expected [1, N, D] attrs, got {base_cpu.shape}")
    header_size = ENTROPY_HEADER_STRUCT.size
    (
        magic,
        version,
        side_bits,
        delta_bits,
        _flags,
        attr_dim,
        gaussian_count,
        keep_count,
        metadata_z_bytes,
        delta_z_bytes,
        residual_z_bytes,
    ) = ENTROPY_HEADER_STRUCT.unpack(payload[:header_size])
    if magic != ENTROPY_MAGIC:
        raise ValueError("invalid entropy residual side-info magic")
    if version != VERSION:
        raise ValueError(f"unsupported entropy residual side-info version: {version}")
    if int(base_cpu.shape[1]) != gaussian_count or int(base_cpu.shape[2]) != attr_dim:
        raise ValueError("entropy payload shape does not match base attrs")

    offset = header_size
    metadata_z = payload[offset:offset + int(metadata_z_bytes)]
    offset += int(metadata_z_bytes)
    delta_z = payload[offset:offset + int(delta_z_bytes)]
    offset += int(delta_z_bytes)
    residual_z = payload[offset:offset + int(residual_z_bytes)]

    metadata = zlib.decompress(metadata_z)
    expected_metadata_bytes = int(attr_dim) * 2 * FLOAT16_BYTES
    if len(metadata) != expected_metadata_bytes:
        raise ValueError("invalid entropy metadata length")
    mins = np.frombuffer(metadata[:int(attr_dim) * FLOAT16_BYTES], dtype="<f2").astype("<f4").copy()
    maxs = np.frombuffer(metadata[int(attr_dim) * FLOAT16_BYTES:], dtype="<f2").astype("<f4").copy()
    delta_payload = zlib.decompress(delta_z)
    residual_payload = zlib.decompress(residual_z)

    deltas = _unpack_ints(delta_payload, int(keep_count), int(delta_bits))
    keep_idx = _indices_from_deltas(deltas)
    residual_values = int(keep_count) * int(attr_dim)
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
