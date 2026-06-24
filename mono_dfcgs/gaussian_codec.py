from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import torch


STATIC_ANCHOR_DIMS = {
    "rgb": 3,
    "opacity": 1,
    "scale": 2,
    "xyz": 3,
    "rot": 4,
}


@dataclass(frozen=True)
class PayloadEstimate:
    values: int
    bytes: int
    mib: float


def flatten_static_anchor(anchor: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten a static Gaussian anchor dict into `[B, N, 13]` attributes."""
    parts = []
    for key, dim in STATIC_ANCHOR_DIMS.items():
        if key not in anchor:
            raise KeyError(f"Missing anchor field: {key}")
        value = anchor[key]
        if value.shape[-1] != dim:
            raise ValueError(f"Field {key} expected dim {dim}, got {value.shape[-1]}")
        parts.append(value)
    return torch.cat(parts, dim=-1)


def unflatten_static_anchor(attrs: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Convert `[B, N, 13]` attributes back into a static anchor dict."""
    expected = sum(STATIC_ANCHOR_DIMS.values())
    if attrs.shape[-1] != expected:
        raise ValueError(f"Expected last dim {expected}, got {attrs.shape[-1]}")
    out = {}
    start = 0
    for key, dim in STATIC_ANCHOR_DIMS.items():
        out[key] = attrs[..., start:start + dim]
        start += dim
    return out


def estimate_static_anchor_payload(
    anchor: Dict[str, torch.Tensor],
    codec: str = "float16",
    opacity_threshold: float = 0.0,
) -> PayloadEstimate:
    """Estimate transmitted payload size for a static keyframe Gaussian anchor."""
    attrs = flatten_static_anchor(anchor)
    opacity = anchor["opacity"][..., 0]
    keep = opacity >= opacity_threshold
    values = int(attrs[keep].numel())
    if codec == "float32":
        byte_count = values * 4
    elif codec == "float16":
        byte_count = values * 2
    elif codec.startswith("q"):
        bits = int(codec[1:])
        byte_count = (values * bits + 7) // 8
    else:
        raise ValueError(f"Unsupported codec: {codec}")
    return PayloadEstimate(values=values, bytes=byte_count, mib=byte_count / (1024.0 * 1024.0))


def uniform_quantize(attrs: torch.Tensor, bits: int = 8, eps: float = 1e-8):
    """Per-channel uniform quantization for smoke experiments."""
    if bits <= 0 or bits > 16:
        raise ValueError(f"bits should be in [1, 16], got {bits}")
    qmax = (1 << bits) - 1
    mins = attrs.amin(dim=(-3, -2), keepdim=True)
    maxs = attrs.amax(dim=(-3, -2), keepdim=True)
    scales = (maxs - mins).clamp_min(eps) / qmax
    q = torch.round((attrs - mins) / scales).clamp(0, qmax).to(torch.int32)
    return q, mins, scales


def uniform_dequantize(q: torch.Tensor, mins: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    return q.to(torch.float32) * scales + mins


def assert_anchor_fields(anchor: Dict[str, torch.Tensor], required: Iterable[str] = STATIC_ANCHOR_DIMS):
    for key in required:
        if key not in anchor:
            raise KeyError(f"Missing anchor field: {key}")
