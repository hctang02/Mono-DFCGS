from __future__ import annotations

from typing import Dict

import torch


def ensure_batched_anchor(anchor: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for key, value in anchor.items():
        if value.dim() == 2:
            out[key] = value.unsqueeze(0)
        elif value.dim() == 3:
            out[key] = value
        else:
            raise ValueError(f"Expected {key} to have 2 or 3 dims, got {value.shape}")
    return out


def static_anchor_to_single_frame_gaussians(anchor: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Wrap a static anchor as a zero-dynamic StreamSplat Gaussian dict.

    The dynamic renderer expects `xyz` and `rot` to include static and dynamic
    components. For a single predicted intermediate anchor, the dynamic
    component can be zero and `timestamps=None` renders one static frame.
    """
    anchor = ensure_batched_anchor(anchor)
    xyz_static = anchor["xyz"]
    rot_static = anchor["rot"]
    return {
        "rgb": anchor["rgb"],
        "opacity": anchor["opacity"],
        "scale": anchor["scale"],
        "xyz": torch.stack([xyz_static, torch.zeros_like(xyz_static)], dim=2),
        "rot": torch.stack([rot_static, torch.zeros_like(rot_static)], dim=2),
    }
