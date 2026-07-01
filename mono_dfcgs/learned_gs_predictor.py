from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gaussian_codec import flatten_static_anchor, unflatten_static_anchor


def linear_static_anchor(
    left_anchor: Dict[str, torch.Tensor],
    right_anchor: Dict[str, torch.Tensor],
    t: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    left = flatten_static_anchor(left_anchor)
    right = flatten_static_anchor(right_anchor)
    if left.shape != right.shape:
        raise ValueError(f"Anchor shapes differ: {left.shape} vs {right.shape}")
    if t.dim() == 0:
        t = t[None].expand(left.shape[0])
    t_scalar = t.reshape(left.shape[0], 1, 1).to(left.device, left.dtype)
    return unflatten_static_anchor(left * (1.0 - t_scalar) + right * t_scalar)


class TemporalBasisGSRefiner(nn.Module):
    """Factorized temporal-basis refiner for static GS anchor attributes.

    The residual is multiplied by `t * (1 - t)`, so endpoint frames decode
    exactly to the transmitted keyframes when output constraints are disabled.
    """

    def __init__(
        self,
        attr_dim: int = 13,
        hidden_dim: int = 192,
        global_dim: int = 64,
        residual_scale: float = 0.1,
        apply_output_constraints: bool = False,
        zero_init_residual: bool = True,
    ):
        super().__init__()
        self.attr_dim = int(attr_dim)
        self.hidden_dim = int(hidden_dim)
        self.global_dim = int(global_dim)
        self.residual_scale = float(residual_scale)
        self.apply_output_constraints = bool(apply_output_constraints)
        time_dim = 5
        local_dim = self.attr_dim * 5 + time_dim
        global_raw_dim = self.attr_dim * 8
        self.local_mlp = nn.Sequential(
            nn.Linear(local_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(global_raw_dim, self.global_dim),
            nn.SiLU(),
            nn.Linear(self.global_dim, self.global_dim),
            nn.SiLU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(self.hidden_dim + self.global_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.attr_dim),
        )
        if zero_init_residual:
            nn.init.zeros_(self.fuse[-1].weight)
            nn.init.zeros_(self.fuse[-1].bias)

    @staticmethod
    def time_features(t: torch.Tensor, batch: int, count: int, device, dtype) -> torch.Tensor:
        if t.dim() == 0:
            t = t[None].expand(batch)
        t = t.reshape(batch, 1, 1).to(device=device, dtype=dtype)
        t = t.expand(batch, count, 1)
        return torch.cat([
            t,
            t * t,
            torch.sin(torch.pi * t),
            torch.cos(torch.pi * t),
            t * (1.0 - t),
        ], dim=-1)

    @staticmethod
    def endpoint_gate(t: torch.Tensor, batch: int, device, dtype) -> torch.Tensor:
        if t.dim() == 0:
            t = t[None].expand(batch)
        t = t.reshape(batch, 1, 1).to(device=device, dtype=dtype)
        return t * (1.0 - t)

    @staticmethod
    def global_stats(left: torch.Tensor, right: torch.Tensor, diff: torch.Tensor, absdiff: torch.Tensor) -> torch.Tensor:
        parts = [
            left.mean(dim=1),
            right.mean(dim=1),
            diff.mean(dim=1),
            absdiff.mean(dim=1),
            left.std(dim=1, unbiased=False),
            right.std(dim=1, unbiased=False),
            diff.std(dim=1, unbiased=False),
            absdiff.std(dim=1, unbiased=False),
        ]
        return torch.cat(parts, dim=-1)

    def forward(
        self,
        left_anchor: Dict[str, torch.Tensor],
        right_anchor: Dict[str, torch.Tensor],
        t: torch.Tensor,
        apply_output_constraints: bool | None = None,
    ) -> Dict[str, torch.Tensor]:
        left = flatten_static_anchor(left_anchor)
        right = flatten_static_anchor(right_anchor)
        if left.shape != right.shape:
            raise ValueError(f"Anchor shapes differ: {left.shape} vs {right.shape}")
        batch, count, _ = left.shape
        if t.dim() == 0:
            t = t[None].expand(batch)
        t_scalar = t.reshape(batch, 1, 1).to(left.device, left.dtype)
        base = left * (1.0 - t_scalar) + right * t_scalar
        diff = right - left
        absdiff = diff.abs()
        time = self.time_features(t, batch, count, left.device, left.dtype)
        local = torch.cat([left, right, base, diff, absdiff, time], dim=-1)
        local_feat = self.local_mlp(local)
        global_feat = self.global_mlp(self.global_stats(left, right, diff, absdiff))[:, None, :]
        global_feat = global_feat.expand(batch, count, -1)
        residual = self.fuse(torch.cat([local_feat, global_feat], dim=-1))
        pred = base + self.residual_scale * self.endpoint_gate(t, batch, left.device, left.dtype) * residual
        out = unflatten_static_anchor(pred)
        if apply_output_constraints is None:
            apply_output_constraints = self.apply_output_constraints
        if not apply_output_constraints:
            return out
        out["rgb"] = out["rgb"].sigmoid()
        out["opacity"] = out["opacity"].sigmoid()
        out["scale"] = F.softplus(out["scale"])
        out["rot"] = F.normalize(out["rot"], dim=-1)
        return out
