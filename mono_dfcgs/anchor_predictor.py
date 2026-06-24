from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gaussian_codec import flatten_static_anchor, unflatten_static_anchor


class GaussianAnchorEncoder(nn.Module):
    def __init__(self, in_dim: int = 13, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

    def forward(self, attrs: torch.Tensor) -> torch.Tensor:
        return self.net(attrs)


class GaussianAnchorDynamicPredictor(nn.Module):
    """Minimal Gaussian-anchor-conditioned dynamic predictor.

    This is a stage-4 smoke model. It predicts a residual over linear endpoint
    interpolation and keeps the output in the static anchor field format.
    """

    def __init__(self, attr_dim: int = 13, hidden_dim: int = 128):
        super().__init__()
        self.encoder = GaussianAnchorEncoder(attr_dim, hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, attr_dim),
        )

    @staticmethod
    def time_features(t: torch.Tensor, n: int) -> torch.Tensor:
        if t.dim() == 1:
            t = t[:, None, None]
        elif t.dim() == 2:
            t = t[:, :, None]
        t = t.expand(-1, n, -1)
        return torch.cat([t, t * t, torch.sin(torch.pi * t), torch.cos(torch.pi * t)], dim=-1)

    def forward(
        self,
        left_anchor: Dict[str, torch.Tensor],
        right_anchor: Dict[str, torch.Tensor],
        t: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        left = flatten_static_anchor(left_anchor)
        right = flatten_static_anchor(right_anchor)
        if left.shape != right.shape:
            raise ValueError(f"Anchor shapes differ: {left.shape} vs {right.shape}")
        n = left.shape[1]
        if t.dim() == 0:
            t = t[None].expand(left.shape[0])
        t_scalar = t.reshape(left.shape[0], 1, 1).to(left.device, left.dtype)
        base = left * (1.0 - t_scalar) + right * t_scalar
        left_feat = self.encoder(left)
        right_feat = self.encoder(right)
        time_feat = self.time_mlp(self.time_features(t.to(left.device, left.dtype), n))
        residual = self.fuse(torch.cat([left_feat, right_feat, time_feat], dim=-1))
        pred = base + 0.05 * residual
        out = unflatten_static_anchor(pred)
        out["rgb"] = out["rgb"].sigmoid()
        out["opacity"] = out["opacity"].sigmoid()
        out["scale"] = F.softplus(out["scale"])
        out["rot"] = torch.nn.functional.normalize(out["rot"], dim=-1)
        return out
