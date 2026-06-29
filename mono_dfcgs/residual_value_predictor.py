import torch
import torch.nn as nn


def residual_time_features(t, count, device=None, dtype=None):
    t_value = torch.as_tensor(t, device=device, dtype=dtype).reshape(1)
    feats = torch.stack([
        t_value[0],
        t_value[0] * t_value[0],
        torch.sin(torch.pi * t_value[0]),
        torch.cos(torch.pi * t_value[0]),
    ]).reshape(1, 4)
    return feats.expand(int(count), -1)


def endpoint_diff_topk_indices(left_attrs, right_attrs, keep_fraction):
    """Return decoder-reproducible endpoint-diff top-k indices."""
    if left_attrs.shape != right_attrs.shape or left_attrs.ndim != 3 or left_attrs.shape[0] != 1:
        raise ValueError(f"expected matching [1, N, D] attrs, got {left_attrs.shape} and {right_attrs.shape}")
    gaussian_count = int(left_attrs.shape[1])
    keep_count = min(max(int(round(gaussian_count * float(keep_fraction))), 0), gaussian_count)
    if keep_count <= 0:
        return torch.empty((0,), dtype=torch.int64)
    scores = torch.sum((right_attrs[0].float() - left_attrs[0].float()) ** 2, dim=-1)
    return torch.sort(torch.topk(scores, k=keep_count, largest=True).indices.to(torch.int64)).values.detach().cpu()


def selected_residual_values_from_prediction(base_attrs, predicted_attrs, selected_indices):
    """Predict residual values at caller-selected indices from a feed-forward anchor prediction."""
    if base_attrs.shape != predicted_attrs.shape or base_attrs.ndim != 3 or base_attrs.shape[0] != 1:
        raise ValueError(f"expected matching [1, N, D] attrs, got {base_attrs.shape} and {predicted_attrs.shape}")
    keep_idx = torch.as_tensor(selected_indices, dtype=torch.int64, device=base_attrs.device).reshape(-1)
    if keep_idx.numel() == 0:
        return torch.empty((0, int(base_attrs.shape[-1])), dtype=base_attrs.dtype, device=base_attrs.device)
    if int(keep_idx.min().item()) < 0 or int(keep_idx.max().item()) >= int(base_attrs.shape[1]):
        raise ValueError("selected index out of range")
    return predicted_attrs[0, keep_idx, :] - base_attrs[0, keep_idx, :]


def apply_selected_residual_values(base_attrs, selected_indices, residual_values):
    """Apply predicted selected residual values to a base anchor attribute tensor."""
    if base_attrs.ndim != 3 or base_attrs.shape[0] != 1:
        raise ValueError(f"expected [1, N, D] base attrs, got {base_attrs.shape}")
    keep_idx = torch.as_tensor(selected_indices, dtype=torch.int64, device=base_attrs.device).reshape(-1)
    residual_values = torch.as_tensor(residual_values, dtype=base_attrs.dtype, device=base_attrs.device)
    expected_shape = (int(keep_idx.numel()), int(base_attrs.shape[-1]))
    if tuple(residual_values.shape) != expected_shape:
        raise ValueError(f"expected residual values shape {expected_shape}, got {tuple(residual_values.shape)}")
    out = base_attrs.clone()
    if keep_idx.numel() > 0:
        if int(keep_idx.min().item()) < 0 or int(keep_idx.max().item()) >= int(base_attrs.shape[1]):
            raise ValueError("selected index out of range")
        out[0, keep_idx, :] += residual_values
    return out


def selected_residual_feature_matrix(left_attrs, right_attrs, base_attrs, selected_indices, t):
    """Build per-selected-Gaussian decoder-side features for residual value prediction."""
    if left_attrs.shape != right_attrs.shape or left_attrs.shape != base_attrs.shape:
        raise ValueError(f"expected matching attrs, got {left_attrs.shape}, {right_attrs.shape}, {base_attrs.shape}")
    if left_attrs.ndim != 3 or left_attrs.shape[0] != 1:
        raise ValueError(f"expected [1, N, D] attrs, got {left_attrs.shape}")
    keep_idx = torch.as_tensor(selected_indices, dtype=torch.int64, device=left_attrs.device).reshape(-1)
    attr_dim = int(left_attrs.shape[-1])
    if keep_idx.numel() == 0:
        return torch.empty((0, attr_dim * 4 + 4), dtype=left_attrs.dtype, device=left_attrs.device)
    if int(keep_idx.min().item()) < 0 or int(keep_idx.max().item()) >= int(left_attrs.shape[1]):
        raise ValueError("selected index out of range")
    left = left_attrs[0, keep_idx, :]
    right = right_attrs[0, keep_idx, :]
    base = base_attrs[0, keep_idx, :]
    diff = right - left
    time = residual_time_features(t, int(keep_idx.numel()), device=left_attrs.device, dtype=left_attrs.dtype)
    return torch.cat([left, right, base, diff, time], dim=-1)


class SelectedResidualValueMLP(nn.Module):
    """Small per-Gaussian residual value predictor for selected deterministic indices."""

    def __init__(self, feature_dim=56, residual_dim=13, hidden_dim=128):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.residual_dim = int(residual_dim)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.residual_dim),
        )

    def forward(self, features):
        return self.net(features)
