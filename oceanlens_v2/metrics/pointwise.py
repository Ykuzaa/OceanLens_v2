"""Pointwise field metrics."""

from __future__ import annotations

import torch


def masked_mae(predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.to(dtype=predicted.dtype)
    return ((predicted - target).abs() * valid).sum() / (valid.sum().clamp(min=1.0) * predicted.shape[1])


def masked_rmse(predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.to(dtype=predicted.dtype)
    mse = ((predicted - target).square() * valid).sum() / (valid.sum().clamp(min=1.0) * predicted.shape[1])
    return torch.sqrt(mse)


def masked_correlation(predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average per-sample, per-channel correlation over ocean pixels."""
    valid = mask.bool()
    correlations = []
    for batch in range(predicted.shape[0]):
        for channel in range(predicted.shape[1]):
            pred_values = predicted[batch, channel][valid[batch, 0]]
            true_values = target[batch, channel][valid[batch, 0]]
            if pred_values.numel() < 2:
                continue
            pred_values = pred_values - pred_values.mean()
            true_values = true_values - true_values.mean()
            denom = pred_values.std(unbiased=False) * true_values.std(unbiased=False) + 1.0e-8
            correlations.append((pred_values * true_values).mean() / denom)
    if not correlations:
        return predicted.new_tensor(0.0)
    return torch.stack(correlations).mean()

