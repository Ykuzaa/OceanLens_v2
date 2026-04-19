"""Thermal-front diagnostics."""

from __future__ import annotations

import torch

from oceanlens_v2.losses.gradient import centered_gradient_magnitude
from oceanlens_v2.metrics.pointwise import masked_correlation, masked_rmse


def log_temperature_gradient_field(field: torch.Tensor, mask: torch.Tensor, temperature_channel: int, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    magnitude, valid = centered_gradient_magnitude(field[:, temperature_channel : temperature_channel + 1], mask, eps)
    return torch.log(magnitude + eps), valid


def log_temperature_gradient_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    temperature_channel: int,
    eps: float = 1.0e-6,
) -> dict[str, torch.Tensor]:
    pred_log, valid = log_temperature_gradient_field(predicted, mask, temperature_channel, eps)
    true_log, _ = log_temperature_gradient_field(target, mask, temperature_channel, eps)
    return {
        "loggrad_thetao_rmse": masked_rmse(pred_log, true_log, valid),
        "loggrad_thetao_corr": masked_correlation(pred_log, true_log, valid),
    }

