"""Front-aware temperature-gradient losses."""

from __future__ import annotations

import torch


def centered_gradient_magnitude(field: torch.Tensor, ocean_mask: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute land-safe centered gradient magnitude.

    Args:
        field: tensor shaped (B, 1, H, W).
        ocean_mask: tensor shaped (B, 1, H, W), 1 over ocean.
    """
    dx = 0.5 * (field[:, :, 1:-1, 2:] - field[:, :, 1:-1, :-2])
    dy = 0.5 * (field[:, :, 2:, 1:-1] - field[:, :, :-2, 1:-1])
    valid = (
        ocean_mask[:, :, 1:-1, 1:-1].bool()
        & ocean_mask[:, :, 1:-1, 2:].bool()
        & ocean_mask[:, :, 1:-1, :-2].bool()
        & ocean_mask[:, :, 2:, 1:-1].bool()
        & ocean_mask[:, :, :-2, 1:-1].bool()
    )
    magnitude = torch.sqrt(dx.square() + dy.square() + 1.0e-12)
    return magnitude, valid.to(dtype=field.dtype)


def temperature_log_gradient_loss(
    predicted_field: torch.Tensor,
    target_field: torch.Tensor,
    ocean_mask: torch.Tensor,
    temperature_channel: int,
    eps: float,
) -> torch.Tensor:
    """L1 loss between log(|grad thetao| + eps) maps."""
    pred_mag, valid = centered_gradient_magnitude(
        predicted_field[:, temperature_channel : temperature_channel + 1],
        ocean_mask,
        eps,
    )
    true_mag, _ = centered_gradient_magnitude(
        target_field[:, temperature_channel : temperature_channel + 1],
        ocean_mask,
        eps,
    )
    pred_log = torch.log(pred_mag + eps)
    true_log = torch.log(true_mag + eps)
    denom = valid.sum().clamp(min=1.0)
    return ((pred_log - true_log).abs() * valid).sum() / denom

