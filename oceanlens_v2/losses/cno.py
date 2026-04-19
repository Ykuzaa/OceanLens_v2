"""CNO branch losses."""

from __future__ import annotations

import torch

from oceanlens_v2.losses.gradient import temperature_log_gradient_loss


def masked_residual_cno_loss(
    predicted_residual: torch.Tensor,
    hr_target: torch.Tensor,
    lr_on_hr_grid: torch.Tensor,
    ocean_mask: torch.Tensor,
    cfg,
) -> torch.Tensor:
    """Train CNO to predict HR-LR residual and optionally match thermal fronts."""
    target_residual = (hr_target - lr_on_hr_grid) * ocean_mask
    weights = torch.tensor(cfg.loss.cno.variable_weights, device=predicted_residual.device, dtype=predicted_residual.dtype)
    loss = predicted_residual.new_tensor(0.0)
    denom = ocean_mask.sum().clamp(min=1.0)
    for channel in range(predicted_residual.shape[1]):
        channel_error = (predicted_residual[:, channel] - target_residual[:, channel]).abs() * ocean_mask[:, 0]
        loss = loss + weights[channel] * channel_error.sum() / denom
    loss = loss / weights.sum().clamp(min=1.0)

    grad_cfg = cfg.loss.cno.log_temperature_gradient
    if bool(grad_cfg.enabled) and float(grad_cfg.weight) > 0.0:
        variables = list(cfg.data.variables)
        temperature_channel = variables.index(str(grad_cfg.variable))
        mu = (lr_on_hr_grid + predicted_residual) * ocean_mask
        grad_loss = temperature_log_gradient_loss(
            mu,
            hr_target * ocean_mask,
            ocean_mask,
            temperature_channel,
            float(grad_cfg.eps),
        )
        loss = loss + float(grad_cfg.weight) * grad_loss
    return loss

