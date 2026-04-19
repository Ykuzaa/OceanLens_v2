"""Flow Matching losses."""

from __future__ import annotations

import torch


def flow_matching_velocity_loss(
    fm_model,
    target_residual: torch.Tensor,
    mu_condition: torch.Tensor,
    ocean_mask: torch.Tensor,
    mask_ocean: bool,
) -> torch.Tensor:
    """OT-CFM velocity loss for residual target HR-mu."""
    batch_size = target_residual.shape[0]
    t = torch.rand(batch_size, device=target_residual.device)
    t_view = t.view(batch_size, 1, 1, 1)
    x0 = torch.randn_like(target_residual)
    x_t = (1.0 - t_view) * x0 + t_view * target_residual
    target_velocity = target_residual - x0
    predicted_velocity = fm_model(x_t, t, mu_condition)
    loss = (predicted_velocity - target_velocity).square()
    if mask_ocean:
        loss = loss * ocean_mask
        return loss.sum() / (ocean_mask.sum().clamp(min=1.0) * target_residual.shape[1])
    return loss.mean()

