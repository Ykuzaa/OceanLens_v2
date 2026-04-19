"""Current diagnostics."""

from __future__ import annotations

import torch

from oceanlens_v2.metrics.pointwise import masked_correlation, masked_rmse


def speed_from_uv(field: torch.Tensor, u_channel: int, v_channel: int) -> torch.Tensor:
    return torch.sqrt(field[:, u_channel : u_channel + 1].square() + field[:, v_channel : v_channel + 1].square() + 1.0e-12)


def kinetic_energy_from_uv(field: torch.Tensor, u_channel: int, v_channel: int) -> torch.Tensor:
    return 0.5 * (field[:, u_channel : u_channel + 1].square() + field[:, v_channel : v_channel + 1].square())


def relative_vorticity_from_uv(field: torch.Tensor, u_channel: int, v_channel: int) -> torch.Tensor:
    """Compute dv/dx - du/dy with unit grid spacing."""
    u = field[:, u_channel : u_channel + 1]
    v = field[:, v_channel : v_channel + 1]
    dvdx = 0.5 * (v[:, :, 1:-1, 2:] - v[:, :, 1:-1, :-2])
    dudy = 0.5 * (u[:, :, 2:, 1:-1] - u[:, :, :-2, 1:-1])
    return dvdx - dudy


def current_summary_metrics(predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, u_channel: int, v_channel: int) -> dict[str, torch.Tensor]:
    speed_pred = speed_from_uv(predicted, u_channel, v_channel)
    speed_true = speed_from_uv(target, u_channel, v_channel)
    ke_pred = kinetic_energy_from_uv(predicted, u_channel, v_channel)
    ke_true = kinetic_energy_from_uv(target, u_channel, v_channel)
    vort_pred = relative_vorticity_from_uv(predicted, u_channel, v_channel)
    vort_true = relative_vorticity_from_uv(target, u_channel, v_channel)
    vort_mask = mask[:, :, 1:-1, 1:-1]
    return {
        "speed_rmse": masked_rmse(speed_pred, speed_true, mask),
        "ke_rmse": masked_rmse(ke_pred, ke_true, mask),
        "vorticity_corr": masked_correlation(vort_pred, vort_true, vort_mask),
    }

