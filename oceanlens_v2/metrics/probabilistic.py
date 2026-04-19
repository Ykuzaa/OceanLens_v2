"""Probabilistic ensemble metrics."""

from __future__ import annotations

import torch


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / (mask.sum().clamp(min=1.0) * values.shape[-3])


def masked_ensemble_crps(ensemble: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute ensemble CRPS averaged over channels and ocean pixels."""
    if ensemble.shape[0] < 2:
        return ensemble.new_tensor(float("nan"))
    absolute_error = (ensemble - target.unsqueeze(0)).abs().mean(dim=0)
    pairwise_distance = (ensemble[:, None] - ensemble[None, :]).abs().mean(dim=(0, 1))
    return _masked_mean(absolute_error - 0.5 * pairwise_distance, mask)


def masked_spread_skill_ratio(ensemble: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute ensemble spread divided by RMSE of the ensemble mean."""
    if ensemble.shape[0] < 2:
        return ensemble.new_tensor(float("nan"))
    mean_prediction = ensemble.mean(dim=0)
    variance = ensemble.var(dim=0, unbiased=False)
    spread = torch.sqrt(_masked_mean(variance, mask).clamp(min=0.0))
    squared_error = (mean_prediction - target).square()
    skill = torch.sqrt(_masked_mean(squared_error, mask).clamp(min=1.0e-12))
    return spread / skill

