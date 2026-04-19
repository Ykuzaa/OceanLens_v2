"""Normalization statistics."""

from __future__ import annotations

import numpy as np


def compute_ocean_mean_std(values: np.ndarray, mask: np.ndarray, eps: float = 1.0e-8) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean/std over ocean pixels.

    Args:
        values: array shaped (time, channel, y, x).
        mask: array shaped (y, x) or (time, y, x), where 1 means ocean.
    """
    if mask.ndim == 2:
        valid = mask[None, None, :, :].astype(bool)
    elif mask.ndim == 3:
        valid = mask[:, None, :, :].astype(bool)
    else:
        raise ValueError(f"Expected 2D or 3D mask, got shape {mask.shape}")

    means = []
    stds = []
    for channel in range(values.shape[1]):
        channel_values = values[:, channel : channel + 1]
        ocean_values = channel_values[valid]
        means.append(float(np.nanmean(ocean_values)))
        stds.append(float(np.nanstd(ocean_values) + eps))
    return np.asarray(means, dtype=np.float32), np.asarray(stds, dtype=np.float32)


def replace_nan_with_channel_mean(values: np.ndarray, means: np.ndarray) -> np.ndarray:
    """Fill NaNs with the channel mean so land becomes zero after normalization."""
    filled = values.copy()
    for channel, mean in enumerate(means):
        channel_values = filled[:, channel]
        channel_values[np.isnan(channel_values)] = mean
        filled[:, channel] = channel_values
    return filled


def normalize_channels(values: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    """Apply per-channel normalization to an array shaped (time, channel, y, x)."""
    return (values - means[None, :, None, None]) / (stds[None, :, None, None] + 1.0e-8)

