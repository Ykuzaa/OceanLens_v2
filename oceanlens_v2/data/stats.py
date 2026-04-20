"""Normalization statistics."""

from __future__ import annotations

import numpy as np


class PerPixelWelford:
    """Streaming Welford accumulator for per-pixel ocean mean/std.

    Statistics are accumulated per (channel, y, x) using only pixels where the
    provided ocean mask is true. The count is shared across channels because
    callers build the mask from pixels where all channels are finite.
    """

    def __init__(self, shape: tuple[int, int, int]) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.m2 = np.zeros(shape, dtype=np.float64)
        self.count = np.zeros(shape[1:], dtype=np.int64)

    def update(self, values: np.ndarray, mask: np.ndarray) -> None:
        """Update with one sample shaped (channel, y, x)."""
        if values.shape != self.mean.shape:
            raise ValueError(f"Expected values shape {self.mean.shape}, got {values.shape}")
        if mask.shape != self.count.shape:
            raise ValueError(f"Expected mask shape {self.count.shape}, got {mask.shape}")

        bool_mask = mask.astype(bool)
        safe_values = np.where(np.isnan(values), 0.0, values).astype(np.float64, copy=False)
        old_count = self.count
        new_count = old_count + bool_mask.astype(np.int64)
        safe_new_count = np.where(bool_mask, new_count, 1)

        delta = safe_values - self.mean
        self.mean += np.where(bool_mask[None, :, :], delta / safe_new_count[None, :, :], 0.0)
        delta2 = safe_values - self.mean
        self.m2 += np.where(bool_mask[None, :, :], delta * delta2, 0.0)
        self.count = new_count

    def finalize(self, std_floor: float = 1.0e-3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return mean/std maps and the ever-ocean valid mask."""
        valid = self.count > 0
        safe_count = np.where(valid, self.count, 1)
        variance = self.m2 / safe_count[None, :, :]
        variance = np.maximum(variance, 0.0)
        std = np.maximum(np.sqrt(variance), float(std_floor))
        mean = np.where(valid[None, :, :], self.mean, 0.0)
        std = np.where(valid[None, :, :], std, 1.0)
        return mean.astype(np.float32), std.astype(np.float32), valid


def normalize_per_pixel(values: np.ndarray, mean_map: np.ndarray, std_map: np.ndarray) -> np.ndarray:
    """Apply per-pixel normalization to (time, channel, y, x) or (channel, y, x)."""
    if values.ndim == 4:
        return (values - mean_map[None, :, :, :]) / std_map[None, :, :, :]
    if values.ndim == 3:
        return (values - mean_map) / std_map
    raise ValueError(f"Expected ndim 3 or 4, got {values.ndim}")


def denormalize_per_pixel(values: np.ndarray, mean_map: np.ndarray, std_map: np.ndarray) -> np.ndarray:
    """Reverse per-pixel normalization."""
    if values.ndim == 4:
        return values * std_map[None, :, :, :] + mean_map[None, :, :, :]
    if values.ndim == 3:
        return values * std_map + mean_map
    raise ValueError(f"Expected ndim 3 or 4, got {values.ndim}")


def replace_nan_with_pixel_mean(values: np.ndarray, mean_map: np.ndarray) -> np.ndarray:
    """Replace NaNs with the per-pixel climatological mean."""
    filled = values.copy()
    if values.ndim == 4:
        mean_broadcast = np.broadcast_to(mean_map[None, :, :, :], filled.shape)
    elif values.ndim == 3:
        mean_broadcast = np.broadcast_to(mean_map, filled.shape)
    else:
        raise ValueError(f"Expected ndim 3 or 4, got {values.ndim}")
    nan_positions = np.isnan(filled)
    filled[nan_positions] = mean_broadcast[nan_positions]
    return filled


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
