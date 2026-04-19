"""Simple kinetic-energy spectra utilities."""

from __future__ import annotations

import numpy as np


def radial_kinetic_energy_spectrum(u: np.ndarray, v: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Compute an isotropic KE spectrum from 2D u/v fields.

    Args:
        u, v: arrays shaped (H, W).
        mask: optional ocean mask shaped (H, W).
    """
    if mask is not None:
        u = np.where(mask > 0, u, 0.0)
        v = np.where(mask > 0, v, 0.0)

    u_hat = np.fft.fft2(u)
    v_hat = np.fft.fft2(v)
    energy_2d = 0.5 * (np.abs(u_hat) ** 2 + np.abs(v_hat) ** 2)

    height, width = u.shape
    ky = np.fft.fftfreq(height) * height
    kx = np.fft.fftfreq(width) * width
    grid_kx, grid_ky = np.meshgrid(kx, ky)
    radius = np.sqrt(grid_kx**2 + grid_ky**2).astype(np.int64)
    max_radius = int(radius.max())

    spectrum = np.zeros(max_radius + 1, dtype=np.float64)
    counts = np.zeros(max_radius + 1, dtype=np.float64)
    np.add.at(spectrum, radius.ravel(), energy_2d.ravel())
    np.add.at(counts, radius.ravel(), 1.0)
    spectrum = spectrum / np.maximum(counts, 1.0)
    return np.arange(max_radius + 1), spectrum

