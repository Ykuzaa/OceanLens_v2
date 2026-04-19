"""Offline coarsening and interpolation utilities."""

from __future__ import annotations

import xarray as xr


def coarsen_to_target_resolution(
    dataset: xr.Dataset,
    target_resolution_deg: float,
    coarse_resolution_deg: float,
) -> xr.Dataset:
    """Coarsen a regular lat/lon dataset with area-like block averages.

    This assumes the source grid is approximately regular and that
    `coarse_resolution_deg / target_resolution_deg` is close to an integer.
    """
    factor = int(round(coarse_resolution_deg / target_resolution_deg))
    if factor < 1:
        raise ValueError("coarse_resolution_deg must be larger than target_resolution_deg")
    return dataset.coarsen(latitude=factor, longitude=factor, boundary="trim").mean()


def interpolate_to_reference_grid(dataset: xr.Dataset, reference_grid: xr.Dataset) -> xr.Dataset:
    """Interpolate `dataset` onto the latitude/longitude grid of `reference_grid`."""
    return dataset.interp(
        latitude=reference_grid.latitude,
        longitude=reference_grid.longitude,
        method="linear",
    )

