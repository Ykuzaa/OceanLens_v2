"""Preprocess GLORYS daily files into ready-to-train Zarr tensors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from oceanlens_v2.data.coarsen import coarsen_to_target_resolution, interpolate_to_reference_grid
from oceanlens_v2.data.stats import compute_ocean_mean_std, normalize_channels, replace_nan_with_channel_mean


def _load_daily_glorys_file(path: Path, variables: list[str]) -> xr.Dataset:
    dataset = xr.open_dataset(path)
    selected = {}
    for variable in variables:
        data = dataset[variable]
        if "depth" in data.dims:
            data = data.isel(depth=0)
        selected[variable] = data
    return xr.Dataset(selected, coords={key: dataset.coords[key] for key in dataset.coords})


def _dataset_to_channel_array(dataset: xr.Dataset, variables: list[str]) -> np.ndarray:
    arrays = [dataset[variable].values for variable in variables]
    stacked = np.stack(arrays, axis=1).astype(np.float32)
    if stacked.ndim != 4:
        raise ValueError(f"Expected (time, channel, y, x), got {stacked.shape}")
    return stacked


def build_processed_store(cfg) -> None:
    """Create a Zarr store with HR, experimental LR and masks."""
    raw_dir = Path(cfg.data.raw_dir)
    output_store = Path(cfg.data.processed_store)
    variables = list(cfg.data.variables)
    years = sorted(set(cfg.data.train_years + cfg.data.val_years + cfg.data.test_years))

    hr_arrays = []
    lr_arrays = []
    time_arrays = []
    year_arrays = []
    hr_latitude = None
    hr_longitude = None
    lr_latitude = None
    lr_longitude = None

    for year in years:
        files = sorted(raw_dir.glob(f"**/*{year}*.nc"))
        if not files:
            raise FileNotFoundError(f"No GLORYS files found for year {year} in {raw_dir}")

        for path in files:
            hr_dataset = _load_daily_glorys_file(path, variables)
            if hr_latitude is None:
                hr_latitude = hr_dataset.latitude.values
                hr_longitude = hr_dataset.longitude.values

            coarse_dataset = coarsen_to_target_resolution(
                hr_dataset,
                float(cfg.data.target_resolution_deg),
                float(cfg.data.coarse_resolution_deg),
            )

            experimental_grid = hr_dataset.coarsen(
                latitude=int(round(float(cfg.data.experimental_lr_resolution_deg) / float(cfg.data.target_resolution_deg))),
                longitude=int(round(float(cfg.data.experimental_lr_resolution_deg) / float(cfg.data.target_resolution_deg))),
                boundary="trim",
            ).mean()
            lr_dataset = interpolate_to_reference_grid(coarse_dataset, experimental_grid)
            if lr_latitude is None:
                lr_latitude = lr_dataset.latitude.values
                lr_longitude = lr_dataset.longitude.values

            hr_arrays.append(_dataset_to_channel_array(hr_dataset, variables))
            lr_arrays.append(_dataset_to_channel_array(lr_dataset, variables))
            time_arrays.append(hr_dataset.time.values)
            year_arrays.append(np.full(hr_dataset.sizes["time"], year, dtype=np.int16))
            hr_dataset.close()
            coarse_dataset.close()
            experimental_grid.close()
            lr_dataset.close()

    hr = np.concatenate(hr_arrays, axis=0)
    lr = np.concatenate(lr_arrays, axis=0)
    times = np.concatenate(time_arrays, axis=0)
    years_by_sample = np.concatenate(year_arrays, axis=0)

    hr_mask = np.all(~np.isnan(hr), axis=1).astype(np.float32)
    lr_mask = np.all(~np.isnan(lr), axis=1).astype(np.float32)

    train_selector = np.isin(years_by_sample, np.asarray(cfg.data.train_years))
    means, stds = compute_ocean_mean_std(hr[train_selector], hr_mask[train_selector])

    hr = normalize_channels(replace_nan_with_channel_mean(hr, means), means, stds) * hr_mask[:, None]
    lr = normalize_channels(replace_nan_with_channel_mean(lr, means), means, stds) * lr_mask[:, None]

    output_store.parent.mkdir(parents=True, exist_ok=True)
    output = xr.Dataset(
        {
            "hr": (("time", "channel", "y_hr", "x_hr"), hr),
            "lr": (("time", "channel", "y_lr", "x_lr"), lr),
            "hr_mask": (("time", "y_hr", "x_hr"), hr_mask),
            "lr_mask": (("time", "y_lr", "x_lr"), lr_mask),
            "mean": (("channel",), means),
            "std": (("channel",), stds),
            "year": (("time",), years_by_sample),
        },
        coords={
            "time": times,
            "channel": variables,
            "lat_hr": (("y_hr",), hr_latitude),
            "lon_hr": (("x_hr",), hr_longitude),
            "lat_lr": (("y_lr",), lr_latitude),
            "lon_lr": (("x_lr",), lr_longitude),
        },
    )
    output.to_zarr(output_store, mode="w")
