"""Preprocess GLORYS daily files into ready-to-train Zarr tensors."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import xarray as xr

from oceanlens_v2.data.coarsen import coarsen_to_target_resolution, interpolate_to_reference_grid
from oceanlens_v2.data.stats import normalize_channels, replace_nan_with_channel_mean


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


def _year_files(raw_dir: Path, year: int) -> list[Path]:
    files = sorted(raw_dir.glob(f"**/*{year}*.nc"))
    if not files:
        raise FileNotFoundError(f"No GLORYS files found for year {year} in {raw_dir}")
    return files


def _build_lr_dataset(hr_dataset: xr.Dataset, cfg) -> xr.Dataset:
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
    try:
        return interpolate_to_reference_grid(coarse_dataset, experimental_grid)
    finally:
        coarse_dataset.close()
        experimental_grid.close()


def _compute_streaming_train_stats(raw_dir: Path, train_years: list[int], variables: list[str]) -> tuple[np.ndarray, np.ndarray]:
    sums = np.zeros(len(variables), dtype=np.float64)
    sumsq = np.zeros(len(variables), dtype=np.float64)
    counts = np.zeros(len(variables), dtype=np.int64)

    for year in train_years:
        files = _year_files(raw_dir, year)
        print(f"[stats] year={year} files={len(files)}", flush=True)
        for index, path in enumerate(files, start=1):
            print(f"[stats] {year} {index}/{len(files)} {path.name}", flush=True)
            hr_dataset = _load_daily_glorys_file(path, variables)
            try:
                hr = _dataset_to_channel_array(hr_dataset, variables)
                hr_mask = np.all(~np.isnan(hr), axis=1)
                for channel in range(hr.shape[1]):
                    values = hr[:, channel][hr_mask]
                    valid = values[~np.isnan(values)].astype(np.float64, copy=False)
                    sums[channel] += valid.sum()
                    sumsq[channel] += np.square(valid).sum()
                    counts[channel] += valid.size
            finally:
                hr_dataset.close()

    if np.any(counts == 0):
        raise ValueError(f"Cannot compute normalization stats; empty channels: {np.flatnonzero(counts == 0).tolist()}")

    means = sums / counts
    variances = np.maximum(sumsq / counts - np.square(means), 0.0)
    stds = np.sqrt(variances) + 1.0e-8
    return means.astype(np.float32), stds.astype(np.float32)


def _normalize_sample(values: np.ndarray, mask: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    return normalize_channels(replace_nan_with_channel_mean(values, means), means, stds) * mask[:, None]


def _output_dataset(
    hr: np.ndarray,
    lr: np.ndarray,
    hr_mask: np.ndarray,
    lr_mask: np.ndarray,
    times: np.ndarray,
    years: np.ndarray,
    variables: list[str],
    hr_latitude: np.ndarray,
    hr_longitude: np.ndarray,
    lr_latitude: np.ndarray,
    lr_longitude: np.ndarray,
    means: np.ndarray | None = None,
    stds: np.ndarray | None = None,
) -> xr.Dataset:
    data_vars = {
        "hr": (("time", "channel", "y_hr", "x_hr"), hr),
        "lr": (("time", "channel", "y_lr", "x_lr"), lr),
        "hr_mask": (("time", "y_hr", "x_hr"), hr_mask),
        "lr_mask": (("time", "y_lr", "x_lr"), lr_mask),
        "year": (("time",), years),
    }
    if means is not None and stds is not None:
        data_vars["mean"] = (("channel",), means)
        data_vars["std"] = (("channel",), stds)

    return xr.Dataset(
        data_vars,
        coords={
            "time": times,
            "channel": variables,
            "lat_hr": (("y_hr",), hr_latitude),
            "lon_hr": (("x_hr",), hr_longitude),
            "lat_lr": (("y_lr",), lr_latitude),
            "lon_lr": (("x_lr",), lr_longitude),
        },
    )


def _zarr_encoding(dataset: xr.Dataset) -> dict[str, dict[str, tuple[int, ...]]]:
    encoding = {}
    for name, data_array in dataset.data_vars.items():
        if "time" in data_array.dims:
            encoding[name] = {"chunks": tuple(1 if dim == "time" else size for dim, size in data_array.sizes.items())}
    return encoding


def build_processed_store(cfg) -> None:
    """Create a Zarr store with HR, experimental LR and masks.

    The global GLORYS dataset is too large to keep all years in memory. This
    implementation streams over daily files twice: first to compute train stats,
    then to append normalized samples to the output Zarr store.
    """
    raw_dir = Path(cfg.data.raw_dir)
    output_store = Path(cfg.data.processed_store)
    variables = list(cfg.data.variables)
    years = sorted(set(cfg.data.train_years + cfg.data.val_years + cfg.data.test_years))

    print(f"[preprocess] raw_dir={raw_dir}", flush=True)
    print(f"[preprocess] output_store={output_store}", flush=True)
    print(f"[preprocess] years={years}", flush=True)

    means, stds = _compute_streaming_train_stats(raw_dir, list(cfg.data.train_years), variables)
    print(f"[preprocess] means={means.tolist()}", flush=True)
    print(f"[preprocess] stds={stds.tolist()}", flush=True)

    if output_store.exists():
        print(f"[preprocess] removing existing store: {output_store}", flush=True)
        shutil.rmtree(output_store)
    output_store.parent.mkdir(parents=True, exist_ok=True)

    hr_latitude = None
    hr_longitude = None
    lr_latitude = None
    lr_longitude = None
    wrote_first = False

    for year in years:
        files = _year_files(raw_dir, year)
        print(f"[write] year={year} files={len(files)}", flush=True)

        for index, path in enumerate(files, start=1):
            print(f"[write] {year} {index}/{len(files)} {path.name}", flush=True)
            hr_dataset = _load_daily_glorys_file(path, variables)
            try:
                if hr_latitude is None:
                    hr_latitude = hr_dataset.latitude.values
                    hr_longitude = hr_dataset.longitude.values

                lr_dataset = _build_lr_dataset(hr_dataset, cfg)
                try:
                    if lr_latitude is None:
                        lr_latitude = lr_dataset.latitude.values
                        lr_longitude = lr_dataset.longitude.values

                    hr = _dataset_to_channel_array(hr_dataset, variables)
                    lr = _dataset_to_channel_array(lr_dataset, variables)
                    hr_mask = np.all(~np.isnan(hr), axis=1).astype(np.float32)
                    lr_mask = np.all(~np.isnan(lr), axis=1).astype(np.float32)
                    hr = _normalize_sample(hr, hr_mask, means, stds)
                    lr = _normalize_sample(lr, lr_mask, means, stds)
                    times = hr_dataset.time.values
                    years_by_sample = np.full(hr_dataset.sizes["time"], year, dtype=np.int16)

                    output = _output_dataset(
                        hr,
                        lr,
                        hr_mask,
                        lr_mask,
                        times,
                        years_by_sample,
                        variables,
                        hr_latitude,
                        hr_longitude,
                        lr_latitude,
                        lr_longitude,
                        means if not wrote_first else None,
                        stds if not wrote_first else None,
                    )
                    if wrote_first:
                        output.to_zarr(output_store, mode="a", append_dim="time")
                    else:
                        output.to_zarr(output_store, mode="w", encoding=_zarr_encoding(output))
                        wrote_first = True
                finally:
                    lr_dataset.close()
            finally:
                hr_dataset.close()

    print(f"[preprocess] done: {output_store}", flush=True)
