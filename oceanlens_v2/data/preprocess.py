"""Preprocess GLORYS daily files into ready-to-train Zarr tensors."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import xarray as xr

from oceanlens_v2.data.coarsen import coarsen_to_target_resolution, interpolate_to_reference_grid
from oceanlens_v2.data.stats import PerPixelWelford, normalize_per_pixel, replace_nan_with_pixel_mean


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


def _compute_perpixel_train_stats(
    raw_dir: Path,
    train_years: list[int],
    variables: list[str],
    cfg,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-pixel mean/std maps for HR and LR grids, streaming."""
    hr_welford: PerPixelWelford | None = None
    lr_welford: PerPixelWelford | None = None

    for year in train_years:
        files = _year_files(raw_dir, year)
        print(f"[stats] year={year} files={len(files)}", flush=True)
        for index, path in enumerate(files, start=1):
            print(f"[stats] {year} {index}/{len(files)} {path.name}", flush=True)
            hr_dataset = _load_daily_glorys_file(path, variables)
            try:
                lr_dataset = _build_lr_dataset(hr_dataset, cfg)
                try:
                    hr = _dataset_to_channel_array(hr_dataset, variables)
                    lr = _dataset_to_channel_array(lr_dataset, variables)

                    if hr_welford is None:
                        hr_welford = PerPixelWelford(hr.shape[1:])
                    if lr_welford is None:
                        lr_welford = PerPixelWelford(lr.shape[1:])

                    for t in range(hr.shape[0]):
                        hr_frame = hr[t]
                        lr_frame = lr[t]
                        hr_mask = np.all(~np.isnan(hr_frame), axis=0).astype(np.float32)
                        lr_mask = np.all(~np.isnan(lr_frame), axis=0).astype(np.float32)
                        hr_welford.update(hr_frame, hr_mask)
                        lr_welford.update(lr_frame, lr_mask)
                finally:
                    lr_dataset.close()
            finally:
                hr_dataset.close()

    if hr_welford is None or lr_welford is None:
        raise ValueError("Cannot compute normalization stats; no training samples were found")

    hr_mean, hr_std, _ = hr_welford.finalize()
    lr_mean, lr_std, _ = lr_welford.finalize()
    return hr_mean, hr_std, lr_mean, lr_std


def _normalize_sample_perpixel(values: np.ndarray, mask: np.ndarray, mean_map: np.ndarray, std_map: np.ndarray) -> np.ndarray:
    """Normalize with per-pixel maps; land pixels are zeroed by mask multiplication."""
    filled = replace_nan_with_pixel_mean(values, mean_map)
    normalized = normalize_per_pixel(filled, mean_map, std_map)
    return normalized * mask[:, None]


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
    hr_mean_map: np.ndarray | None = None,
    hr_std_map: np.ndarray | None = None,
    lr_mean_map: np.ndarray | None = None,
    lr_std_map: np.ndarray | None = None,
) -> xr.Dataset:
    data_vars = {
        "hr": (("time", "channel", "y_hr", "x_hr"), hr),
        "lr": (("time", "channel", "y_lr", "x_lr"), lr),
        "hr_mask": (("time", "y_hr", "x_hr"), hr_mask),
        "lr_mask": (("time", "y_lr", "x_lr"), lr_mask),
        "year": (("time",), years),
    }
    if hr_mean_map is not None:
        if hr_std_map is None or lr_mean_map is None or lr_std_map is None:
            raise ValueError("All per-pixel normalization maps must be provided together")
        data_vars["hr_mean_map"] = (("channel", "y_hr", "x_hr"), hr_mean_map)
        data_vars["hr_std_map"] = (("channel", "y_hr", "x_hr"), hr_std_map)
        data_vars["lr_mean_map"] = (("channel", "y_lr", "x_lr"), lr_mean_map)
        data_vars["lr_std_map"] = (("channel", "y_lr", "x_lr"), lr_std_map)

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
        elif name.endswith("_map"):
            encoding[name] = {
                "chunks": tuple(
                    min(512, size) if dim.startswith(("y_", "x_")) else size for dim, size in data_array.sizes.items()
                )
            }
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

    hr_mean_map, hr_std_map, lr_mean_map, lr_std_map = _compute_perpixel_train_stats(
        raw_dir,
        list(cfg.data.train_years),
        variables,
        cfg,
    )
    print(f"[preprocess] hr_mean_map shape={hr_mean_map.shape}", flush=True)
    print(f"[preprocess] hr_std_map min={float(hr_std_map.min())} max={float(hr_std_map.max())}", flush=True)
    print(f"[preprocess] lr_mean_map shape={lr_mean_map.shape}", flush=True)
    print(f"[preprocess] lr_std_map min={float(lr_std_map.min())} max={float(lr_std_map.max())}", flush=True)

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
                    hr = _normalize_sample_perpixel(hr, hr_mask, hr_mean_map, hr_std_map)
                    lr = _normalize_sample_perpixel(lr, lr_mask, lr_mean_map, lr_std_map)
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
                        hr_mean_map if not wrote_first else None,
                        hr_std_map if not wrote_first else None,
                        lr_mean_map if not wrote_first else None,
                        lr_std_map if not wrote_first else None,
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
