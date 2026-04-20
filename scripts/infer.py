#!/usr/bin/env python
"""Run CNO+FM inference and save compact NPZ outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import xarray as xr

from oceanlens_v2.data.dataset import PreprocessedOceanDataset
from oceanlens_v2.training.system import OceanLensV1System
from oceanlens_v2.utils.config import load_config


def tile_starts(size: int, tile_size: int, overlap: int) -> list[int]:
    """Return starts that cover an axis exactly once at the boundaries."""
    if size <= tile_size:
        return [0]
    stride = max(1, tile_size - overlap)
    starts = list(range(0, size - tile_size + 1, stride))
    if starts[-1] != size - tile_size:
        starts.append(size - tile_size)
    return starts


def gaussian_tile_weight(height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Build a smooth positive 2D blending window."""
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    weight = torch.exp(-2.0 * (xx.square() + yy.square()))
    return weight.clamp(min=1.0e-3).view(1, 1, height, width)


def denormalize_tensor_per_pixel(values: torch.Tensor, mean_map: torch.Tensor, std_map: torch.Tensor) -> torch.Tensor:
    """Reverse per-pixel normalization for tensors with channel/y/x trailing axes."""
    if values.ndim == 5:
        return values * std_map[None, None, :, :, :] + mean_map[None, None, :, :, :]
    if values.ndim == 4:
        return values * std_map[None, :, :, :] + mean_map[None, :, :, :]
    if values.ndim == 3:
        return values * std_map + mean_map
    raise ValueError(f"Expected ndim 3, 4 or 5, got {values.ndim}")


def load_normalization_maps(processed_store: str, device: torch.device) -> dict[str, torch.Tensor] | None:
    """Load per-pixel normalization maps from the processed Zarr store."""
    store = xr.open_zarr(processed_store)
    try:
        required = ["hr_mean_map", "hr_std_map", "lr_mean_map", "lr_std_map"]
        if not all(name in store for name in required):
            return None
        return {
            name: torch.from_numpy(store[name].values.astype(np.float32)).to(device)
            for name in required
        }
    finally:
        store.close()


@torch.no_grad()
def sample_tiled(
    system: OceanLensV1System,
    mu: torch.Tensor,
    mask: torch.Tensor,
    n_steps: int,
    ensemble_members: int,
    tile_size: int,
    tile_overlap: int,
) -> torch.Tensor:
    """Sample FM residuals tile-by-tile and blend overlaps."""
    _, channels, height, width = mu.shape
    if tile_size <= 0 or (height <= tile_size and width <= tile_size):
        predictions = []
        for _ in range(int(ensemble_members)):
            residual = system.integrate_fm_residual(mu, mask, n_steps=n_steps)
            predictions.append((mu + residual) * mask)
        return torch.stack(predictions, dim=0)

    y_starts = tile_starts(height, tile_size, tile_overlap)
    x_starts = tile_starts(width, tile_size, tile_overlap)
    accumulated = torch.zeros((ensemble_members, mu.shape[0], channels, height, width), device=mu.device, dtype=mu.dtype)
    accumulated_weight = torch.zeros((1, 1, 1, height, width), device=mu.device, dtype=mu.dtype)

    for top in y_starts:
        bottom = min(top + tile_size, height)
        for left in x_starts:
            right = min(left + tile_size, width)
            mu_tile = mu[:, :, top:bottom, left:right]
            mask_tile = mask[:, :, top:bottom, left:right]
            weight = gaussian_tile_weight(bottom - top, right - left, mu.device, mu.dtype)
            tile_predictions = []
            for _ in range(int(ensemble_members)):
                residual = system.integrate_fm_residual(mu_tile, mask_tile, n_steps=n_steps)
                tile_predictions.append((mu_tile + residual) * mask_tile)
            tile_ensemble = torch.stack(tile_predictions, dim=0)
            accumulated[:, :, :, top:bottom, left:right] += tile_ensemble * weight.unsqueeze(0)
            accumulated_weight[:, :, :, top:bottom, left:right] += weight.unsqueeze(0)

    return accumulated / accumulated_weight.clamp(min=1.0e-6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v1.yaml")
    parser.add_argument("--cno_ckpt", required=True)
    parser.add_argument("--fm_ckpt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ensemble_members", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = PreprocessedOceanDataset(
        processed_store=cfg.data.processed_store,
        years=list(cfg.data.test_years),
        patch_size=None,
        min_ocean_fraction=float(cfg.data.min_ocean_fraction),
        deterministic_crop=True,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    system = OceanLensV1System(cfg, phase="fm")
    system.load_cno_weights(args.cno_ckpt)
    system.load_fm_weights(args.fm_ckpt)
    system.eval().to(args.device)
    device = torch.device(args.device)
    normalization_maps = load_normalization_maps(cfg.data.processed_store, device)
    if normalization_maps is None:
        print("[infer] normalization maps not found; saving normalized arrays", flush=True)

    ensemble_members = args.ensemble_members or int(cfg.inference.ensemble_members)
    with torch.no_grad():
        for sample_id, batch in enumerate(loader):
            if sample_id >= args.max_samples:
                break
            lr = batch["lr"].to(args.device)
            hr = batch["hr"].to(args.device)
            mask = batch["hr_mask"].to(args.device)
            mu, _ = system.compute_mu(lr, hr.shape[-2:], mask)
            ensemble = sample_tiled(
                system,
                mu,
                mask,
                int(cfg.inference.n_steps),
                ensemble_members,
                int(cfg.inference.tile_size),
                int(cfg.inference.tile_overlap),
            )
            pred = ensemble.mean(dim=0)
            lr_on_hr_grid = system.prepare_lr_on_hr_grid(lr, hr.shape[-2:])
            if normalization_maps is not None:
                hr_mean_map = normalization_maps["hr_mean_map"]
                hr_std_map = normalization_maps["hr_std_map"]
                lr_mean_map = normalization_maps["lr_mean_map"]
                lr_std_map = normalization_maps["lr_std_map"]
                lr_physical = denormalize_tensor_per_pixel(lr, lr_mean_map, lr_std_map)
                lr_on_hr_grid = system.prepare_lr_on_hr_grid(lr_physical, hr.shape[-2:])
                mu = denormalize_tensor_per_pixel(mu, hr_mean_map, hr_std_map)
                pred = denormalize_tensor_per_pixel(pred, hr_mean_map, hr_std_map)
                ensemble = denormalize_tensor_per_pixel(ensemble, hr_mean_map, hr_std_map)
                hr = denormalize_tensor_per_pixel(hr, hr_mean_map, hr_std_map)
            np.savez_compressed(
                output_dir / f"sample_{sample_id:04d}.npz",
                lr=lr_on_hr_grid.cpu().numpy(),
                mu=mu.cpu().numpy(),
                pred=pred.cpu().numpy(),
                ensemble=ensemble.cpu().numpy(),
                hr=hr.cpu().numpy(),
                mask=mask.cpu().numpy(),
            )
            print(f"Wrote {output_dir / f'sample_{sample_id:04d}.npz'}")


if __name__ == "__main__":
    main()
