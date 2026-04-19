"""Zarr-backed dataset for OceanLens v2."""

from __future__ import annotations

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset


class PreprocessedOceanDataset(Dataset):
    """Return LR/HR/mask samples from a processed Zarr store."""

    def __init__(
        self,
        processed_store: str,
        years: list[int],
        patch_size: int | None,
        min_ocean_fraction: float,
        deterministic_crop: bool,
    ) -> None:
        self.dataset = xr.open_zarr(processed_store)
        year_values = self.dataset["year"].values
        self.indices = np.flatnonzero(np.isin(year_values, np.asarray(years))).astype(np.int64)
        self.patch_size = patch_size
        self.min_ocean_fraction = float(min_ocean_fraction)
        self.deterministic_crop = deterministic_crop

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        sample_index = int(self.indices[item])
        lr = torch.from_numpy(self.dataset["lr"].isel(time=sample_index).values.astype(np.float32))
        hr = torch.from_numpy(self.dataset["hr"].isel(time=sample_index).values.astype(np.float32))
        hr_mask = torch.from_numpy(self.dataset["hr_mask"].isel(time=sample_index).values.astype(np.float32)).unsqueeze(0)
        lr_mask = torch.from_numpy(self.dataset["lr_mask"].isel(time=sample_index).values.astype(np.float32)).unsqueeze(0)

        if self.patch_size:
            lr, lr_mask, hr, hr_mask = self._crop_aligned_lr_hr_patch(lr, lr_mask, hr, hr_mask, item)

        return {
            "lr": lr,
            "hr": hr,
            "hr_mask": hr_mask,
            "lr_mask": lr_mask,
            "sample_index": torch.tensor(sample_index, dtype=torch.long),
        }

    def _crop_aligned_lr_hr_patch(
        self,
        lr: torch.Tensor,
        lr_mask: torch.Tensor,
        hr: torch.Tensor,
        hr_mask: torch.Tensor,
        item: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        patch_size = int(self.patch_size)
        _, height, width = hr.shape
        if height <= patch_size or width <= patch_size:
            return lr, lr_mask, hr, hr_mask

        generator = None
        if self.deterministic_crop:
            generator = torch.Generator()
            generator.manual_seed(int(item))

        max_retries = 20
        top = 0
        left = 0
        for _ in range(max_retries):
            top = int(torch.randint(0, height - patch_size + 1, (1,), generator=generator).item())
            left = int(torch.randint(0, width - patch_size + 1, (1,), generator=generator).item())
            candidate_mask = hr_mask[:, top : top + patch_size, left : left + patch_size]
            if float(candidate_mask.mean()) >= self.min_ocean_fraction:
                break

        hr = hr[:, top : top + patch_size, left : left + patch_size]
        hr_mask = hr_mask[:, top : top + patch_size, left : left + patch_size]

        lr_height, lr_width = lr.shape[-2:]
        lr_top = int(round(top * lr_height / height))
        lr_left = int(round(left * lr_width / width))
        lr_patch_h = max(2, int(round(patch_size * lr_height / height)))
        lr_patch_w = max(2, int(round(patch_size * lr_width / width)))
        lr_top = min(lr_top, max(0, lr_height - lr_patch_h))
        lr_left = min(lr_left, max(0, lr_width - lr_patch_w))

        lr = lr[:, lr_top : lr_top + lr_patch_h, lr_left : lr_left + lr_patch_w]
        lr_mask = lr_mask[:, lr_top : lr_top + lr_patch_h, lr_left : lr_left + lr_patch_w]
        return lr, lr_mask, hr, hr_mask
