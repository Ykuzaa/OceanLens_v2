"""Lightning data module for preprocessed OceanLens tensors."""

from __future__ import annotations

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from oceanlens_v2.data.dataset import PreprocessedOceanDataset


class OceanLensDataModule(pl.LightningDataModule):
    """Build train/validation loaders from a processed Zarr store."""

    def __init__(self, cfg, phase: str) -> None:
        super().__init__()
        self.cfg = cfg
        self.phase = phase

    def setup(self, stage: str | None = None) -> None:
        patch_size = int(self.cfg.data.patch_size)
        common = {
            "processed_store": self.cfg.data.processed_store,
            "patch_size": patch_size,
            "min_ocean_fraction": float(self.cfg.data.min_ocean_fraction),
        }
        self.train_dataset = PreprocessedOceanDataset(
            years=list(self.cfg.data.train_years),
            deterministic_crop=False,
            **common,
        )
        self.val_dataset = PreprocessedOceanDataset(
            years=list(self.cfg.data.val_years),
            deterministic_crop=True,
            **common,
        )

    def train_dataloader(self) -> DataLoader:
        training_cfg = self.cfg.training[self.phase]
        return DataLoader(
            self.train_dataset,
            batch_size=int(training_cfg.batch_size),
            shuffle=True,
            num_workers=int(self.cfg.data.num_workers),
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        training_cfg = self.cfg.training[self.phase]
        return DataLoader(
            self.val_dataset,
            batch_size=int(training_cfg.batch_size),
            shuffle=False,
            num_workers=int(self.cfg.data.num_workers),
            pin_memory=True,
            drop_last=False,
        )

