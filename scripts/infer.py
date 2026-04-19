#!/usr/bin/env python
"""Run CNO+FM inference and save compact NPZ outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from oceanlens_v2.data.dataset import PreprocessedOceanDataset
from oceanlens_v2.training.system import OceanLensV1System
from oceanlens_v2.utils.config import load_config


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

    ensemble_members = args.ensemble_members or int(cfg.inference.ensemble_members)
    with torch.no_grad():
        for sample_id, batch in enumerate(loader):
            if sample_id >= args.max_samples:
                break
            lr = batch["lr"].to(args.device)
            hr = batch["hr"].to(args.device)
            mask = batch["hr_mask"].to(args.device)
            mu, _ = system.compute_mu(lr, hr.shape[-2:], mask)
            pred = system.sample(lr, mask, int(cfg.inference.n_steps), ensemble_members)
            np.savez_compressed(
                output_dir / f"sample_{sample_id:04d}.npz",
                lr=system.prepare_lr_on_hr_grid(lr, hr.shape[-2:]).cpu().numpy(),
                mu=mu.cpu().numpy(),
                pred=pred.cpu().numpy(),
                hr=hr.cpu().numpy(),
                mask=mask.cpu().numpy(),
            )
            print(f"Wrote {output_dir / f'sample_{sample_id:04d}.npz'}")


if __name__ == "__main__":
    main()

