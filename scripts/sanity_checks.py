#!/usr/bin/env python
"""Quick checks before training."""

from __future__ import annotations

import argparse

import xarray as xr

from oceanlens_v2.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v1.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    dataset = xr.open_zarr(cfg.data.processed_store)

    print("Store:", cfg.data.processed_store)
    print("HR shape:", dataset["hr"].shape)
    print("LR shape:", dataset["lr"].shape)
    print("HR mask ocean fraction:", float(dataset["hr_mask"].mean()))
    print("LR mask ocean fraction:", float(dataset["lr_mask"].mean()))
    print("Variables:", list(dataset["channel"].values))
    print("Years:", sorted(set(dataset["year"].values.tolist())))
    print("HR mean_map shape:", dataset["hr_mean_map"].shape)
    print("HR mean_map: min", float(dataset["hr_mean_map"].min()), "max", float(dataset["hr_mean_map"].max()))
    print("HR std_map: min", float(dataset["hr_std_map"].min()), "max", float(dataset["hr_std_map"].max()))
    print("LR mean_map shape:", dataset["lr_mean_map"].shape)
    print("LR mean_map: min", float(dataset["lr_mean_map"].min()), "max", float(dataset["lr_mean_map"].max()))
    print("LR std_map: min", float(dataset["lr_std_map"].min()), "max", float(dataset["lr_std_map"].max()))


if __name__ == "__main__":
    main()
