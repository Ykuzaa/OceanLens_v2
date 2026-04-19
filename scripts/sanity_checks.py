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
    print("Mean:", dataset["mean"].values)
    print("Std:", dataset["std"].values)


if __name__ == "__main__":
    main()

