#!/usr/bin/env python
"""Build the processed Zarr store used by OceanLens_v2."""

from __future__ import annotations

import argparse

from oceanlens_v2.data.preprocess import build_processed_store
from oceanlens_v2.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v1.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    build_processed_store(cfg)


if __name__ == "__main__":
    main()

