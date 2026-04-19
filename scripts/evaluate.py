#!/usr/bin/env python
"""Evaluate saved inference NPZ files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from oceanlens_v2.losses.gradient import temperature_log_gradient_loss
from oceanlens_v2.metrics.currents import current_summary_metrics
from oceanlens_v2.metrics.pointwise import masked_correlation, masked_mae, masked_rmse


VARIABLES = ["thetao", "so", "zos", "uo", "vo"]


def tensor_from_npz(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array).float()


def evaluate_file(path: Path) -> list[dict[str, float | str]]:
    data = np.load(path)
    rows = []
    target = tensor_from_npz(data["hr"])
    mask = tensor_from_npz(data["mask"])
    for name in ["lr", "mu", "pred"]:
        predicted = tensor_from_npz(data[name])
        rows.append(
            {
                "sample": path.stem,
                "variant": name,
                "mae": float(masked_mae(predicted, target, mask)),
                "rmse": float(masked_rmse(predicted, target, mask)),
                "corr": float(masked_correlation(predicted, target, mask)),
                "loggrad_thetao": float(temperature_log_gradient_loss(predicted, target, mask, VARIABLES.index("thetao"), 1.0e-6)),
            }
        )
        current_metrics = current_summary_metrics(predicted, target, mask, VARIABLES.index("uo"), VARIABLES.index("vo"))
        rows[-1].update({key: float(value) for key, value in current_metrics.items()})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", required=True)
    args = parser.parse_args()
    result_dir = Path(args.result_dir)
    rows = []
    for path in sorted(result_dir.glob("sample_*.npz")):
        rows.extend(evaluate_file(path))

    output_path = result_dir / "metrics.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

