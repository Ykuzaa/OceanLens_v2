#!/usr/bin/env python
"""Train the CNO or FM phase."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from oceanlens_v2.data.datamodule import OceanLensDataModule
from oceanlens_v2.training.ema import FlowMatchingEMACallback
from oceanlens_v2.training.system import OceanLensV1System
from oceanlens_v2.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v1.yaml")
    parser.add_argument("--phase", choices=["cno", "fm"], required=True)
    parser.add_argument("--cno_ckpt", default=None)
    parser.add_argument("--fm_ckpt", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    pl.seed_everything(int(cfg.experiment.seed), workers=True)

    datamodule = OceanLensDataModule(cfg, phase=args.phase)
    system = OceanLensV1System(cfg, phase=args.phase)
    if args.cno_ckpt:
        system.load_cno_weights(args.cno_ckpt)
    if args.fm_ckpt:
        system.load_fm_weights(args.fm_ckpt)

    run_dir = Path(cfg.paths.run_dir) / args.phase
    checkpoint = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename=f"{cfg.experiment.name}-{args.phase}-{{epoch:03d}}",
        save_last=True,
        save_top_k=3,
        monitor=f"val/{args.phase}_loss",
        mode="min",
    )
    training_cfg = cfg.training[args.phase]
    callbacks = [checkpoint, LearningRateMonitor(logging_interval="epoch")]
    ema_cfg = getattr(training_cfg, "ema", None)
    if args.phase == "fm" and ema_cfg is not None and bool(getattr(ema_cfg, "enabled", False)):
        callbacks.append(FlowMatchingEMACallback(decay=float(ema_cfg.decay)))
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[args.gpu],
        max_epochs=int(training_cfg.max_epochs),
        accumulate_grad_batches=int(training_cfg.accumulate_grad_batches),
        gradient_clip_val=float(training_cfg.gradient_clip_val),
        default_root_dir=str(run_dir),
        callbacks=callbacks,
        log_every_n_steps=20,
    )
    trainer.fit(system, datamodule=datamodule)


if __name__ == "__main__":
    main()
