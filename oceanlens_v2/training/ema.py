"""Exponential moving average callback for the FM branch."""

from __future__ import annotations

import torch
import pytorch_lightning as pl


class FlowMatchingEMACallback(pl.Callback):
    """Maintain EMA weights for FM training and use them for validation."""

    def __init__(self, decay: float) -> None:
        super().__init__()
        self.decay = float(decay)
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}

    def _enabled(self, pl_module: pl.LightningModule) -> bool:
        return getattr(pl_module, "phase", None) == "fm"

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self._enabled(pl_module):
            return
        if self.shadow:
            self.shadow = {
                name: self.shadow[name].to(device=parameter.device, dtype=parameter.dtype)
                for name, parameter in pl_module.fm.named_parameters()
                if parameter.requires_grad and name in self.shadow
            }
            return
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in pl_module.fm.named_parameters()
            if parameter.requires_grad
        }

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if not self._enabled(pl_module):
            return
        with torch.no_grad():
            for name, parameter in pl_module.fm.named_parameters():
                if parameter.requires_grad:
                    if self.shadow[name].device != parameter.device:
                        self.shadow[name] = self.shadow[name].to(device=parameter.device, dtype=parameter.dtype)
                    self.shadow[name].mul_(self.decay).add_(parameter.detach(), alpha=1.0 - self.decay)

    def _swap_to_ema(self, pl_module: pl.LightningModule) -> None:
        if self.backup or not self.shadow:
            return
        self.backup = {}
        for name, parameter in pl_module.fm.named_parameters():
            if name in self.shadow:
                self.backup[name] = parameter.detach().clone()
                parameter.data.copy_(self.shadow[name].to(device=parameter.device, dtype=parameter.dtype))

    def _restore(self, pl_module: pl.LightningModule) -> None:
        for name, parameter in pl_module.fm.named_parameters():
            if name in self.backup:
                parameter.data.copy_(self.backup[name].to(device=parameter.device, dtype=parameter.dtype))
        self.backup = {}

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._enabled(pl_module):
            self._swap_to_ema(pl_module)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._enabled(pl_module):
            self._restore(pl_module)

    def on_save_checkpoint(self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: dict) -> None:
        if self._enabled(pl_module) and self.shadow:
            checkpoint["fm_ema_state_dict"] = {name: value.detach().cpu() for name, value in self.shadow.items()}

    def on_load_checkpoint(self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: dict) -> None:
        if self._enabled(pl_module) and "fm_ema_state_dict" in checkpoint:
            self.shadow = {name: value.detach().clone() for name, value in checkpoint["fm_ema_state_dict"].items()}
