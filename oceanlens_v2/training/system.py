"""Lightning system for OceanLens_v2 v1."""

from __future__ import annotations

import math

import torch
import pytorch_lightning as pl

from oceanlens_v2.losses.cno import masked_residual_cno_loss
from oceanlens_v2.losses.fm import flow_matching_velocity_loss
from oceanlens_v2.metrics.currents import current_summary_metrics
from oceanlens_v2.metrics.pointwise import masked_correlation, masked_mae, masked_rmse
from oceanlens_v2.metrics.probabilistic import masked_ensemble_crps, masked_spread_skill_ratio
from oceanlens_v2.models.cno_resampling import filtered_resize_to_shape
from oceanlens_v2.models.factory import build_cno_from_config, build_fm_from_config


class OceanLensV1System(pl.LightningModule):
    """Residual CNO + FM conditioned on mu."""

    def __init__(self, cfg, phase: str) -> None:
        super().__init__()
        self.cfg = cfg
        self.phase = phase
        self.cno = build_cno_from_config(cfg)
        self.fm = build_fm_from_config(cfg)
        self.save_hyperparameters(ignore=["cfg"])

    def prepare_lr_on_hr_grid(self, lr: torch.Tensor, hr_shape: tuple[int, int]) -> torch.Tensor:
        """Project experimental LR to the HR grid using the configured CNO filter."""
        return filtered_resize_to_shape(lr, hr_shape, filter_name=str(self.cfg.model.cno.resample_filter))

    def compute_mu(self, lr: torch.Tensor, hr_shape: tuple[int, int], hr_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute deterministic field mu = LR + CNO(LR)."""
        lr_on_hr_grid = self.prepare_lr_on_hr_grid(lr, hr_shape)
        predicted_residual = self.cno(lr_on_hr_grid)
        mu = (lr_on_hr_grid + predicted_residual) * hr_mask
        return mu, lr_on_hr_grid

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        lr, hr, hr_mask = batch["lr"], batch["hr"], batch["hr_mask"]
        if self.phase == "cno":
            mu, lr_on_hr_grid = self.compute_mu(lr, hr.shape[-2:], hr_mask)
            predicted_residual = mu - lr_on_hr_grid
            loss = masked_residual_cno_loss(predicted_residual, hr, lr_on_hr_grid, hr_mask, self.cfg)
            self.log("train/cno_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
            return loss

        if self.phase == "fm":
            with torch.no_grad():
                mu, _ = self.compute_mu(lr, hr.shape[-2:], hr_mask)
            target_residual = (hr - mu) * hr_mask
            loss = flow_matching_velocity_loss(
                self.fm,
                target_residual,
                mu,
                hr_mask,
                bool(self.cfg.loss.fm.mask_ocean),
            )
            self.log("train/fm_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
            return loss

        raise ValueError(f"Unknown phase: {self.phase}")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        lr, hr, hr_mask = batch["lr"], batch["hr"], batch["hr_mask"]
        mu, lr_on_hr_grid = self.compute_mu(lr, hr.shape[-2:], hr_mask)
        cno_residual = mu - lr_on_hr_grid
        cno_loss = masked_residual_cno_loss(cno_residual, hr, lr_on_hr_grid, hr_mask, self.cfg)
        self.log("val/cno_loss", cno_loss, prog_bar=True, sync_dist=True)

        if self.phase == "fm":
            target_residual = (hr - mu) * hr_mask
            fm_loss = flow_matching_velocity_loss(self.fm, target_residual, mu, hr_mask, bool(self.cfg.loss.fm.mask_ocean))
            self.log("val/fm_loss", fm_loss, prog_bar=True, sync_dist=True)

        if batch_idx == 0:
            if self.phase == "cno":
                prediction = mu
                ensemble = None
            else:
                validation_members = int(getattr(self.cfg.inference, "validation_ensemble_members", 1))
                ensemble = self.sample_ensemble(lr, hr_mask, n_steps=int(self.cfg.inference.n_steps), ensemble_members=validation_members)
                prediction = ensemble.mean(dim=0)
            self.log("val/sample_mae", masked_mae(prediction, hr, hr_mask), sync_dist=True)
            self.log("val/sample_rmse", masked_rmse(prediction, hr, hr_mask), sync_dist=True)
            self.log("val/sample_corr", masked_correlation(prediction, hr, hr_mask), sync_dist=True)
            if ensemble is not None and ensemble.shape[0] > 1:
                self.log("val/sample_crps", masked_ensemble_crps(ensemble, hr, hr_mask), sync_dist=True)
                self.log("val/spread_skill_ratio", masked_spread_skill_ratio(ensemble, hr, hr_mask), sync_dist=True)
            variables = list(self.cfg.data.variables)
            current_metrics = current_summary_metrics(
                prediction,
                hr,
                hr_mask,
                variables.index("uo"),
                variables.index("vo"),
            )
            for name, value in current_metrics.items():
                self.log(f"val/{name}", value, sync_dist=True)

    @torch.no_grad()
    def sample(self, lr: torch.Tensor, hr_mask: torch.Tensor, n_steps: int, ensemble_members: int) -> torch.Tensor:
        """Generate HR prediction as ensemble mean over FM residual samples."""
        return self.sample_ensemble(lr, hr_mask, n_steps=n_steps, ensemble_members=ensemble_members).mean(dim=0)

    @torch.no_grad()
    def sample_ensemble(self, lr: torch.Tensor, hr_mask: torch.Tensor, n_steps: int, ensemble_members: int) -> torch.Tensor:
        """Generate HR predictions for each FM ensemble member."""
        hr_shape = hr_mask.shape[-2:]
        mu, _ = self.compute_mu(lr, hr_shape, hr_mask)
        predictions = []
        for _ in range(int(ensemble_members)):
            residual = self.integrate_fm_residual(mu, hr_mask, n_steps=n_steps)
            predictions.append((mu + residual) * hr_mask)
        return torch.stack(predictions, dim=0)

    @torch.no_grad()
    def integrate_fm_residual(self, mu: torch.Tensor, ocean_mask: torch.Tensor, n_steps: int) -> torch.Tensor:
        """Integrate the FM velocity field from Gaussian noise to residual."""
        x = torch.randn_like(mu) * ocean_mask
        dt = 1.0 / float(n_steps)
        solver = str(self.cfg.inference.solver)
        for step in range(n_steps):
            t = torch.full((x.shape[0],), step * dt, device=x.device, dtype=x.dtype)
            if solver == "midpoint":
                v1 = self.fm(x, t, mu)
                t_mid = torch.clamp(t + 0.5 * dt, max=1.0)
                v2 = self.fm(x + 0.5 * dt * v1, t_mid, mu)
                x = x + dt * v2
            elif solver == "heun":
                v1 = self.fm(x, t, mu)
                t_next = torch.clamp(t + dt, max=1.0)
                x_euler = (x + dt * v1) * ocean_mask
                v2 = self.fm(x_euler, t_next, mu)
                x = x + 0.5 * dt * (v1 + v2)
            else:
                x = x + dt * self.fm(x, t, mu)
            x = x * ocean_mask
        return x

    def configure_optimizers(self):
        training_cfg = self.cfg.training[self.phase]
        parameters = self.cno.parameters() if self.phase == "cno" else self.fm.parameters()
        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(training_cfg.learning_rate),
            weight_decay=float(training_cfg.weight_decay),
        )
        if str(getattr(training_cfg, "scheduler", "none")) != "cosine":
            return optimizer

        warmup_steps = int(getattr(training_cfg, "warmup_steps", 0))
        total_steps = max(warmup_steps + 1, int(getattr(self.trainer, "estimated_stepping_batches", 0) or 0))

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            progress = min(1.0, max(0.0, progress))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def load_cno_weights(self, checkpoint_path: str) -> None:
        state = torch.load(checkpoint_path, map_location="cpu")
        state_dict = state.get("state_dict", state)
        cno_state = {key.replace("cno.", "", 1): value for key, value in state_dict.items() if key.startswith("cno.")}
        missing, unexpected = self.cno.load_state_dict(cno_state, strict=False)
        if missing or unexpected:
            print(f"CNO load: missing={missing}, unexpected={unexpected}")

    def load_fm_weights(self, checkpoint_path: str) -> None:
        state = torch.load(checkpoint_path, map_location="cpu")
        if "fm_ema_state_dict" in state:
            fm_state = state["fm_ema_state_dict"]
        else:
            state_dict = state.get("state_dict", state)
            fm_state = {key.replace("fm.", "", 1): value for key, value in state_dict.items() if key.startswith("fm.")}
        missing, unexpected = self.fm.load_state_dict(fm_state, strict=False)
        if missing or unexpected:
            print(f"FM load: missing={missing}, unexpected={unexpected}")
