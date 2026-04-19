"""Model factory functions."""

from __future__ import annotations

from oceanlens_v2.models.cno_neurips import NeuripsCNO2d
from oceanlens_v2.models.fm_unet import FlowMatchingUNet


def build_cno_from_config(cfg) -> NeuripsCNO2d:
    cno_cfg = cfg.model.cno
    return NeuripsCNO2d(
        in_channels=int(cno_cfg.in_channels),
        out_channels=int(cno_cfg.out_channels),
        hidden_channels=list(cno_cfg.hidden_channels),
        n_res_blocks=int(cno_cfg.n_res_blocks),
        kernel_size=int(cno_cfg.kernel_size),
        filter_name=str(cno_cfg.resample_filter),
    )


def build_fm_from_config(cfg) -> FlowMatchingUNet:
    fm_cfg = cfg.model.fm
    return FlowMatchingUNet(
        in_channels=int(fm_cfg.in_channels),
        out_channels=int(fm_cfg.out_channels),
        hidden_channels=list(fm_cfg.hidden_channels),
        time_dim=int(fm_cfg.time_dim),
        n_res_blocks=int(fm_cfg.n_res_blocks),
        attention_heads=int(fm_cfg.attention_heads),
    )

