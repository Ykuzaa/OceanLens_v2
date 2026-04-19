"""NeurIPS-style CNO branch for deterministic residual prediction."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from oceanlens_v2.models.cno_resampling import filtered_downsample_by_two, filtered_upsample_to_match


class BandLimitedActivation(nn.Module):
    """Upsample -> LeakyReLU -> filtered downsample."""

    def __init__(self, filter_name: str = "lanczos", up_factor: int = 2) -> None:
        super().__init__()
        self.filter_name = filter_name
        self.up_factor = up_factor
        self.activation = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_shape = (x.shape[-2] * self.up_factor, x.shape[-1] * self.up_factor)
        from oceanlens_v2.models.cno_resampling import filtered_resize_to_shape

        high_resolution = filtered_resize_to_shape(x, target_shape, self.filter_name)
        activated = self.activation(high_resolution)
        return filtered_resize_to_shape(activated, x.shape[-2:], self.filter_name)


class CNOResidualBlock(nn.Module):
    """Residual CNO block with stable skip scaling."""

    def __init__(self, channels: int, kernel_size: int, filter_name: str) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.act1 = BandLimitedActivation(filter_name)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.act2 = BandLimitedActivation(filter_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        return (x + residual) / math.sqrt(2.0)


class CNODownBlock(nn.Module):
    """Filtered downsampling followed by convolution and band-limited activation."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, filter_name: str) -> None:
        super().__init__()
        self.filter_name = filter_name
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.act = BandLimitedActivation(filter_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = filtered_downsample_by_two(x, self.filter_name)
        return self.act(self.conv(x))


class CNOUpBlock(nn.Module):
    """Filtered upsampling, skip merge and band-limited activation."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, kernel_size: int, filter_name: str) -> None:
        super().__init__()
        self.filter_name = filter_name
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size, padding=padding)
        self.act = BandLimitedActivation(filter_name)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = filtered_upsample_to_match(x, skip, self.filter_name)
        x = torch.cat([x, skip], dim=1)
        return self.act(self.conv(x))


class NeuripsCNO2d(nn.Module):
    """CNO that predicts the deterministic HR residual."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: list[int],
        n_res_blocks: int,
        kernel_size: int,
        filter_name: str,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.lift = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels[0], kernel_size, padding=padding),
            BandLimitedActivation(filter_name),
        )

        self.encoder_residuals = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        for level in range(len(hidden_channels) - 1):
            self.encoder_residuals.append(
                nn.Sequential(*[CNOResidualBlock(hidden_channels[level], kernel_size, filter_name) for _ in range(n_res_blocks)])
            )
            self.down_blocks.append(CNODownBlock(hidden_channels[level], hidden_channels[level + 1], kernel_size, filter_name))

        self.bottleneck = nn.Sequential(
            *[CNOResidualBlock(hidden_channels[-1], kernel_size, filter_name) for _ in range(n_res_blocks)]
        )

        self.up_blocks = nn.ModuleList()
        self.decoder_residuals = nn.ModuleList()
        for level in range(len(hidden_channels) - 1, 0, -1):
            self.up_blocks.append(
                CNOUpBlock(hidden_channels[level], hidden_channels[level - 1], hidden_channels[level - 1], kernel_size, filter_name)
            )
            self.decoder_residuals.append(
                nn.Sequential(*[CNOResidualBlock(hidden_channels[level - 1], kernel_size, filter_name) for _ in range(n_res_blocks)])
            )

        self.project = nn.Conv2d(hidden_channels[0], out_channels, kernel_size=1)

    def forward(self, lr_on_hr_grid: torch.Tensor) -> torch.Tensor:
        x = self.lift(lr_on_hr_grid)
        skips = []
        for residual_block, down_block in zip(self.encoder_residuals, self.down_blocks):
            x = residual_block(x)
            skips.append(x)
            x = down_block(x)
        x = self.bottleneck(x)
        for up_block, residual_block, skip in zip(self.up_blocks, self.decoder_residuals, reversed(skips)):
            x = up_block(x, skip)
            x = residual_block(x)
        return self.project(x)

