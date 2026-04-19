"""Flow Matching U-Net conditioned on the deterministic CNO field mu."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Encode t in [0, 1]."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        frequencies = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        values = t[:, None] * frequencies[None, :]
        return self.mlp(torch.cat([torch.cos(values), torch.sin(values)], dim=-1))


class TimeConditionedResBlock(nn.Module):
    """Residual block with zero-initialized FiLM time conditioning."""

    def __init__(self, channels: int, time_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.activation = nn.GELU()
        self.time_to_scale_shift = nn.Linear(time_dim, channels * 2)
        nn.init.zeros_(self.time_to_scale_shift.weight)
        nn.init.zeros_(self.time_to_scale_shift.bias)

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.activation(self.norm1(self.conv1(x)))
        scale_shift = self.time_to_scale_shift(time_embedding)[:, :, None, None]
        scale, shift = scale_shift.chunk(2, dim=1)
        x = x * (1.0 + scale) + shift
        x = self.activation(self.norm2(self.conv2(x)))
        return (x + residual) / math.sqrt(2.0)


class BottleneckAttention(nn.Module):
    """Self-attention at the bottleneck resolution."""

    def __init__(self, channels: int, n_heads: int) -> None:
        super().__init__()
        if channels % n_heads != 0:
            raise ValueError("channels must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = channels // n_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = self.head_dim**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        residual = x
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        q = q.reshape(batch, self.n_heads, self.head_dim, height * width).transpose(-2, -1)
        k = k.reshape(batch, self.n_heads, self.head_dim, height * width).transpose(-2, -1)
        v = v.reshape(batch, self.n_heads, self.head_dim, height * width).transpose(-2, -1)
        attention = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        out = torch.matmul(attention, v)
        out = out.transpose(-2, -1).reshape(batch, channels, height, width)
        return residual + self.proj(out)


class DownBlock(nn.Module):
    """Downsample with stride convolution, then apply residual blocks."""

    def __init__(self, in_channels: int, out_channels: int, time_dim: int, n_res_blocks: int) -> None:
        super().__init__()
        self.downsample = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.blocks = nn.ModuleList([TimeConditionedResBlock(out_channels, time_dim) for _ in range(n_res_blocks)])

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        for block in self.blocks:
            x = block(x, time_embedding)
        return x


class UpBlock(nn.Module):
    """Upsample with interpolation+conv, merge skip with 3x3 convolution."""

    def __init__(self, in_channels: int, out_channels: int, time_dim: int, n_res_blocks: int) -> None:
        super().__init__()
        self.upsample_conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.merge_skip = nn.Conv2d(out_channels * 2, out_channels, 3, padding=1)
        self.blocks = nn.ModuleList([TimeConditionedResBlock(out_channels, time_dim) for _ in range(n_res_blocks)])

    def forward(self, x: torch.Tensor, skip: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = self.upsample_conv(x)
        x = self.merge_skip(torch.cat([x, skip], dim=1))
        for block in self.blocks:
            x = block(x, time_embedding)
        return x


class FlowMatchingUNet(nn.Module):
    """Predict FM velocity for residual target HR - mu."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: list[int],
        time_dim: int,
        n_res_blocks: int,
        attention_heads: int,
    ) -> None:
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.input_conv = nn.Conv2d(in_channels, hidden_channels[0], 3, padding=1)

        self.encoder_residuals = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        for level in range(len(hidden_channels) - 1):
            self.encoder_residuals.append(
                nn.ModuleList([TimeConditionedResBlock(hidden_channels[level], time_dim) for _ in range(n_res_blocks)])
            )
            self.down_blocks.append(DownBlock(hidden_channels[level], hidden_channels[level + 1], time_dim, n_res_blocks))

        self.bottleneck = nn.ModuleList(
            [
                TimeConditionedResBlock(hidden_channels[-1], time_dim),
                BottleneckAttention(hidden_channels[-1], attention_heads),
                TimeConditionedResBlock(hidden_channels[-1], time_dim),
                BottleneckAttention(hidden_channels[-1], attention_heads),
            ]
        )

        self.up_blocks = nn.ModuleList()
        for level in range(len(hidden_channels) - 1, 0, -1):
            self.up_blocks.append(UpBlock(hidden_channels[level], hidden_channels[level - 1], time_dim, n_res_blocks))

        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, hidden_channels[0]),
            nn.GELU(),
            nn.Conv2d(hidden_channels[0], out_channels, 1),
        )
        nn.init.zeros_(self.output_conv[-1].weight)
        nn.init.zeros_(self.output_conv[-1].bias)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, mu_condition: torch.Tensor) -> torch.Tensor:
        time_embedding = self.time_embedding(t)
        x = self.input_conv(torch.cat([x_t, mu_condition], dim=1))
        skips = []
        for residual_blocks, down_block in zip(self.encoder_residuals, self.down_blocks):
            for block in residual_blocks:
                x = block(x, time_embedding)
            skips.append(x)
            x = down_block(x, time_embedding)
        for block in self.bottleneck:
            x = block(x, time_embedding) if isinstance(block, TimeConditionedResBlock) else block(x)
        for up_block, skip in zip(self.up_blocks, reversed(skips)):
            x = up_block(x, skip, time_embedding)
        return self.output_conv(x)

