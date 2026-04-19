"""Filtered resampling blocks used by the CNO branch."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def lanczos_kernel_1d(scale_factor: int, radius: int = 3, device=None, dtype=None) -> torch.Tensor:
    """Build a normalized 1D Lanczos interpolation kernel."""
    if scale_factor < 1:
        raise ValueError("scale_factor must be >= 1")
    half_width = radius * scale_factor
    x = torch.arange(-half_width, half_width + 1, device=device, dtype=dtype) / float(scale_factor)
    kernel = torch.sinc(x) * torch.sinc(x / float(radius))
    kernel = torch.where(x.abs() < radius, kernel, torch.zeros_like(kernel))
    return kernel / kernel.sum().clamp(min=1.0e-12)


def depthwise_separable_filter_2d(x: torch.Tensor, kernel_1d: torch.Tensor) -> torch.Tensor:
    """Apply a separable depthwise filter with reflect padding."""
    channels = x.shape[1]
    kernel_y = kernel_1d.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    kernel_x = kernel_1d.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    pad = kernel_1d.numel() // 2
    x = F.pad(x, (0, 0, pad, pad), mode="reflect")
    x = F.conv2d(x, kernel_y, groups=channels)
    x = F.pad(x, (pad, pad, 0, 0), mode="reflect")
    return F.conv2d(x, kernel_x, groups=channels)


def filtered_resize_to_shape(x: torch.Tensor, target_shape: tuple[int, int], filter_name: str = "lanczos") -> torch.Tensor:
    """Resize tensors with an explicit low-pass filtering path when possible."""
    height, width = x.shape[-2:]
    target_height, target_width = target_shape
    if (height, width) == (target_height, target_width):
        return x

    integer_upscale = target_height % height == 0 and target_width % width == 0
    same_factor = integer_upscale and target_height // height == target_width // width
    if filter_name == "lanczos" and same_factor:
        scale = target_height // height
        upsampled = torch.zeros(
            x.shape[0],
            x.shape[1],
            target_height,
            target_width,
            device=x.device,
            dtype=x.dtype,
        )
        upsampled[:, :, ::scale, ::scale] = x * float(scale * scale)
        kernel = lanczos_kernel_1d(scale, device=x.device, dtype=x.dtype)
        return depthwise_separable_filter_2d(upsampled, kernel)

    return F.interpolate(x, size=target_shape, mode="bicubic", align_corners=False, antialias=True)


def filtered_downsample_by_two(x: torch.Tensor, filter_name: str = "lanczos") -> torch.Tensor:
    """Low-pass filter then downsample by a factor of two."""
    if filter_name == "lanczos":
        kernel = lanczos_kernel_1d(2, device=x.device, dtype=x.dtype)
        x = depthwise_separable_filter_2d(x, kernel)
    return F.interpolate(x, scale_factor=0.5, mode="bicubic", align_corners=False, antialias=True)


def filtered_upsample_to_match(x: torch.Tensor, reference: torch.Tensor, filter_name: str = "lanczos") -> torch.Tensor:
    """Upsample `x` to the spatial shape of `reference`."""
    return filtered_resize_to_shape(x, reference.shape[-2:], filter_name=filter_name)

