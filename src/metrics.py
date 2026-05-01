# -*- coding: utf-8 -*-
"""
Utilities for evaluating diffusion model samples.

This module provides small reusable helpers for image quality metrics used in
the project notebooks. The full metric pipelines are intentionally executed in
the notebooks inside measures/.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def to_01_range(images: Tensor, assume_minus1_1: bool = True) -> Tensor:
    """Map images to [0, 1] and clamp numerical outliers.

    Args:
        images: Tensor of shape (B, C, H, W).
        assume_minus1_1: If True, convert from [-1, 1] to [0, 1].
    """
    x = images.float()
    if assume_minus1_1:
        x = (x + 1.0) / 2.0
    return x.clamp(0.0, 1.0)


def inception_uint8(images_01: Tensor) -> Tensor:
    """Prepare a batch for Inception-based metrics as uint8 tensors.

    TorchMetrics' FID/IS can consume uint8 tensors in [0, 255].
    """
    x = images_01
    if x.ndim != 4:
        raise ValueError("Expected images with shape (B, C, H, W).")

    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    elif x.shape[1] != 3:
        raise ValueError("Expected images with 1 or 3 channels.")

    return (x * 255.0).round().clamp(0, 255).to(torch.uint8)


def load_saved_samples(sample_file: str, sampler_key: str) -> Tensor:
    """Load saved final samples from a .pt file."""
    payload = torch.load(sample_file, map_location="cpu")

    if sampler_key not in payload:
        raise KeyError(
            f"Sampler key '{sampler_key}' not found in file. "
            f"Available keys: {list(payload.keys())}"
        )

    return payload[sampler_key].float()


def gaussian_log_prob(x: Tensor, std: float | Tensor = 1.0) -> Tensor:
    """Compute log probability under N(0, std^2 I), per sample."""
    if not torch.is_tensor(std):
        std = torch.tensor(std, device=x.device, dtype=x.dtype)
    else:
        std = std.to(device=x.device, dtype=x.dtype)

    dim = x[0].numel()
    x_flat = x.reshape(x.shape[0], -1)

    return -0.5 * (
        torch.sum((x_flat / std) ** 2, dim=1)
        + dim * torch.log(torch.tensor(2.0 * math.pi, device=x.device, dtype=x.dtype))
        + 2.0 * dim * torch.log(std)
    )


def hutchinson_divergence(drift_function, x: Tensor, t: Tensor) -> Tensor:
    """Estimate divergence of drift_function(x, t) with Hutchinson estimator."""
    x = x.detach().requires_grad_(True)
    epsilon = torch.randn_like(x)

    drift = drift_function(x, t)
    inner = torch.sum(drift * epsilon)

    grad = torch.autograd.grad(
        inner,
        x,
        create_graph=False,
        retain_graph=False,
    )[0]

    divergence = torch.sum(grad * epsilon, dim=(1, 2, 3))
    return divergence.detach()


def bpd_probability_flow_ode(
    x_0: Tensor,
    probability_flow_drift,
    terminal_std: float | Tensor = 1.0,
    t_0: float = 1.0e-3,
    t_end: float = 1.0,
    n_steps: int = 100,
) -> tuple[Tensor, Tensor]:
    """Compute log-likelihood and BPD using the probability flow ODE.

    The ODE is integrated forward from data x_0 to the terminal distribution.
    The divergence term is estimated with the Hutchinson estimator.

    Args:
        x_0: Real images with shape (B, C, H, W), normally in [-1, 1].
        probability_flow_drift: Drift function of the probability flow ODE.
        terminal_std: Standard deviation of the terminal Gaussian distribution.
            For VP/OU this is usually 1.0. For VE/BM use sigma_t(T).
        t_0: Initial integration time.
        t_end: Final integration time.
        n_steps: Number of Euler integration steps.

    Returns:
        log_p_0: Estimated log-likelihood per image.
        bpd: Estimated bits per dimension per image.
    """
    device = x_0.device
    dtype = x_0.dtype
    batch_size = x_0.shape[0]

    times = torch.linspace(t_0, t_end, n_steps + 1, device=device, dtype=dtype)
    dt = times[1] - times[0]

    x = x_0.clone()
    accumulated_divergence = torch.zeros(batch_size, device=device, dtype=dtype)

    for t_scalar in times[:-1]:
        t = torch.ones(batch_size, device=device, dtype=dtype) * t_scalar

        divergence = hutchinson_divergence(probability_flow_drift, x, t)
        accumulated_divergence += divergence * dt

        with torch.no_grad():
            x = x + probability_flow_drift(x, t) * dt

    log_p_T = gaussian_log_prob(x, std=terminal_std)
    log_p_0 = log_p_T + accumulated_divergence

    dim = x_0[0].numel()
    bpd = -log_p_0 / (dim * math.log(2.0))

    return log_p_0, bpd


def mean_bpd_probability_flow_ode(
    x_0: Tensor,
    probability_flow_drift,
    terminal_std: float | Tensor = 1.0,
    t_0: float = 1.0e-3,
    t_end: float = 1.0,
    n_steps: int = 100,
) -> float:
    """Compute the average BPD over a batch of real images."""
    _, bpd = bpd_probability_flow_ode(
        x_0=x_0,
        probability_flow_drift=probability_flow_drift,
        terminal_std=terminal_std,
        t_0=t_0,
        t_end=t_end,
        n_steps=n_steps,
    )
    return bpd.mean().item()