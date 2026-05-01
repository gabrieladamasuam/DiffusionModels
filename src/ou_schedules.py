# -*- coding: utf-8 -*-
from __future__ import annotations

from functools import partial
from typing import Callable, Dict

import torch
from torch import Tensor


def _ensure_1d_t(t: Tensor) -> Tensor:
    if t.ndim == 0:
        return t.view(1)
    return t


def _clamp_unit_interval(t: Tensor) -> Tensor:
    return torch.clamp(t, 0.0, 1.0)


def linear_beta(t: Tensor, beta_min: float = 0.1, beta_max: float = 20.0) -> Tensor:
    t = _clamp_unit_interval(_ensure_1d_t(t))
    return beta_min + (beta_max - beta_min) * t


def linear_int_beta(t: Tensor, beta_min: float = 0.1, beta_max: float = 20.0) -> Tensor:
    t = _clamp_unit_interval(_ensure_1d_t(t))
    return beta_min * t + 0.5 * (beta_max - beta_min) * t**2


def cosine_beta(t: Tensor, s: float = 0.008, beta_cap: float = 50.0) -> Tensor:
    t = _clamp_unit_interval(_ensure_1d_t(t))
    theta = (t + s) / (1.0 + s) * torch.pi / 2.0
    beta = (torch.pi / (1.0 + s)) * torch.tan(theta)
    return torch.clamp(beta, min=1.0e-5, max=beta_cap)


def cosine_int_beta(t: Tensor, s: float = 0.008) -> Tensor:
    t = _clamp_unit_interval(_ensure_1d_t(t))
    theta_t = (t + s) / (1.0 + s) * torch.pi / 2.0
    theta_0 = torch.tensor(s / (1.0 + s) * torch.pi / 2.0, device=t.device, dtype=t.dtype)

    alpha_bar_t = (torch.cos(theta_t) ** 2) / (torch.cos(theta_0) ** 2)
    alpha_bar_t = torch.clamp(alpha_bar_t, min=1.0e-12)
    return -torch.log(alpha_bar_t)


def sigmoid_beta(
    t: Tensor,
    beta_min: float = 0.1,
    beta_max: float = 20.0,
    k: float = 10.0,
) -> Tensor:
    t = _clamp_unit_interval(_ensure_1d_t(t))
    return beta_min + (beta_max - beta_min) * torch.sigmoid(k * (t - 0.5))


def sigmoid_int_beta(
    t: Tensor,
    beta_min: float = 0.1,
    beta_max: float = 20.0,
    k: float = 10.0,
) -> Tensor:
    t = _clamp_unit_interval(_ensure_1d_t(t))
    c = beta_max - beta_min
    term_t = torch.log1p(torch.exp(k * (t - 0.5))) / k
    term_0 = torch.log1p(torch.exp(torch.tensor(-0.5 * k, device=t.device, dtype=t.dtype))) / k
    return beta_min * t + c * (term_t - term_0)


def make_schedule(
    name: str,
    **kwargs,
) -> tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
    schedule = name.lower().strip()

    if schedule == "linear":
        return partial(linear_beta, **kwargs), partial(linear_int_beta, **kwargs)
    if schedule == "cosine":
        return partial(cosine_beta, **kwargs), partial(cosine_int_beta, **kwargs)
    if schedule == "sigmoid":
        return partial(sigmoid_beta, **kwargs), partial(sigmoid_int_beta, **kwargs)

    raise ValueError(f"Unknown schedule '{name}'. Use one of: linear, cosine, sigmoid.")


def make_ou_process_functions(
    schedule_name: str,
    **schedule_kwargs,
) -> dict[str, Callable]:
    beta_t, int_beta_t = make_schedule(schedule_name, **schedule_kwargs)

    def drift_coefficient(x_t: Tensor, t: Tensor) -> Tensor:
        beta = beta_t(t).view(-1, 1, 1, 1)
        return -0.5 * beta * x_t

    def diffusion_coefficient(t: Tensor) -> Tensor:
        beta = beta_t(t)
        return torch.sqrt(torch.clamp(beta, min=1.0e-8))

    def mu_t(x_0: Tensor, t: Tensor) -> Tensor:
        integral = int_beta_t(t).view(-1, 1, 1, 1)
        return x_0 * torch.exp(-0.5 * integral)

    def sigma_t(t: Tensor) -> Tensor:
        integral = int_beta_t(t)
        return torch.sqrt(torch.clamp(1.0 - torch.exp(-integral), min=1.0e-8))

    return {
        "beta_t": beta_t,
        "int_beta_t": int_beta_t,
        "drift_coefficient": drift_coefficient,
        "diffusion_coefficient": diffusion_coefficient,
        "mu_t": mu_t,
        "sigma_t": sigma_t,
    }


def build_schedule_map() -> Dict[str, dict]:
    return {
        "linear": make_ou_process_functions(
            "linear",
            beta_min=0.1,
            beta_max=20.0,
        ),
        "cosine": make_ou_process_functions(
            "cosine",
            s=0.008,
            beta_cap=50.0,
        ),
        "sigmoid": make_ou_process_functions(
            "sigmoid",
            beta_min=0.1,
            beta_max=20.0,
            k=10.0,
        ),
    }


def make_reverse_drift_coefficient(
    schedule_dict: dict,
    score_model,
) -> Callable:
    beta_t = schedule_dict["beta_t"]

    def backward_drift_coefficient(x_t: Tensor, t: Tensor) -> Tensor:
        beta = beta_t(t).view(-1, 1, 1, 1)
        return -0.5 * beta * x_t - beta * score_model(x_t, t)

    return backward_drift_coefficient


def make_ou_probability_flow_drift(
    schedule_dict: dict,
    score_model,
) -> Callable:
    beta_t = schedule_dict["beta_t"]

    def probability_flow_drift(x_t: Tensor, t: Tensor) -> Tensor:
        beta = beta_t(t).view(-1, 1, 1, 1)
        return -0.5 * beta * x_t - 0.5 * beta * score_model(x_t, t)

    return probability_flow_drift