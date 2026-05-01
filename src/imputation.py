# -*- coding: utf-8 -*-

import torch
from torch import Tensor

from src.score_model import ScoreNet
from src.ou_schedules import make_ou_process_functions


def create_center_mask(image_shape, box_size=12, device=None):
    """Create a binary mask: 1 = known pixels, 0 = missing pixels."""
    _, _, height, width = image_shape

    mask = torch.ones(image_shape, device=device)

    h0 = height // 2 - box_size // 2
    h1 = h0 + box_size
    w0 = width // 2 - box_size // 2
    w1 = w0 + box_size

    mask[:, :, h0:h1, w0:w1] = 0.0
    return mask


def impute_image_ou(
    image: Tensor,
    mask: Tensor,
    checkpoint_path: str,
    n_steps: int = 1000,
    n_corrector_steps: int = 1,
    snr: float = 0.16,
    device=None,
):
    """Impute missing pixels using a trained grayscale OU-cosine model."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image = image.to(device)
    mask = mask.to(device)

    sched = make_ou_process_functions("cosine")

    score_model = ScoreNet(
        marginal_prob_std=sched["sigma_t"],
        in_channels=1,
        out_channels=1,
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    score_model.load_state_dict(state_dict)
    score_model.eval()

    beta_t = sched["beta_t"]
    diffusion_coefficient = sched["diffusion_coefficient"]

    batch_size = image.shape[0]
    dtype = image.dtype

    times = torch.linspace(1.0, 1.0e-3, n_steps + 1, device=device, dtype=dtype)
    dt = times[1] - times[0]

    x = torch.randn_like(image)

    with torch.no_grad():
        for t_scalar in times[:-1]:
            t = torch.ones(batch_size, device=device, dtype=dtype) * t_scalar

            known_noisy = (
                sched["mu_t"](image, t)
                + sched["sigma_t"](t).view(-1, 1, 1, 1) * torch.randn_like(image)
            )
            x = x * (1.0 - mask) + known_noisy * mask

            for _ in range(n_corrector_steps):
                grad = score_model(x, t)
                noise = torch.randn_like(x)

                grad_norm = torch.norm(grad.reshape(batch_size, -1), dim=1).mean()
                noise_norm = torch.norm(noise.reshape(batch_size, -1), dim=1).mean()

                step_size = 2.0 * (snr * noise_norm / (grad_norm + 1.0e-12)) ** 2

                x = x + step_size * grad + torch.sqrt(2.0 * step_size) * noise

                known_noisy = (
                    sched["mu_t"](image, t)
                    + sched["sigma_t"](t).view(-1, 1, 1, 1) * torch.randn_like(image)
                )
                x = x * (1.0 - mask) + known_noisy * mask

            beta = beta_t(t).view(-1, 1, 1, 1)
            drift = -0.5 * beta * x - beta * score_model(x, t)

            g = diffusion_coefficient(t).view(-1, 1, 1, 1)
            z = torch.randn_like(x)

            x = x + drift * dt + g * torch.sqrt(torch.abs(dt)) * z

            known_noisy = (
                sched["mu_t"](image, t)
                + sched["sigma_t"](t).view(-1, 1, 1, 1) * torch.randn_like(image)
            )
            x = x * (1.0 - mask) + known_noisy * mask

        imputed = x * (1.0 - mask) + image * mask

    return imputed