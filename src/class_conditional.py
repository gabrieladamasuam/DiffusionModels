# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np

from torch import Tensor
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import Compose, Resize, ToTensor

from src.ou_schedules import make_ou_process_functions
from src.diffusion_process import predictor_corrector_sampler

COLOR_PALETTE = {
    0: ("red", torch.tensor([1.0, 0.0, 0.0])),
    1: ("green", torch.tensor([0.0, 1.0, 0.0])),
    2: ("blue", torch.tensor([0.0, 0.0, 1.0])),
    3: ("yellow", torch.tensor([1.0, 1.0, 0.0])),
    4: ("cyan", torch.tensor([0.0, 1.0, 1.0])),
    5: ("magenta", torch.tensor([1.0, 0.0, 1.0])),
    6: ("orange", torch.tensor([1.0, 0.5, 0.0])),
    7: ("purple", torch.tensor([0.5, 0.0, 1.0])),
}

COLOR_NAME_TO_ID = {
    name: idx for idx, (name, _) in COLOR_PALETTE.items()
}


class ConditionalColorMNISTDataset(Dataset):
    """MNIST dataset with digit and color labels.

    Each image is converted to RGB and colored using one color sampled from a
    fixed palette. The model receives both the digit label and the color label.
    """

    def __init__(
        self,
        root: str = "./data",
        train: bool = True,
        image_size: int = 32,
        download: bool = True,
    ) -> None:
        self.transform = Compose([
            Resize((image_size, image_size)),
            ToTensor(),
        ])

        self.dataset = datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=self.transform,
        )

        self.n_colors = len(COLOR_PALETTE)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        image_gray, digit_label = self.dataset[index]

        color_id = torch.randint(0, self.n_colors, size=(1,)).item()
        _, color = COLOR_PALETTE[color_id]

        color = color.view(3, 1, 1)
        image_rgb = image_gray.repeat(3, 1, 1) * color

        # Normalize from [0, 1] to [-1, 1].
        image_rgb = image_rgb * 2.0 - 1.0

        return (
            image_rgb.float(),
            torch.tensor(digit_label, dtype=torch.long),
            torch.tensor(color_id, dtype=torch.long),
        )


class GaussianRandomFourierFeatures(nn.Module):
    """Gaussian random Fourier features for encoding time."""

    def __init__(self, embed_dim: int, scale: float = 30.0) -> None:
        super().__init__()
        self.rff_weights = nn.Parameter(
            torch.randn(embed_dim // 2) * scale,
            requires_grad=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        x_proj = x[:, None] * self.rff_weights[None, :] * 2.0 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """Fully connected layer reshaped as a feature-map bias."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.dense(x)[..., None, None]


class ConditionalScoreNet(nn.Module):
    """U-Net score network conditioned on time, digit and color."""

    def __init__(
        self,
        marginal_prob_std,
        in_channels: int = 3,
        out_channels: int = 3,
        channels: tuple[int, int, int, int] = (32, 64, 128, 256),
        embed_dim: int = 256,
        n_digits: int = 10,
        n_colors: int = 8,
    ) -> None:
        super().__init__()

        self.time_embed = nn.Sequential(
            GaussianRandomFourierFeatures(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

        self.digit_embed = nn.Embedding(n_digits, embed_dim)
        self.color_embed = nn.Embedding(n_colors, embed_dim)

        self.condition_mlp = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.conv1 = nn.Conv2d(in_channels, channels[0], 3, stride=1, padding=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, channels[0])

        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, padding=1, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, channels[1])

        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, padding=1, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, channels[2])

        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, padding=1, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, channels[3])

        self.tconv4 = nn.ConvTranspose2d(
            channels[3],
            channels[2],
            3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False,
        )
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, channels[2])

        self.tconv3 = nn.ConvTranspose2d(
            channels[2] + channels[2],
            channels[1],
            3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False,
        )
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, channels[1])

        self.tconv2 = nn.ConvTranspose2d(
            channels[1] + channels[1],
            channels[0],
            3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False,
        )
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, channels[0])

        self.tconv1 = nn.ConvTranspose2d(
            channels[0] + channels[0],
            out_channels,
            3,
            stride=1,
            padding=1,
        )

        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        digit_labels: Tensor,
        color_labels: Tensor,
    ) -> Tensor:
        time_emb = self.act(self.time_embed(t))
        digit_emb = self.digit_embed(digit_labels)
        color_emb = self.color_embed(color_labels)

        embed = torch.cat([time_emb, digit_emb, color_emb], dim=1)
        embed = self.condition_mlp(embed)
        embed = self.act(embed)

        h1 = self.conv1(x)
        h1 += self.dense1(embed)
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)

        h2 = self.conv2(h1)
        h2 += self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)

        h3 = self.conv3(h2)
        h3 += self.dense3(embed)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)

        h4 = self.conv4(h3)
        h4 += self.dense4(embed)
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)

        h = self.tconv4(h4)
        h += self.dense5(embed)
        h = self.tgnorm4(h)
        h = self.act(h)

        h = self.tconv3(torch.cat([h, h3], dim=1))
        h += self.dense6(embed)
        h = self.tgnorm3(h)
        h = self.act(h)

        h = self.tconv2(torch.cat([h, h2], dim=1))
        h += self.dense7(embed)
        h = self.tgnorm2(h)
        h = self.act(h)

        h = self.tconv1(torch.cat([h, h1], dim=1))

        h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h


def conditional_loss_function(
    diffusion_process,
    score_model: nn.Module,
    x_0: Tensor,
    digit_labels: Tensor,
    color_labels: Tensor,
    eps: float = 1.0e-5,
) -> Tensor:
    """Conditional denoising score matching loss."""

    batch_size = x_0.shape[0]

    t = torch.rand(
        batch_size,
        device=x_0.device,
        dtype=x_0.dtype,
    ) * (1.0 - eps) + eps

    z = torch.randn_like(x_0)

    mu_t = diffusion_process.mu_t(x_0, t)
    sigma_t = diffusion_process.sigma_t(t)
    sigma_t_view = sigma_t.view(-1, 1, 1, 1)

    x_t = mu_t + sigma_t_view * z

    score = score_model(
        x_t,
        t,
        digit_labels,
        color_labels,
    )

    per_sample_sq_error = torch.sum(
        (sigma_t_view * score + z) ** 2,
        dim=(1, 2, 3),
    )

    return torch.mean(per_sample_sq_error)


def generate_conditional_images(
    digit: int,
    color_name: str,
    checkpoint_path: str,
    n_images: int = 9,
    n_steps: int = 1000,
    n_corrector_steps: int = 1,
    snr: float = 0.16,
    device: torch.device | None = None,
) -> Tensor:
    """Generate colored MNIST digits using OU cosine + predictor-corrector."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if digit < 0 or digit > 9:
        raise ValueError("digit must be between 0 and 9.")

    color_name = color_name.lower().strip()

    if color_name not in COLOR_NAME_TO_ID:
        raise ValueError(
            f"Unknown color '{color_name}'. "
            f"Available colors: {list(COLOR_NAME_TO_ID.keys())}"
        )

    color_id = COLOR_NAME_TO_ID[color_name]

    sched = make_ou_process_functions("cosine")

    score_model = ConditionalScoreNet(
        marginal_prob_std=sched["sigma_t"],
        in_channels=3,
        out_channels=3,
        n_digits=10,
        n_colors=len(COLOR_NAME_TO_ID),
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    score_model.load_state_dict(state_dict)
    score_model.eval()

    digit_labels = torch.full(
        (n_images,),
        digit,
        dtype=torch.long,
        device=device,
    )

    color_labels = torch.full(
        (n_images,),
        color_id,
        dtype=torch.long,
        device=device,
    )

    def conditional_score(x_t: Tensor, t: Tensor) -> Tensor:
        return score_model(x_t, t, digit_labels, color_labels)

    beta_t = sched["beta_t"]

    def reverse_drift(x_t: Tensor, t: Tensor) -> Tensor:
        beta = beta_t(t).view(-1, 1, 1, 1)
        return -0.5 * beta * x_t - beta * conditional_score(x_t, t)

    x_T = torch.randn(n_images, 3, 32, 32, device=device)

    with torch.no_grad():
        _, x_t = predictor_corrector_sampler(
            x_0=x_T,
            t_0=1.0,
            t_end=1.0e-3,
            n_steps=n_steps,
            drift_coefficient=reverse_drift,
            diffusion_coefficient=sched["diffusion_coefficient"],
            score_model=conditional_score,
            n_corrector_steps=n_corrector_steps,
            snr=snr,
        )

    return x_t[..., -1]