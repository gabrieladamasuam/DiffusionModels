# -*- coding: utf-8 -*-
"""
Used for plotting image grids and their evolution during the diffusion process.

@author: <gabriela.damas@estudiante.uam.es> and <eva.blazquez@estudiante.uam.es>
"""

from numpy.typing import ArrayLike

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from matplotlib.colors import Colormap

import torch
from torchvision.utils import make_grid
from torchvision.transforms import functional


def plot_image_grid( 
    images: torch.Tensor, 
    figsize: tuple,
    n_rows: int,
    n_cols: int,
    padding: int = 2,
    pad_value: int = 1.0,
    cmap: Colormap | None = "gray",
    normalized: bool = False,
    axis_on_off: str = "off",
):
    # Convert list of tensors to a single tensor
    if isinstance(images, list):
        images = torch.stack(images)

    # Detach and move to CPU for plotting
    images = images.detach().cpu()

    # If images are in the range [-1, 1], convert them to [0, 1] for visualization
    if not normalized:
        images = images * 0.5 + 0.5
        images = images.clamp(0.0, 1.0)

    grid = make_grid(
        images, 
        nrow=n_cols, 
        padding=padding, 
        normalize=normalized,
        pad_value=pad_value,
    )

    # Convert to PIL Image and display
    fig, ax = plt.subplots(figsize=figsize)

    if grid.shape[0] == 1:
        ax.imshow(grid[0], cmap="gray")
    elif grid.shape[0] == 3:
        ax.imshow(grid.permute(1, 2, 0))
    else:
        raise ValueError("Unsupported number of channels.")

    ax.axis(axis_on_off)
    return fig, ax


def plot_image_evolution(
    images: torch.Tensor,
    n_images: int,
    n_intermediate_steps: ArrayLike,
    figsize: tuple,
    cmap: Colormap | None = "gray",
):
    fig, axs = plt.subplots(
        n_images, 
        len(n_intermediate_steps), 
        figsize=figsize,
    )

    images_cpu = images.detach().cpu()

    if n_images == 1:
        axs = np.expand_dims(axs, axis=0)

    for n_image in range(n_images):
        for i, ax in enumerate(axs[n_image, :]):
            img = images_cpu[n_image, :, :, :, n_intermediate_steps[i]]

            # If images are in the range [-1, 1], convert them to [0, 1] for visualization
            img = img * 0.5 + 0.5
            img = img.clamp(0.0, 1.0)

            if img.shape[0] == 1:
                ax.imshow(img[0], cmap="gray")
            elif img.shape[0] == 3:
                ax.imshow(img.permute(1, 2, 0))
            else:
                raise ValueError("Unsupported number of channels.")

            ax.set_axis_off()

    return fig, axs


def animation_images(
        images_t, 
        interval,
        figsize,
    ): 
    images_t = images_t.detach().cpu()

    # Create a figure and axes.  
    fig, ax = plt.subplots(figsize=figsize)

    first_frame = images_t[:, :, :, 0]

    # If images are in the range [-1, 1], convert them to [0, 1] for visualization
    first_frame = first_frame * 0.5 + 0.5
    first_frame = first_frame.clamp(0.0, 1.0)

    if first_frame.shape[0] == 1:
        img_display = ax.imshow(first_frame[0], cmap="gray")
    elif first_frame.shape[0] == 3:
        img_display = ax.imshow(first_frame.permute(1, 2, 0))
    else:
        raise ValueError("Unsupported number of channels.")

    def update(t):
        """Update function for the animation."""
        frame = images_t[:, :, :, t]

        # If images are in the range [-1, 1], convert them to [0, 1] for visualization
        frame = frame * 0.5 + 0.5
        frame = frame.clamp(0.0, 1.0)

        if frame.shape[0] == 1:
            img_display.set_array(frame[0])
        elif frame.shape[0] == 3:
            img_display.set_array(frame.permute(1, 2, 0))
        else:
            raise ValueError("Unsupported number of channels.")

        return [img_display]

    return ( 
        fig, 
        ax, 
        animation.FuncAnimation(
            fig, 
            update, 
            frames=images_t.shape[-1], 
            interval=interval, 
            blit=False)
    )