# -*- coding: utf-8 -*-
"""
Simulate Gaussian processes.

@author: <gabriela.damas@estudiante.uam.es> and <eva.blazquez@estudiante.uam.es>
"""

from __future__ import annotations

from typing import Callable, Union

import torch
from torch import Tensor



def euler_maruyama_integrator(
    x_0: Tensor,
    t_0: float,
    t_end: float,
    n_steps: int,
    drift_coefficient: Callable[[Tensor, Tensor], Tensor],
    diffusion_coefficient: Callable[[Tensor], Tensor],
    seed: Union[int, None] = None,
) -> tuple[Tensor, Tensor]:
    """Euler-Maruyama integrator (approximate)

     Args:
        x_0: The initial images of dimensions 
            (batch_size, n_channels, image_height, image_width)
        t_0: float,
        t_end: endpoint of the integration interval    
        n_steps: number of integration steps 
        drift_coefficient: Function of :math`(x(t), t)` that defines the drift term            
        diffusion_coefficient: Function of :math`(t)` that defines the diffusion term  
        seed: Seed for the random number generator
        
    Returns:
        times:
            Tensor with the integration time grid of shape (n_steps + 1,).
        x_t:
            Trajectories that result from the integration of the SDE.
            The shape is (*x_0.shape, n_steps + 1).
            
    Notes:
        The implementation is fully vectorized except for a loop over time.

    Examples:
        >>> drift_coefficient = lambda x_t, t: - x_t
        >>> diffusion_coefficient = lambda t: torch.ones_like(t)
        >>> x_0 = torch.arange(120, dtype=torch.float32).reshape(2, 3, 5, 4)
        >>> t_0, t_end = 0.0, 3.0
        >>> n_steps = 6
        >>> times, x_t = euler_maruyama_integrator(
        ...     x_0, t_0, t_end, n_steps, drift_coefficient, diffusion_coefficient, 
        ... )
        >>> print(times)
        tensor([0.0000, 0.5000, 1.0000, 1.5000, 2.0000, 2.5000, 3.0000])
        >>> print(x_t.shape)
        torch.Size([2, 3, 5, 4, 7])
    """
    # Select device (CPU or GPU) from the input tensor
    device = x_0.device

    # Set random seed for reproducibility (if provided).
    if seed is not None:
        torch.manual_seed(seed)

    # Create a uniform time grid between t_0 and t_end
    times = torch.linspace(t_0, t_end, n_steps + 1, device=device, dtype=x_0.dtype)

    # Compute time step size
    dt = times[1] - times[0]

    # Initialize tensor to store the trajectory at all time steps
    x_t = torch.empty(*x_0.shape, len(times), dtype=x_0.dtype, device=device)

    # Set initial condition x(t_0) = x_0
    x_t[..., 0] = x_0

    # Sample Gaussian noise for all time steps
    z = torch.randn_like(x_t)
    z[..., -1] = 0.0 # No noise injection in the last step

    batch_size = x_0.shape[0]

    # Iterate over time steps using Euler-Maruyama scheme
    for n, t_scalar in enumerate(times[:-1]):

        # Create a batch vector of current time t
        t = torch.ones(batch_size, device=device, dtype=x_0.dtype) * t_scalar

        # Euler-Maruyama update:
        # x_{t+dt} = x_t + drift * dt + diffusion * sqrt(dt) * noise
        x_t[..., n + 1] = (
            x_t[..., n]
            + drift_coefficient(x_t[..., n], t) * dt
            + diffusion_coefficient(t).view(-1, 1, 1, 1)
            * torch.sqrt(torch.abs(dt))
            * z[..., n]
        )

    return times, x_t


def euler_ode_integrator(
    x_0: Tensor,
    t_0: float,
    t_end: float,
    n_steps: int,
    drift_coefficient: Callable[[Tensor, Tensor], Tensor],
) -> tuple[Tensor, Tensor]:
    """Euler integrator for ordinary differential equations."""
    device = x_0.device

    times = torch.linspace(t_0, t_end, n_steps + 1, device=device, dtype=x_0.dtype)
    dt = times[1] - times[0]

    x_t = torch.empty(*x_0.shape, len(times), dtype=x_0.dtype, device=device)
    x_t[..., 0] = x_0

    batch_size = x_0.shape[0]

    for n, t_scalar in enumerate(times[:-1]):
        t = torch.ones(batch_size, device=device, dtype=x_0.dtype) * t_scalar
        x_t[..., n + 1] = x_t[..., n] + drift_coefficient(x_t[..., n], t) * dt

    return times, x_t


def predictor_corrector_sampler(
    x_0: Tensor,
    t_0: float,
    t_end: float,
    n_steps: int,
    drift_coefficient: Callable[[Tensor, Tensor], Tensor],
    diffusion_coefficient: Callable[[Tensor], Tensor],
    score_model,
    n_corrector_steps: int = 1,
    snr: float = 0.16,
    seed: Union[int, None] = None,
) -> tuple[Tensor, Tensor]:
    """
    Predictor-corrector sampler for reverse-time generation.

    Args:
        x_0: initial samples at time t_0, shape (B, C, H, W)
        t_0: initial time
        t_end: final time
        n_steps: number of predictor time steps
        drift_coefficient: reverse drift f(x,t)
        diffusion_coefficient: diffusion coefficient g(t)
        score_model: trained score network
        n_corrector_steps: number of Langevin correction steps per time level
        snr: signal-to-noise ratio controlling corrector step size
        seed: optional random seed

    Returns:
        times: tensor of shape (n_steps + 1,)
        x_t: trajectory tensor of shape (*x_0.shape, n_steps + 1)
    """
    device = x_0.device
    dtype = x_0.dtype

    if seed is not None:
        torch.manual_seed(seed)

    times = torch.linspace(t_0, t_end, n_steps + 1, device=device, dtype=dtype)
    dt = times[1] - times[0]

    x_t = torch.empty(*x_0.shape, len(times), device=device, dtype=dtype)
    x_t[..., 0] = x_0

    batch_size = x_0.shape[0]

    for n, t_scalar in enumerate(times[:-1]):
        x = x_t[..., n]
        t = torch.ones(batch_size, device=device, dtype=dtype) * t_scalar

        # ---------- Corrector: Langevin steps ----------
        for _ in range(n_corrector_steps):
            grad = score_model(x, t)
            noise = torch.randn_like(x)

            grad_norm = torch.norm(grad.reshape(batch_size, -1), dim=1).mean()
            noise_norm = torch.norm(noise.reshape(batch_size, -1), dim=1).mean()

            step_size = 2.0 * (snr * noise_norm / (grad_norm + 1.0e-12)) ** 2

            x = x + step_size * grad + torch.sqrt(2.0 * step_size) * noise

        # ---------- Predictor: Euler-Maruyama reverse SDE ----------
        z = torch.randn_like(x)
        g = diffusion_coefficient(t).view(-1, 1, 1, 1)

        x = (
            x
            + drift_coefficient(x, t) * dt
            + g * torch.sqrt(torch.abs(dt)) * z
        )

        x_t[..., n + 1] = x

    return times, x_t


class DiffusionProcess:

    def __init__(
        self,
        drift_coefficient: Callable[[Tensor, Tensor], Tensor] = lambda x_t, t: torch.zeros_like(x_t),
        diffusion_coefficient: Callable[[Tensor], Tensor] = lambda t: torch.ones_like(t),
    ) -> None:
        self.drift_coefficient = drift_coefficient
        self.diffusion_coefficient = diffusion_coefficient
     

        
class GaussianDiffusionProcess(DiffusionProcess):
    """
    Gaussian diffusion process.

    This class represents a stochastic diffusion process whose marginal
    distribution at time t, conditioned on the initial data x_0, is Gaussian:

        x_t | x_0 ~ N(mu_t(x_0, t), sigma_t(t)^2 I)

    where:
        - mu_t(x_0, t) is the mean of the perturbed data at time t
        - sigma_t(t) is the standard deviation of the perturbation kernel

    The process is defined through:
        - a drift coefficient f(x, t)
        - a diffusion coefficient g(t)
        - closed-form expressions for mu_t and sigma_t

    This formulation is commonly used in score-based generative models,
    where data is progressively perturbed with Gaussian noise and a neural
    network is trained to estimate the score function ∇_x log p_t(x).

    The class also provides a loss function for training a score-based model
    using denoising score matching.

    Args:
        drift_coefficient: Function f(x, t) defining the drift term of the SDE.
        diffusion_coefficient: Function g(t) defining the diffusion term.
        mu_t: Function that computes the marginal mean at time t given x_0.
        sigma_t: Function that computes the marginal standard deviation at time t.
    
    Example 1:
        >>> mu, sigma = 1.5, 2.0
        >>> bm = GaussianDiffusionProcess(
        ...     drift_coefficient=lambda x_t, t: mu * torch.ones_like(x_t),
        ...     diffusion_coefficient=lambda t: sigma * torch.ones_like(t),
        ...     mu_t=lambda x_0, t: x_0 + mu * t.view(-1, 1, 1, 1),
        ...     sigma_t=lambda t: torch.sqrt(2.0 * t),
        ... )
        >>> x = torch.tensor([[[[3.0]]]])
        >>> t = torch.tensor([10.0])
        >>> print(bm.drift_coefficient(x, t))
        tensor([[[[1.5000]]]])

    """
    kind = "Gaussian"
    
    def __init__(
        self,
        drift_coefficient: Callable[[Tensor, Tensor], Tensor] = lambda x_t, t: torch.zeros_like(x_t),
        diffusion_coefficient: Callable[[Tensor], Tensor] = lambda t: torch.ones_like(t),
        mu_t: Callable[[Tensor, Tensor], Tensor] = lambda x_0, t: x_0,
        sigma_t: Callable[[Tensor], Tensor] = lambda t: torch.sqrt(t),
    ) -> None:
        super().__init__(drift_coefficient, diffusion_coefficient)
        self.mu_t = mu_t
        self.sigma_t = sigma_t
    

    def loss_function(
        self,
        score_model,
        x_0: Tensor,
        eps: float = 1.0e-5,
    ) -> Tensor:
        """The loss function for training score-based generative models.

          Args:
              score_model:  A PyTorch model instance that represents a 
                            time-dependent score-based model.
          x_0: A mini-batch of training data.    
          eps: A tolerance value for numerical stability.
        """
        batch_size = x_0.shape[0]

        # Sample one diffusion time per training example: t_i ~ U[eps, 1].
        t = torch.rand(batch_size, device=x_0.device, dtype=x_0.dtype) * (1.0 - eps) + eps

        # Sample Gaussian noise z_i ~ N(0, I).
        z = torch.randn_like(x_0)

        # Compute the conditional mean and standard deviation.
        mu_t = self.mu_t(x_0, t)
        sigma_t = self.sigma_t(t)
        sigma_t_view = sigma_t.view(-1, 1, 1, 1)

        # Generate perturbed samples x_t = mu_t + sigma_t * z.
        x_t = mu_t + sigma_t_view * z

        # Evaluate the score model.
        score = score_model(x_t, t)

        # Denoising score matching loss.
        per_sample_sq_error = torch.sum(
            (sigma_t_view * score + z) ** 2,
            dim=(1, 2, 3),
        )

        loss = torch.mean(per_sample_sq_error)
        return loss


if __name__ == "__main__":
    import doctest
    doctest.testmod()