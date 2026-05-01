import torch
import numpy as np 

def make_bm_process_functions(sigma):
    def drift_coefficient(x_t, t):
        return torch.zeros_like(x_t)

    def diffusion_coefficient(t):
        sigma_tensor = torch.tensor(sigma, device=t.device, dtype=t.dtype)
        return sigma_tensor ** t

    def mu_t(x_0, t):
        return x_0

    def sigma_t(t):
        sigma_tensor = torch.tensor(sigma, device=t.device, dtype=t.dtype)
        return torch.sqrt(
            0.5 * (sigma_tensor ** (2 * t) - 1.0) / torch.log(sigma_tensor)
        )

    return {
        "drift_coefficient": drift_coefficient,
        "diffusion_coefficient": diffusion_coefficient,
        "mu_t": mu_t,
        "sigma_t": sigma_t,
    }

def make_bm_backward_drift(diffusion_coefficient, score_model):

    def backward_drift_coefficient(x_t, t):
        g = diffusion_coefficient(t).view(-1, 1, 1, 1)
        return -(g ** 2) * score_model(x_t, t)

    return backward_drift_coefficient

def make_bm_probability_flow_drift(diffusion_coefficient, score_model):

    def probability_flow_drift(x_t, t):
        g = diffusion_coefficient(t).view(-1, 1, 1, 1)
        return -0.5 * (g ** 2) * score_model(x_t, t)

    return probability_flow_drift