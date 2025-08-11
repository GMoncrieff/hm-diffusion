# src/models/diffusion/ddpm.py
from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn


def make_beta_schedule(T: int, schedule: str = "cosine") -> torch.Tensor:
    if schedule == "linear":
        beta_start, beta_end = 1e-4, 2e-2
        return torch.linspace(beta_start, beta_end, T, dtype=torch.float32)
    # cosine (Nichol & Dhariwal)
    s = 0.008
    t = torch.linspace(0, T, T + 1, dtype=torch.float32) / T
    alphas_bar = torch.cos(((t + s) / (1 + s)) * 0.5 * torch.pi) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    return betas.clamp(1e-5, 0.999)


class DDPM(nn.Module):
    def __init__(self, model: nn.Module, timesteps: int = 1000, schedule: str = "cosine", parameterization: str = "eps"):
        super().__init__()
        self.model = model
        self.parameterization = parameterization
        betas = make_beta_schedule(timesteps, schedule)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_bar", alphas_bar)
        self.T = timesteps

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x_t = sqrt(a_bar)*x0 + sqrt(1-a_bar)*noise
        a_bar = self.alphas_bar[t][:, None, None, None]
        return torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise

    def predict_eps(self, x_t: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.model(x_t, cond, t)

    def p_mean_variance(self, x_t: torch.Tensor, cond: torch.Tensor, t: torch.Tensor):
        beta_t = self.betas[t][:, None, None, None]
        alpha_t = self.alphas[t][:, None, None, None]
        a_bar_t = self.alphas_bar[t][:, None, None, None]
        eps = self.predict_eps(x_t, cond, t)
        # estimate x0
        x0_hat = (x_t - torch.sqrt(1.0 - a_bar_t) * eps) / torch.sqrt(a_bar_t)
        # posterior q(x_{t-1}|x_t, x0)
        mean = (
            torch.sqrt(self.alphas_bar[t - 1][:, None, None, None].clamp(min=1e-8)) * x0_hat
            + torch.sqrt(1 - self.alphas_bar[t - 1][:, None, None, None].clamp(min=1e-8)) * eps
            if (t > 0).all()
            else x0_hat
        )
        var = beta_t
        return mean, var, x0_hat

    # ---- training loss (ε prediction) ----
    def loss(self, x0: torch.Tensor, cond: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = x0.size(0)
        device = x0.device
        t = torch.randint(0, self.T, (B,), device=device, dtype=torch.long)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        eps = self.predict_eps(x_t, cond, t)
        loss_eps = torch.mean((eps - noise) ** 2)
        return {"loss_eps": loss_eps}

    # ---- ancestral sampler ----
    @torch.no_grad()
    def sample(self, cond: torch.Tensor, steps: int = 30, tau: float = 1.0) -> torch.Tensor:
        """Return one sample x0 ~ p(x0|cond)."""
        device = cond.device
        B, C, H, W = cond.shape[0], 1, cond.shape[-2], cond.shape[-1]
        x_t = torch.randn(B, C, H, W, device=device)

        # convert steps to a sub-sequence of timesteps
        ts = torch.linspace(self.T - 1, 0, steps, dtype=torch.long, device=device)
        ts = torch.clamp(ts, 0, self.T - 1)

        for i in range(steps):
            t = ts[i].repeat(B)
            beta_t = self.betas[t][:, None, None, None]
            alpha_t = self.alphas[t][:, None, None, None]
            a_bar_t = self.alphas_bar[t][:, None, None, None]
            eps = self.predict_eps(x_t, cond, t)
            x0_hat = (x_t - torch.sqrt(1.0 - a_bar_t) * eps) / torch.sqrt(a_bar_t)
            if i < steps - 1:
                noise = tau * torch.randn_like(x_t)
            else:
                noise = torch.zeros_like(x_t)
            # DDPM update
            coef1 = (1.0 / torch.sqrt(alpha_t))
            coef2 = ((1.0 - alpha_t) / torch.sqrt(1.0 - a_bar_t))
            x_t = coef1 * (x_t - coef2 * eps) + torch.sqrt(beta_t) * noise
        return x0_hat
