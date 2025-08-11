# src/utils/samplers.py
from __future__ import annotations
from typing import Dict
import torch


@torch.no_grad()
def sample_K(ddpm, cond: torch.Tensor, K: int, sampler_cfg: Dict) -> torch.Tensor:
    """Draw K samples from DDPM given conditioning.
    Returns [K,B,1,H,W]
    """
    method = sampler_cfg.get("method", "ddpm")
    steps = int(sampler_cfg.get("steps", 30))
    tau = float(sampler_cfg.get("tau", 1.0))

    batch = cond.shape[0]
    out = []
    for k in range(K):
        x0 = ddpm.sample(cond.to(ddpm.betas.device), steps=steps, tau=tau)
        out.append(x0.cpu())
    return torch.stack(out, dim=0)  # [K,B,1,H,W]
