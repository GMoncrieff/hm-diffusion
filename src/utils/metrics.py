# src/utils/metrics.py
from __future__ import annotations
from typing import Dict
import torch
import torch.nn.functional as F


def mae(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(y_pred - y_true))


def rmse(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2))


def crps_ensemble(y_samps: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """CRPS per pixel, sample-based; returns scalar mean over all dims.
    y_samps: [K,B,1,H,W], y_true: [B,1,H,W]
    """
    K = y_samps.size(0)
    B = y_samps.size(1)
    yt = y_true.unsqueeze(0).expand_as(y_samps)  # [K,B,1,H,W]
    term1 = torch.mean(torch.abs(y_samps - yt))
    # E|X - X'|
    diffs = y_samps.unsqueeze(0) - y_samps.unsqueeze(1)  # [K,K,B,1,H,W]
    term2 = torch.mean(torch.abs(diffs))
    return term1 - 0.5 * term2


def compute_basic_metrics(y_mean: torch.Tensor, y_true: torch.Tensor) -> Dict[str, float]:
    return {"mae": float(mae(y_mean, y_true)), "rmse": float(rmse(y_mean, y_true))}


def compute_prob_scores(y_samps: torch.Tensor, y_true: torch.Tensor) -> Dict[str, float]:
    return {"crps": float(crps_ensemble(y_samps, y_true))}


def radial_spectrum(x: torch.Tensor) -> torch.Tensor:
    """Return isotropic power spectrum, radially binned; x: [B,1,H,W], out: [R] normalized."""
    # single-batch approximation
    x = x.mean(dim=0)  # [1,H,W]
    X = torch.fft.rfft2(x, norm="ortho")
    P = (X.real ** 2 + X.imag ** 2).squeeze(0)  # [H, W//2+1]
    H, W = P.shape
    ky = torch.fft.fftfreq(H, d=1.0, device=P.device)[:, None].abs()
    kx = torch.fft.rfftfreq((W - 1) * 2, d=1.0, device=P.device)[None, :].abs()
    kr = torch.sqrt(ky ** 2 + kx ** 2)
    R = H // 2
    bins = torch.clamp((kr * R).long(), 0, R - 1)
    out = torch.zeros(R, device=P.device)
    counts = torch.zeros(R, device=P.device)
    out.index_add_(0, bins.reshape(-1), P.reshape(-1))
    counts.index_add_(0, bins.reshape(-1), torch.ones_like(P).reshape(-1))
    out = out / (counts + 1e-6)
    out = out / (out.sum() + 1e-6)
    return out


def compute_spectrum_summary(y_mean: torch.Tensor, y_true: torch.Tensor) -> Dict[str, float]:
    sp_p = radial_spectrum(y_mean)
    sp_t = radial_spectrum(y_true)
    # simple L2 distance between normalized spectra
    return {"spec_l2": float(torch.mean((sp_p - sp_t) ** 2))}
