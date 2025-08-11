# src/utils/wandb_log.py
from __future__ import annotations
from typing import Optional
import torch

def log_backtest_panels(wandb_logger, preds: torch.Tensor, target: torch.Tensor, meta: Optional[dict] = None):
    """Minimal stub to avoid tight coupling with W&B APIs.
    You can extend this to log images/tables; baseline keeps it light.
    """
    if not hasattr(wandb_logger, "experiment"):
        return
    try:
        mean = preds.mean(dim=0)
        std = preds.std(dim=0)
        wandb_logger.experiment.log({
            "val/preview/mean_min": float(mean.min()),
            "val/preview/mean_max": float(mean.max()),
            "val/preview/std_mean": float(std.mean()),
        })
    except Exception:
        pass
