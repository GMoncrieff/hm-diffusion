#!/usr/bin/env python3
"""
Backtest on 1990→2020 pairs using a small ensemble (K=4 by default).
Logs pixel metrics and probabilistic scores to W&B; emits quick diagnostic rasters.

Expected imports (implemented next):
- src.data.pair_view: PairedStepDataModule (val split only)
- src.models.diffusion.lightning_module: DiffusionLightningModule
- src.utils.metrics: mae, rmse, crps_per_pixel, energy_score, variogram_score, radial_spectrum
- src.utils.wandb_log: helpers for panels
"""

from pathlib import Path
import argparse
from omegaconf import OmegaConf
import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger

from src.data.pair_view import PairedStepDataModule
from src.models.diffusion.lightning_module import DiffusionLightningModule
from src.utils.metrics import (
    compute_basic_metrics, compute_prob_scores, compute_spectrum_summary
)
from src.utils.wandb_log import log_backtest_panels


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--train", type=str, required=True)
    p.add_argument("--override", type=str, nargs="*", default=[])
    return p.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()

    cfg_data = OmegaConf.load(args.data)
    cfg_model = OmegaConf.load(args.model)
    cfg_train = OmegaConf.load(args.train)
    cfg = OmegaConf.merge(cfg_data, cfg_model, cfg_train)
    if args.override:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.override))

    seed_everything(cfg.get("repro", {}).get("seed", 1337), workers=True)

    wandb_cfg = cfg.logging.wandb
    wandb_logger = WandbLogger(
        project=wandb_cfg.project,
        entity=wandb_cfg.get("entity", None),
        name=f"{wandb_cfg.run_name}_backtest",
        mode=wandb_cfg.get("mode", "online"),
        save_code=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DiffusionLightningModule.load_from_checkpoint(
        args.ckpt, cfg=cfg, map_location=device
    ).eval().to(device)

    # val-only datamodule emitting (X_1990, static, exo_1990_2020, Y_2020)
    dm = PairedStepDataModule(cfg)
    dm.setup(stage="validate")

    # Iterate validation tiles; sample K members per tile; accumulate metrics
    K = cfg.evaluation.ensemble_K_val
    sampler_cfg = cfg.evaluation.sampler

    metrics_accum = []
    prob_accum = []
    spec_accum = []

    for batch in dm.val_dataloader():
        # batch: dict with 'cond' (tensor), 'target' (tensor), 'meta' (geo info)
        preds = model.sample(cond=batch["cond"], K=K, sampler_cfg=sampler_cfg)  # [K,B,C,H,W]
        target = batch["target"]  # [B,C,H,W]

        # mean reconstruction (if learning Δ, adapter/model handles restoring HFI_t + Δ)
        mean_pred = preds.mean(dim=0)

        metrics_accum.append(compute_basic_metrics(mean_pred, target))
        prob_accum.append(compute_prob_scores(preds, target))
        spec_accum.append(compute_spectrum_summary(mean_pred, target))

        # optional: log a small panel for first few batches
        log_backtest_panels(wandb_logger, preds, target, batch.get("meta", None))

    # Aggregate and log
    # (Implementations will stack dicts and log scalar means)
    print("Backtest complete; see W&B for metrics and panels.")


if __name__ == "__main__":
    main()
