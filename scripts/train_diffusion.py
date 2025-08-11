#!/usr/bin/env python3
"""
Train the baseline conditional DDPM for 30-year HFI change.
Keeps your existing data pipeline intact by using a thin pair-view adapter.

Expected imports (implemented in next step):
- src.data.pair_view: PairedStepDataModule
- src.models.diffusion.lightning_module: DiffusionLightningModule
"""

from pathlib import Path
import argparse
from omegaconf import OmegaConf
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

# ---- local imports (to be implemented next) ----
from src.data.pair_view import PairedStepDataModule
from src.models.diffusion.lightning_module import DiffusionLightningModule


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="config/data/*.yaml")
    p.add_argument("--model", type=str, required=True, help="config/model/*.yaml")
    p.add_argument("--train", type=str, required=True, help="config/training/*.yaml")
    p.add_argument("--override", type=str, nargs="*", default=[],
                   help="Inline overrides like key1.sub=val key2=val")
    return p.parse_args()


def main():
    args = parse_args()

    cfg_data = OmegaConf.load(args.data)
    cfg_model = OmegaConf.load(args.model)
    cfg_train = OmegaConf.load(args.train)
    cfg = OmegaConf.merge(cfg_data, cfg_model, cfg_train)
    if args.override:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.override))

    seed_everything(cfg.get("repro", {}).get("seed", 1337), workers=True)

    # --- loggers & callbacks ---
    wandb_cfg = cfg.logging.wandb
    wandb_logger = WandbLogger(
        project=wandb_cfg.project,
        entity=wandb_cfg.get("entity", None),
        name=wandb_cfg.run_name,
        mode=wandb_cfg.get("mode", "online"),
        save_code=wandb_cfg.get("save_code", True),
    )

    ckpt_dir = Path(cfg.checkpoints.dirpath)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        monitor=cfg.checkpoints.monitor,
        mode=cfg.checkpoints.mode,
        save_top_k=cfg.checkpoints.save_top_k,
        every_n_train_steps=cfg.checkpoints.every_n_train_steps,
        # keep filename simple to avoid metric name formatting issues
        filename="ddpm-{step:06d}",
        auto_insert_metric_name=False,
    )
    lr_cb = LearningRateMonitor(logging_interval="step")

    trainer = Trainer(
        max_steps=cfg.trainer.max_steps,
        precision=cfg.trainer.precision,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        val_check_interval=cfg.trainer.val_check_interval,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        deterministic=cfg.trainer.deterministic,
        logger=wandb_logger,
        callbacks=[ckpt_cb, lr_cb],
    )

    # --- data & model ---
    dm = PairedStepDataModule(cfg)
    # ensure splits are prepared so cond_channels is correct
    dm.setup("fit")
    # derive cond_channels (1 + #static [+ #exo if later added])
    if "arch" in cfg and hasattr(cfg["arch"], "cond_channels"):
        cfg.arch.cond_channels = int(dm.cond_channels)
    elif "arch" in cfg:
        # dict-style
        cfg["arch"]["cond_channels"] = int(dm.cond_channels)
    model = DiffusionLightningModule(cfg)

    # --- train ---
    trainer.fit(model, datamodule=dm)

    # --- end-of-fit panel handled inside LightningModule's on_validation_end ---
    print("Finished training.")


if __name__ == "__main__":
    main()
