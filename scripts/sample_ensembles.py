#!/usr/bin/env python3
"""
Sample a K-member ensemble for 2020→2050 conditioned on HFI_2020 (+ static + exogenous deltas).
Writes mean, std, percentiles, and (optionally) members.

Expected imports (implemented next):
- src.data.pair_view: ConditioningRasterLoader (for 2020 tiles)
- src.models.diffusion.lightning_module: DiffusionLightningModule.load_from_checkpoint
- src.utils.stitch: sliding_window_sample_and_stitch
"""

from pathlib import Path
import argparse
from omegaconf import OmegaConf
import torch

from src.data.pair_view import ConditioningRasterLoader
from src.models.diffusion.lightning_module import DiffusionLightningModule
from src.utils.stitch import sliding_window_sample_and_stitch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    p.add_argument("--data", type=str, required=True, help="config/data/*.yaml")
    p.add_argument("--sampling", type=str, required=True, help="config/sampling/*.yaml")
    p.add_argument("--override", type=str, nargs="*", default=[])
    return p.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()

    cfg_data = OmegaConf.load(args.data)
    cfg_sampling = OmegaConf.load(args.sampling)
    cfg = OmegaConf.merge(cfg_data, cfg_sampling)
    if args.override:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.override))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = DiffusionLightningModule.load_from_checkpoint(
        args.ckpt, cfg=cfg, map_location=device
    )
    model.eval()
    model.to(device)

    # Build conditioning (2020 map + static + exogenous deltas for 2020→2050 if provided)
    cond = ConditioningRasterLoader(cfg)

    # Sliding-window sampling and stitching
    sliding_window_sample_and_stitch(
        model=model,
        cond=cond,
        tile_size=cfg.inference_tiles.size,
        stride=cfg.inference_tiles.stride,
        feather=cfg.inference_tiles.feather,
        sampler_cfg=cfg.sampler,
        ensemble_cfg=cfg.ensemble,
        output_cfg=cfg.output,
    )

    print(f"Wrote outputs to: {cfg.output.dir}")


if __name__ == "__main__":
    main()
