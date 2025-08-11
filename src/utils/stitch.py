# src/utils/stitch.py
from __future__ import annotations
from typing import Dict, Tuple
from pathlib import Path

import numpy as np
import torch
import rasterio as rio
from rasterio.transform import Affine
from tqdm import tqdm


def _percentiles(x: np.ndarray, qs=(10, 50, 90)) -> Tuple[np.ndarray, ...]:
    return tuple(np.percentile(x, q, axis=0).astype(np.float32) for q in qs)


def _open_writer(path: Path, height: int, width: int, transform: Affine, crs, dtype="float32", compress="deflate"):
    path.parent.mkdir(parents=True, exist_ok=True)
    return rio.open(
        str(path),
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
        tiled=True,
        compress=compress,
        BIGTIFF="IF_SAFER",
    )


@torch.inference_mode()
def sliding_window_sample_and_stitch(
    model,
    cond,
    tile_size: int,
    stride: int,
    feather: str,
    sampler_cfg: Dict,
    ensemble_cfg: Dict,
    output_cfg: Dict,
):
    # note: for baseline we avoid complicated feathering; recommend stride == size.
    meta = cond.geo_meta()
    H, W = meta["height"], meta["width"]
    transform, crs = meta["transform"], meta["crs"]

    out_dir = Path(output_cfg["dir"])
    fmt = output_cfg.get("format", "gtiff")
    compress = output_cfg.get("compress", "deflate")

    write_mean = bool(output_cfg["write"].get("mean", True))
    write_std = bool(output_cfg["write"].get("std", True))
    write_p10 = bool(output_cfg["write"].get("p10", True))
    write_med = bool(output_cfg["write"].get("median", True))
    write_p90 = bool(output_cfg["write"].get("p90", True))

    writers = {}
    if fmt != "gtiff":
        raise NotImplementedError("Only GeoTIFF is implemented in baseline.")

    if write_mean:
        writers["mean"] = _open_writer(out_dir / "mean.tif", H, W, transform, crs, compress=compress)
    if write_std:
        writers["std"] = _open_writer(out_dir / "std.tif", H, W, transform, crs, compress=compress)
    if write_p10:
        writers["p10"] = _open_writer(out_dir / "p10.tif", H, W, transform, crs, compress=compress)
    if write_med:
        writers["median"] = _open_writer(out_dir / "median.tif", H, W, transform, crs, compress=compress)
    if write_p90:
        writers["p90"] = _open_writer(out_dir / "p90.tif", H, W, transform, crs, compress=compress)

    K = int(ensemble_cfg.get("K", 32))
    device = next(model.parameters()).device

    # baseline: recommend stride == tile for clean writing (no feather).
    assert stride == tile_size or feather == "none", "Baseline stitcher expects stride==size or feather=none."

    # access normalizer to invert model-space to data-space
    norm = getattr(cond, "norm", None)
    norm_cfg = getattr(cond, "cfg", {}).get("norm", {}) if hasattr(cond, "cfg") else {}
    hfi_norm = norm_cfg.get("hfi", {}) if isinstance(norm_cfg, dict) else {}
    hfi_type = hfi_norm.get("type", "minmax")
    hfi_range = hfi_norm.get("range", [0.0, 1.0])
    learn_delta = False
    if hasattr(cond, "cfg") and isinstance(cond.cfg, dict):
        learn_delta = bool(cond.cfg.get("target", {}).get("learn_delta", False))

    for i in tqdm(range(len(cond)), desc="Sampling tiles"):
        tile = cond.tile(i)
        c = tile["cond"].unsqueeze(0).to(device)  # [1,C,H,W]
        preds = model.sample(cond=c, K=K, sampler_cfg=sampler_cfg)  # [K,1,1,h,w] (Lightning wraps ddpm)
        preds = preds.squeeze(1).cpu().numpy()  # [K,1,h,w] -> [K,h,w] if squeeze
        preds = preds[:, 0, :, :] if preds.ndim == 4 else preds  # ensure [K,h,w]

        r, col, h, w = tile["window"]
        mean = preds.mean(axis=0).astype(np.float32)
        std = preds.std(axis=0).astype(np.float32)
        p10, med, p90 = _percentiles(preds, qs=(10, 50, 90))

        # if learning delta, add back x_t (2020) in model space before inverse-transform
        if learn_delta:
            x_t_m = c[0, 0].detach().cpu().numpy()
            mean = mean + x_t_m
            p10 = p10 + x_t_m
            med = med + x_t_m
            p90 = p90 + x_t_m

        # inverse-normalize from model-space to data-space
        if norm is not None:
            mean = norm.hfi_from_model(mean)
            p10 = norm.hfi_from_model(p10)
            med = norm.hfi_from_model(med)
            p90 = norm.hfi_from_model(p90)
            # std: for minmax only, linear scale; for logit leave as model-space (nonlinear)
            if hfi_type == "minmax":
                lo, hi = (hfi_range[0], hfi_range[1]) if isinstance(hfi_range, (list, tuple)) else (0.0, 1.0)
                std = std * float(hi - lo)

        if write_mean: writers["mean"].write(mean, 1, window=rio.windows.Window(col, r, w, h))
        if write_std:  writers["std"].write(std, 1, window=rio.windows.Window(col, r, w, h))
        if write_p10:  writers["p10"].write(p10, 1, window=rio.windows.Window(col, r, w, h))
        if write_med:  writers["median"].write(med, 1, window=rio.windows.Window(col, r, w, h))
        if write_p90:  writers["p90"].write(p90, 1, window=rio.windows.Window(col, r, w, h))

    for w in writers.values():
        w.close()
