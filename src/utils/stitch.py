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

        if write_mean: writers["mean"].write(mean, 1, window=rio.windows.Window(col, r, w, h))
        if write_std:  writers["std"].write(std, 1, window=rio.windows.Window(col, r, w, h))
        if write_p10:  writers["p10"].write(p10, 1, window=rio.windows.Window(col, r, w, h))
        if write_med:  writers["median"].write(med, 1, window=rio.windows.Window(col, r, w, h))
        if write_p90:  writers["p90"].write(p90, 1, window=rio.windows.Window(col, r, w, h))

    for w in writers.values():
        w.close()
