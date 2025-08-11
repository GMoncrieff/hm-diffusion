# src/data/pair_view.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import rasterio as rio
from rasterio.windows import Window
from affine import Affine


def _open(path: str | Path):
    return rio.open(str(path))


@dataclass
class TileSpec:
    row: int
    col: int
    height: int
    width: int


def _enumerate_tiles(height: int, width: int, size: int, stride: int) -> List[TileSpec]:
    tiles: List[TileSpec] = []
    for r in range(0, max(1, height - size + 1), stride):
        for c in range(0, max(1, width - size + 1), stride):
            tiles.append(TileSpec(r, c, size, size))
    # ensure coverage to bottom/right edges
    if (height - size) % stride != 0:
        for c in range(0, max(1, width - size + 1), stride):
            tiles.append(TileSpec(height - size, c, size, size))
    if (width - size) % stride != 0:
        for r in range(0, max(1, height - size + 1), stride):
            tiles.append(TileSpec(r, width - size, size, size))
    if (height - size) % stride != 0 and (width - size) % stride != 0:
        tiles.append(TileSpec(height - size, width - size, size, size))
    # deduplicate
    seen = set()
    uniq: List[TileSpec] = []
    for t in tiles:
        key = (t.row, t.col)
        if key not in seen:
            seen.add(key)
            uniq.append(t)
    return uniq


class _Normalizer:
    """Minimal per-config transforms. HFI is assumed to be in [0,1] already if minmax is chosen.
    Supports: minmax (with [0,1] range) and logit on HFI. Static/exogenous use tile-wise zscore by default.
    """
    def __init__(self, cfg: Dict):
        self.cfg = cfg

    # ----- HFI (targets + state) -----
    def hfi_to_model(self, x: np.ndarray) -> np.ndarray:
        conf = self.cfg["norm"]["hfi"]
        t = conf["type"]
        if t == "minmax":
            lo, hi = conf.get("range", [0.0, 1.0])
            x = (x - lo) / (hi - lo + 1e-9)
            x = np.clip(x, 0.0, 1.0)
            return x
        elif t == "logit":
            lo, hi = conf.get("range", [0.0, 1.0])
            eps = float(conf.get("logit_epsilon", 1e-4))
            x = (x - lo) / (hi - lo + 1e-9)
            x = np.clip(x, eps, 1 - eps)
            return np.log(x / (1 - x))
        else:
            return x

    def hfi_from_model(self, x: np.ndarray) -> np.ndarray:
        conf = self.cfg["norm"]["hfi"]
        t = conf["type"]
        if t == "minmax":
            lo, hi = conf.get("range", [0.0, 1.0])
            x = np.clip(x, 0.0, 1.0)
            return x * (hi - lo) + lo
        elif t == "logit":
            lo, hi = conf.get("range", [0.0, 1.0])
            x = 1.0 / (1.0 + np.exp(-x))
            return x * (hi - lo) + lo
        else:
            return x

    # ----- static/exogenous (tile-wise zscore by default) -----
    @staticmethod
    def zscore_tile(x: np.ndarray) -> np.ndarray:
        m = np.nanmean(x, dtype=np.float64)
        s = np.nanstd(x, dtype=np.float64)
        s = s if s > 1e-6 else 1.0
        return (x - m) / s


class PairedTilesDataset(Dataset):
    """Reads tiles for (X_t -> Y_{t+Δ}) pairs with stacked static and exogenous channels.
    - target x0 is Y_{t+Δ} in model space; reconstruction handles Δ if requested.
    - cond contains state at t (model-space) + normalized static/exo channels.

    Rasters must be aligned on the same grid/CRS.
    """

    def __init__(self, cfg: Dict, stage: str = "train"):
        self.cfg = cfg
        self.stage = stage
        self.paths = cfg["paths"]
        self.static_paths: List[str] = cfg["static"]["channels"]
        self.size = int(cfg["tiles"]["size"])
        self.stride = int(cfg["tiles"]["stride"])
        self.target_delta = bool(cfg["target"]["learn_delta"])

        with _open(self.paths["hfi_1990"]) as src:
            self.height, self.width = src.height, src.width
            self.transform: Affine = src.transform
            self.crs = src.crs

        self.tiles = _enumerate_tiles(self.height, self.width, self.size, self.stride)

        # geoblock split
        split_cfg = cfg.get("split", {"strategy": "geoblock"})
        if split_cfg["strategy"] == "geoblock":
            block_deg = float(split_cfg["geoblock"]["block_deg"])
            val_fraction = float(split_cfg["geoblock"]["val_fraction"])
            self.train_ids, self.val_ids = self._geoblock_split(block_deg, val_fraction)
        else:
            # fallback: first 90% train, last 10% val
            n = len(self.tiles)
            cut = int(0.9 * n)
            self.train_ids, self.val_ids = list(range(cut)), list(range(cut, n))

        self.norm = _Normalizer(cfg)

    # --- split helpers ---
    def _tile_center_ll(self, t: TileSpec) -> Tuple[float, float]:
        r_center = t.row + t.height / 2
        c_center = t.col + t.width / 2
        x, y = self.transform * (c_center, r_center)
        # assume CRS is geographic; if not, we still just do block partitioning in that space
        return (x, y)

    def _geoblock_split(self, block_deg: float, val_fraction: float) -> Tuple[List[int], List[int]]:
        # bucket tiles by integer block index
        blocks: Dict[Tuple[int, int], List[int]] = {}
        for idx, t in enumerate(self.tiles):
            lon, lat = self._tile_center_ll(t)
            bi = (int(math.floor(lon / block_deg)), int(math.floor(lat / block_deg)))
            blocks.setdefault(bi, []).append(idx)
        keys = list(blocks.keys())
        rng = np.random.default_rng(self.cfg.get("seed", 1337))
        rng.shuffle(keys)
        n_val = max(1, int(len(keys) * val_fraction))
        val_keys = set(keys[:n_val])
        train_ids, val_ids = [], []
        for k, indices in blocks.items():
            (val_ids if k in val_keys else train_ids).extend(indices)
        return train_ids, val_ids

    # --- IO ---
    def _read_stack(self, win: Window, paths: List[str]) -> np.ndarray:
        arrs = []
        for p in paths:
            with _open(p) as src:
                a = src.read(1, window=win, boundless=True, fill_value=np.nan)
                arrs.append(a)
        return np.stack(arrs, axis=0)  # [C,H,W]

    def _read_single(self, win: Window, path: str) -> np.ndarray:
        with _open(path) as src:
            return src.read(1, window=win, boundless=True, fill_value=np.nan)

    def __len__(self) -> int:
        if self.stage == "train":
            return len(self.train_ids)
        elif self.stage == "validate":
            return len(self.val_ids)
        return len(self.tiles)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.stage == "train":
            t = self.tiles[self.train_ids[idx]]
        elif self.stage == "validate":
            t = self.tiles[self.val_ids[idx]]
        else:
            t = self.tiles[idx]
        win = Window(t.col, t.row, t.width, t.height)

        x_t = self._read_single(win, self.paths["hfi_1990"])  # training pair is 1990->2020
        y_tp = self._read_single(win, self.paths["hfi_2020"])
        static = self._read_stack(win, self.static_paths) if self.static_paths else np.zeros((0, t.height, t.width), np.float32)

        # normalize to model space
        x_t_m = self.norm.hfi_to_model(x_t)
        y_tp_m = self.norm.hfi_to_model(y_tp)

        # exogenous deltas (for training backtest use 1990->2020 deltas if provided; otherwise zeros)
        exo_cfg = self.paths.get("exogenous_2020_2050", {})
        exo_train = []  # zeros by default at train time
        if exo_cfg:
            # these are for inference; training uses zeros (baseline). Can extend later.
            pass

        # static / exo normalization (tile-wise zscore for baseline)
        if static.shape[0] > 0:
            static = np.stack([_Normalizer.zscore_tile(s) for s in static], axis=0)
        exo = np.stack(exo_train, axis=0) if exo_train else np.zeros((0, t.height, t.width), np.float32)

        # target can be delta or absolute depending on config
        if self.target_delta:
            target_arr = (y_tp_m - x_t_m).astype(np.float32)[None, ...]
        else:
            target_arr = y_tp_m.astype(np.float32)[None, ...]

        cond = np.concatenate([x_t_m[None, ...], static, exo], axis=0).astype(np.float32)  # [C_cond,H,W]

        # sanitize NaNs to avoid propagating through convs; optional mask channel can be added later
        cond = np.nan_to_num(cond, nan=0.0, posinf=0.0, neginf=0.0)
        target_arr = np.nan_to_num(target_arr, nan=0.0, posinf=0.0, neginf=0.0)

        sample = {
            "cond": torch.from_numpy(cond),
            "target": torch.from_numpy(target_arr),
            "meta": {
                "row": t.row, "col": t.col, "height": t.height, "width": t.width,
            }
        }
        return sample


class PairedStepDataModule:
    """Lightning-style DataModule facade without importing Lightning.
    Scripts construct DataLoader from here.
    """

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.train_ds: Optional[PairedTilesDataset] = None
        self.val_ds: Optional[PairedTilesDataset] = None

        # count conditioning channels: 1 (HFI_t) + #static + #exo
        n_static = len(cfg["static"]["channels"])
        # exogenous deltas are optional and fed at inference; set to 0 for baseline training
        self.cond_channels = 1 + n_static

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit", "train"):
            self.train_ds = PairedTilesDataset(self.cfg, stage="train")
        if stage in (None, "fit", "validate", "validate_only"):
            self.val_ds = PairedTilesDataset(self.cfg, stage="validate")

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None
        bs = int(self.cfg["loader"]["train"]["batch_size"])
        nw = int(self.cfg["loader"]["train"]["num_workers"])
        return DataLoader(self.train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True, drop_last=self.cfg["loader"]["train"]["drop_last"])

    def val_dataloader(self) -> DataLoader:
        assert self.val_ds is not None
        bs = int(self.cfg["loader"]["val"]["batch_size"])
        nw = int(self.cfg["loader"]["val"]["num_workers"])
        return DataLoader(self.val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)


# ----- Inference-time conditioning over full raster -----

class ConditioningRasterLoader:
    """Provides sliding-window conditioning tiles for inference (2020→2050)."""
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.size = int(cfg["inference_tiles"]["size"])
        self.stride = int(cfg["inference_tiles"]["stride"])
        paths = cfg["paths"]
        self.paths = paths
        self.norm = _Normalizer(cfg)

        with _open(paths["hfi_2020"]) as src:
            self.height, self.width = src.height, src.width
            self.transform = src.transform
            self.crs = src.crs

        self.tiles = _enumerate_tiles(self.height, self.width, self.size, self.stride)
        self.static_paths: List[str] = cfg["static"]["channels"]

        # exogenous deltas 2020→2050 (optional)
        exo_cfg = paths.get("exogenous_2020_2050", {})
        self.exo_paths: List[str] = list(exo_cfg.values()) if exo_cfg else []

    def __len__(self): return len(self.tiles)

    def geo_meta(self):
        return dict(height=self.height, width=self.width, transform=self.transform, crs=self.crs)

    def tile(self, idx: int) -> Dict[str, torch.Tensor]:
        t = self.tiles[idx]
        win = Window(t.col, t.row, t.width, t.height)
        x_t = self._read_single(win, self.paths["hfi_2020"])
        static = self._read_stack(win, self.static_paths) if self.static_paths else np.zeros((0, t.height, t.width), np.float32)
        exo = self._read_stack(win, self.exo_paths) if self.exo_paths else np.zeros((0, t.height, t.width), np.float32)

        x_t_m = self.norm.hfi_to_model(x_t)
        if static.shape[0] > 0:
            static = np.stack([_Normalizer.zscore_tile(s) for s in static], axis=0)
        if exo.shape[0] > 0:
            exo = np.stack([_Normalizer.zscore_tile(s) for s in exo], axis=0)

        cond = np.concatenate([x_t_m[None, ...], static, exo], axis=0).astype(np.float32)
        cond = np.nan_to_num(cond, nan=0.0, posinf=0.0, neginf=0.0)
        return {
            "cond": torch.from_numpy(cond),
            "window": (t.row, t.col, t.height, t.width),
        }

    def _read_single(self, win: Window, path: str) -> np.ndarray:
        with _open(path) as src:
            return src.read(1, window=win, boundless=True, fill_value=np.nan)

    def _read_stack(self, win: Window, paths: List[str]) -> np.ndarray:
        arrs = []
        for p in paths:
            with _open(p) as src:
                a = src.read(1, window=win, boundless=True, fill_value=np.nan)
                arrs.append(a)
        return np.stack(arrs, axis=0)
