# Diffusion Baseline for 30-Year Gridded Forecasts

A minimal, extensible **conditional diffusion** baseline to predict **30-year changes** in a global, 1 km gridded **Human Footprint Index (HFI)** map, with **ensembles** out of the box.

* **Train:** learn a single-step jump (e.g., **1990 → 2020**).
* **Forecast:** condition on **2020** (+ static covariates, optional exogenous deltas) to sample **2050** ensembles.
* **Keep it simple:** plain 2-D U-Net + DDPM ε-loss, Lightning plumbing, W\&B logging, raster IO with Rasterio.
* **Future-proof:** hooks for ensemble tuning, calibration, and location encodings.

---

## Project structure

```
config/
  data/hfi_1km_pairs.yaml         # paths, tiling, normalization, splits
  model/ddpm_unet.yaml            # U-Net + diffusion hyperparams
  training/baseline.yaml          # trainer, logging, eval settings
  sampling/baseline.yaml          # ensemble sampling config
scripts/
  train_diffusion.py              # train 1990→2020 baseline
  sample_ensembles.py             # sample K-member 2020→2050
  eval_backtest.py                # backtest metrics on 1990→2020
src/
  data/pair_view.py               # thin paired-tiles adapter + inference loader
  models/diffusion/unet2d.py      # conditional U-Net (FiLM conditioning)
  models/diffusion/ddpm.py        # DDPM core (cosine schedule)
  models/diffusion/lightning_module.py
  utils/metrics.py                # MAE/RMSE, CRPS, simple spectra
  utils/samplers.py               # K-sample helper (DDPM steps/τ)
  utils/stitch.py                 # sliding-window sampling + GeoTIFF writing
  utils/wandb_log.py              # minimal W&B panels
requirements.txt
```

---

## Data expectations

* **Aligned rasters on the same grid/CRS**:

  * `hfi_1990.tif`, `hfi_2020.tif` — single-band, scaled to `[0, 1]` (or raw; see normalization below).
  * **Static covariates**: any number of single-band rasters (elevation, slope, biome one-hots, etc.).
  * **Optional exogenous deltas 2020→2050**: population, climate deltas, etc. (can start as zeros).
* **Normalization** (configurable):

  * HFI: `minmax` (default) or `logit`.
  * Static & exogenous: per-tile z-score (baseline).

Update paths in `config/data/hfi_1km_pairs.yaml`.

---

## Quickstart

### 1) Create env & install

```bash
# (Recommended) create a fresh Conda env with CUDA-matched PyTorch preinstalled
# then:
pip install -r requirements.txt

# login to Weights & Biases (optional)
wandb login
```

### 2) Train (1990→2020)

```bash
python scripts/train_diffusion.py \
  --data config/data/hfi_1km_pairs.yaml \
  --model config/model/ddpm_unet.yaml \
  --train config/training/baseline.yaml
```

### 3) Sample ensembles (2020→2050)

```bash
python scripts/sample_ensembles.py \
  --ckpt models/checkpoints/ddpm-<best>.ckpt \
  --data config/data/hfi_1km_pairs.yaml \
  --sampling config/sampling/baseline.yaml
```

Outputs (GeoTIFFs) land in `models/forecasts/2050_baseline/`:
`mean.tif`, `std.tif`, `p10.tif`, `median.tif`, `p90.tif`.

### 4) Backtest (optional)

```bash
python scripts/eval_backtest.py \
  --ckpt models/checkpoints/ddpm-<best>.ckpt \
  --data config/data/hfi_1km_pairs.yaml \
  --model config/model/ddpm_unet.yaml \
  --train config/training/baseline.yaml
```

---

## Configuration highlights

* **Pairs & tiling**: `config/data/hfi_1km_pairs.yaml`

  * `tiles.size` (default **128**) and `tiles.stride` (default **64**).
  * `split.strategy: geoblock` (10° blocks) to reduce leakage.
  * `target.learn_delta: true` if you prefer learning Δ directly (baseline trains on `Y_{t+Δ}`; Δ reconstruction is trivial to add later).
* **Model**: `config/model/ddpm_unet.yaml`

  * 4-scale U-Net, cosine schedule, ε-param.
  * Aux L1 on `x0` (light weight) for stability.
* **Training**: `config/training/baseline.yaml`

  * Mixed precision, periodic small-K validation sampling (K=4) to ensure the ensemble path works during training.
* **Sampling**: `config/sampling/baseline.yaml`

  * `K=32`, `steps=30`, `tau=1.0` (noise temperature), `cfg_scale=0.0` (off).

---

## Notes & limitations (baseline)

* **Feathering**: the baseline stitcher assumes `stride == size` (no overlap). Enable overlap later with real feathering if needed.
* **Static z-score**: done per tile; swap to global stats if you have them.
* **Exogenous deltas**: training uses zeros by default. Plug 1990→2020 deltas if you want strict consistency checks during backtests.

---

## Planned extensions

1. **Location embeddings**

   * Add lat/lon channels (normalized or sinusoidal) in `transforms.py`.
   * Optional **spherical harmonics** features as low-frequency global context.
   * Light wiring: increment `cond_channels`; no architectural upheaval.

2. **Ensemble tuning hooks**

   * **Noise temperature (τ)** and **early-stop** depth exposed in `utils/samplers.py`.
   * **Noise shaping** (Fourier-space radial gain) to match observed spectra.
   * **Conditioning dropout / low-CFG** during sampling for diversity.
   * **Input perturbations** from residual covariances (IC-style spread).

3. **Calibration & evaluation**

   * Optimize τ/steps via **CRPS**, **rank histograms**, and **Energy/Variogram Scores**.
   * Add **Fractions Skill Score (FSS)** / spatial thresholding metrics if needed.
   * Save **member stacks** in chunked **Zarr** for scalable analysis.

4. **Model/loss upgrades (kept modest)**

   * Optional **MS-SSIM** and **spectral** auxiliary losses (low weights).
   * **EMA weights** for sampling stability.
   * **DPMSolver++** sampler (stochastic mode) for fewer steps with similar skill.

5. **Data & scenarios**

   * Wire **SSP/RCP** exogenous deltas for 2020→2050 runs via config only.
   * Add simple **masking** for oceans / protected areas if required.

6. **Performance & UX**

   * True **overlap-feathering** in the stitcher.
   * **Multi-GPU / DDP** via Lightning flags.
   * Robust **resume** / auto-checkpoint loading.

---

## License & citation

* TBD


