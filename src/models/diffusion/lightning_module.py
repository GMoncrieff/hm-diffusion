# src/models/diffusion/lightning_module.py
from __future__ import annotations
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from src.models.diffusion.unet2d import UNet2D
from src.models.diffusion.ddpm import DDPM
from src.utils.metrics import mae, rmse
from src.utils.samplers import sample_K


class DiffusionLightningModule(L.LightningModule):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters({"cfg": cfg})
        self.cfg = cfg

        arch = cfg["arch"]
        diff = cfg["diffusion"]
        loss_cfg = cfg.get("loss", {})

        # model
        self.net = UNet2D(
            in_channels=1,
            out_channels=1,
            base_channels=int(arch["base_channels"]),
            channel_mults=tuple(arch["channel_mults"]),
            num_res_blocks=int(arch["num_res_blocks"]),
            time_embed_dim=int(arch["time_embed_dim"]),
            cond_channels=int(arch.get("cond_channels", 1)),
            dropout=float(arch.get("dropout", 0.0)),
            use_attention=bool(arch.get("use_attention", False)),
        )
        self.ddpm = DDPM(self.net, timesteps=int(diff["timesteps"]), schedule=diff["schedule"], parameterization=diff["parameterization"])

        # aux loss on x0
        self.aux_l1_enabled = bool(loss_cfg.get("aux_l1_on_x0", {}).get("enabled", False))
        self.aux_l1_w = float(loss_cfg.get("aux_l1_on_x0", {}).get("weight", 0.0))

        # optimizer settings
        self.lr = float(cfg["optim"]["lr"])
        self.weight_decay = float(cfg["optim"].get("weight_decay", 0.0))
        betas = cfg["optim"].get("betas", [0.9, 0.999])
        self.betas = (float(betas[0]), float(betas[1]))
        self.grad_clip = float(cfg["optim"].get("grad_clip_norm", 0.0))

    # ------------- Lightning API -------------
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=self.betas, weight_decay=self.weight_decay)
        return opt

    def on_before_optimizer_step(self, optimizer):
        if self.grad_clip and self.grad_clip > 0:
            self.clip_gradients(optimizer, gradient_clip_val=self.grad_clip, gradient_clip_algorithm="norm")

    def training_step(self, batch, batch_idx):
        cond, x0 = batch["cond"], batch["target"]  # x0 is Y_{t+Δ} in model space
        losses = self.ddpm.loss(x0, cond)
        loss = losses["loss_eps"]

        # optional aux L1 on x0 from a single-step reconstruction (teacher-forced)
        if self.aux_l1_enabled:
            # one-step estimate of x0 via posterior mean at random t (rough proxy; cheap)
            B = x0.shape[0]
            t = torch.randint(0, self.ddpm.T, (B,), device=self.device, dtype=torch.long)
            noise = torch.randn_like(x0)
            x_t = self.ddpm.q_sample(x0, t, noise)
            _, _, x0_hat = self.ddpm.p_mean_variance(x_t, cond, t)
            loss_l1 = F.l1_loss(x0_hat, x0)
            loss = loss + self.aux_l1_w * loss_l1
            self.log("train/aux_l1", loss_l1, prog_bar=False, on_step=True, on_epoch=True, batch_size=x0.size(0))

        self.log("train/loss_eps", losses["loss_eps"], prog_bar=True, on_step=True, on_epoch=True, batch_size=x0.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        cond, y_true = batch["cond"], batch["target"]
        # small ensemble for sanity
        K = int(self.cfg["evaluation"]["ensemble_K_val"])
        sampler_cfg = self.cfg["evaluation"]["sampler"]
        preds = sample_K(self.ddpm, cond, K=K, sampler_cfg=sampler_cfg)  # [K,B,1,H,W]
        y_mean = preds.mean(dim=0)

        self.log("val/mae", mae(y_mean, y_true), prog_bar=True, on_step=False, on_epoch=True, batch_size=y_true.size(0))
        self.log("val/rmse", rmse(y_mean, y_true), prog_bar=False, on_step=False, on_epoch=True, batch_size=y_true.size(0))
        # recon MAE can be identical here (we're modeling y directly). Left for compatibility:
        self.log("val/mae_recon", mae(y_mean, y_true), prog_bar=False, on_step=False, on_epoch=True, batch_size=y_true.size(0))

    # ------------- Inference API used by scripts -------------
    @torch.inference_mode()
    def sample(self, cond: torch.Tensor, K: int, sampler_cfg: Dict[str, Any]) -> torch.Tensor:
        """Return K samples: [K,B,1,H,W] in model space (normalized)."""
        return sample_K(self.ddpm, cond, K=K, sampler_cfg=sampler_cfg)
