# src/models/diffusion/unet2d.py
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """[B] -> [B, dim] sinusoidal time embedding."""
    device = timesteps.device
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, device=device, dtype=torch.float32) * (-math.log(10000.0) / (half - 1))
    )
    args = timesteps.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0,1))
    return emb


class FiLM(nn.Module):
    """Feature-wise linear modulation from a conditioning vector."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.to_scale = nn.Linear(in_dim, out_dim)
        self.to_shift = nn.Linear(in_dim, out_dim)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # h: [B,C,H,W], z: [B, in_dim]
        scale = self.to_scale(z)[:, :, None, None]
        shift = self.to_shift(z)[:, :, None, None]
        return h * (1 + scale) + shift


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int, cond_dim: int, dropout: float = 0.0, groups: int = 8):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.time = nn.Linear(t_dim, out_ch)
        self.cond = FiLM(cond_dim, out_ch)

        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x, t_emb, c_emb):
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time(t_emb)[:, :, None, None]
        h = self.cond(h, c_emb)
        h = self.conv2(self.dropout(self.act(self.norm2(h))))
        return h + self.skip(x)


class Down(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.pool = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x): return self.pool(x)


class Up(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(ch, ch, 2, stride=2)

    def forward(self, x): return self.up(x)


class CondEncoder(nn.Module):
    """Turn spatial cond maps into a global vector via lightweight CNN + GAP."""
    def __init__(self, in_ch: int, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, dim, 3, padding=1), nn.SiLU(),
            nn.Conv2d(dim, dim, 3, padding=1), nn.SiLU(),
        )

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        # c: [B,C,H,W] -> [B,dim]
        h = self.net(c)
        h = h.mean(dim=(2,3))
        return h


class UNet2D(nn.Module):
    """Plain 2D U-Net with FiLM conditioning and time embeddings.
    Input channel is only x_t (1); conditioning is fed via FiLM (no concat).
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 96,
        channel_mults=(1, 2, 4, 4),
        num_res_blocks: int = 2,
        time_embed_dim: int = 384,
        cond_channels: int = 1,
        dropout: float = 0.0,
        use_attention: bool = False,  # reserved for later
    ):
        super().__init__()
        self.time_embed_dim = time_embed_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
        )
        self.cond_enc = CondEncoder(cond_channels, time_embed_dim)

        # input stem
        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        chs = [base_channels * m for m in channel_mults]
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.res_down = nn.ModuleList()
        self.res_up = nn.ModuleList()

        ch = base_channels
        # Down path
        for i, cm in enumerate(channel_mults):
            out_ch = base_channels * cm
            for _ in range(num_res_blocks):
                self.res_down.append(ResBlock(ch, out_ch, time_embed_dim, time_embed_dim, dropout))
                ch = out_ch
            self.downs.append(Down(ch) if i < len(channel_mults) - 1 else nn.Identity())

        # Mid
        self.mid1 = ResBlock(ch, ch, time_embed_dim, time_embed_dim, dropout)
        self.mid2 = ResBlock(ch, ch, time_embed_dim, time_embed_dim, dropout)

        # Up path
        for i, cm in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * cm
            for _ in range(num_res_blocks):
                self.res_up.append(ResBlock(ch + out_ch, out_ch, time_embed_dim, time_embed_dim, dropout))
                ch = out_ch
            self.ups.append(Up(ch) if i > 0 else nn.Identity())

        self.out_norm = nn.GroupNorm(8, ch)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(self, x_t: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x_t: [B,1,H,W], cond: [B,Cc,H,W], t: [B]
        t_emb = self.time_mlp(sinusoidal_embedding(t, self.time_embed_dim))
        c_emb = self.cond_enc(cond)

        hs = []
        h = self.in_conv(x_t)

        # Down
        di = 0
        for i, down in enumerate(self.downs):
            for _ in range(len(self._res_down_group(i))):
                h = self.res_down[di](h, t_emb, c_emb); di += 1
                hs.append(h)
            h = down(h)

        # Mid
        h = self.mid1(h, t_emb, c_emb)
        h = self.mid2(h, t_emb, c_emb)

        # Up
        ui = 0
        for i, up in enumerate(self.ups):
            # concat skip
            for _ in range(len(self._res_up_group(i))):
                skip = hs.pop()
                h = torch.cat([h, skip], dim=1)
                h = self.res_up[ui](h, t_emb, c_emb); ui += 1
            h = up(h)

        h = self.out_conv(self.out_act(self.out_norm(h)))
        return h

    def _res_down_group(self, i: int):
        # helper to know how many res blocks per scale (we appended sequentially)
        return [None] * 2  # num_res_blocks fixed = 2 in constructor; keep simple

    def _res_up_group(self, i: int):
        return [None] * 2
