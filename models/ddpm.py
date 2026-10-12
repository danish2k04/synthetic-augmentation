"""
Lightweight conditional UNet for DDPM / DDIM on 128×128 Pap-smear cells.

Architecture (from config.DDPM):
  unet_channels   = (64, 128, 256)   — 3 resolution stages
  attn_resolutions= (16,)            — self-attention only at 16×16
  res_blocks      = 2                — ResBlocks per stage
  num_classes     = 3                — class-conditional via embedding

Spatial path: 128 → 64 → 32 → 16 (bottleneck) → 32 → 64 → 128
Skip connections: one per encoder stage (saved after res-blocks, before downsample)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(32, ch), ch)


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal timestep → MLP → (B, emb_dim*4)."""
    def __init__(self, base_ch: int):
        super().__init__()
        half = base_ch // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half) / max(half - 1, 1))
        self.register_buffer("freqs", freqs)
        self.proj = nn.Sequential(
            nn.Linear(base_ch, base_ch * 4),
            nn.SiLU(),
            nn.Linear(base_ch * 4, base_ch * 4),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = t[:, None].float() * self.freqs[None]
        x = torch.cat([x.sin(), x.cos()], dim=-1)
        return self.proj(x)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """ResBlock with scale-shift time+class conditioning."""
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int):
        super().__init__()
        self.norm1    = _norm(in_ch)
        self.conv1    = nn.Conv2d(in_ch,  out_ch, 3, padding=1)
        self.norm2    = _norm(out_ch)
        self.conv2    = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_ch * 2)
        self.skip     = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h  = self.conv1(F.silu(self.norm1(x)))
        ss = self.emb_proj(F.silu(emb))[:, :, None, None]
        scale, shift = ss.chunk(2, dim=1)
        h  = self.norm2(h) * (1 + scale) + shift
        h  = self.conv2(F.silu(h))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Single-head self-attention with pre-norm."""
    def __init__(self, ch: int):
        super().__init__()
        self.norm  = _norm(ch)
        self.qkv   = nn.Conv2d(ch, ch * 3, 1)
        self.proj  = nn.Conv2d(ch, ch, 1)
        self.scale = ch ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h   = self.norm(x)
        qkv = self.qkv(h).view(B, 3, C, H * W)
        q, k, v = qkv.unbind(1)
        attn = torch.softmax(torch.bmm(q.transpose(1, 2), k) * self.scale, dim=-1)
        out  = torch.bmm(v, attn.transpose(1, 2)).view(B, C, H, W)
        return x + self.proj(out)


def _res_stack(in_ch: int, out_ch: int, emb_dim: int,
               n: int, use_attn: bool) -> tuple[nn.ModuleList, nn.ModuleList]:
    """Returns (resnets, attns) for a stack of n ResBlocks."""
    resnets = nn.ModuleList([
        ResBlock(in_ch if i == 0 else out_ch, out_ch, emb_dim)
        for i in range(n)
    ])
    attns = nn.ModuleList([
        AttentionBlock(out_ch) if use_attn else nn.Identity()
        for _ in range(n)
    ])
    return resnets, attns


def _apply_stack(x, emb, resnets, attns):
    for res, attn in zip(resnets, attns):
        x = attn(res(x, emb))
    return x


# ---------------------------------------------------------------------------
# UNet
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """
    Encoder saves ONE skip per stage (after res blocks, before downsample).
    Decoder: upsample → concat skip → res blocks.

    Channel sizes at each resolution:
      128×128 : ch0 = 64
       64× 64 : ch1 = 128
       32× 32 : ch2 = 256
       16× 16 : ch2 = 256  (bottleneck, attention here)
    """
    def __init__(self):
        super().__init__()
        cfg   = config.DDPM
        ch    = cfg["unet_channels"]          # (64, 128, 256)
        nr    = cfg["res_blocks"]             # 2
        attn  = set(cfg["attn_resolutions"])  # {16}
        emb_d = ch[0] * 4                     # 256 — embedding dimension

        self.time_emb  = SinusoidalEmbedding(ch[0])
        self.class_emb = nn.Embedding(cfg["num_classes"], emb_d)
        self.in_conv   = nn.Conv2d(3, ch[0], 3, padding=1)

        # ---- Encoder ----
        # Stage 1: 128×128, ch0→ch0
        self.enc1_res, self.enc1_attn = _res_stack(
            ch[0], ch[0], emb_d, nr, use_attn=(128 in attn))
        self.down1 = nn.Conv2d(ch[0], ch[0], 3, stride=2, padding=1)

        # Stage 2: 64×64, ch0→ch1
        self.enc2_res, self.enc2_attn = _res_stack(
            ch[0], ch[1], emb_d, nr, use_attn=(64 in attn))
        self.down2 = nn.Conv2d(ch[1], ch[1], 3, stride=2, padding=1)

        # Stage 3: 32×32, ch1→ch2
        self.enc3_res, self.enc3_attn = _res_stack(
            ch[1], ch[2], emb_d, nr, use_attn=(32 in attn))
        self.down3 = nn.Conv2d(ch[2], ch[2], 3, stride=2, padding=1)

        # ---- Bottleneck: 16×16 with attention ----
        self.mid_res1 = ResBlock(ch[2], ch[2], emb_d)
        self.mid_attn = AttentionBlock(ch[2])
        self.mid_res2 = ResBlock(ch[2], ch[2], emb_d)

        # ---- Decoder ----
        # Stage 3 up: upsample 16→32, cat skip(ch2) → ch2+ch2, res → ch2
        self.up3_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(ch[2], ch[2], 3, padding=1),
        )
        self.dec3_res, self.dec3_attn = _res_stack(
            ch[2] + ch[2], ch[2], emb_d, nr, use_attn=(32 in attn))

        # Stage 2 up: upsample 32→64, cat skip(ch1) → ch2+ch1, res → ch1
        self.up2_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(ch[2], ch[2], 3, padding=1),
        )
        self.dec2_res, self.dec2_attn = _res_stack(
            ch[2] + ch[1], ch[1], emb_d, nr, use_attn=(64 in attn))

        # Stage 1 up: upsample 64→128, cat skip(ch0) → ch1+ch0, res → ch0
        self.up1_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(ch[1], ch[1], 3, padding=1),
        )
        self.dec1_res, self.dec1_attn = _res_stack(
            ch[1] + ch[0], ch[0], emb_d, nr, use_attn=(128 in attn))

        # Output
        self.out_norm = _norm(ch[0])
        self.out_conv = nn.Conv2d(ch[0], 3, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        emb = self.time_emb(t) + self.class_emb(labels)   # (B, emb_d)

        # Encoder
        x  = self.in_conv(x)                              # (B, 64, 128, 128)
        x  = _apply_stack(x, emb, self.enc1_res, self.enc1_attn)
        s1 = x                                             # skip (B, 64, 128, 128)
        x  = self.down1(x)                                 # (B, 64, 64, 64)

        x  = _apply_stack(x, emb, self.enc2_res, self.enc2_attn)
        s2 = x                                             # skip (B, 128, 64, 64)
        x  = self.down2(x)                                 # (B, 128, 32, 32)

        x  = _apply_stack(x, emb, self.enc3_res, self.enc3_attn)
        s3 = x                                             # skip (B, 256, 32, 32)
        x  = self.down3(x)                                 # (B, 256, 16, 16)

        # Bottleneck
        x = self.mid_res2(self.mid_attn(self.mid_res1(x, emb)), emb)

        # Decoder
        x = self.up3_up(x)                                 # (B, 256, 32, 32)
        x = torch.cat([x, s3], dim=1)                     # (B, 512, 32, 32)
        x = _apply_stack(x, emb, self.dec3_res, self.dec3_attn)

        x = self.up2_up(x)                                 # (B, 256, 64, 64)
        x = torch.cat([x, s2], dim=1)                     # (B, 384, 64, 64)
        x = _apply_stack(x, emb, self.dec2_res, self.dec2_attn)

        x = self.up1_up(x)                                 # (B, 128, 128, 128)
        x = torch.cat([x, s1], dim=1)                     # (B, 192, 128, 128)
        x = _apply_stack(x, emb, self.dec1_res, self.dec1_attn)

        return self.out_conv(F.silu(self.out_norm(x)))     # (B, 3, 128, 128)
