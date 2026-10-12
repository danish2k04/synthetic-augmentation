"""
Conditional UNet for DDPM / DDIM — supports both lightweight and heavy configs.

Pass cfg=config.DDPM  for the lightweight model (3 stages, attn@16, T=500).
Pass cfg=config.DDPM_HEAVY for the heavy model  (4 stages, attn@8+16, T=1000).

Encoder saves one skip per stage (after res-blocks, before downsample).
Decoder: upsample → concat skip → res-blocks.

Resolution chart (128×128 input):
  n_stages=3 : 128→64→32→16(bot)→32→64→128
  n_stages=4 : 128→64→32→16→8(bot)→16→32→64→128
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import config as _config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(32, ch), ch)


class SinusoidalEmbedding(nn.Module):
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


def _res_stack(in_ch, out_ch, emb_dim, n, use_attn):
    resnets = nn.ModuleList([
        ResBlock(in_ch if i == 0 else out_ch, out_ch, emb_dim) for i in range(n)
    ])
    attns = nn.ModuleList([
        AttentionBlock(out_ch) if use_attn else nn.Identity() for _ in range(n)
    ])
    return resnets, attns


def _apply_stack(x, emb, resnets, attns):
    for res, attn in zip(resnets, attns):
        x = attn(res(x, emb))
    return x


# ---------------------------------------------------------------------------
# Generic UNet
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """
    cfg must contain:
      unet_channels    : tuple of ints, length = n_stages
      attn_resolutions : iterable of ints (feature-map sizes where attn is applied)
      res_blocks       : int
      num_classes      : int
      image_size       : int
    """
    def __init__(self, cfg: dict | None = None):
        super().__init__()
        if cfg is None:
            cfg = _config.DDPM

        ch      = cfg["unet_channels"]          # e.g. (64,128,256) or (64,128,256,512)
        n       = len(ch)                        # number of stages
        nr      = cfg["res_blocks"]
        attn    = set(cfg["attn_resolutions"])
        emb_d   = ch[0] * 4
        img_sz  = cfg["image_size"]             # 128

        self._ch    = ch
        self._n     = n
        self._emb_d = emb_d

        self.time_emb  = SinusoidalEmbedding(ch[0])
        self.class_emb = nn.Embedding(cfg["num_classes"], emb_d)
        self.in_conv   = nn.Conv2d(3, ch[0], 3, padding=1)

        # ---- Encoder ----
        # Stage j (j=0..n-1): processes resolution img_sz // (2^j)
        #   in_ch  = ch[0]      if j==0
        #          = ch[j-1]    if j>0  (previous stage output, same channel count after downsample)
        #   out_ch = ch[j]
        self.enc_res   = nn.ModuleList()
        self.enc_attn  = nn.ModuleList()
        self.enc_down  = nn.ModuleList()
        for j in range(n):
            res = img_sz // (2 ** j)
            in_c  = ch[0] if j == 0 else ch[j - 1]
            out_c = ch[j]
            r, a = _res_stack(in_c, out_c, emb_d, nr, use_attn=(res in attn))
            self.enc_res.append(r)
            self.enc_attn.append(a)
            self.enc_down.append(nn.Conv2d(out_c, out_c, 3, stride=2, padding=1))

        # ---- Bottleneck ----
        # at resolution img_sz // 2^n, channels = ch[n-1]
        bot_res = img_sz // (2 ** n)
        self.mid_res1 = ResBlock(ch[-1], ch[-1], emb_d)
        self.mid_attn = AttentionBlock(ch[-1]) if (bot_res in attn) else nn.Identity()
        self.mid_res2 = ResBlock(ch[-1], ch[-1], emb_d)

        # ---- Decoder ----
        # Stage j (j=n-1..0, iterating from deep to shallow):
        #   target resolution after upsample: img_sz // (2^j)
        #   upsample input channels:
        #     j==n-1: ch[n-1]   (bottleneck output)
        #     j< n-1: ch[j+1]   (previous decoder output)
        #   skip channels: ch[j]  (s[j] from encoder stage j)
        #   dec in_ch  = upsample_ch + ch[j]
        #   dec out_ch = ch[j]
        self.dec_up   = nn.ModuleList()
        self.dec_res  = nn.ModuleList()
        self.dec_attn = nn.ModuleList()
        for j in range(n - 1, -1, -1):
            res    = img_sz // (2 ** j)
            up_in  = ch[n - 1] if j == n - 1 else ch[j + 1]
            skip_c = ch[j]
            dec_in = up_in + skip_c
            dec_out= ch[j]
            self.dec_up.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(up_in, up_in, 3, padding=1),
            ))
            r, a = _res_stack(dec_in, dec_out, emb_d, nr, use_attn=(res in attn))
            self.dec_res.append(r)
            self.dec_attn.append(a)

        # Output
        self.out_norm = _norm(ch[0])
        self.out_conv = nn.Conv2d(ch[0], 3, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        emb = self.time_emb(t) + self.class_emb(labels)

        # Encoder
        x = self.in_conv(x)
        skips = []
        for j in range(self._n):
            x = _apply_stack(x, emb, self.enc_res[j], self.enc_attn[j])
            skips.append(x)
            x = self.enc_down[j](x)

        # Bottleneck
        x = self.mid_res1(x, emb)
        x = self.mid_attn(x) if isinstance(self.mid_attn, AttentionBlock) else self.mid_attn(x)
        x = self.mid_res2(x, emb)

        # Decoder (dec_up/res/attn lists run from j=n-1 down to j=0)
        for k, j in enumerate(range(self._n - 1, -1, -1)):
            x = self.dec_up[k](x)
            x = torch.cat([x, skips[j]], dim=1)
            x = _apply_stack(x, emb, self.dec_res[k], self.dec_attn[k])

        return self.out_conv(F.silu(self.out_norm(x)))
