"""
Conditional DCGAN for 128×128 Pap-smear cell synthesis.

Generator  : z(latent_dim) + label_emb → 4×4 → 8 → 16 → 32 → 64 → 128×128×3
Discriminator: img(3+1) + label spatial map → 5 strided convs → scalar logit
"""
import torch
import torch.nn as nn
import config


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        cfg   = config.DCGAN
        g_ch  = cfg["g_channels"]          # [512, 256, 128, 64]
        in_d  = cfg["latent_dim"] + cfg["label_emb_dim"]

        self.label_emb = nn.Embedding(cfg["num_classes"], cfg["label_emb_dim"])

        # z + label_emb → reshape to (g_ch[0], 4, 4)
        self.proj = nn.Sequential(
            nn.Linear(in_d, g_ch[0] * 4 * 4, bias=False),
            nn.BatchNorm1d(g_ch[0] * 4 * 4),
            nn.ReLU(True),
        )

        # 5 upsampling blocks → 4→8→16→32→64→128
        self.blocks = nn.Sequential(
            self._up(g_ch[0], g_ch[1]),           # 4→8:   512→256
            self._up(g_ch[1], g_ch[2]),           # 8→16:  256→128
            self._up(g_ch[2], g_ch[3]),           # 16→32: 128→64
            self._up(g_ch[3], g_ch[3] // 2),     # 32→64:  64→32
            self._up(g_ch[3] // 2, 3, final=True),  # 64→128: 32→3
        )

    @staticmethod
    def _up(in_c: int, out_c: int, final: bool = False) -> nn.Sequential:
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False)
        ]
        if final:
            layers.append(nn.Tanh())
        else:
            layers += [nn.BatchNorm2d(out_c), nn.ReLU(True)]
        return nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        emb = self.label_emb(labels)                        # (B, label_emb_dim)
        x   = self.proj(torch.cat([z, emb], dim=1))        # (B, 512*4*4)
        x   = x.view(x.size(0), config.DCGAN["g_channels"][0], 4, 4)
        return self.blocks(x)                               # (B, 3, 128, 128)


# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        cfg  = config.DCGAN
        d_ch = cfg["d_channels"]           # [64, 128, 256, 512]
        sz   = cfg["image_size"]           # 128

        self.label_emb  = nn.Embedding(cfg["num_classes"], cfg["label_emb_dim"])
        self.label_proj = nn.Linear(cfg["label_emb_dim"], sz * sz)

        # Input: img (3 ch) + label spatial map (1 ch) = 4 ch
        self.blocks = nn.Sequential(
            self._down(4,       d_ch[0], bn=False),  # 128→64
            self._down(d_ch[0], d_ch[1]),             # 64→32
            self._down(d_ch[1], d_ch[2]),             # 32→16
            self._down(d_ch[2], d_ch[3]),             # 16→8
            self._down(d_ch[3], d_ch[3]),             # 8→4
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_ch[3] * 4 * 4, 1),
        )

    @staticmethod
    def _down(in_c: int, out_c: int, bn: bool = True) -> nn.Sequential:
        layers: list[nn.Module] = [
            nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False)
        ]
        if bn:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2, True))
        return nn.Sequential(*layers)

    def forward(self, img: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        sz  = config.DCGAN["image_size"]
        emb = self.label_emb(labels)                                  # (B, label_emb_dim)
        prj = self.label_proj(emb).view(-1, 1, sz, sz)               # (B, 1, 128, 128)
        x   = self.blocks(torch.cat([img, prj], dim=1))              # (B, 512, 4, 4)
        return self.head(x)                                           # (B, 1)
