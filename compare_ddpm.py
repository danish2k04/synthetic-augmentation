"""
Compare lightweight DDPM vs heavy DDPM:
  - Generate 500 unhealthy images from each
  - Compute FID against real unhealthy images
  - Save side-by-side 8×8 sample grids
  - Print comparison table

Usage (run after both DDPM checkpoints exist):
    python compare_ddpm.py

Outputs:
  outputs/generated_images/ddpm/unhealthy_500/        (if not already present)
  outputs/generated_images/ddpm_heavy/unhealthy_500/
  outputs/plots/ddpm_comparison_grid.png
  outputs/metrics/ddpm_fid_comparison.json
"""
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import linalg
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

import config
from models.ddpm import UNet
from train_ddpm import NoiseSchedule, ddim_sample

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DDPM_LIGHT_CKPT  = config.CHECKPOINT_DIR / "ddpm"       / "ddpm_ep0300.pth"
DDPM_HEAVY_CKPT  = config.CHECKPOINT_DIR / "ddpm_heavy" / "ddpm_heavy_ep0300.pth"

LIGHT_OUT_DIR = config.GEN_IMAGE_DIR / "ddpm"       / "unhealthy_500"
HEAVY_OUT_DIR = config.GEN_IMAGE_DIR / "ddpm_heavy" / "unhealthy_500"

PLOT_DIR = config.PLOTS_DIR
PLOT_DIR.mkdir(parents=True, exist_ok=True)
config.METRICS_DIR.mkdir(parents=True, exist_ok=True)

N_GEN  = 500
BATCH  = 32
SEED   = 42


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(s=SEED):
    import random
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ---------------------------------------------------------------------------
# Generate from a DDPM checkpoint
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate(ckpt_path: Path, cfg: dict, out_dir: Path,
             device: torch.device) -> torch.Tensor:
    """Load checkpoint, generate N_GEN images, save PNGs, return (N,3,128,128)."""
    model = UNet(cfg=cfg).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    sched = NoiseSchedule(cfg["timesteps"], cfg["beta_start"],
                          cfg["beta_end"], device)
    cls   = config.UNHEALTHY_IDX
    imgs  = []
    for start in tqdm(range(0, N_GEN, BATCH), desc=f"  sampling {ckpt_path.stem}", leave=False):
        bsz = min(BATCH, N_GEN - start)
        z   = torch.randn(bsz, 3, cfg["image_size"], cfg["image_size"], device=device)
        lbl = torch.full((bsz,), cls, dtype=torch.long, device=device)
        out = ddim_sample(model, sched, z, lbl, cfg["ddim_steps"])
        imgs.append((out * 0.5 + 0.5).clamp(0, 1).cpu())

    tensors = torch.cat(imgs)

    out_dir.mkdir(parents=True, exist_ok=True)
    for i, t in enumerate(tensors):
        transforms.ToPILImage()(t).save(out_dir / f"{i:05d}.png")
    print(f"  saved {N_GEN} images → {out_dir}")

    return tensors


# ---------------------------------------------------------------------------
# InceptionV3 feature extractor
# ---------------------------------------------------------------------------
class InceptionFeatures(nn.Module):
    def __init__(self, device):
        super().__init__()
        net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        net.aux_logits = False
        net.AuxLogits  = None
        net.fc = nn.Identity()
        self.net    = net.to(device).eval()
        self.device = device

    @torch.no_grad()
    def extract(self, tensors: torch.Tensor) -> np.ndarray:
        feats = []
        for start in range(0, len(tensors), BATCH):
            b = tensors[start:start + BATCH].to(self.device)
            b = F.interpolate(b, size=(299, 299), mode="bilinear", align_corners=False)
            feats.append(self.net(b).cpu().numpy())
        return np.concatenate(feats, axis=0)

    @torch.no_grad()
    def extract_from_dir(self, folder: Path, max_n: int = 3000) -> np.ndarray:
        paths = sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg"))
        paths = paths[:max_n]
        tf    = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
        tensors = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths])
        return self.extract(tensors)


# ---------------------------------------------------------------------------
# FID
# ---------------------------------------------------------------------------
def compute_fid(real: np.ndarray, gen: np.ndarray) -> float:
    mu1, s1 = real.mean(0), np.cov(real.T)
    mu2, s2 = gen.mean(0),  np.cov(gen.T)
    diff    = mu1 - mu2
    cov     = linalg.sqrtm(s1 @ s2)
    if np.iscomplexobj(cov):
        cov = cov.real
    return round(float(diff @ diff + np.trace(s1 + s2 - 2 * cov)), 2)


# ---------------------------------------------------------------------------
# Side-by-side grid  (lightweight left, heavy right)
# ---------------------------------------------------------------------------
def save_comparison_grid(light: torch.Tensor, heavy: torch.Tensor, path: Path):
    n_show = 64   # 8×8 each
    idx_l  = torch.randperm(len(light))[:n_show]
    idx_h  = torch.randperm(len(heavy))[:n_show]

    grid_l = make_grid(light[idx_l], nrow=8, padding=2, normalize=False)
    grid_h = make_grid(heavy[idx_h], nrow=8, padding=2, normalize=False)

    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    axes[0].imshow(grid_l.permute(1, 2, 0).clamp(0, 1).numpy())
    axes[0].set_title("Lightweight DDPM\n(64-128-256, T=500, 2 res-blocks)",
                      fontsize=13, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(grid_h.permute(1, 2, 0).clamp(0, 1).numpy())
    axes[1].set_title("Heavy DDPM\n(64-128-256-512, T=1000, 3 res-blocks)",
                      fontsize=13, fontweight="bold")
    axes[1].axis("off")

    fig.suptitle("Generated unhealthy Pap-smear cells — DDPM variant comparison",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Grid   → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}\n")

    # Confirm both checkpoints exist
    for p in [DDPM_LIGHT_CKPT, DDPM_HEAVY_CKPT]:
        if not p.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {p}\n"
                "Train the missing model first, then re-run compare_ddpm.py"
            )

    # ---- Generate ----
    print("Generating lightweight DDPM images …")
    light_imgs = generate(DDPM_LIGHT_CKPT, config.DDPM,       LIGHT_OUT_DIR, device)

    print("Generating heavy DDPM images …")
    heavy_imgs = generate(DDPM_HEAVY_CKPT, config.DDPM_HEAVY, HEAVY_OUT_DIR, device)

    # ---- Inception features ----
    print("\nExtracting InceptionV3 features …")
    incep      = InceptionFeatures(device)
    real_dir   = config.TRAIN_DIR / "unhealthy"
    feats_real = incep.extract_from_dir(real_dir)
    feats_light = incep.extract(light_imgs)
    feats_heavy = incep.extract(heavy_imgs)
    print(f"  real      : {feats_real.shape[0]}")
    print(f"  lightweight: {feats_light.shape[0]}")
    print(f"  heavy     : {feats_heavy.shape[0]}")

    # ---- FID ----
    fid_light = compute_fid(feats_real, feats_light)
    fid_heavy = compute_fid(feats_real, feats_heavy)

    results = {
        "ddpm_lightweight_500": fid_light,
        "ddpm_heavy_500":       fid_heavy,
    }
    out_path = config.METRICS_DIR / "ddpm_fid_comparison.json"
    out_path.write_text(json.dumps(results, indent=2))

    # ---- Grid ----
    save_comparison_grid(light_imgs, heavy_imgs,
                         PLOT_DIR / "ddpm_comparison_grid.png")

    # ---- Summary ----
    print("\n" + "=" * 48)
    print("  DDPM FID Comparison  (500 generated vs real)")
    print("-" * 48)
    print(f"  Lightweight  FID = {fid_light:.2f}")
    print(f"  Heavy        FID = {fid_heavy:.2f}")
    winner = "Lightweight" if fid_light < fid_heavy else "Heavy"
    print(f"  → {winner} wins on FID")
    print("=" * 48)
    print(f"\nFID saved → {out_path}")


if __name__ == "__main__":
    main()
