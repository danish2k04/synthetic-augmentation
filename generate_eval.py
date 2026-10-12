"""
Generate synthetic unhealthy-class images from DCGAN and DDPM,
compute FID against real unhealthy images, run t-SNE, and
produce side-by-side sample grids.

Usage:
    python generate_eval.py

Outputs:
  outputs/generated_images/dcgan/unhealthy_250/   (250 PNGs)
  outputs/generated_images/dcgan/unhealthy_500/   (500 PNGs)
  outputs/generated_images/ddpm/unhealthy_250/    (250 PNGs)
  outputs/generated_images/ddpm/unhealthy_500/    (500 PNGs)
  outputs/plots/sample_grid_comparison.png
  outputs/plots/tsne.png
  outputs/metrics/fid_scores.json
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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.utils import make_grid
from tqdm import tqdm

import config
from dataset import UnlabelledFolderDataset
from models.dcgan import Generator
from models.ddpm import UNet
from train_ddpm import NoiseSchedule, ddim_sample


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DCGAN_CKPT = config.CHECKPOINT_DIR / "dcgan" / "dcgan_ep0200.pth"
DDPM_CKPT  = config.CHECKPOINT_DIR / "ddpm"  / "ddpm_ep0300.pth"
PLOT_DIR   = config.OUTPUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
config.METRICS_DIR.mkdir(parents=True, exist_ok=True)

COUNTS = [250, 500]
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
# Image transform (PIL PNG → tensor in [0,1])
# ---------------------------------------------------------------------------
_to_tensor = transforms.ToTensor()


def load_pil_to_tensor(path: Path) -> torch.Tensor:
    return _to_tensor(Image.open(path).convert("RGB"))


# ---------------------------------------------------------------------------
# Generate from DCGAN
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_dcgan(n: int, device: torch.device) -> torch.Tensor:
    """Returns (n, 3, 128, 128) float tensor in [0, 1]."""
    G = Generator().to(device)
    ckpt = torch.load(DCGAN_CKPT, map_location=device, weights_only=False)
    G.load_state_dict(ckpt["G_state"])
    G.eval()

    cls = config.UNHEALTHY_IDX
    imgs = []
    for start in range(0, n, BATCH):
        bsz = min(BATCH, n - start)
        z   = torch.randn(bsz, config.DCGAN["latent_dim"], device=device)
        lbl = torch.full((bsz,), cls, dtype=torch.long, device=device)
        out = G(z, lbl)                          # [-1, 1]
        imgs.append((out * 0.5 + 0.5).clamp(0, 1).cpu())
    return torch.cat(imgs)


# ---------------------------------------------------------------------------
# Generate from DDPM (DDIM)
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_ddpm(n: int, device: torch.device) -> torch.Tensor:
    """Returns (n, 3, 128, 128) float tensor in [0, 1]."""
    cfg   = config.DDPM
    model = UNet().to(device)
    ckpt  = torch.load(DDPM_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    sched = NoiseSchedule(cfg["timesteps"], cfg["beta_start"],
                          cfg["beta_end"], device)
    cls   = config.UNHEALTHY_IDX
    imgs  = []
    for start in tqdm(range(0, n, BATCH), desc="  DDIM sampling", leave=False):
        bsz = min(BATCH, n - start)
        z   = torch.randn(bsz, 3, cfg["image_size"], cfg["image_size"], device=device)
        lbl = torch.full((bsz,), cls, dtype=torch.long, device=device)
        out = ddim_sample(model, sched, z, lbl, cfg["ddim_steps"])
        imgs.append((out * 0.5 + 0.5).clamp(0, 1).cpu())
    return torch.cat(imgs)


# ---------------------------------------------------------------------------
# Save generated images to disk
# ---------------------------------------------------------------------------
def save_images(tensors: torch.Tensor, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, t in enumerate(tensors):
        img = transforms.ToPILImage()(t)
        img.save(out_dir / f"{i:05d}.png")


# ---------------------------------------------------------------------------
# InceptionV3 feature extractor (for FID + t-SNE)
# ---------------------------------------------------------------------------
class InceptionFeatures(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        net.aux_logits = False
        net.AuxLogits  = None
        net.fc = nn.Identity()
        self.net = net.to(device).eval()
        self.device = device

    @torch.no_grad()
    def extract(self, tensors: torch.Tensor) -> np.ndarray:
        """tensors: (N, 3, H, W) in [0,1]. Returns (N, 2048)."""
        feats = []
        for start in range(0, len(tensors), BATCH):
            batch = tensors[start:start + BATCH].to(self.device)
            batch = F.interpolate(batch, size=(299, 299),
                                  mode="bilinear", align_corners=False)
            out = self.net(batch)
            feats.append(out.cpu().numpy())
        return np.concatenate(feats, axis=0)

    @torch.no_grad()
    def extract_from_dir(self, folder: Path, max_n: int = 3000) -> np.ndarray:
        paths = sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg"))
        paths = paths[:max_n]
        tf    = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        tensors = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths])
        return self.extract(tensors)


# ---------------------------------------------------------------------------
# FID
# ---------------------------------------------------------------------------
def compute_fid(feats_real: np.ndarray, feats_gen: np.ndarray) -> float:
    mu1, sigma1 = feats_real.mean(0), np.cov(feats_real.T)
    mu2, sigma2 = feats_gen.mean(0),  np.cov(feats_gen.T)
    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))
    return round(fid, 2)


# ---------------------------------------------------------------------------
# t-SNE plot
# ---------------------------------------------------------------------------
def plot_tsne(feature_sets: dict[str, np.ndarray], out_path: Path):
    """feature_sets: {label: (N, 2048)} — PCA→50 then t-SNE→2D."""
    labels, arrays = zip(*feature_sets.items())
    n_per = 200
    sampled = [a[np.random.choice(len(a), min(n_per, len(a)), replace=False)]
               for a in arrays]

    combined = np.concatenate(sampled, axis=0)
    pca      = PCA(n_components=50, random_state=SEED)
    reduced  = pca.fit_transform(combined)
    embedded = TSNE(n_components=2, perplexity=30, max_iter=1000,
                    random_state=SEED).fit_transform(reduced)

    colours = ["#2196F3", "#F44336", "#FF9800", "#4CAF50", "#9C27B0"]
    fig, ax = plt.subplots(figsize=(8, 7))
    offset  = 0
    for i, (lbl, s) in enumerate(zip(labels, sampled)):
        n = len(s)
        ax.scatter(embedded[offset:offset + n, 0],
                   embedded[offset:offset + n, 1],
                   s=12, alpha=0.65, color=colours[i % len(colours)],
                   label=lbl, linewidths=0)
        offset += n
    ax.legend(fontsize=9, markerscale=2)
    ax.set_title("t-SNE of InceptionV3 features\n(real vs synthetic unhealthy)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"t-SNE  → {out_path}")


# ---------------------------------------------------------------------------
# Side-by-side sample grid
# ---------------------------------------------------------------------------
def plot_grid_comparison(tensors: dict[str, torch.Tensor], out_path: Path):
    """4-panel grid: DCGAN-250 | DCGAN-500 | DDPM-250 | DDPM-500."""
    n_show = 16   # 4×4 per panel
    keys   = ["DCGAN-250", "DCGAN-500", "DDPM-250", "DDPM-500"]

    fig = plt.figure(figsize=(22, 6))
    gs  = gridspec.GridSpec(1, 4, wspace=0.04)

    for col, key in enumerate(keys):
        t    = tensors[key]
        idx  = torch.randperm(len(t))[:n_show]
        grid = make_grid(t[idx], nrow=4, padding=2, normalize=False)
        ax   = fig.add_subplot(gs[col])
        ax.imshow(grid.permute(1, 2, 0).clamp(0, 1).numpy())
        ax.set_title(key, fontsize=13, fontweight="bold", pad=6)
        ax.axis("off")

    fig.suptitle("Generated unhealthy Pap-smear cells", fontsize=14, y=1.01)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Grid   → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}\n")

    # ---- Generate images ----
    gen_tensors: dict[str, torch.Tensor] = {}
    gen_dirs:    dict[str, Path]         = {}

    for n in COUNTS:
        print(f"Generating DCGAN-{n} …")
        t = generate_dcgan(n, device)
        d = config.GEN_IMAGE_DIR / "dcgan" / f"unhealthy_{n}"
        save_images(t, d)
        gen_tensors[f"DCGAN-{n}"] = t
        gen_dirs[f"DCGAN-{n}"]    = d
        print(f"  saved {n} images → {d}")

    for n in COUNTS:
        print(f"Generating DDPM-{n} …")
        t = generate_ddpm(n, device)
        d = config.GEN_IMAGE_DIR / "ddpm" / f"unhealthy_{n}"
        save_images(t, d)
        gen_tensors[f"DDPM-{n}"] = t
        gen_dirs[f"DDPM-{n}"]    = d
        print(f"  saved {n} images → {d}")

    # ---- Inception features ----
    print("\nExtracting InceptionV3 features …")
    incep = InceptionFeatures(device)

    real_dir   = config.TRAIN_DIR / "unhealthy"
    feats_real = incep.extract_from_dir(real_dir)
    print(f"  real unhealthy : {feats_real.shape[0]} images")

    feats_gen: dict[str, np.ndarray] = {}
    for key, t in gen_tensors.items():
        feats_gen[key] = incep.extract(t)
        print(f"  {key:12s} : {feats_gen[key].shape[0]} images")

    # ---- FID ----
    print("\nComputing FID …")
    fid_scores: dict[str, float] = {}
    for key, fg in feats_gen.items():
        fid = compute_fid(feats_real, fg)
        fid_scores[key] = fid
        print(f"  FID {key:12s} = {fid:.2f}")

    fid_path = config.METRICS_DIR / "fid_scores.json"
    fid_path.write_text(json.dumps(fid_scores, indent=2))
    print(f"FID    → {fid_path}")

    # ---- t-SNE ----
    print("\nRunning t-SNE …")
    tsne_sets = {"Real (unhealthy)": feats_real}
    tsne_sets.update(feats_gen)
    plot_tsne(tsne_sets, PLOT_DIR / "tsne.png")

    # ---- Side-by-side grid ----
    print("\nSaving sample grid …")
    plot_grid_comparison(gen_tensors, PLOT_DIR / "sample_grid_comparison.png")

    # ---- Summary ----
    print("\n" + "=" * 50)
    print("  FID Summary")
    print("-" * 50)
    for k, v in fid_scores.items():
        print(f"  {k:12s}  FID = {v:.2f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
