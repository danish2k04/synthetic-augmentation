"""
Full evaluation: FID + t-SNE + side-by-side grid for all three generators.

Reads pre-generated images from disk (no re-generation).

Outputs:
  outputs/metrics/fid_all.json               — FID for DCGAN, DDPM-light, DDPM-heavy
  outputs/plots/full_comparison_grid.png     — 3-panel 8x8 sample grid
  outputs/plots/tsne_all.png                 — t-SNE: real + 3 generated sets
"""
import json
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

SEED  = 42
BATCH = 32

# Image folders (all already on disk)
DIRS = {
    "DCGAN-500":      config.GEN_IMAGE_DIR / "dcgan"      / "unhealthy_500",
    "DDPM-light-500": config.GEN_IMAGE_DIR / "ddpm"       / "unhealthy_500",
    "DDPM-heavy-500": config.GEN_IMAGE_DIR / "ddpm_heavy" / "unhealthy_500",
}
REAL_DIR = config.TRAIN_DIR / "unhealthy"

config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
config.METRICS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(s=SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ---------------------------------------------------------------------------
# Load a folder of images into a (N,3,128,128) tensor
# ---------------------------------------------------------------------------
_tf = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

def load_dir(folder: Path, max_n: int = 500) -> torch.Tensor:
    paths = sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg"))
    paths = paths[:max_n]
    return torch.stack([_tf(Image.open(p).convert("RGB")) for p in
                        tqdm(paths, desc=f"  loading {folder.name}", leave=False)])


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
        self.net = net.to(device).eval()
        self.device = device

    @torch.no_grad()
    def extract(self, tensors: torch.Tensor) -> np.ndarray:
        feats = []
        for i in range(0, len(tensors), BATCH):
            b = tensors[i:i + BATCH].to(self.device)
            b = F.interpolate(b, size=(299, 299), mode="bilinear", align_corners=False)
            feats.append(self.net(b).cpu().numpy())
        return np.concatenate(feats, axis=0)


# ---------------------------------------------------------------------------
# FID
# ---------------------------------------------------------------------------
def fid(real: np.ndarray, gen: np.ndarray) -> float:
    mu1, s1 = real.mean(0), np.cov(real.T)
    mu2, s2 = gen.mean(0),  np.cov(gen.T)
    diff = mu1 - mu2
    cov  = linalg.sqrtm(s1 @ s2)
    if np.iscomplexobj(cov):
        cov = cov.real
    return round(float(diff @ diff + np.trace(s1 + s2 - 2 * cov)), 2)


# ---------------------------------------------------------------------------
# t-SNE plot
# ---------------------------------------------------------------------------
PALETTE = {
    "Real":           "#333333",
    "DCGAN-500":      "#F44336",
    "DDPM-light-500": "#2196F3",
    "DDPM-heavy-500": "#4CAF50",
}

def plot_tsne(feat_dict: dict, out: Path):
    n_per = 200
    keys, arrays = zip(*feat_dict.items())
    sampled = [a[np.random.choice(len(a), min(n_per, len(a)), replace=False)]
               for a in arrays]

    combined = np.concatenate(sampled, axis=0)
    reduced  = PCA(n_components=50, random_state=SEED).fit_transform(combined)
    emb      = TSNE(n_components=2, perplexity=30, max_iter=1000,
                    random_state=SEED).fit_transform(reduced)

    fig, ax = plt.subplots(figsize=(9, 8))
    offset  = 0
    for key, s in zip(keys, sampled):
        n   = len(s)
        col = PALETTE.get(key, "#9C27B0")
        ax.scatter(emb[offset:offset + n, 0], emb[offset:offset + n, 1],
                   s=14, alpha=0.70, color=col, label=key, linewidths=0)
        offset += n

    ax.legend(fontsize=10, markerscale=2, framealpha=0.85)
    ax.set_title("t-SNE of InceptionV3 features\n"
                 "Real unhealthy vs. three generators (500 samples each)",
                 fontsize=12, pad=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"t-SNE  → {out}")


# ---------------------------------------------------------------------------
# Side-by-side 3-panel grid
# ---------------------------------------------------------------------------
def plot_grid(tensors: dict, fid_scores: dict, out: Path):
    n_show = 64   # 8 × 8
    keys   = list(tensors.keys())

    fig, axes = plt.subplots(1, 3, figsize=(24, 9))
    for ax, key in zip(axes, keys):
        t    = tensors[key]
        idx  = torch.randperm(len(t))[:n_show]
        grid = make_grid(t[idx], nrow=8, padding=2, normalize=False)
        ax.imshow(grid.permute(1, 2, 0).clamp(0, 1).numpy())
        fid_val = fid_scores.get(key, "?")
        ax.set_title(f"{key}\nFID = {fid_val:.2f}", fontsize=13, fontweight="bold", pad=8)
        ax.axis("off")

    fig.suptitle("Generated unhealthy Pap-smear cells — all generators (500 samples)",
                 fontsize=14, y=1.005)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Grid   → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}\n")

    # Verify all dirs exist
    for name, d in DIRS.items():
        pngs = list(d.glob("*.png"))
        if not pngs:
            raise FileNotFoundError(f"No PNGs in {d} — generate {name} images first.")
        print(f"  {name}: {len(pngs)} images found")

    # ---- Load generated images from disk ----
    print("\nLoading generated images …")
    gen_tensors = {k: load_dir(v) for k, v in DIRS.items()}

    # ---- Load real images ----
    print("Loading real unhealthy images …")
    real_tensors = load_dir(REAL_DIR, max_n=3000)
    print(f"  Real: {len(real_tensors)} images")

    # ---- InceptionV3 features ----
    print("\nExtracting InceptionV3 features …")
    incep = InceptionFeatures(device)

    feats_real = incep.extract(real_tensors)
    print(f"  real      : {feats_real.shape[0]}")

    feats_gen = {}
    for key, t in gen_tensors.items():
        feats_gen[key] = incep.extract(t)
        print(f"  {key}: {feats_gen[key].shape[0]}")

    # ---- FID ----
    print("\nComputing FID …")
    fid_scores = {}
    for key, fg in feats_gen.items():
        score = fid(feats_real, fg)
        fid_scores[key] = score
        print(f"  FID {key:<18} = {score:.2f}")

    fid_path = config.METRICS_DIR / "fid_all.json"
    fid_path.write_text(json.dumps(fid_scores, indent=2))
    print(f"FID    → {fid_path}")

    # ---- t-SNE ----
    print("\nRunning t-SNE (PCA 2048→50, then TSNE 50→2) …")
    tsne_feats = {"Real": feats_real}
    tsne_feats.update(feats_gen)
    plot_tsne(tsne_feats, config.PLOTS_DIR / "tsne_all.png")

    # ---- Side-by-side grid ----
    print("\nSaving comparison grid …")
    plot_grid(gen_tensors, fid_scores, config.PLOTS_DIR / "full_comparison_grid.png")

    # ---- Summary ----
    print("\n" + "=" * 52)
    print("  FID Summary  (500 generated vs 3000 real)")
    print("-" * 52)
    best = min(fid_scores, key=fid_scores.get)
    for k, v in fid_scores.items():
        marker = " ← best" if k == best else ""
        print(f"  {k:<20}  FID = {v:.2f}{marker}")
    print("=" * 52)


if __name__ == "__main__":
    main()
