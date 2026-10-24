"""
FID and t-SNE evaluated against REAL TEST SET unhealthy images.

Reference : 576 unhealthy images from the annotated test CSV (flat folder)
Generated : 500 images each from DCGAN, DDPM-light, Full-size DDPM

Outputs:
  outputs/metrics/fid_test.json
  outputs/plots/tsne_test.png

Usage:
    python fid_tsne_test.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from scipy import linalg
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torchvision import transforms
from torchvision.models import inception_v3
from tqdm import tqdm
import json

import config

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TEST_CSV    = config.TEST_CSV
TEST_DIR    = config.TEST_DIR
GEN_DIR     = config.GEN_IMAGE_DIR

GENERATED = {
    "DCGAN-500":          GEN_DIR / "dcgan"      / "unhealthy_500",
    "DDPM-light-500":     GEN_DIR / "ddpm"       / "unhealthy_500",
    "Full-size-DDPM-500": GEN_DIR / "ddpm_heavy" / "unhealthy_500",
}

METRICS_OUT = config.METRICS_DIR / "fid_test.json"
TSNE_OUT    = config.PLOTS_DIR   / "tsne_test.png"

# ---------------------------------------------------------------------------
# InceptionV3 feature extractor (pool3 layer, 2048-dim)
# ---------------------------------------------------------------------------
class InceptionFeatures(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        net = inception_v3(weights="IMAGENET1K_V1", transform_input=False)
        net.fc = nn.Identity()
        net.aux_logits = False
        self.net = net.to(device).eval()

    @torch.no_grad()
    def extract(self, img_paths: list, desc: str = "") -> np.ndarray:
        tf = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
        feats = []
        batch, bs = [], 32
        def _flush():
            t = torch.stack(batch).to(self.device)
            feats.append(self.net(t).cpu().numpy())
            batch.clear()

        for p in tqdm(img_paths, desc=f"  {desc}", leave=False):
            img = Image.open(p).convert("RGB")
            batch.append(tf(img))
            if len(batch) == bs:
                _flush()
        if batch:
            _flush()
        return np.concatenate(feats, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# FID
# ---------------------------------------------------------------------------
def compute_fid(feats_real: np.ndarray, feats_gen: np.ndarray) -> float:
    mu1, s1 = feats_real.mean(0), np.cov(feats_real.T)
    mu2, s2 = feats_gen.mean(0),  np.cov(feats_gen.T)
    diff = mu1 - mu2
    cov  = linalg.sqrtm(s1 @ s2)
    if np.iscomplexobj(cov):
        cov = cov.real
    return float(diff @ diff + np.trace(s1 + s2 - 2 * cov))


# ---------------------------------------------------------------------------
# t-SNE
# ---------------------------------------------------------------------------
def plot_tsne(feats_dict: dict, out: Path):
    COLOURS = {
        "Real (test)":        "#333333",
        "DCGAN-500":          "#E53935",
        "DDPM-light-500":     "#1E88E5",
        "Full-size-DDPM-500": "#43A047",
    }
    MARKERS = {
        "Real (test)":        "o",
        "DCGAN-500":          "s",
        "DDPM-light-500":     "^",
        "Full-size-DDPM-500": "D",
    }

    # Sub-sample to 200 per set for t-SNE speed
    rng = np.random.default_rng(42)
    sub, labels = [], []
    for name, f in feats_dict.items():
        idx = rng.choice(len(f), min(200, len(f)), replace=False)
        sub.append(f[idx])
        labels.extend([name] * len(idx))
    all_feats = np.concatenate(sub, axis=0)

    print("  PCA 2048 → 50 …")
    pca  = PCA(n_components=50, random_state=42)
    low  = pca.fit_transform(all_feats)

    print("  t-SNE 50 → 2 …")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42,
                max_iter=1000, init="pca")
    emb  = tsne.fit_transform(low)

    fig, ax = plt.subplots(figsize=(8, 7), facecolor="white")
    ax.set_title(
        "t-SNE of InceptionV3 features\n"
        "Real test unhealthy vs. three generators (500 samples each)",
        fontsize=12, fontweight="bold",
    )

    offset = 0
    for name, f in feats_dict.items():
        n   = min(200, len(f))
        pts = emb[offset: offset + n]
        ax.scatter(pts[:, 0], pts[:, 1],
                   c=COLOURS[name], marker=MARKERS[name],
                   s=22, alpha=0.75, label=name, linewidths=0)
        offset += n

    ax.legend(fontsize=10, framealpha=0.9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[["top","right","bottom","left"]].set_visible(False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load real test unhealthy image paths from CSV ──────────────────
    print("\n[1/4] Loading real test unhealthy paths …")
    df = pd.read_csv(TEST_CSV)
    df_uh = df[df["label"] == "unhealthy"].reset_index(drop=True)
    real_paths = [TEST_DIR / row["image_name"] for _, row in df_uh.iterrows()]
    print(f"  Real test unhealthy : {len(real_paths)} images")

    # ── 2. Load generated image paths ─────────────────────────────────────
    print("\n[2/4] Collecting generated image paths …")
    gen_paths = {}
    for name, folder in GENERATED.items():
        paths = sorted(folder.glob("*.png"))
        gen_paths[name] = paths
        print(f"  {name:<22} : {len(paths)} images")

    # ── 3. Extract InceptionV3 features ───────────────────────────────────
    print("\n[3/4] Extracting InceptionV3 features …")
    incep = InceptionFeatures(device)

    feats_real = incep.extract(real_paths, "Real (test)")
    print(f"  real features shape : {feats_real.shape}")

    feats_gen = {}
    for name, paths in gen_paths.items():
        feats_gen[name] = incep.extract(paths, name)
        print(f"  {name:<22} : {feats_gen[name].shape}")

    # ── 4. FID ─────────────────────────────────────────────────────────────
    print("\n[4/4] Computing FID (vs 576 real test unhealthy) …")
    fid_scores = {}
    for name, fg in feats_gen.items():
        score = compute_fid(feats_real, fg)
        fid_scores[name] = round(score, 2)
        print(f"  FID {name:<22} = {score:.2f}")

    METRICS_OUT.write_text(json.dumps(fid_scores, indent=2))
    print(f"\n  → {METRICS_OUT}")

    best = min(fid_scores, key=fid_scores.get)
    print(f"  Best : {best}  (FID = {fid_scores[best]:.2f})")

    # ── 5. t-SNE ──────────────────────────────────────────────────────────
    print("\nt-SNE …")
    tsne_feats = {"Real (test)": feats_real, **feats_gen}
    plot_tsne(tsne_feats, TSNE_OUT)

    print("\nDone.")


if __name__ == "__main__":
    main()
