"""
3-group comparison grid: 9 DCGAN | 9 DDPM Lightweight | 9 Full-size DDPM
Equal spacing between all images; group titles placed via fig.text().

Output: outputs/plots/gen_samples_grid.png  (300 DPI)
"""
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DCGAN_DIR = Path("outputs/generated_images/dcgan/unhealthy_500")
DDPM_DIR  = Path("outputs/generated_images/ddpm/unhealthy_500")
DDPMH_DIR = Path("outputs/generated_images/ddpm_heavy/unhealthy_500")
OUT_PATH  = Path("outputs/plots/gen_samples_grid.png")

SEED = 42
N    = 9

GROUPS = [
    ("DCGAN",               DCGAN_DIR),
    ("DDPM (Lightweight)",  DDPM_DIR),
    ("Full-size DDPM",      DDPMH_DIR),
]


def pick_images(folder: Path, n: int, seed: int) -> list:
    paths = sorted(folder.glob("*.png"))
    return random.Random(seed).sample(paths, min(n, len(paths)))


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_groups = [(title, pick_images(d, N, SEED)) for title, d in GROUPS]

    # Flat 3-row × 9-col grid — all cells identical size
    fig, axes = plt.subplots(
        3, 9,
        figsize=(13, 5),
        facecolor="white",
        gridspec_kw={"hspace": 0.05, "wspace": 0.05},
    )

    for g_idx, (title, paths) in enumerate(all_groups):
        col_start = g_idx * 3
        for i, p in enumerate(paths):
            row, col = divmod(i, 3)
            ax = axes[row, col_start + col]
            ax.imshow(np.array(Image.open(p).convert("RGB")))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#aaaaaa")
                spine.set_linewidth(0.6)

    # Equal margins; leave room at top for group titles
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.02,
                        hspace=0.05, wspace=0.05)

    # Each group spans exactly 1/3 of the figure width → centres at 1/6, 1/2, 5/6
    title_x = [1/6, 1/2, 5/6]
    for g_idx, (title, _) in enumerate(all_groups):
        fig.text(
            title_x[g_idx], 0.93, title,
            ha="center", va="bottom",
            fontsize=11, fontweight="bold",
        )

    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
