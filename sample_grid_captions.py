"""
Four training images in a row with captions (a), (b), (c), (d).

Output: outputs/plots/sample_grid_captions.png  (300 DPI)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

TRAIN_DIR = Path(r"C:\Users\danish\Desktop\pap_cell_classification\data\isbi2025-ps3c-train-dataset")
OUT_PATH  = Path("outputs/plots/sample_grid_captions.png")

IMAGES = [
    (TRAIN_DIR / "healthy" / "isbi2025_ps3c_train_image_03534.png", "(a)"),
    (TRAIN_DIR / "unhealthy" / "isbi2025_ps3c_train_image_82751.png", "(b)"),
    (TRAIN_DIR / "rubbish"   / "isbi2025_ps3c_train_image_32540.png", "(c)"),
    (TRAIN_DIR / "bothcells"   / "isbi2025_ps3c_train_image_00023.png", "(d)"),
]


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(10, 3.2), facecolor="white")

    for ax, (path, caption) in zip(axes, IMAGES):
        img = Image.open(path).convert("RGB")
        ax.imshow(np.array(img))
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#aaaaaa")
            spine.set_linewidth(0.8)
        ax.set_xlabel(caption, fontsize=13, fontweight="bold", labelpad=6)

    fig.tight_layout(pad=0.5)
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
