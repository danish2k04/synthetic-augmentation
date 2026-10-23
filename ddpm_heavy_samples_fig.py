"""
Figure 3: 4 selected heavy-DDPM generated samples in a single row.

Output: outputs/plots/ddpm_heavy_samples_fig3.png  (300 DPI)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

SAMPLE_DIR = Path("outputs/generated_images/ddpm_heavy/unhealthy_500")
IMAGES     = ["00105.png", "00497.png", "00008.png", "00334.png"]
OUT_PATH   = Path("outputs/plots/ddpm_heavy_samples_fig3.png")

def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5), facecolor="white")
    fig.suptitle("Figure 3: 128×128 DDPM Generated Samples",
                 fontsize=13, fontweight="bold", y=1.02)

    for ax, fname in zip(axes, IMAGES):
        img = Image.open(SAMPLE_DIR / fname).convert("RGB")
        ax.imshow(np.array(img))
        for spine in ax.spines.values():
            spine.set_edgecolor("#888888")
            spine.set_linewidth(1.0)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout(pad=0.5)
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {OUT_PATH}")

if __name__ == "__main__":
    main()
