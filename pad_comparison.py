"""
Visualise the PadToSquareWhite transform.

Picks one unhealthy cell image, shows Original vs After PadToSquareWhite
side-by-side with thin grey borders so both shapes are visible.

Output: outputs/plots/pad_comparison.png  (300 DPI)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
UNHEALTHY_DIR = Path(
    r"C:\Users\danish\Desktop\pap_cell_classification\data"
    r"\isbi2025-ps3c-train-dataset\unhealthy"
)
OUT_PATH = Path("outputs/plots/pad_comparison.png")

# ---------------------------------------------------------------------------
# Padding logic: white-fill shorter side, keep cell centred
# ---------------------------------------------------------------------------
def pad_to_square_white(img: Image.Image) -> Image.Image:
    w, h = img.size
    side  = max(w, h)
    result = Image.new("RGB", (side, side), (255, 255, 255))
    left  = (side - w) // 2
    top   = (side - h) // 2
    result.paste(img, (left, top))
    return result

# ---------------------------------------------------------------------------
# Pick a sample that is actually non-square (more interesting to show)
# ---------------------------------------------------------------------------
def pick_nonsquare(folder: Path, fallback_first=True) -> Path:
    for p in sorted(folder.glob("*.png")):
        w, h = Image.open(p).size
        if w != h:
            return p
    # all square — just return first one
    return next(folder.glob("*.png"))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    img_path = pick_nonsquare(UNHEALTHY_DIR)
    original = Image.open(img_path).convert("RGB")
    padded   = pad_to_square_white(original)

    orig_arr = np.array(original)
    pad_arr  = np.array(padded)

    ow, oh = original.size   # width, height
    pw, ph = padded.size

    fig, axes = plt.subplots(1, 2, figsize=(9, 5),
                             facecolor="white")
    fig.suptitle("Figure 2: PadToSquareWhite Transform",
                 fontsize=13, fontweight="bold", y=1.01)

    for ax, arr, title, dims in [
        (axes[0], orig_arr,  "Original",               f"{ow} × {oh} px"),
        (axes[1], pad_arr,   "After PadToSquareWhite", f"{pw} × {ph} px"),
    ]:
        ax.imshow(arr)
        # thin grey border so extent of image is visible against white page
        for spine in ax.spines.values():
            spine.set_edgecolor("#888888")
            spine.set_linewidth(1.2)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{title}\n{dims}", fontsize=11, fontweight="bold", pad=6)

    fig.tight_layout(pad=1.5)
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {OUT_PATH}")
    print(f"  Source : {img_path.name}")
    print(f"  Original size : {ow} × {oh}")
    print(f"  Padded size   : {pw} × {ph}")

if __name__ == "__main__":
    main()
