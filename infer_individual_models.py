"""
Run inference for Model A and Model B individually on the test set.
Saves JSON and plots matching the exact schema/style of all other condition files.

Outputs:
  outputs/metrics/model_a_individual.json
  outputs/metrics/model_b_individual.json
  outputs/plots/model_a_confusion_matrix.png
  outputs/plots/model_a_metrics.png
  outputs/plots/model_b_confusion_matrix.png
  outputs/plots/model_b_metrics.png

Usage:
    python infer_individual_models.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    f1_score, precision_recall_fscore_support,
)
from tqdm import tqdm

import config
from dataset import build_test_loader, load_train_df
from models.classifier import load_checkpoint

CLASSES = config.CLASSES   # ["healthy", "rubbish", "unhealthy"]
config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict(model, loader, device, desc):
    model.eval()
    preds, labels = [], []
    for imgs, lbls in tqdm(loader, desc=f"  {desc}", leave=False):
        preds.extend(model(imgs.to(device)).argmax(1).cpu().numpy())
        labels.extend(lbls.numpy())
    return np.array(preds), np.array(labels)


# ---------------------------------------------------------------------------
# Metrics — exact same schema as baseline_ensemble_seed42_metrics.json
# ---------------------------------------------------------------------------
def build_metrics(y_true, y_pred, condition: str) -> dict:
    wf1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    mf1 = float(f1_score(y_true, y_pred, average="macro",    zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    cm  = confusion_matrix(y_true, y_pred,
                           labels=list(range(config.NUM_CLASSES))).tolist()
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred,
        labels=list(range(config.NUM_CLASSES)),
        zero_division=0,
    )
    uh = CLASSES.index("unhealthy")
    return {
        "weighted_f1":         round(wf1,         6),
        "macro_f1":            round(mf1,         6),
        "accuracy":            round(acc,         6),
        "per_class_precision": {c: round(float(p[i]), 6) for i, c in enumerate(CLASSES)},
        "per_class_recall":    {c: round(float(r[i]), 6) for i, c in enumerate(CLASSES)},
        "per_class_f1":        {c: round(float(f[i]), 6) for i, c in enumerate(CLASSES)},
        "confusion_matrix":    cm,
        "unhealthy_recall":    round(float(r[uh]), 6),
        "condition":           condition,
        "seed":                42,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def save_confusion_matrix(metrics: dict, title: str, path: Path):
    cm   = np.array(metrics["confusion_matrix"])
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(CLASSES, fontsize=10)
    ax.set_yticklabels(CLASSES, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True",      fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    for i in range(3):
        for j in range(3):
            colour = "white" if norm[i, j] > 0.55 else "black"
            ax.text(j, i,
                    f"{int(cm[i, j])}\n({norm[i, j]*100:.1f}%)",
                    ha="center", va="center",
                    fontsize=9, color=colour, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


def save_metrics_plot(metrics: dict, title: str, path: Path):
    x = np.arange(len(CLASSES))
    w = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # Left panel — overall metrics
    ax = axes[0]
    vals  = [metrics["weighted_f1"], metrics["macro_f1"], metrics["accuracy"]]
    clrs  = ["#1565C0", "#0288D1", "#00897B"]
    xlbls = ["Weighted F1", "Macro F1", "Accuracy"]
    bars  = ax.bar(np.arange(3), vals, color=clrs, alpha=0.88, width=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.005,
                f"{v:.4f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(xlbls, fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("Overall Metrics", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    # Right panel — per-class P / R / F1
    ax = axes[1]
    prec = [metrics["per_class_precision"][c] for c in CLASSES]
    rec  = [metrics["per_class_recall"][c]    for c in CLASSES]
    f1   = [metrics["per_class_f1"][c]        for c in CLASSES]
    for off, vals2, col, lbl in zip(
        [-w, 0, w],
        [prec, rec, f1],
        ["#43A047", "#FB8C00", "#E53935"],
        ["Precision", "Recall", "F1"],
    ):
        bars = ax.bar(x + off, vals2, w, color=col, alpha=0.88, label=lbl)
        for b, v in zip(bars, vals2):
            if v > 0.01:
                ax.text(b.get_x() + b.get_width()/2, v + 0.005,
                        f"{v:.3f}", ha="center", va="bottom",
                        fontsize=8, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, fontsize=10)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("Per-class Precision / Recall / F1", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_one(name, ckpt, condition, device, loader):
    print(f"\n--- {name} ---")
    model   = load_checkpoint(ckpt, device)
    preds, labels = predict(model, loader, device, name)
    m = build_metrics(labels, preds, condition)

    json_path = config.METRICS_DIR / f"{condition}.json"
    json_path.write_text(json.dumps(m, indent=2))
    print(f"  JSON  → {json_path}")
    print(f"  wF1={m['weighted_f1']:.4f}  "
          f"macro_f1={m['macro_f1']:.4f}  "
          f"acc={m['accuracy']:.4f}  "
          f"unhealthy_F1={m['per_class_f1']['unhealthy']:.4f}")

    save_confusion_matrix(m, f"{name} — Confusion Matrix",
                          config.PLOTS_DIR / f"{condition}_confusion_matrix.png")
    save_metrics_plot(m, f"{name} — Individual Inference",
                      config.PLOTS_DIR / f"{condition}_metrics.png")
    return m


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    _, class_to_idx = load_train_df()
    loader, _       = build_test_loader(class_to_idx)
    print(f"Test set : {len(loader.dataset)} images")

    ma = run_one("Model A (pretrained)", config.PRETRAINED_MODEL_A,
                 "model_a_individual", device, loader)
    mb = run_one("Model B (pretrained)", config.PRETRAINED_MODEL_B,
                 "model_b_individual", device, loader)

    print("\n" + "="*50)
    print(f"  Model A  unhealthy-F1 = {ma['per_class_f1']['unhealthy']:.4f}")
    print(f"  Model B  unhealthy-F1 = {mb['per_class_f1']['unhealthy']:.4f}")
    print("="*50)


if __name__ == "__main__":
    main()
