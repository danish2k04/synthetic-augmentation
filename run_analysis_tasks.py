"""
Three-task analysis — inference only, no training.

Task 1 — Individual model inference: Model A alone, Model B alone.
Task 2 — Enlarged ensemble (4 models, softmax-averaged):
            1. Model A          (config.PRETRAINED_MODEL_A)
            2. Model B original (config.PRETRAINED_MODEL_B, heavy aug)
            3. Model B ddpm-heavy-500
            4. Model B combined (ddpm-500 + heavy aug)
          NOTE: Model B checkpoints for gan_500 and ddpm_500 conditions were
          not saved to disk during their training runs, so they cannot be
          included without retraining.
Task 3 — Four comparison plots.

Usage:
    python run_analysis_tasks.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    f1_score, precision_recall_fscore_support,
)
from tqdm import tqdm

import config
from dataset import build_test_loader, load_train_df
from models.classifier import load_checkpoint

# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
CKPT_A       = config.PRETRAINED_MODEL_A
CKPT_B       = config.PRETRAINED_MODEL_B
CKPT_B_GAN500  = config.CHECKPOINT_DIR / "gan_500_ensemble_seed42_modelB.pt"
CKPT_B_DDPM500 = config.CHECKPOINT_DIR / "ddpm_500_ensemble_seed42_modelB.pt"
CKPT_B_DDPH    = config.CHECKPOINT_DIR / "ddpm_heavy_500_ensemble" / "model_b_seed42.pth"

config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
config.METRICS_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = config.CLASSES   # ["healthy", "rubbish", "unhealthy"]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def metrics_dict(y_true, y_pred, condition: str, seed: int = 42) -> dict:
    wf1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    mf1 = float(f1_score(y_true, y_pred, average="macro",    zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    cm  = confusion_matrix(y_true, y_pred,
                           labels=list(range(config.NUM_CLASSES))).tolist()
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(config.NUM_CLASSES)), zero_division=0)
    uh_idx = CLASSES.index("unhealthy")
    # Key order matches existing condition files exactly
    return {
        "weighted_f1":         round(wf1, 6),
        "macro_f1":            round(mf1, 6),
        "accuracy":            round(acc, 6),
        "per_class_precision": {c: round(float(p[i]), 6) for i, c in enumerate(CLASSES)},
        "per_class_recall":    {c: round(float(r[i]), 6) for i, c in enumerate(CLASSES)},
        "per_class_f1":        {c: round(float(f[i]), 6) for i, c in enumerate(CLASSES)},
        "confusion_matrix":    cm,
        "unhealthy_recall":    round(float(r[uh_idx]), 6),
        "condition":           condition,
        "seed":                seed,
    }


# unified getters — handle both nested and flat JSON schemas
def _wf1(m):        return float(m.get("weighted_f1", 0))
def _mf1(m):        return float(m.get("macro_f1",    0))
def _acc(m):        return float(m.get("accuracy",    0))
def _cm(m):         return np.array(m.get("confusion_matrix", [[0]*3]*3))

def _cls_f1(m, cls):
    v = m.get("per_class_f1", {}).get(cls)
    return float(v) if v is not None else float(m.get(f"{cls}_f1", 0))

def _cls_p(m, cls):
    v = m.get("per_class_precision", {}).get(cls)
    if v is not None: return float(v)
    import re
    match = re.search(rf"\s{cls}\s+([\d.]+)", m.get("classification_report", ""))
    return float(match.group(1)) if match else 0.0

def _cls_r(m, cls):
    v = m.get("per_class_recall", {}).get(cls)
    if v is not None: return float(v)
    import re
    match = re.search(rf"\s{cls}\s+[\d.]+\s+([\d.]+)", m.get("classification_report", ""))
    return float(match.group(1)) if match else 0.0


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_single(model, loader, device, desc="predicting"):
    model.eval()
    preds, labels = [], []
    for imgs, lbls in tqdm(loader, desc=f"  {desc}", leave=False):
        preds.extend(model(imgs.to(device)).argmax(1).cpu().numpy())
        labels.extend(lbls.numpy())
    return np.array(preds), np.array(labels)


@torch.no_grad()
def predict_ensemble(models, loader, device):
    for m in models: m.eval()
    preds, labels = [], []
    for imgs, lbls in tqdm(loader, desc="  ensemble", leave=False):
        imgs = imgs.to(device)
        avg  = sum(F.softmax(m(imgs), dim=1) for m in models) / len(models)
        preds.extend(avg.argmax(1).cpu().numpy())
        labels.extend(lbls.numpy())
    return np.array(preds), np.array(labels)


# ---------------------------------------------------------------------------
# Load existing condition JSONs
# ---------------------------------------------------------------------------
def load_conditions() -> dict:
    results = {}
    for path in sorted(config.METRICS_DIR.glob("*_metrics.json")):
        if "_seed" in path.stem:
            cond = path.stem[: -len("_metrics")].rsplit("_seed", 1)[0]
        elif path.stem == "pretrained_ensemble_metrics":
            cond = "pretrained_ensemble"
        else:
            continue
        try:
            results[cond] = json.loads(path.read_text())
        except Exception:
            pass
    return results


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def _bar(ax, x, groups, colours, leg_labels, w=0.32, ylim=(0, 1.10),
         ylabel="Score", val_fs=8):
    n = len(groups)
    offs = np.linspace(-(n-1)*w/2, (n-1)*w/2, n)
    for off, vals, col, lbl in zip(offs, groups, colours, leg_labels):
        bars = ax.bar(x + off, vals, w, color=col, alpha=0.88, label=lbl)
        for b in bars:
            h = b.get_height()
            if h > 0.005:
                ax.text(b.get_x() + b.get_width()/2, h + 0.006,
                        f"{h:.3f}", ha="center", va="bottom",
                        fontsize=val_fs, rotation=90)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def _xticks(ax, x, labels, rot=30):
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rot, ha="right", fontsize=8.5)


# ===========================================================================
# TASK 1
# ===========================================================================
def _save_confusion_matrix(m: dict, title: str, out: Path):
    cm   = _cm(m)
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(CLASSES, fontsize=9)
    ax.set_yticklabels(CLASSES, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("True",      fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    for i in range(3):
        for j in range(3):
            colour = "white" if norm[i, j] > 0.55 else "black"
            ax.text(j, i, f"{int(cm[i,j])}\n({norm[i,j]*100:.1f}%)",
                    ha="center", va="center", fontsize=9,
                    color=colour, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


def _save_bar_plot(m: dict, title: str, out: Path):
    """Weighted F1, macro F1, accuracy + per-class F1/P/R in two panels."""
    classes = CLASSES
    x = np.arange(len(classes))
    w = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    # Left: overall metrics
    ax = axes[0]
    overall = [_wf1(m), _mf1(m), _acc(m)]
    cols    = ["#1565C0", "#0288D1", "#00897B"]
    lbls    = ["Weighted F1", "Macro F1", "Accuracy"]
    bars = ax.bar(np.arange(3), overall, color=cols, alpha=0.88, width=0.5)
    for b, v in zip(bars, overall):
        ax.text(b.get_x() + b.get_width()/2, v + 0.005,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(lbls, fontsize=9)
    ax.set_ylim(0, 1.12); ax.set_ylabel("Score", fontsize=9)
    ax.set_title("Overall Metrics", fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.3); ax.spines[["top","right"]].set_visible(False)

    # Right: per-class P / R / F1
    ax = axes[1]
    prec = [_cls_p(m, c) for c in classes]
    rec  = [_cls_r(m, c) for c in classes]
    f1   = [_cls_f1(m, c) for c in classes]
    offs = [-w, 0, w]
    for off, vals, col, lbl in zip(offs,
                                   [prec, rec, f1],
                                   ["#43A047","#FB8C00","#E53935"],
                                   ["Precision","Recall","F1"]):
        bars = ax.bar(x + off, vals, w, color=col, alpha=0.88, label=lbl)
        for b, v in zip(bars, vals):
            if v > 0.01:
                ax.text(b.get_x() + b.get_width()/2, v + 0.005,
                        f"{v:.3f}", ha="center", va="bottom",
                        fontsize=7.5, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(classes, fontsize=9)
    ax.set_ylim(0, 1.18); ax.set_ylabel("Score", fontsize=9)
    ax.set_title("Per-class Precision / Recall / F1", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


def task1(device, loader):
    print("\n=== TASK 1: Individual model inference ===")

    ma_model = load_checkpoint(CKPT_A, device)
    mb_model = load_checkpoint(CKPT_B, device)

    pa, la = predict_single(ma_model, loader, device, "Model A")
    ma = metrics_dict(la, pa, "model_a_individual")
    (config.METRICS_DIR / "model_a_individual.json").write_text(json.dumps(ma, indent=2))
    print(f"  Model A   wF1={ma['weighted_f1']:.4f}  unhealthy-F1={_cls_f1(ma,'unhealthy'):.4f}")
    _save_confusion_matrix(ma, "Model A — Confusion Matrix",
                           config.PLOTS_DIR / "model_a_confusion_matrix.png")
    _save_bar_plot(ma, "Model A — Individual Inference",
                   config.PLOTS_DIR / "model_a_metrics.png")

    pb, lb = predict_single(mb_model, loader, device, "Model B")
    mb = metrics_dict(lb, pb, "model_b_individual")
    (config.METRICS_DIR / "model_b_individual.json").write_text(json.dumps(mb, indent=2))
    print(f"  Model B   wF1={mb['weighted_f1']:.4f}  unhealthy-F1={_cls_f1(mb,'unhealthy'):.4f}")
    _save_confusion_matrix(mb, "Model B — Confusion Matrix",
                           config.PLOTS_DIR / "model_b_confusion_matrix.png")
    _save_bar_plot(mb, "Model B — Individual Inference",
                   config.PLOTS_DIR / "model_b_metrics.png")

    return ma_model, mb_model, ma, mb


# ===========================================================================
# TASK 2
# ===========================================================================
def task2(ma_model, mb_model, device, loader):
    print("\n=== TASK 2: Enlarged 5-model ensemble ===")

    mb_gan500  = load_checkpoint(CKPT_B_GAN500,  device)
    mb_ddpm500 = load_checkpoint(CKPT_B_DDPM500, device)
    mb_ddph    = load_checkpoint(CKPT_B_DDPH,    device)

    members = [
        (ma_model,   "Model A (std aug)"),
        (mb_model,   "Model B orig (heavy aug)"),
        (mb_gan500,  "Model B gan-500"),
        (mb_ddpm500, "Model B ddpm-500"),
        (mb_ddph,    "Model B ddpm-heavy-500"),
    ]
    print("  Members:")
    for _, n in members: print(f"    • {n}")

    pe, le = predict_ensemble([m for m, _ in members], loader, device)
    me = metrics_dict(le, pe, "enlarged_ensemble_5model")
    me["ensemble_members"] = [n for _, n in members]
    (config.METRICS_DIR / "enlarged_ensemble_5model.json").write_text(
        json.dumps(me, indent=2))
    print(f"  Ensemble  wF1={me['weighted_f1']:.4f}  "
          f"unhealthy-F1={_cls_f1(me,'unhealthy'):.4f}")
    return me


# ===========================================================================
# TASK 3
# ===========================================================================
def task3(ma, mb, me, conds):
    print("\n=== TASK 3: Plots ===")

    # ── Plot 1: individual vs ensemble ──────────────────────────────────────
    entries = [
        ("Model A",        ma),
        ("Model B",        mb),
        ("Baseline\n2-model",       conds.get("baseline_ensemble", {})),
        ("Enlarged\n5-model",       me),
    ]
    xlbls = [e[0] for e in entries]
    x     = np.arange(len(xlbls))
    wf1   = [_wf1(e[1]) for e in entries]
    uf1   = [_cls_f1(e[1], "unhealthy") for e in entries]

    fig, ax = plt.subplots(figsize=(10, 6))
    _bar(ax, x, [wf1, uf1], ["#1565C0", "#E53935"],
         ["Weighted F1", "Unhealthy F1"], w=0.30)
    _xticks(ax, x, xlbls, rot=0)
    ax.set_title("Individual Models vs Ensemble Comparison",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    p = config.PLOTS_DIR / "individual_vs_ensemble_comparison.png"
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  → {p}")

    # ── Plot 2: enlarged vs all 7 conditions ────────────────────────────────
    COND_ORDER = [
        ("baseline_ensemble",        "Baseline"),
        ("heavy_aug_ensemble",       "Heavy Aug"),
        ("gan_250_ensemble",         "GAN-250"),
        ("gan_500_ensemble",         "GAN-500"),
        ("ddpm_250_ensemble",        "DDPM-250"),
        ("ddpm_500_ensemble",        "DDPM-500"),
        ("ddpm_heavy_500_ensemble",  "DDPM-H-500"),
        ("__enlarged__",             "Enlarged\n5-model"),
    ]
    all_data = {**conds, "__enlarged__": me}
    rows     = [(lbl, all_data[k]) for k, lbl in COND_ORDER if k in all_data]
    xlbls2   = [r[0] for r in rows]
    x2       = np.arange(len(xlbls2))
    wf1_2    = [_wf1(r[1])              for r in rows]
    uf1_2    = [_cls_f1(r[1],"unhealthy") for r in rows]
    n_conds  = len(rows) - 1   # last entry is enlarged

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    for ax, vals, base_col, hi_col, title, ylim in [
        (axes[0], wf1_2, "#1565C0", "#0D47A1", "Weighted F1",  (0.82, 0.88)),
        (axes[1], uf1_2, "#E53935", "#B71C1C", "Unhealthy F1", (0.28, 0.62)),
    ]:
        colours = [base_col] * n_conds + [hi_col]
        bars = ax.bar(x2, vals, color=colours, alpha=0.88, width=0.55)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 0.002,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        _xticks(ax, x2, xlbls2, rot=30)
        ax.set_ylim(*ylim)
        ax.set_ylabel("Score", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linewidth=0.7)
        ax.spines[["top","right"]].set_visible(False)
        ax.legend(handles=[
            mpatches.Patch(color=base_col, label="2-model condition"),
            mpatches.Patch(color=hi_col,   label="Enlarged 5-model"),
        ], fontsize=8)
    fig.suptitle("Enlarged 5-Model Ensemble vs All 2-Model Conditions",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = config.PLOTS_DIR / "enlarged_ensemble_vs_conditions.png"
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  → {p}")

    # ── Plot 3: all unhealthy F1 sorted ─────────────────────────────────────
    all_uf1 = {
        "Model A":   _cls_f1(ma, "unhealthy"),
        "Model B":   _cls_f1(mb, "unhealthy"),
        "Enlarged 5-model":  _cls_f1(me, "unhealthy"),
    }
    COND_LABELS = {
        "baseline_ensemble":        "Baseline 2-model",
        "heavy_aug_ensemble":       "Heavy Aug",
        "gan_250_ensemble":         "GAN-250",
        "gan_500_ensemble":         "GAN-500",
        "ddpm_250_ensemble":        "DDPM-250",
        "ddpm_500_ensemble":        "DDPM-500",
        "ddpm_heavy_500_ensemble":  "DDPM-H-500",
    }
    for k, lbl in COND_LABELS.items():
        if k in conds:
            all_uf1[lbl] = _cls_f1(conds[k], "unhealthy")

    sorted_entries = sorted(all_uf1.items(), key=lambda x: x[1], reverse=True)
    xlbls3 = [e[0] for e in sorted_entries]
    vals3  = [e[1] for e in sorted_entries]
    cols3  = ["#B71C1C" if lbl == "Enlarged 5-model" else "#1565C0"
              for lbl in xlbls3]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(np.arange(len(xlbls3)), vals3, color=cols3, alpha=0.88, width=0.6)
    for b, v in zip(bars, vals3):
        ax.text(b.get_x() + b.get_width()/2, v + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    _xticks(ax, np.arange(len(xlbls3)), xlbls3, rot=35)
    ax.set_ylim(0, max(vals3) * 1.20)
    ax.set_ylabel("Unhealthy F1", fontsize=10)
    ax.set_title("Unhealthy-Class F1 — All Models & Conditions (sorted descending)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linewidth=0.7)
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(handles=[
        mpatches.Patch(color="#1565C0", label="Single / 2-model ensemble"),
        mpatches.Patch(color="#B71C1C", label="Enlarged 5-model ensemble"),
    ], fontsize=9)
    fig.tight_layout()
    p = config.PLOTS_DIR / "all_models_unhealthy_f1.png"
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  → {p}")

    # ── Plot 4: confusion matrices 2×2 ──────────────────────────────────────
    panels = [
        ("Model A",      _cm(ma)),
        ("Model B",      _cm(mb)),
        ("Baseline 2-model",     _cm(conds.get("baseline_ensemble", {}))),
        ("Enlarged 5-model",     _cm(me)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")
    for ax, (title, cm) in zip(axes.flat, panels):
        row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
        norm = cm / row_sums
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(CLASSES, fontsize=9)
        ax.set_yticklabels(CLASSES, fontsize=9)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("True",      fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        for i in range(3):
            for j in range(3):
                colour = "white" if norm[i, j] > 0.55 else "black"
                ax.text(j, i, f"{int(cm[i,j])}\n({norm[i,j]*100:.1f}%)",
                        ha="center", va="center", fontsize=8.5,
                        color=colour, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    p = config.PLOTS_DIR / "confusion_matrices.png"
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  → {p}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    _, class_to_idx = load_train_df()
    loader, _       = build_test_loader(class_to_idx)
    print(f"Test set : {len(loader.dataset)} images")

    conds = load_conditions()
    print(f"Loaded {len(conds)} condition files: {', '.join(sorted(conds))}\n")

    ma_model, mb_model, ma, mb = task1(device, loader)
    me                          = task2(ma_model, mb_model, device, loader)
    task3(ma, mb, me, conds)

    print("\n" + "="*54)
    print("  SUMMARY")
    print("-"*54)
    print(f"  Model A            unhealthy-F1 = {_cls_f1(ma,'unhealthy'):.4f}")
    print(f"  Model B            unhealthy-F1 = {_cls_f1(mb,'unhealthy'):.4f}")
    print(f"  Enlarged 5-model   unhealthy-F1 = {_cls_f1(me,'unhealthy'):.4f}")
    print("="*54)


if __name__ == "__main__":
    main()
