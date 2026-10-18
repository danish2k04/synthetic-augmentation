"""
Train combined condition: DDPM-light-500 synthetic images + heavy augmentation.

  Model A : loaded from config.PRETRAINED_MODEL_A  (not retrained)
  Model B : EfficientNetV2-S
              - real images : dual aug (heavy for unhealthy, standard for others)
              - synthetic   : 500 DDPM-light unhealthy PNGs, standard aug
              - loader      : build_model_b_loaders_synth_heavy()

Usage:
    python train_combined_ddpm500_heavy.py [--seed S]

Outputs:
  outputs/checkpoints/combined_ddpm500_heavy_aug/model_b_seed42.pth
  outputs/logs/combined_ddpm500_heavy_aug_ensemble_seed42_modelB_log.csv
  outputs/metrics/combined_ddpm500_heavy_aug_ensemble_seed42_metrics.json
  outputs/plots/combined_ddpm500_heavy_aug_ensemble_seed42_curves.png
  outputs/metrics/comparison_table.csv  (row appended)
"""
import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    f1_score, precision_recall_fscore_support,
)
from tqdm import tqdm

import config
from dataset import (
    build_model_b_loaders_synth_heavy,
    build_test_loader,
    compute_class_weights,
    load_train_df,
    split_train_val,
)
from models.classifier import load_checkpoint, train_model_b

CONDITION  = "combined_ddpm500_heavy_aug_ensemble"
SYNTH_DIR  = config.GEN_IMAGE_DIR / "ddpm" / "unhealthy_500"
CKPT_PATH  = config.CHECKPOINT_DIR / "combined_ddpm500_heavy_aug" / "model_b_seed42.pth"
LOG_PATH   = config.LOG_DIR  / f"{CONDITION}_seed42_modelB_log.csv"
METRICS_PATH = config.METRICS_DIR / f"{CONDITION}_seed42_metrics.json"
CURVES_PATH  = config.PLOTS_DIR   / f"{CONDITION}_seed42_curves.png"
TABLE_PATH   = config.METRICS_DIR / "comparison_table.csv"


# ---------------------------------------------------------------------------
# Ensemble evaluation
# ---------------------------------------------------------------------------
def evaluate_ensemble(model_a, model_b, test_loader, device) -> dict:
    model_a.eval(); model_b.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="  evaluating", leave=False):
            imgs  = imgs.to(device)
            probs = (F.softmax(model_a(imgs), dim=1) +
                     F.softmax(model_b(imgs), dim=1)) / 2
            all_preds.extend(probs.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    classes = config.CLASSES   # ["healthy", "rubbish", "unhealthy"]

    wf1   = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    mf1   = float(f1_score(y_true, y_pred, average="macro",    zero_division=0))
    acc   = float(accuracy_score(y_true, y_pred))
    cm    = confusion_matrix(y_true, y_pred,
                             labels=list(range(config.NUM_CLASSES))).tolist()

    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred,
        labels=list(range(config.NUM_CLASSES)),
        zero_division=0,
    )

    return {
        "condition":          CONDITION,
        "seed":               42,
        "weighted_f1":        round(wf1, 6),
        "macro_f1":           round(mf1, 6),
        "accuracy":           round(acc, 6),
        "per_class_precision": {c: round(float(p[i]), 6) for i, c in enumerate(classes)},
        "per_class_recall":    {c: round(float(r[i]), 6) for i, c in enumerate(classes)},
        "per_class_f1":        {c: round(float(f[i]), 6) for i, c in enumerate(classes)},
        "confusion_matrix":   cm,
    }


# ---------------------------------------------------------------------------
# Training-curve plot
# ---------------------------------------------------------------------------
def plot_curves(log_path: Path, out_path: Path):
    rows = []
    with open(log_path, newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append({k: float(v) if k not in ("phase",) else v
                         for k, v in row.items()})

    epochs     = [r["epoch"]     for r in rows]
    train_loss = [r["train_loss"] for r in rows]
    val_loss   = [r["val_loss"]   for r in rows]
    val_wf1    = [r["val_wf1"]    for r in rows]
    lr         = [r["lr"]         for r in rows]
    phases     = [r["phase"]      for r in rows]

    # Mark phase boundary (first phase2 row)
    phase2_start = next((r["epoch"] for r in rows if r["phase"] == "phase2"), None)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"{CONDITION}\nTraining curves (seed 42)", fontsize=11, y=1.01)

    for ax, y, ylabel, colour, title in [
        (axes[0], train_loss, "Loss",        "#1565C0", "Train Loss"),
        (axes[1], val_loss,   "Loss",        "#E53935", "Val Loss"),
        (axes[2], val_wf1,    "Weighted F1", "#2E7D32", "Val Weighted F1"),
    ]:
        ax.plot(epochs, y, color=colour, linewidth=1.5)
        if phase2_start:
            ax.axvline(phase2_start, color="grey", linestyle="--",
                       linewidth=0.9, label=f"Phase 2 start (ep {int(phase2_start)})")
            ax.legend(fontsize=8)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Curves  → {out_path}")


# ---------------------------------------------------------------------------
# Append row to comparison_table.csv
# ---------------------------------------------------------------------------
def append_to_table(metrics: dict, training_minutes: float):
    row = {
        "Condition":    CONDITION,
        "wF1":          f"{metrics['weighted_f1']:.4f}",
        "mF1":          f"{metrics['macro_f1']:.4f}",
        "Acc":          f"{metrics['accuracy']:.4f}",
        "Unhealthy P":  f"{metrics['per_class_precision']['unhealthy']:.4f}",
        "Unhealthy R":  f"{metrics['per_class_recall']['unhealthy']:.4f}",
        "Unhealthy F1": f"{metrics['per_class_f1']['unhealthy']:.4f}",
        "Time(min)":    f"{training_minutes:.2f}",
    }

    header = ["Condition", "wF1", "mF1", "Acc",
              "Unhealthy P", "Unhealthy R", "Unhealthy F1", "Time(min)"]

    # Read existing table (if any) so we don't duplicate
    existing_conditions = set()
    existing_rows = []
    if TABLE_PATH.exists():
        with open(TABLE_PATH, newline="") as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                existing_conditions.add(r.get("Condition", ""))
                existing_rows.append(r)

    write_header = not TABLE_PATH.exists()
    mode = "a" if TABLE_PATH.exists() else "w"

    if CONDITION in existing_conditions:
        # Overwrite the existing row
        updated = [row if r["Condition"] == CONDITION else r
                   for r in existing_rows]
        with open(TABLE_PATH, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=header)
            w.writeheader()
            w.writerows(updated)
    else:
        with open(TABLE_PATH, mode, newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=header)
            if write_header:
                w.writeheader()
            w.writerow(row)

    print(f"Table   → {TABLE_PATH}  (row {'updated' if CONDITION in existing_conditions else 'appended'})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    SEED = args.seed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device    : {device}")
    print(f"Condition : {CONDITION}")
    print(f"Seed      : {SEED}")

    synth_paths = sorted(SYNTH_DIR.glob("*.png"))
    if not synth_paths:
        raise FileNotFoundError(f"No PNGs in {SYNTH_DIR}")
    print(f"Synthetic : {len(synth_paths)} images  ({SYNTH_DIR})\n")

    for p in [config.CHECKPOINT_DIR, config.LOG_DIR,
              config.METRICS_DIR, config.PLOTS_DIR]:
        p.mkdir(parents=True, exist_ok=True)
    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ---- Data split ----
    df, class_to_idx = load_train_df()
    df_train, df_val  = split_train_val(df, random_state=SEED)
    cw_tensor = torch.tensor(compute_class_weights(df_train))

    print(f"Train split : {len(df_train)} real images")
    print(f"Val split   : {len(df_val)}   images")

    # ---- Loaders (dual aug on real + std aug on synthetic) ----
    train_loader, val_loader = build_model_b_loaders_synth_heavy(
        df_train, df_val,
        synth_paths=[str(p) for p in synth_paths],
    )
    print(f"Train dataset : {len(train_loader.dataset)} samples "
          f"({len(df_train)} real + {len(synth_paths)} synthetic)\n")

    # ---- Train Model B ----
    t0 = time.time()
    train_model_b(
        train_loader, val_loader, cw_tensor,
        ckpt_path=CKPT_PATH,
        log_path=LOG_PATH,
        seed=SEED,
    )
    training_minutes = (time.time() - t0) / 60
    print(f"\nTraining time : {training_minutes:.1f} min")

    # ---- Load Model A (pre-trained, fixed) ----
    print(f"\nLoading Model A from {config.PRETRAINED_MODEL_A}")
    model_a = load_checkpoint(config.PRETRAINED_MODEL_A, device)

    # ---- Load freshly trained Model B ----
    model_b = load_checkpoint(CKPT_PATH, device)

    # ---- Evaluate ensemble on test set ----
    print("\nEvaluating ensemble on full test set …")
    test_loader, _ = build_test_loader(class_to_idx)
    metrics = evaluate_ensemble(model_a, model_b, test_loader, device)
    metrics["training_time_minutes"] = round(training_minutes, 2)

    # ---- Save metrics JSON ----
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Metrics → {METRICS_PATH}")

    # ---- Plot training curves ----
    plot_curves(LOG_PATH, CURVES_PATH)

    # ---- Append to comparison table ----
    append_to_table(metrics, training_minutes)

    # ---- Summary ----
    print("\n" + "=" * 54)
    print(f"  {CONDITION}")
    print("-" * 54)
    print(f"  Weighted F1  : {metrics['weighted_f1']:.4f}")
    print(f"  Macro F1     : {metrics['macro_f1']:.4f}")
    print(f"  Accuracy     : {metrics['accuracy']:.4f}")
    print(f"  Healthy  F1  : {metrics['per_class_f1']['healthy']:.4f}")
    print(f"  Rubbish  F1  : {metrics['per_class_f1']['rubbish']:.4f}")
    print(f"  Unhealthy F1 : {metrics['per_class_f1']['unhealthy']:.4f}")
    print(f"  Unhealthy P  : {metrics['per_class_precision']['unhealthy']:.4f}")
    print(f"  Unhealthy R  : {metrics['per_class_recall']['unhealthy']:.4f}")
    print("=" * 54)


if __name__ == "__main__":
    main()
