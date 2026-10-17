"""
Train ddpm_heavy_500_ensemble condition.

  Model A : loaded from config.PRETRAINED_MODEL_A  (no retraining)
  Model B : EfficientNetV2-S, standard aug on all real images
            + 500 synthetic unhealthy images from DDPM-heavy

Evaluates the ensemble (Model A + new Model B) on the full test set.

Usage:
    python train_ddpm_heavy_500_ensemble.py [--seed S]

Outputs:
  outputs/checkpoints/ddpm_heavy_500_ensemble/model_b_seed42.pth
  outputs/logs/ddpm_heavy_500_ensemble_seed42.csv
  outputs/metrics/ddpm_heavy_500_ensemble_seed42_metrics.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
    accuracy_score,
)
from tqdm import tqdm

import config
from dataset import (
    load_train_df, split_train_val,
    compute_class_weights,
    build_model_b_loaders_synth,
    build_test_loader,
)
from models.classifier import train_model_b, load_checkpoint


SYNTH_DIR  = config.GEN_IMAGE_DIR / "ddpm_heavy" / "unhealthy_500"
CKPT_DIR   = config.CHECKPOINT_DIR / "ddpm_heavy_500_ensemble"
METRICS_OUT = config.METRICS_DIR / "ddpm_heavy_500_ensemble_seed42_metrics.json"


# ---------------------------------------------------------------------------
# Ensemble evaluation (returns full metrics dict)
# ---------------------------------------------------------------------------
def evaluate_ensemble(model_a, model_b, test_loader, device) -> dict:
    model_a.eval(); model_b.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="  evaluating ensemble", leave=False):
            imgs = imgs.to(device)
            sa   = F.softmax(model_a(imgs), dim=1)
            sb   = F.softmax(model_b(imgs), dim=1)
            preds = ((sa + sb) / 2).argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    per_f1 = f1_score(all_labels, all_preds, average=None,
                      labels=list(range(config.NUM_CLASSES)), zero_division=0)
    wf1    = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    acc    = accuracy_score(all_labels, all_preds)
    cm     = confusion_matrix(all_labels, all_preds,
                              labels=list(range(config.NUM_CLASSES)))
    report = classification_report(all_labels, all_preds,
                                   target_names=config.CLASSES, zero_division=0)

    return {
        "condition":      "ddpm_heavy_500_ensemble",
        "weighted_f1":    round(float(wf1),      4),
        "accuracy":       round(float(acc),       4),
        "healthy_f1":     round(float(per_f1[0]), 4),
        "rubbish_f1":     round(float(per_f1[1]), 4),
        "unhealthy_f1":   round(float(per_f1[2]), 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


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
    print(f"Seed      : {SEED}")

    # Verify synthetic images
    synth_paths = sorted(SYNTH_DIR.glob("*.png"))
    if not synth_paths:
        raise FileNotFoundError(f"No PNGs found in {SYNTH_DIR}")
    print(f"Synthetic : {len(synth_paths)} images from {SYNTH_DIR}\n")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Data split ----
    df, class_to_idx = load_train_df()
    df_train, df_val = split_train_val(df, random_state=SEED)
    cw = compute_class_weights(df_train)
    cw_tensor = torch.tensor(cw)

    print(f"Train split : {len(df_train)} real images")
    print(f"Val split   : {len(df_val)} images")

    # ---- Build Model B loaders (std aug + 500 synthetic unhealthy) ----
    train_loader, val_loader = build_model_b_loaders_synth(
        df_train, df_val,
        sampler=None,
        synth_paths=[str(p) for p in synth_paths],
    )
    print(f"Model B train dataset : {len(train_loader.dataset)} samples "
          f"({len(df_train)} real + {len(synth_paths)} synthetic)\n")

    # ---- Train Model B ----
    ckpt_path = CKPT_DIR / f"model_b_seed{SEED}.pth"
    log_path  = config.LOG_DIR / f"ddpm_heavy_500_ensemble_seed{SEED}.csv"

    train_model_b(
        train_loader, val_loader, cw_tensor,
        ckpt_path=ckpt_path,
        log_path=log_path,
        seed=SEED,
    )

    # ---- Load Model A (pre-trained, unchanged) ----
    print(f"\nLoading Model A from {config.PRETRAINED_MODEL_A}")
    model_a = load_checkpoint(config.PRETRAINED_MODEL_A, device)

    # ---- Load freshly trained Model B ----
    model_b = load_checkpoint(ckpt_path, device)

    # ---- Evaluate ensemble on test set ----
    print("\nEvaluating ensemble on full test set …")
    test_loader, _ = build_test_loader(class_to_idx)
    metrics = evaluate_ensemble(model_a, model_b, test_loader, device)

    # ---- Save metrics ----
    METRICS_OUT.write_text(json.dumps(metrics, indent=2))

    # ---- Print summary ----
    print("\n" + "=" * 52)
    print("  ddpm_heavy_500_ensemble  results")
    print("-" * 52)
    print(f"  Weighted F1  : {metrics['weighted_f1']:.4f}")
    print(f"  Accuracy     : {metrics['accuracy']:.4f}")
    print(f"  Healthy  F1  : {metrics['healthy_f1']:.4f}")
    print(f"  Rubbish  F1  : {metrics['rubbish_f1']:.4f}")
    print(f"  Unhealthy F1 : {metrics['unhealthy_f1']:.4f}")
    print("=" * 52)
    print(f"\n{metrics['classification_report']}")
    print(f"Metrics saved → {METRICS_OUT}")


if __name__ == "__main__":
    main()
