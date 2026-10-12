"""
Downstream F1 comparison: does the heavy DDPM improve classification?

Trains one fresh Model B per DDPM variant using 500 synthetic unhealthy images,
then evaluates the two-model ensemble (pre-trained Model A + new Model B) on the
full test set and prints per-class F1.

Conditions evaluated:
  ddpm_lightweight_500 — Model B + 500 lightweight DDPM images
  ddpm_heavy_500       — Model B + 500 heavy DDPM images

Model A is the same for both (loaded from config.PRETRAINED_MODEL_A).

Usage:
    python run_downstream_f1.py [--seed S]

Pre-requisites:
  outputs/generated_images/ddpm/unhealthy_500/       (from compare_ddpm.py or generate_eval.py)
  outputs/generated_images/ddpm_heavy/unhealthy_500/ (from compare_ddpm.py)
  config.PRETRAINED_MODEL_A                          (pre-trained Model A checkpoint)

Outputs:
  outputs/checkpoints/ddpm_downstream/model_b_light.pth
  outputs/checkpoints/ddpm_downstream/model_b_heavy.pth
  outputs/metrics/downstream_f1_comparison.json
  outputs/logs/downstream_light_log.csv
  outputs/logs/downstream_heavy_log.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix
)

import config
from dataset import (
    load_train_df, load_test_df, split_train_val,
    compute_class_weights,
    build_model_b_loaders_synth, build_test_loader,
)
from models.classifier import train_model_b, load_checkpoint
from models.ensemble import ensemble_predict


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LIGHT_SYNTH_DIR = config.GEN_IMAGE_DIR / "ddpm"       / "unhealthy_500"
HEAVY_SYNTH_DIR = config.GEN_IMAGE_DIR / "ddpm_heavy" / "unhealthy_500"

CKPT_DIR = config.CHECKPOINT_DIR / "ddpm_downstream"
LOG_DIR  = config.LOG_DIR

CKPT_LIGHT = CKPT_DIR / "model_b_light.pth"
CKPT_HEAVY = CKPT_DIR / "model_b_heavy.pth"

LOG_LIGHT  = LOG_DIR / "downstream_light_log.csv"
LOG_HEAVY  = LOG_DIR / "downstream_heavy_log.csv"


# ---------------------------------------------------------------------------
# Evaluate ensemble on test set, return per-class and weighted F1
# ---------------------------------------------------------------------------
def evaluate(model_a: torch.nn.Module, model_b: torch.nn.Module,
             test_loader, device) -> dict:
    # ensemble_predict takes the full loader and returns (preds, labels) arrays
    all_preds, all_labels = ensemble_predict(model_a, model_b, test_loader, device)
    wf1  = f1_score(all_labels, all_preds, average="weighted",      zero_division=0)
    per  = f1_score(all_labels, all_preds, average=None,            zero_division=0,
                    labels=list(range(config.NUM_CLASSES)))
    cm   = confusion_matrix(all_labels, all_preds,
                            labels=list(range(config.NUM_CLASSES)))
    report = classification_report(all_labels, all_preds,
                                   target_names=config.CLASSES,
                                   zero_division=0)
    return {
        "weighted_f1":   round(float(wf1), 4),
        "healthy_f1":    round(float(per[0]), 4),
        "rubbish_f1":    round(float(per[1]), 4),
        "unhealthy_f1":  round(float(per[2]), 4),
        "confusion_matrix": cm.tolist(),
        "report":        report,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}\n")

    # Verify synthetic image directories exist
    for d, name in [(LIGHT_SYNTH_DIR, "lightweight"), (HEAVY_SYNTH_DIR, "heavy")]:
        pngs = sorted(d.glob("*.png"))
        if len(pngs) == 0:
            raise FileNotFoundError(
                f"No PNGs found in {d}\n"
                f"Run compare_ddpm.py first to generate {name} DDPM images."
            )
        print(f"Found {len(pngs)} {name} synthetic images in {d}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Data split (same as all other conditions) ----
    df, class_to_idx = load_train_df()
    df_train, df_val = split_train_val(df, random_state=args.seed)
    cw = compute_class_weights(df_train)
    cw_tensor = torch.tensor(cw)

    test_loader, _ = build_test_loader(class_to_idx)

    # ---- Load pre-trained Model A (shared for both conditions) ----
    print(f"\nLoading Model A from {config.PRETRAINED_MODEL_A}")
    model_a = load_checkpoint(config.PRETRAINED_MODEL_A, device)

    results = {}

    for tag, synth_dir, ckpt_path, log_path in [
        ("ddpm_lightweight_500", LIGHT_SYNTH_DIR, CKPT_LIGHT, LOG_LIGHT),
        ("ddpm_heavy_500",       HEAVY_SYNTH_DIR, CKPT_HEAVY, LOG_HEAVY),
    ]:
        print(f"\n{'='*60}")
        print(f"  Condition: {tag}")
        print(f"{'='*60}")

        synth_paths = sorted(synth_dir.glob("*.png"))
        train_loader, val_loader = build_model_b_loaders_synth(
            df_train, df_val,
            sampler=None,          # build_model_b_loaders_synth builds its own sampler
            synth_paths=[str(p) for p in synth_paths],
        )

        print(f"  Synthetic images : {len(synth_paths)}")
        print(f"  Train dataset    : {len(train_loader.dataset)}")

        train_model_b(
            train_loader, val_loader, cw_tensor,
            ckpt_path=ckpt_path,
            log_path=log_path,
            seed=args.seed,
        )

        print(f"\n  Evaluating ensemble (Model A + new Model B) …")
        model_b = load_checkpoint(ckpt_path, device)
        metrics = evaluate(model_a, model_b, test_loader, device)
        results[tag] = metrics

        print(f"  weighted F1  = {metrics['weighted_f1']:.4f}")
        print(f"  healthy  F1  = {metrics['healthy_f1']:.4f}")
        print(f"  rubbish  F1  = {metrics['rubbish_f1']:.4f}")
        print(f"  unhealthy F1 = {metrics['unhealthy_f1']:.4f}")
        print(f"\n{metrics['report']}")

    # ---- Save results ----
    out_path = config.METRICS_DIR / "downstream_f1_comparison.json"
    # Strip the full report text for clean JSON storage
    for v in results.values():
        v.pop("report", None)
    out_path.write_text(json.dumps(results, indent=2))

    # ---- Final summary ----
    print("\n" + "=" * 60)
    print("  Downstream F1 Comparison (ensemble on full test set)")
    print("-" * 60)
    print(f"  {'Condition':<28}  wF1    unhealthy-F1")
    print("-" * 60)
    for tag, m in results.items():
        print(f"  {tag:<28}  {m['weighted_f1']:.4f}  {m['unhealthy_f1']:.4f}")
    print("=" * 60)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
