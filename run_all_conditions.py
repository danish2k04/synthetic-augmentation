"""
Run all downstream classifier conditions sequentially.

Trains a fresh Model B for each condition, evaluates the two-model ensemble
(pre-trained Model A + new Model B) on the full test set, and saves:
  outputs/metrics/{condition}_seed{N}_metrics.json
  outputs/checkpoints/{condition}_seed{N}_modelB.pt
  outputs/logs/{condition}_seed{N}_modelB_log.csv

Conditions:
  baseline_ensemble   — Model B: standard aug only, no synthetic data
  heavy_aug_ensemble  — Model B: dual aug (heavy for unhealthy)
  gan_250_ensemble    — Model B: standard aug + 250 DCGAN synthetic unhealthy
  gan_500_ensemble    — Model B: standard aug + 500 DCGAN synthetic unhealthy
  ddpm_250_ensemble   — Model B: standard aug + 250 DDPM synthetic unhealthy
  ddpm_500_ensemble   — Model B: standard aug + 500 DDPM synthetic unhealthy

Model A is always loaded from config.PRETRAINED_MODEL_A (shared, never retrained).

Usage:
    python run_all_conditions.py [--seed 42]
    python run_all_conditions.py --conditions heavy_aug_ensemble ddpm_500_ensemble
    python run_all_conditions.py --skip_existing
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score, confusion_matrix,
)

import config
from dataset import (
    load_train_df, split_train_val, compute_class_weights, make_sampler,
    build_model_b_loaders_std, build_model_b_loaders_heavy,
    build_model_b_loaders_synth, build_test_loader,
)
from models.classifier import train_model_b, load_checkpoint
from models.ensemble import ensemble_predict


ALL_CONDITIONS = [
    "baseline_ensemble",
    "heavy_aug_ensemble",
    "gan_250_ensemble",
    "gan_500_ensemble",
    "ddpm_250_ensemble",
    "ddpm_500_ensemble",
]

SYNTH_DIRS = {
    "gan_250_ensemble":  config.GEN_IMAGE_DIR / "dcgan" / "unhealthy_250",
    "gan_500_ensemble":  config.GEN_IMAGE_DIR / "dcgan" / "unhealthy_500",
    "ddpm_250_ensemble": config.GEN_IMAGE_DIR / "ddpm"  / "unhealthy_250",
    "ddpm_500_ensemble": config.GEN_IMAGE_DIR / "ddpm"  / "unhealthy_500",
}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_ensemble(model_a, model_b, test_loader, device) -> dict:
    preds, labels = ensemble_predict(model_a, model_b, test_loader, device)
    wf1      = f1_score(labels, preds, average="weighted", zero_division=0)
    mf1      = f1_score(labels, preds, average="macro",    zero_division=0)
    acc      = accuracy_score(labels, preds)
    prec_cls = precision_score(labels, preds, average=None, zero_division=0).tolist()
    rec_cls  = recall_score(labels, preds,    average=None, zero_division=0).tolist()
    f1_cls   = f1_score(labels, preds,        average=None, zero_division=0).tolist()
    cm       = confusion_matrix(labels, preds).tolist()
    return {
        "weighted_f1":         round(float(wf1), 6),
        "macro_f1":            round(float(mf1), 6),
        "accuracy":            round(float(acc), 6),
        "per_class_precision": {c: round(prec_cls[i], 6) for i, c in enumerate(config.CLASSES)},
        "per_class_recall":    {c: round(rec_cls[i],  6) for i, c in enumerate(config.CLASSES)},
        "per_class_f1":        {c: round(f1_cls[i],   6) for i, c in enumerate(config.CLASSES)},
        "confusion_matrix":    cm,
        "unhealthy_recall":    round(float(rec_cls[config.UNHEALTHY_IDX]), 6),
    }


# ---------------------------------------------------------------------------
# Single-condition runner
# ---------------------------------------------------------------------------

def run_condition(condition: str, seed: int,
                  df_train, df_val, cw: np.ndarray, cw_tensor: torch.Tensor,
                  model_a, test_loader, device) -> dict:

    ckpt_path = config.CHECKPOINT_DIR / f"{condition}_seed{seed}_modelB.pt"
    log_path  = config.LOG_DIR        / f"{condition}_seed{seed}_modelB_log.csv"
    out_path  = config.METRICS_DIR    / f"{condition}_seed{seed}_metrics.json"

    for d in (config.CHECKPOINT_DIR, config.LOG_DIR, config.METRICS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  Condition : {condition}  (seed={seed})")
    print(f"{'='*65}")

    # Build loaders
    if condition == "baseline_ensemble":
        sampler = make_sampler(df_train, cw)
        train_loader, val_loader = build_model_b_loaders_std(df_train, df_val, sampler)

    elif condition == "heavy_aug_ensemble":
        sampler = make_sampler(df_train, cw)
        train_loader, val_loader = build_model_b_loaders_heavy(df_train, df_val, sampler)

    else:
        synth_dir   = SYNTH_DIRS[condition]
        synth_paths = sorted(str(p) for p in synth_dir.glob("*.png"))
        if not synth_paths:
            raise FileNotFoundError(f"No PNG files found in {synth_dir}")
        print(f"  Synthetic images : {len(synth_paths)} from {synth_dir.name}")
        # build_model_b_loaders_synth builds its own sampler internally
        train_loader, val_loader = build_model_b_loaders_synth(
            df_train, df_val, sampler=None, synth_paths=synth_paths,
        )

    print(f"  Train batches : {len(train_loader)}  |  Val batches : {len(val_loader)}")

    t0 = time.time()
    train_model_b(
        train_loader, val_loader, cw_tensor,
        ckpt_path=ckpt_path, log_path=log_path, seed=seed,
    )
    training_minutes = (time.time() - t0) / 60.0

    print(f"\n  Evaluating ensemble (Model A + {condition} Model B) …")
    model_b = load_checkpoint(ckpt_path, device)
    metrics = evaluate_ensemble(model_a, model_b, test_loader, device)
    metrics["training_time_minutes"] = round(training_minutes, 2)
    metrics["condition"] = condition
    metrics["seed"]      = seed

    out_path.write_text(json.dumps(metrics, indent=2))

    p = metrics["per_class_precision"]
    r = metrics["per_class_recall"]
    f = metrics["per_class_f1"]
    print(f"\n  Weighted F1      : {metrics['weighted_f1']:.4f}")
    print(f"  Macro F1         : {metrics['macro_f1']:.4f}")
    print(f"  Accuracy         : {metrics['accuracy']:.4f}")
    print(f"  Unhealthy Recall : {metrics['unhealthy_recall']:.4f}")
    print(f"  Training time    : {training_minutes:.1f} min")
    print(f"  Per-class  P / R / F1:")
    for cls in config.CLASSES:
        print(f"    {cls:<10}  {p[cls]:.4f} / {r[cls]:.4f} / {f[cls]:.4f}")
    print(f"  Metrics -> {out_path}")

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",         type=int,  default=42)
    parser.add_argument("--conditions",   nargs="+", default=ALL_CONDITIONS,
                        choices=ALL_CONDITIONS, metavar="COND",
                        help="Which conditions to run (default: all)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip conditions whose metrics JSON already exists")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"Seed       : {args.seed}")
    print(f"Conditions : {args.conditions}")

    # Resolve which conditions to actually run
    to_run = []
    for cond in args.conditions:
        out_path = config.METRICS_DIR / f"{cond}_seed{args.seed}_metrics.json"
        if args.skip_existing and out_path.exists():
            print(f"  [skip] {cond} — metrics file already exists")
        else:
            to_run.append(cond)

    if not to_run:
        print("Nothing to run.")
        return

    # Shared data setup (done once)
    print("\nLoading data …")
    df, class_to_idx = load_train_df()
    df_train, df_val = split_train_val(df, random_state=args.seed)
    cw               = compute_class_weights(df_train)
    cw_tensor        = torch.tensor(cw)
    test_loader, _   = build_test_loader(class_to_idx)
    print(f"  Train : {len(df_train)}  Val : {len(df_val)}  Test : {len(test_loader.dataset)}")

    print(f"\nLoading Model A from {config.PRETRAINED_MODEL_A}")
    model_a = load_checkpoint(config.PRETRAINED_MODEL_A, device)

    # Run conditions
    wall_start   = time.time()
    all_results  = {}
    for condition in to_run:
        metrics = run_condition(
            condition, args.seed, df_train, df_val, cw, cw_tensor,
            model_a, test_loader, device,
        )
        all_results[condition] = metrics

    total_hours = (time.time() - wall_start) / 3600.0

    # Final summary table
    print(f"\n{'='*75}")
    print(f"  ALL-CONDITIONS SUMMARY  (seed={args.seed})")
    print(f"{'='*75}")
    hdr = f"  {'Condition':<28}  {'wF1':>6}  {'mF1':>6}  {'Acc':>6}  {'Unhlthy-R':>10}  {'Time(m)':>7}"
    print(hdr)
    print(f"  {'-'*70}")
    for cond in args.conditions:
        if cond not in all_results:
            continue
        m = all_results[cond]
        print(f"  {cond:<28}  {m['weighted_f1']:>6.4f}  {m['macro_f1']:>6.4f}  "
              f"{m['accuracy']:>6.4f}  {m['unhealthy_recall']:>10.4f}  "
              f"{m['training_time_minutes']:>7.1f}")
    print(f"{'='*75}")
    print(f"  Total wall time : {total_hours:.2f} h")


if __name__ == "__main__":
    main()
