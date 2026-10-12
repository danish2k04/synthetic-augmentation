"""
Evaluate the two-model ensemble on the test set.

Usage:
    python evaluate_ensemble.py \
        --model_a <path_to_model_a.pth> \
        --model_b <path_to_model_b.pth> \
        [--out <output_json_path>]

With no arguments, loads the pre-trained checkpoints from config.py
(PRETRAINED_MODEL_A / PRETRAINED_MODEL_B) and prints results.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
)

import config
from dataset import build_test_loader, load_train_df
from models.ensemble import load_ensemble, ensemble_predict


def evaluate(model_a_path: Path, model_b_path: Path, out_path: Path | None = None) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    _, class_to_idx = load_train_df()
    test_loader, _  = build_test_loader(class_to_idx)

    # Load models
    print(f"Loading Model A: {model_a_path}")
    print(f"Loading Model B: {model_b_path}")
    model_a, model_b = load_ensemble(model_a_path, model_b_path, device)

    # Run ensemble inference
    preds, labels = ensemble_predict(model_a, model_b, test_loader, device)

    # Metrics
    wf1      = f1_score(labels, preds, average="weighted", zero_division=0)
    mf1      = f1_score(labels, preds, average="macro",    zero_division=0)
    acc      = accuracy_score(labels, preds)
    prec_cls = precision_score(labels, preds, average=None, zero_division=0).tolist()
    rec_cls  = recall_score(labels, preds, average=None, zero_division=0).tolist()
    f1_cls   = f1_score(labels, preds, average=None, zero_division=0).tolist()
    cm       = confusion_matrix(labels, preds).tolist()
    unhealthy_recall = rec_cls[config.UNHEALTHY_IDX]

    results = {
        "weighted_f1":          round(wf1,  6),
        "macro_f1":             round(mf1,  6),
        "accuracy":             round(acc,  6),
        "per_class_precision":  {c: round(prec_cls[i], 6) for i, c in enumerate(config.CLASSES)},
        "per_class_recall":     {c: round(rec_cls[i],  6) for i, c in enumerate(config.CLASSES)},
        "per_class_f1":         {c: round(f1_cls[i],   6) for i, c in enumerate(config.CLASSES)},
        "confusion_matrix":     cm,
        "unhealthy_recall":     round(unhealthy_recall, 6),
        "model_a":              str(model_a_path),
        "model_b":              str(model_b_path),
    }

    # Pretty print
    print("\n" + "=" * 55)
    print(f"  Weighted F1      : {wf1:.4f}")
    print(f"  Macro F1         : {mf1:.4f}")
    print(f"  Accuracy         : {acc:.4f}")
    print(f"  Unhealthy Recall : {unhealthy_recall:.4f}")
    print("-" * 55)
    for cls in config.CLASSES:
        print(f"  {cls:10s}  P={results['per_class_precision'][cls]:.4f}  "
              f"R={results['per_class_recall'][cls]:.4f}  "
              f"F1={results['per_class_f1'][cls]:.4f}")
    print("=" * 55)
    print(f"Confusion matrix (rows=true, cols=pred):")
    idx_to_cls = {i: c for c, i in zip(config.CLASSES, range(config.NUM_CLASSES))}
    header = "       " + "  ".join(f"{c[:7]:>7}" for c in config.CLASSES)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {config.CLASSES[i][:7]:>7}  " + "  ".join(f"{v:>7}" for v in row))

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_a", type=Path, default=config.PRETRAINED_MODEL_A)
    parser.add_argument("--model_b", type=Path, default=config.PRETRAINED_MODEL_B)
    parser.add_argument("--out",     type=Path,
                        default=config.METRICS_DIR / "pretrained_ensemble_metrics.json")
    args = parser.parse_args()
    evaluate(args.model_a, args.model_b, args.out)


if __name__ == "__main__":
    main()
