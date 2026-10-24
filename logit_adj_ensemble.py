"""
Post-hoc Logit Adjustment (τ = 0.1, fixed) applied to:
  - Each of 8 Model B variants (standalone)
  - Each of 8 two-model ensembles (Model A + Model B)

Formula (per model before combining):
    adjusted_logit_y  =  f_y(x)  −  τ · log(π_y)

Ensemble:
    probs = ( softmax(adj_logits_a) + softmax(adj_logits_b) ) / 2
    ŷ     = argmax(probs)

Outputs → logit_adjustment_results/
  ensemble_logit_adj.json    — all conditions, before vs after
  ensemble_logit_adj.csv     — compact comparison table

Usage:
    python logit_adj_ensemble.py [--tau 0.1]
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
)
from tqdm import tqdm

import config
from dataset import build_test_loader, load_train_df
from models.classifier import load_checkpoint

CLASSES = config.CLASSES
UH_IDX  = CLASSES.index("unhealthy")
OUT_DIR = Path("logit_adjustment_results")

DISPLAY_NAME = {
    "baseline":                   "standard_aug",
    "heavy_aug":                  "heavy_aug",
    "gan_250":                    "gan_250",
    "gan_500":                    "gan_500",
    "ddpm_250":                   "ddpm_250",
    "ddpm_500":                   "ddpm_500",
    "ddpm_heavy_500":             "ddpm_heavy_500",
    "combined_ddpm500_heavy_aug": "combined_ddpm500_heavy_aug",
}

CONDITIONS = {
    "baseline":                   config.CHECKPOINT_DIR / "baseline_ensemble_seed42_modelB.pt",
    "heavy_aug":                  config.CHECKPOINT_DIR / "heavy_aug_ensemble_seed42_modelB.pt",
    "gan_250":                    config.CHECKPOINT_DIR / "gan_250_ensemble_seed42_modelB.pt",
    "gan_500":                    config.CHECKPOINT_DIR / "gan_500_ensemble_seed42_modelB.pt",
    "ddpm_250":                   config.CHECKPOINT_DIR / "ddpm_250_ensemble_seed42_modelB.pt",
    "ddpm_500":                   config.CHECKPOINT_DIR / "ddpm_500_ensemble_seed42_modelB.pt",
    "ddpm_heavy_500":             config.CHECKPOINT_DIR / "ddpm_heavy_500_ensemble" / "model_b_seed42.pth",
    "combined_ddpm500_heavy_aug": config.CHECKPOINT_DIR / "combined_ddpm500_heavy_aug" / "model_b_seed42.pth",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def get_logits(ckpt_path, loader, device) -> np.ndarray:
    model = load_checkpoint(ckpt_path, device)
    model.eval()
    parts = []
    for imgs, _ in tqdm(loader, desc=f"  {ckpt_path.name[:38]}", leave=False):
        parts.append(model(imgs.to(device)).cpu().numpy())
    del model; torch.cuda.empty_cache()
    return np.concatenate(parts, axis=0).astype(np.float32)


@torch.no_grad()
def get_labels(loader) -> np.ndarray:
    labels = []
    for _, lbl in loader:
        labels.extend(lbl.numpy())
    return np.array(labels, dtype=np.int64)


def compute_metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import confusion_matrix as sk_cm
    wf1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    mf1 = float(f1_score(y_true, y_pred, average="macro",    zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    cm  = sk_cm(y_true, y_pred, labels=list(range(config.NUM_CLASSES))).tolist()
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred,
        labels=list(range(config.NUM_CLASSES)),
        zero_division=0,
    )
    return {
        "weighted_f1":         round(wf1, 6),
        "macro_f1":            round(mf1, 6),
        "accuracy":            round(acc, 6),
        "per_class_precision": {c: round(float(p[i]), 6) for i, c in enumerate(CLASSES)},
        "per_class_recall":    {c: round(float(r[i]), 6) for i, c in enumerate(CLASSES)},
        "per_class_f1":        {c: round(float(f[i]), 6) for i, c in enumerate(CLASSES)},
        "confusion_matrix":    cm,
        "unhealthy_recall":    round(float(r[UH_IDX]), 6),
    }


def adjust(logits: np.ndarray, log_pi: np.ndarray, tau: float) -> np.ndarray:
    return logits - tau * log_pi


def ens_predict(logits_a, logits_b, log_pi, tau) -> np.ndarray:
    adj_a = adjust(logits_a, log_pi, tau)
    adj_b = adjust(logits_b, log_pi, tau)
    probs = (softmax(adj_a) + softmax(adj_b)) / 2
    return probs.argmax(axis=1)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau", type=float, default=0.1)
    args = parser.parse_args()
    TAU = args.tau

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}   τ = {TAU}\n")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Class priors ──────────────────────────────────────────────────────
    df, class_to_idx = load_train_df()
    counts  = np.array([(df["target"] == i).sum()
                        for i in range(config.NUM_CLASSES)], dtype=np.float64)
    priors  = counts / counts.sum()
    log_pi  = np.log(priors).astype(np.float32)
    print("Class priors:")
    for i, c in enumerate(CLASSES):
        print(f"  {c:<12} π={priors[i]:.4f}  −log(π)={-log_pi[i]:.3f}")

    # ── Test loader + labels ───────────────────────────────────────────────
    print("\nBuilding test loader …")
    test_loader, _ = build_test_loader(class_to_idx)
    y_true = get_labels(test_loader)
    print(f"  Test set : {len(y_true)} samples")

    # ── Model A logits (shared across all conditions) ─────────────────────
    print("\nModel A logits …")
    logits_a = get_logits(config.PRETRAINED_MODEL_A, test_loader, device)

    # ── Per-condition loop ────────────────────────────────────────────────
    results = {}
    print(f"\nProcessing 8 conditions (τ = {TAU}) …\n")

    for cond, ckpt in CONDITIONS.items():
        print(f"── {cond} ──")
        logits_b = get_logits(ckpt, test_loader, device)

        # Standalone Model B — baseline (no adjustment)
        solo_base = compute_metrics(y_true, logits_b.argmax(1))

        # Standalone Model B — adjusted
        solo_adj_pred = adjust(logits_b, log_pi, TAU).argmax(1)
        solo_adj      = compute_metrics(y_true, solo_adj_pred)

        # 2-model Ensemble — baseline (standard softmax avg)
        ens_base_probs = (softmax(logits_a) + softmax(logits_b)) / 2
        ens_base = compute_metrics(y_true, ens_base_probs.argmax(1))

        # 2-model Ensemble — adjusted
        ens_adj_pred = ens_predict(logits_a, logits_b, log_pi, TAU)
        ens_adj      = compute_metrics(y_true, ens_adj_pred)

        results[cond] = {
            "tau": TAU,
            "model_b_baseline": solo_base,
            "model_b_adjusted": solo_adj,
            "model_b_delta": {
                k: round(solo_adj[k] - solo_base[k], 4)
                for k in solo_base
                if isinstance(solo_base[k], float)
            },
            "ensemble_baseline": ens_base,
            "ensemble_adjusted": ens_adj,
            "ensemble_delta": {
                k: round(ens_adj[k] - ens_base[k], 4)
                for k in ens_base
                if isinstance(ens_base[k], float)
            },
        }

        print(f"  Solo B    base uh_F1={solo_base['per_class_f1']['unhealthy']:.4f}  "
              f"adj uh_F1={solo_adj['per_class_f1']['unhealthy']:.4f}  "
              f"Δ={solo_adj['per_class_f1']['unhealthy']-solo_base['per_class_f1']['unhealthy']:+.4f}")
        print(f"  Ensemble  base uh_F1={ens_base['per_class_f1']['unhealthy']:.4f}  "
              f"adj uh_F1={ens_adj['per_class_f1']['unhealthy']:.4f}  "
              f"Δ={ens_adj['per_class_f1']['unhealthy']-ens_base['per_class_f1']['unhealthy']:+.4f}\n")

    # ── Save JSON — one entry per unit, matching logit_adj_model_a.json schema ──
    scalar_keys = ["weighted_f1", "macro_f1", "accuracy", "unhealthy_recall"]

    def make_entry(model_label, baseline, adjusted):
        return {
            "method":        "post_hoc_logit_adjustment",
            "model":         model_label,
            "tau":           TAU,
            "baseline_test": baseline,
            "adjusted_test": adjusted,
            "delta": {
                k: round(adjusted[k] - baseline[k], 6)
                for k in scalar_keys
            },
        }

    out = {}
    for cond, r in results.items():
        dname = DISPLAY_NAME[cond]
        out[f"model_b_{cond}"] = make_entry(
            f"Model B — {dname}",
            r["model_b_baseline"],
            r["model_b_adjusted"],
        )
        out[f"ensemble_{cond}"] = make_entry(
            f"Ensemble — {dname}",
            r["ensemble_baseline"],
            r["ensemble_adjusted"],
        )

    json_path = OUT_DIR / "ensemble_logit_adj.json"
    json_path.write_text(json.dumps(out, indent=2))
    print(f"→ {json_path}")

    # ── Save CSV ──────────────────────────────────────────────────────────
    csv_path = OUT_DIR / "ensemble_logit_adj.csv"
    rows = []
    for cond, r in results.items():
        for unit, key in [("model_b", "model_b"), ("ensemble", "ensemble")]:
            base = r[f"{key}_baseline"]
            adj  = r[f"{key}_adjusted"]
            rows.append({
                "condition":         cond,
                "unit":              unit,
                "tau":               TAU,
                "uh_f1_base":        base["per_class_f1"]["unhealthy"],
                "uh_f1_adj":         adj["per_class_f1"]["unhealthy"],
                "uh_f1_delta":       round(adj["per_class_f1"]["unhealthy"] - base["per_class_f1"]["unhealthy"], 4),
                "uh_recall_base":    base["unhealthy_recall"],
                "uh_recall_adj":     adj["unhealthy_recall"],
                "wf1_base":          base["weighted_f1"],
                "wf1_adj":           adj["weighted_f1"],
                "wf1_delta":         round(adj["weighted_f1"] - base["weighted_f1"], 4),
            })
    fields = ["condition","unit","tau",
              "uh_f1_base","uh_f1_adj","uh_f1_delta",
              "uh_recall_base","uh_recall_adj",
              "wf1_base","wf1_adj","wf1_delta"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"→ {csv_path}")

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SUMMARY — Unhealthy F1  (τ = {TAU})")
    print(f"  {'Condition':<30} {'Solo B':>10}  {'Ens':>10}  {'Ens+Adj':>10}")
    print(f"  {'-'*30} {'-'*10}  {'-'*10}  {'-'*10}")
    for cond, r in results.items():
        sb  = r["model_b_baseline"]["per_class_f1"]["unhealthy"]
        eb  = r["ensemble_baseline"]["per_class_f1"]["unhealthy"]
        ea  = r["ensemble_adjusted"]["per_class_f1"]["unhealthy"]
        print(f"  {cond:<30} {sb:>10.4f}  {eb:>10.4f}  {ea:>10.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
