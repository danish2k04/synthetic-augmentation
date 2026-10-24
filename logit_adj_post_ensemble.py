"""
Post-Ensemble Logit Adjustment — three variants compared side by side.

Variant A — per_member (current approach in logit_adj_ensemble.py):
    adj_a  = logits_a − τ · log(π)
    adj_b  = logits_b − τ · log(π)
    ŷ      = argmax( softmax(adj_a) + softmax(adj_b) )

Variant B — post_ensemble_probs (NEW: adjust after averaging softmax):
    probs      = ( softmax(logits_a) + softmax(logits_b) ) / 2
    log_probs  = log(probs)
    ŷ          = argmax( log_probs − τ · log(π) )

Variant C — post_ensemble_logits (NEW: adjust average of raw logits):
    avg_logits = ( logits_a + logits_b ) / 2
    ŷ          = argmax( avg_logits − τ · log(π) )

τ is tuned on the validation split (split_train_val, random_state=42) by
maximising unhealthy F1. The same best τ is then used for all 8 conditions
on the test set.

Also applies all three variants to the 5-model enlarged ensemble:
    members: Model A, pretrained Model B, gan_500 B, ddpm_500 B, ddpm_heavy_500 B

Outputs → logit_adjustment_results/
    post_ensemble_logit_adj.json
    post_ensemble_logit_adj.csv

Usage:
    python logit_adj_post_ensemble.py
    python logit_adj_post_ensemble.py --tau_values 0.1 0.5 1.0 1.5
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, confusion_matrix as sk_cm,
    f1_score, precision_recall_fscore_support,
)
from tqdm import tqdm

import config
from dataset import (
    build_test_loader, load_train_df, split_train_val,
    PapCellDataset, get_eval_tf, make_loader,
)
from models.classifier import load_checkpoint

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASSES  = config.CLASSES
UH_IDX   = CLASSES.index("unhealthy")
OUT_DIR  = Path("logit_adjustment_results")

TAU_GRID = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

# 8 two-model ensemble conditions (Model A + each Model B)
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

# 5-model enlarged ensemble members (Model A + 4 specific Model Bs)
ENLARGED_MEMBERS = {
    "model_a":          config.PRETRAINED_MODEL_A,
    "model_b_pretrained": config.PRETRAINED_MODEL_B,
    "model_b_gan500":   config.CHECKPOINT_DIR / "gan_500_ensemble_seed42_modelB.pt",
    "model_b_ddpm500":  config.CHECKPOINT_DIR / "ddpm_500_ensemble_seed42_modelB.pt",
    "model_b_ddpmheavy": config.CHECKPOINT_DIR / "ddpm_heavy_500_ensemble" / "model_b_seed42.pth",
}

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


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def get_logits(ckpt_path: Path, loader, device) -> np.ndarray:
    model = load_checkpoint(ckpt_path, device)
    model.eval()
    parts = []
    for imgs, _ in tqdm(loader, desc=f"  {ckpt_path.name[:38]}", leave=False):
        parts.append(model(imgs.to(device)).cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(parts, axis=0).astype(np.float32)


@torch.no_grad()
def get_labels(loader) -> np.ndarray:
    labels = []
    for _, lbl in loader:
        labels.extend(lbl.numpy())
    return np.array(labels, dtype=np.int64)


# ---------------------------------------------------------------------------
# Math utilities
# ---------------------------------------------------------------------------
def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def safe_log(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log(np.clip(p, eps, None))


# ---------------------------------------------------------------------------
# Three prediction variants
# ---------------------------------------------------------------------------
def predict_per_member(logits_list: list, log_pi: np.ndarray, tau: float) -> np.ndarray:
    """
    Variant A: adjust each member's logits before softmax, then average probs.
        ŷ = argmax( mean_k[ softmax(logits_k − τ·log(π)) ] )
    """
    adj_probs = [softmax(lg - tau * log_pi) for lg in logits_list]
    return np.mean(adj_probs, axis=0).argmax(axis=1)


def predict_post_probs(logits_list: list, log_pi: np.ndarray, tau: float) -> np.ndarray:
    """
    Variant B: average softmax probs first, then adjust in log-space.
        probs     = mean_k[ softmax(logits_k) ]
        log_probs = log(probs)
        ŷ         = argmax( log_probs − τ·log(π) )
    """
    avg_probs = np.mean([softmax(lg) for lg in logits_list], axis=0)
    return (safe_log(avg_probs) - tau * log_pi).argmax(axis=1)


def predict_post_logits(logits_list: list, log_pi: np.ndarray, tau: float) -> np.ndarray:
    """
    Variant C: average raw logits first, then adjust.
        avg_logits = mean_k[ logits_k ]
        ŷ          = argmax( avg_logits − τ·log(π) )
    """
    avg_logits = np.mean(logits_list, axis=0)
    return (avg_logits - tau * log_pi).argmax(axis=1)


VARIANTS = {
    "per_member":           predict_per_member,
    "post_ensemble_probs":  predict_post_probs,
    "post_ensemble_logits": predict_post_logits,
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
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


# ---------------------------------------------------------------------------
# τ sweep on a loader
# ---------------------------------------------------------------------------
def sweep_tau(logits_list: list, y_val: np.ndarray,
              log_pi: np.ndarray, tau_grid: list) -> dict:
    """
    For each τ and each variant, compute unhealthy F1 on the validation set.
    Returns {variant: {tau: uh_f1}}.
    """
    sweep = {v: {} for v in VARIANTS}
    for tau in tau_grid:
        for vname, fn in VARIANTS.items():
            pred = fn(logits_list, log_pi, tau)
            _, _, f, _ = precision_recall_fscore_support(
                y_val, pred,
                labels=list(range(config.NUM_CLASSES)),
                zero_division=0,
            )
            sweep[vname][tau] = round(float(f[UH_IDX]), 6)
    return sweep


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau_values", type=float, nargs="+", default=TAU_GRID)
    args = parser.parse_args()
    tau_grid = args.tau_values

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Class priors ────────────────────────────────────────────────────
    print("\n[1] Class priors …")
    df, class_to_idx = load_train_df()
    counts  = np.array([(df["target"] == i).sum()
                        for i in range(config.NUM_CLASSES)], dtype=np.float64)
    priors  = counts / counts.sum()
    log_pi  = np.log(priors).astype(np.float32)
    for i, c in enumerate(CLASSES):
        print(f"  {c:<12} pi={priors[i]:.4f}  -log(pi)={-log_pi[i]:.3f}")

    # ── 2. Validation loader for τ tuning ──────────────────────────────────
    print("\n[2] Validation loader (tau tuning) ...")
    df_train, df_val = split_train_val(df, random_state=42)
    val_loader = make_loader(PapCellDataset(df_val, config.TRAIN_DIR, get_eval_tf()))
    y_val      = get_labels(val_loader)
    print(f"  Val set : {len(y_val)} samples")

    # ── 3. Test loader ─────────────────────────────────────────────────────
    print("\n[3] Test loader …")
    test_loader, _ = build_test_loader(class_to_idx)
    y_test = get_labels(test_loader)
    print(f"  Test set : {len(y_test)} samples")

    # ── 4. Model A logits (val + test, reused across all conditions) ───────
    print("\n[4] Model A logits …")
    val_logits_a  = get_logits(config.PRETRAINED_MODEL_A, val_loader, device)
    test_logits_a = get_logits(config.PRETRAINED_MODEL_A, test_loader, device)

    # ── 5. τ tuning: sweep across all 8 conditions, pick global best τ ─────
    # We use the first condition's val logits to find a τ that generalises,
    # then apply the same τ to all conditions (as requested).
    print("\n[5] tau sweep on validation set ...")
    print("  (using baseline condition to find shared best tau per variant)\n")

    val_logits_b_baseline = get_logits(
        CONDITIONS["baseline"], val_loader, device
    )
    val_sweep = sweep_tau(
        [val_logits_a, val_logits_b_baseline], y_val, log_pi, tau_grid
    )

    best_tau = {}
    for vname, scores in val_sweep.items():
        bt = max(scores, key=scores.get)
        best_tau[vname] = bt
        print(f"  {vname:<25}  best tau = {bt}  "
              f"(val uh_F1 = {scores[bt]:.4f})")
        for tau in tau_grid:
            print(f"    tau={tau:.2f}  uh_F1={scores[tau]:.4f}")

    del val_logits_b_baseline   # free memory

    # ── 6. Test evaluation — 8 two-model conditions ────────────────────────
    print("\n[6] Test evaluation — 8 conditions …\n")
    results      = {}   # condition → variant → metrics dict
    csv_rows     = []
    json_out     = {}

    for cond, ckpt in CONDITIONS.items():
        print(f"── {cond} ──")
        test_logits_b = get_logits(ckpt, test_loader, device)
        logits_list   = [test_logits_a, test_logits_b]

        # No adjustment baseline (standard softmax average)
        base_pred = np.mean([softmax(lg) for lg in logits_list], axis=0).argmax(1)
        base_m    = compute_metrics(y_test, base_pred)

        results[cond] = {"no_adjustment": base_m}
        dname = DISPLAY_NAME[cond]

        for vname, fn in VARIANTS.items():
            tau  = best_tau[vname]
            pred = fn(logits_list, log_pi, tau)
            m    = compute_metrics(y_test, pred)
            results[cond][vname] = m

            uh_f1_base = base_m["per_class_f1"]["unhealthy"]
            uh_f1_adj  = m["per_class_f1"]["unhealthy"]
            print(f"  [{vname:<25}] tau={tau}  "
                  f"uh_F1: {uh_f1_base:.4f} -> {uh_f1_adj:.4f}  "
                  f"delta={uh_f1_adj - uh_f1_base:+.4f}")

            csv_rows.append({
                "condition":    cond,
                "tau":          tau,
                "method":       vname,
                "uh_f1":        m["per_class_f1"]["unhealthy"],
                "uh_recall":    m["unhealthy_recall"],
                "uh_precision": m["per_class_precision"]["unhealthy"],
                "wf1":          m["weighted_f1"],
                "mf1":          m["macro_f1"],
                "accuracy":     m["accuracy"],
            })

            scalar_keys = ["weighted_f1", "macro_f1", "accuracy", "unhealthy_recall"]
            json_out[f"{cond}__{vname}"] = {
                "method":        vname,
                "model":         f"Ensemble — {dname}",
                "tau":           tau,
                "baseline_test": base_m,
                "adjusted_test": m,
                "delta": {
                    k: round(m[k] - base_m[k], 6)
                    for k in scalar_keys
                },
            }

        print()

    # ── 7. 5-model enlarged ensemble ───────────────────────────────────────
    print("── 5-model enlarged ensemble ──")
    enlarged_logits = {}
    for name, ckpt in ENLARGED_MEMBERS.items():
        print(f"  Loading {name} …")
        enlarged_logits[name] = get_logits(ckpt, test_loader, device)

    logits_list_5 = list(enlarged_logits.values())

    base_pred_5 = np.mean([softmax(lg) for lg in logits_list_5], axis=0).argmax(1)
    base_m_5    = compute_metrics(y_test, base_pred_5)

    for vname, fn in VARIANTS.items():
        tau  = best_tau[vname]
        pred = fn(logits_list_5, log_pi, tau)
        m    = compute_metrics(y_test, pred)

        uh_f1_base = base_m_5["per_class_f1"]["unhealthy"]
        uh_f1_adj  = m["per_class_f1"]["unhealthy"]
        print(f"  [{vname:<25}] tau={tau}  "
              f"uh_F1: {uh_f1_base:.4f} -> {uh_f1_adj:.4f}  "
              f"delta={uh_f1_adj - uh_f1_base:+.4f}")

        csv_rows.append({
            "condition":    "enlarged_5model",
            "tau":          tau,
            "method":       vname,
            "uh_f1":        m["per_class_f1"]["unhealthy"],
            "uh_recall":    m["unhealthy_recall"],
            "uh_precision": m["per_class_precision"]["unhealthy"],
            "wf1":          m["weighted_f1"],
            "mf1":          m["macro_f1"],
            "accuracy":     m["accuracy"],
        })

        scalar_keys = ["weighted_f1", "macro_f1", "accuracy", "unhealthy_recall"]
        json_out[f"enlarged_5model__{vname}"] = {
            "method":        vname,
            "model":         "5-model Enlarged Ensemble",
            "tau":           tau,
            "baseline_test": base_m_5,
            "adjusted_test": m,
            "delta": {
                k: round(m[k] - base_m_5[k], 6)
                for k in scalar_keys
            },
        }

    # ── 8. Save JSON ───────────────────────────────────────────────────────
    json_path = OUT_DIR / "post_ensemble_logit_adj.json"
    json_path.write_text(json.dumps(json_out, indent=2))
    print(f"\n→ {json_path}")

    # ── 9. Save CSV ────────────────────────────────────────────────────────
    csv_path = OUT_DIR / "post_ensemble_logit_adj.csv"
    fields   = ["condition", "tau", "method",
                "uh_f1", "uh_recall", "uh_precision",
                "wf1", "mf1", "accuracy"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(csv_rows)
    print(f"→ {csv_path}")

    # ── 10. Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  SUMMARY — Unhealthy F1 by method  (best tau per variant, test set)")
    print(f"  {'Condition':<30} {'no_adj':>8} "
          f"{'per_mbr':>8} {'post_prb':>9} {'post_lgt':>9}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*9} {'-'*9}")

    def uh(m): return m["per_class_f1"]["unhealthy"]

    for cond in list(CONDITIONS) + ["enlarged_5model"]:
        if cond == "enlarged_5model":
            base  = base_m_5
            r     = {v: json_out[f"enlarged_5model__{v}"]["adjusted_test"]
                     for v in VARIANTS}
        else:
            base  = results[cond]["no_adjustment"]
            r     = {v: results[cond][v] for v in VARIANTS}
        print(
            f"  {cond:<30} {uh(base):>8.4f} "
            f"{uh(r['per_member']):>8.4f} "
            f"{uh(r['post_ensemble_probs']):>9.4f} "
            f"{uh(r['post_ensemble_logits']):>9.4f}"
        )
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
