"""
Paired bootstrap confidence interval analysis.

Evaluates:
  - Model A (standalone)                         ← reference baseline
  - 8 Model B variants (standalone)
  - 8 two-model ensembles (Model A + each B)
  - 5-model enlarged ensemble

Comparisons:
  1. Every unit vs Ensemble — baseline
  2. Each 2-model ensemble vs its own standalone Model B
  3. 5-model enlarged ensemble vs best 2-model ensemble (by unhealthy F1)

Outputs (in bootstrap_results/):
  metrics_ci.csv      — point estimates + 95% CIs for every evaluation unit
  comparisons_ci.csv  — pairwise diff CIs + significance
  summary.txt         — human-readable report

GPU note: inference runs one model at a time; model is freed before loading the next.

Usage:
    python bootstrap_ci_analysis.py [--n_boot 10000] [--seed 42]
"""
import argparse
import csv
import gc
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from tqdm import tqdm

import config
from dataset import build_test_loader, load_train_df
from models.classifier import load_checkpoint

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CKPT = config.CHECKPOINT_DIR

CONDITIONS = {
    "baseline":                  CKPT / "baseline_ensemble_seed42_modelB.pt",
    "heavy_aug":                 CKPT / "heavy_aug_ensemble_seed42_modelB.pt",
    "gan_250":                   CKPT / "gan_250_ensemble_seed42_modelB.pt",
    "gan_500":                   CKPT / "gan_500_ensemble_seed42_modelB.pt",
    "ddpm_250":                  CKPT / "ddpm_250_ensemble_seed42_modelB.pt",
    "ddpm_500":                  CKPT / "ddpm_500_ensemble_seed42_modelB.pt",
    "ddpm_heavy_500":            CKPT / "ddpm_heavy_500_ensemble" / "model_b_seed42.pth",
    "combined_ddpm500_heavy_aug": CKPT / "combined_ddpm500_heavy_aug" / "model_b_seed42.pth",
}

# The pretrained Model B used in the 5-model enlarged ensemble
# (same as config.PRETRAINED_MODEL_B — separate from the retrained heavy_aug above)
CKPT_MODEL_B_PRETRAINED = config.PRETRAINED_MODEL_B

OUT_DIR = Path("bootstrap_results")
METRICS = ["weighted_f1", "macro_f1", "accuracy",
           "unhealthy_f1", "unhealthy_precision", "unhealthy_recall"]

UH_IDX = config.CLASSES.index("unhealthy")   # 2


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def get_probs(ckpt_path: Path, loader, device) -> np.ndarray:
    """Load checkpoint, run eval, return softmax probs [N, C] as float32 numpy."""
    model = load_checkpoint(ckpt_path, device)
    model.eval()
    all_probs = []
    for imgs, _ in tqdm(loader, desc=f"  {ckpt_path.name[:40]}", leave=False):
        logits = model(imgs.to(device))
        all_probs.append(F.softmax(logits, dim=1).cpu().numpy())
    # Free GPU memory immediately
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return np.concatenate(all_probs, axis=0).astype(np.float32)


@torch.no_grad()
def get_labels(loader) -> np.ndarray:
    labels = []
    for _, lbl in loader:
        labels.extend(lbl.numpy())
    return np.array(labels, dtype=np.int64)


def ensemble_probs(*prob_arrays) -> np.ndarray:
    """Average softmax probs from multiple models."""
    return np.mean(np.stack(prob_arrays, axis=0), axis=0)


# ---------------------------------------------------------------------------
# Metric computation (operates on numpy arrays)
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    wf1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    mf1 = float(f1_score(y_true, y_pred, average="macro",    zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred,
        labels=list(range(config.NUM_CLASSES)),
        zero_division=0,
    )
    return {
        "weighted_f1":          wf1,
        "macro_f1":             mf1,
        "accuracy":             acc,
        "unhealthy_f1":         float(f[UH_IDX]),
        "unhealthy_precision":  float(p[UH_IDX]),
        "unhealthy_recall":     float(r[UH_IDX]),
    }


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray,
                 n_boot: int, seed: int) -> dict:
    """95% CI for each metric via percentile bootstrap."""
    rng = np.random.default_rng(seed)
    n   = len(y_true)
    boot = {m: [] for m in METRICS}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m   = compute_metrics(y_true[idx], y_pred[idx])
        for k in METRICS:
            boot[k].append(m[k])
    return {k: (np.percentile(v, 2.5), np.percentile(v, 97.5))
            for k, v in boot.items()}


def paired_bootstrap_diff(y_true: np.ndarray,
                          pred_ref: np.ndarray, pred_new: np.ndarray,
                          n_boot: int, seed: int) -> dict:
    """95% CI of (metric_new − metric_ref) using paired bootstrap (same indices)."""
    rng = np.random.default_rng(seed)
    n   = len(y_true)
    diffs = {m: [] for m in METRICS}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt  = y_true[idx]
        m_ref = compute_metrics(yt, pred_ref[idx])
        m_new = compute_metrics(yt, pred_new[idx])
        for k in METRICS:
            diffs[k].append(m_new[k] - m_ref[k])
    return {k: (np.percentile(v, 2.5), np.percentile(v, 97.5))
            for k, v in diffs.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_boot", type=int, default=10_000)
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    print(f"n_boot  : {args.n_boot:,}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Build test loader                                                 #
    # ------------------------------------------------------------------ #
    print("\n[1/4] Building test loader …")
    _, class_to_idx = load_train_df()
    loader, _        = build_test_loader(class_to_idx)
    y_true           = get_labels(loader)
    N                = len(y_true)
    print(f"  Test set : {N} samples")

    # ------------------------------------------------------------------ #
    # 2. Run inference — one model at a time, store probs as numpy        #
    # ------------------------------------------------------------------ #
    print("\n[2/4] Running inference …")
    store = {}   # name → probs[N, C]

    # Model A
    print("  Model A …")
    store["_model_a"] = get_probs(config.PRETRAINED_MODEL_A, loader, device)

    # Pretrained Model B (for 5-model ensemble)
    print("  Pretrained Model B …")
    store["_model_b_pretrained"] = get_probs(CKPT_MODEL_B_PRETRAINED, loader, device)

    # Condition Model Bs
    for cond, ckpt in CONDITIONS.items():
        print(f"  Model B [{cond}] …")
        store[f"_b_{cond}"] = get_probs(ckpt, loader, device)

    # ------------------------------------------------------------------ #
    # 3. Build evaluation units (name → pred labels)                      #
    # ------------------------------------------------------------------ #
    print("\n[3/4] Building evaluation units …")
    units = {}   # unit_name → y_pred[N]

    # Standalone Model A
    units["Model A"] = store["_model_a"].argmax(1)

    # Standalone Model Bs + 2-model ensembles
    for cond in CONDITIONS:
        b_key = f"_b_{cond}"
        units[f"Model B — {cond}"]    = store[b_key].argmax(1)
        ens_probs = ensemble_probs(store["_model_a"], store[b_key])
        units[f"Ensemble — {cond}"]   = ens_probs.argmax(1)

    # 5-model enlarged ensemble
    five_probs = ensemble_probs(
        store["_model_a"],
        store["_model_b_pretrained"],
        store["_b_gan_500"],
        store["_b_ddpm_500"],
        store["_b_ddpm_heavy_500"],
    )
    units["5-model Enlarged"] = five_probs.argmax(1)

    # ------------------------------------------------------------------ #
    # 4. Bootstrap                                                         #
    # ------------------------------------------------------------------ #
    print(f"\n[4/4] Bootstrap CIs ({args.n_boot:,} iterations per unit) …")
    point_est = {}   # unit_name → {metric: value}
    cis        = {}  # unit_name → {metric: (lo, hi)}

    for uname, y_pred in tqdm(units.items(), desc="  units"):
        point_est[uname] = compute_metrics(y_true, y_pred)
        cis[uname]       = bootstrap_ci(y_true, y_pred, args.n_boot, args.seed)

    # Determine best 2-model ensemble by unhealthy F1
    ens_names = [n for n in units if n.startswith("Ensemble —")]
    best_ens  = max(ens_names,
                    key=lambda n: point_est[n]["unhealthy_f1"])
    print(f"\n  Best 2-model ensemble : {best_ens}")
    print(f"    unhealthy F1 = {point_est[best_ens]['unhealthy_f1']:.4f}")

    # ------------------------------------------------------------------ #
    # Comparisons                                                          #
    # ------------------------------------------------------------------ #
    comparisons = []   # list of dicts for CSV

    def add_comparison(new_name, ref_name, tag):
        diff_ci = paired_bootstrap_diff(
            y_true,
            units[ref_name], units[new_name],
            args.n_boot, args.seed,
        )
        for metric in METRICS:
            lo, hi   = diff_ci[metric]
            sig      = "Yes" if lo > 0 or hi < 0 else "No"
            comparisons.append({
                "comparison_tag":   tag,
                "new_model":        new_name,
                "ref_model":        ref_name,
                "metric":           metric,
                "new_point":        round(point_est[new_name][metric], 6),
                "ref_point":        round(point_est[ref_name][metric], 6),
                "diff":             round(point_est[new_name][metric]
                                          - point_est[ref_name][metric], 6),
                "diff_ci_lower":    round(float(lo), 6),
                "diff_ci_upper":    round(float(hi), 6),
                "significant":      sig,
            })

    print("\n  Computing paired bootstrap diffs …")

    # A) Every unit vs Ensemble — baseline
    BASELINE = "Ensemble — baseline"
    for uname in tqdm(units, desc="  vs Ensemble-baseline"):
        if uname == BASELINE:
            continue
        add_comparison(uname, BASELINE, "vs_baseline_ensemble")

    # B) Each 2-model ensemble vs its own standalone Model B
    for cond in tqdm(CONDITIONS, desc="  ensemble vs own B"):
        ens  = f"Ensemble — {cond}"
        solo = f"Model B — {cond}"
        add_comparison(ens, solo, "ensemble_vs_solo_b")

    # C) 5-model enlarged vs Ensemble — baseline
    add_comparison("5-model Enlarged", "Ensemble — baseline", "5model_vs_baseline_ensemble")

    # ------------------------------------------------------------------ #
    # Save CSV 1: metrics_ci.csv                                           #
    # ------------------------------------------------------------------ #
    metrics_rows = []
    for uname, pe in point_est.items():
        ci = cis[uname]
        for metric in METRICS:
            lo, hi = ci[metric]
            metrics_rows.append({
                "model":          uname,
                "metric":         metric,
                "point_estimate": round(pe[metric], 6),
                "ci_lower":       round(float(lo), 6),
                "ci_upper":       round(float(hi), 6),
            })

    metrics_path = OUT_DIR / "metrics_ci.csv"
    with open(metrics_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "metric",
                                            "point_estimate", "ci_lower", "ci_upper"])
        w.writeheader()
        w.writerows(metrics_rows)
    print(f"\n  → {metrics_path}")

    # Save CSV 2: comparisons_ci.csv
    comp_path = OUT_DIR / "comparisons_ci.csv"
    comp_fields = ["comparison_tag", "new_model", "ref_model", "metric",
                   "new_point", "ref_point", "diff",
                   "diff_ci_lower", "diff_ci_upper", "significant"]
    with open(comp_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=comp_fields)
        w.writeheader()
        w.writerows(comparisons)
    print(f"  → {comp_path}")

    # Save raw point estimates as JSON for reference
    raw_path = OUT_DIR / "point_estimates.json"
    raw_path.write_text(json.dumps(point_est, indent=2))
    print(f"  → {raw_path}")

    # ------------------------------------------------------------------ #
    # Summary .txt                                                         #
    # ------------------------------------------------------------------ #
    summary_path = OUT_DIR / "summary.txt"
    lines = []
    lines.append("=" * 72)
    lines.append("PAIRED BOOTSTRAP CONFIDENCE INTERVAL ANALYSIS")
    lines.append(f"n_boot={args.n_boot:,}  seed={args.seed}  test_n={N}")
    lines.append("=" * 72)

    # Point estimates table
    lines.append("\n── POINT ESTIMATES (all evaluation units) ──\n")
    header = f"{'Model':<40} {'wF1':>7} {'mF1':>7} {'Acc':>7} {'UhF1':>7} {'UhR':>7}"
    lines.append(header)
    lines.append("-" * len(header))
    for uname, pe in point_est.items():
        lines.append(
            f"{uname:<40} "
            f"{pe['weighted_f1']:>7.4f} "
            f"{pe['macro_f1']:>7.4f} "
            f"{pe['accuracy']:>7.4f} "
            f"{pe['unhealthy_f1']:>7.4f} "
            f"{pe['unhealthy_recall']:>7.4f}"
        )

    # Section A: units vs Ensemble — baseline
    lines.append("\n\n── SECTION A: All units vs Ensemble — baseline ──")
    lines.append("(diff = unit − Ensemble baseline; * = CI excludes zero)\n")
    prev_tag = None
    for row in comparisons:
        if row["comparison_tag"] != "vs_baseline_ensemble":
            continue
        if row["metric"] != "unhealthy_f1":
            continue
        sig = "*" if row["significant"] == "Yes" else " "
        lines.append(
            f"  {sig} {row['new_model']:<38}  "
            f"uh_F1: {row['new_point']:.4f}  "
            f"diff={row['diff']:+.4f}  "
            f"95%CI=[{row['diff_ci_lower']:+.4f}, {row['diff_ci_upper']:+.4f}]  "
            f"sig={row['significant']}"
        )

    # Section B: ensemble vs its own Model B
    lines.append("\n\n── SECTION B: 2-model Ensemble vs standalone Model B ──")
    lines.append("(diff = ensemble − solo B; * = CI excludes zero)\n")
    for row in comparisons:
        if row["comparison_tag"] != "ensemble_vs_solo_b":
            continue
        if row["metric"] != "unhealthy_f1":
            continue
        sig = "*" if row["significant"] == "Yes" else " "
        lines.append(
            f"  {sig} {row['new_model']:<38}  "
            f"uh_F1: {row['new_point']:.4f} vs {row['ref_point']:.4f}  "
            f"diff={row['diff']:+.4f}  "
            f"95%CI=[{row['diff_ci_lower']:+.4f}, {row['diff_ci_upper']:+.4f}]  "
            f"sig={row['significant']}"
        )

    # Section C: 5-model vs Ensemble — baseline
    lines.append("\n\n── SECTION C: 5-model Enlarged vs Ensemble — baseline ──")
    lines.append("(diff = 5-model Enlarged − Ensemble baseline; * = CI excludes zero)\n")
    for row in comparisons:
        if row["comparison_tag"] != "5model_vs_baseline_ensemble":
            continue
        sig = "*" if row["significant"] == "Yes" else " "
        lines.append(
            f"  {sig} [{row['metric']:<22}]  "
            f"5model={row['new_point']:.4f}  baseline_ens={row['ref_point']:.4f}  "
            f"diff={row['diff']:+.4f}  "
            f"95%CI=[{row['diff_ci_lower']:+.4f}, {row['diff_ci_upper']:+.4f}]  "
            f"sig={row['significant']}"
        )

    lines.append("\n" + "=" * 72)
    txt = "\n".join(lines)
    summary_path.write_text(txt, encoding="utf-8")
    print(f"  → {summary_path}")

    # Print summary to terminal as well
    print("\n" + txt)


if __name__ == "__main__":
    main()
