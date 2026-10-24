"""
Post-hoc Logit Adjustment for Model A  (Menon et al., ICLR 2021)
https://arxiv.org/abs/2007.07314

NO RETRAINING — adjusts Model A's raw logits at inference time only.

Formula:
    ŷ = argmax_y [ f_y(x)  −  τ · log(π_y) ]

where π_y = class frequency in training data, τ is a scalar tuned on
the validation split.

Because log(π_unhealthy) ≈ −3.54, subtracting τ·log(π_unhealthy) adds
a large positive offset to the unhealthy logit, correcting the bias
caused by class imbalance during training.

Outputs:
  outputs/metrics/logit_adj_model_a.json   — full metrics for best τ
  outputs/plots/logit_adj_tau_sweep.png    — F1 vs τ sweep plot

Usage:
    python logit_adjustment.py [--tau_values 0.25 0.5 1.0 1.5 2.0]
"""
import argparse
import json
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
    build_test_loader, load_train_df, split_train_val,
    PapCellDataset, get_eval_tf, make_loader,
)
from models.classifier import load_checkpoint

CLASSES  = config.CLASSES   # ["healthy", "rubbish", "unhealthy"]
UH_IDX   = CLASSES.index("unhealthy")
OUT_DIR  = Path("logit_adjustment_results")
OUT_JSON = OUT_DIR / "logit_adj_model_a.json"
OUT_PLOT = OUT_DIR / "logit_adj_tau_sweep.png"


# ---------------------------------------------------------------------------
# Compute class priors from the FULL training CSV (before split)
# ---------------------------------------------------------------------------
def get_class_priors(df) -> np.ndarray:
    counts = np.array([
        (df["target"] == i).sum() for i in range(config.NUM_CLASSES)
    ], dtype=np.float64)
    priors = counts / counts.sum()
    print("  Class priors (training set):")
    for i, c in enumerate(CLASSES):
        print(f"    {c:<12} : {counts[i]:>6.0f}  π={priors[i]:.4f}  "
              f"−log(π)={-np.log(priors[i]):.3f}")
    return priors.astype(np.float32)


# ---------------------------------------------------------------------------
# Inference — returns raw logits [N, C]
# ---------------------------------------------------------------------------
@torch.no_grad()
def get_logits(model, loader, device) -> np.ndarray:
    model.eval()
    all_logits = []
    for imgs, _ in tqdm(loader, desc="  inference", leave=False):
        all_logits.append(model(imgs.to(device)).cpu().numpy())
    return np.concatenate(all_logits, axis=0).astype(np.float32)


@torch.no_grad()
def get_labels(loader) -> np.ndarray:
    labels = []
    for _, lbl in loader:
        labels.extend(lbl.numpy())
    return np.array(labels, dtype=np.int64)


# ---------------------------------------------------------------------------
# Apply logit adjustment: logit_y  −=  τ · log(π_y)
# ---------------------------------------------------------------------------
def adjust_and_predict(logits: np.ndarray, log_priors: np.ndarray,
                       tau: float) -> np.ndarray:
    adjusted = logits - tau * log_priors   # broadcast over N
    return adjusted.argmax(axis=1)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred) -> dict:
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
# τ sweep plot
# ---------------------------------------------------------------------------
def plot_tau_sweep(tau_vals, val_results, test_results_best, best_tau, out):
    uh_f1_val  = [val_results[t]["per_class_f1"]["unhealthy"] for t in tau_vals]
    wf1_val    = [val_results[t]["weighted_f1"]               for t in tau_vals]

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="white")
    ax.plot(tau_vals, uh_f1_val, "o-", color="#E53935", linewidth=2,
            markersize=7, label="Unhealthy F1 (val)")
    ax.plot(tau_vals, wf1_val,   "s--", color="#1565C0", linewidth=2,
            markersize=7, label="Weighted F1 (val)")
    ax.axvline(best_tau, color="#777", linestyle=":", linewidth=1.5,
               label=f"Best τ = {best_tau}")
    ax.set_xlabel("τ (logit adjustment scale)", fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_title("Model A — Post-hoc Logit Adjustment\nValidation F1 vs τ",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau_values", type=float, nargs="+",
                        default=[0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Class priors ────────────────────────────────────────────────────
    print("[1] Computing class priors …")
    df, class_to_idx = load_train_df()
    priors   = get_class_priors(df)
    log_pi   = np.log(priors)   # shape [C], e.g. [-1.04, -0.48, -3.54]
    print(f"\n  log(π) = {log_pi}")

    # ── 2. Build val loader for τ tuning ──────────────────────────────────
    print("\n[2] Building validation loader for τ tuning …")
    df_train, df_val = split_train_val(df, random_state=42)
    val_ds   = PapCellDataset(df_val, config.TRAIN_DIR, get_eval_tf())
    val_loader = make_loader(val_ds)
    print(f"  Val set : {len(val_ds)} images")

    # ── 3. Build test loader ───────────────────────────────────────────────
    print("\n[3] Building test loader …")
    test_loader, _ = build_test_loader(class_to_idx)
    print(f"  Test set : {len(test_loader.dataset)} images")

    # ── 4. Load Model A, get logits once (val + test) ─────────────────────
    print("\n[4] Loading Model A and running inference …")
    model = load_checkpoint(config.PRETRAINED_MODEL_A, device)

    print("  Val logits …")
    val_logits  = get_logits(model, val_loader, device)
    val_labels  = get_labels(val_loader)

    print("  Test logits …")
    test_logits = get_logits(model, test_loader, device)
    test_labels = get_labels(test_loader)

    # Free GPU
    del model
    torch.cuda.empty_cache()

    # ── 5. Baseline (τ=0, no adjustment) ──────────────────────────────────
    baseline_val  = compute_metrics(val_labels,  val_logits.argmax(1))
    baseline_test = compute_metrics(test_labels, test_logits.argmax(1))
    print(f"\n  Baseline (τ=0):")
    print(f"    val   unhealthy F1 = {baseline_val['per_class_f1']['unhealthy']:.4f}")
    print(f"    test  unhealthy F1 = {baseline_test['per_class_f1']['unhealthy']:.4f}")

    # ── 6. τ sweep on validation set ──────────────────────────────────────
    print(f"\n[5] Sweeping τ ∈ {args.tau_values} on validation set …")
    val_results = {}
    for tau in args.tau_values:
        pred = adjust_and_predict(val_logits, log_pi, tau)
        val_results[tau] = compute_metrics(val_labels, pred)
        uh = val_results[tau]["per_class_f1"]["unhealthy"]
        wf = val_results[tau]["weighted_f1"]
        print(f"  τ={tau:.2f}   val unhealthy_F1={uh:.4f}   wF1={wf:.4f}")

    best_tau = max(args.tau_values,
                   key=lambda t: val_results[t]["per_class_f1"]["unhealthy"])
    print(f"\n  Best τ = {best_tau}  "
          f"(val unhealthy F1 = {val_results[best_tau]['per_class_f1']['unhealthy']:.4f})")

    # ── 7. Apply best τ to test set ───────────────────────────────────────
    print(f"\n[6] Evaluating on test set with τ = {best_tau} …")
    test_pred = adjust_and_predict(test_logits, log_pi, best_tau)
    test_adj  = compute_metrics(test_labels, test_pred)

    # ── 8. Save results ───────────────────────────────────────────────────
    output = {
        "method":       "post_hoc_logit_adjustment",
        "model":        "Model A",
        "best_tau":     best_tau,
        "tau_sweep_val": {
            str(t): v for t, v in val_results.items()
        },
        "baseline_test": baseline_test,
        "adjusted_test": test_adj,
        "delta": {
            "unhealthy_f1":    round(test_adj["per_class_f1"]["unhealthy"]
                                     - baseline_test["per_class_f1"]["unhealthy"], 6),
            "unhealthy_recall": round(test_adj["per_class_recall"]["unhealthy"]
                                      - baseline_test["per_class_recall"]["unhealthy"], 6),
            "weighted_f1":     round(test_adj["weighted_f1"]
                                     - baseline_test["weighted_f1"], 6),
        },
    }
    OUT_JSON.write_text(json.dumps(output, indent=2))
    print(f"\n  → {OUT_JSON}")

    # ── 9. Plot ───────────────────────────────────────────────────────────
    plot_tau_sweep(args.tau_values, val_results, test_adj, best_tau, OUT_PLOT)

    # ── 10. Summary ───────────────────────────────────────────────────────
    d = output["delta"]
    print("\n" + "=" * 54)
    print(f"  Model A — Post-hoc Logit Adjustment (τ = {best_tau})")
    print("-" * 54)
    print(f"  {'Metric':<22} {'Baseline':>9} {'Adjusted':>9} {'Δ':>8}")
    print(f"  {'-'*22} {'-'*9} {'-'*9} {'-'*8}")
    for key, label in [
        ("weighted_f1",                "Weighted F1"),
        ("macro_f1",                   "Macro F1"),
        ("accuracy",                   "Accuracy"),
    ]:
        b = baseline_test[key]
        a = test_adj[key]
        print(f"  {label:<22} {b:>9.4f} {a:>9.4f} {a-b:>+8.4f}")
    for cls in CLASSES:
        b = baseline_test["per_class_f1"][cls]
        a = test_adj["per_class_f1"][cls]
        print(f"  {cls+' F1':<22} {b:>9.4f} {a:>9.4f} {a-b:>+8.4f}")
    print("=" * 54)


if __name__ == "__main__":
    main()
