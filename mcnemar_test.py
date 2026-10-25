"""
McNemar's exact test comparing every evaluation unit against Ensemble — baseline
(Model A + baseline Model B, softmax-averaged).

Two tests per unit:
  A) Overall    — all 18,159 test images
  B) Unhealthy  — only the 576 test images whose true label is unhealthy

Outputs (in mcnemar_results/):
  mcnemar_overall.csv    — overall McNemar results
  mcnemar_unhealthy.csv  — unhealthy-only McNemar results
  summary.txt            — human-readable side-by-side table

Usage:
    python mcnemar_test.py
"""
import csv
import gc
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from statsmodels.stats.contingency_tables import mcnemar
from tqdm import tqdm

import config
from dataset import build_test_loader, load_train_df
from models.classifier import load_checkpoint

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CKPT = config.CHECKPOINT_DIR

CONDITIONS = {
    "baseline":                   CKPT / "baseline_ensemble_seed42_modelB.pt",
    "heavy_aug":                  CKPT / "heavy_aug_ensemble_seed42_modelB.pt",
    "gan_250":                    CKPT / "gan_250_ensemble_seed42_modelB.pt",
    "gan_500":                    CKPT / "gan_500_ensemble_seed42_modelB.pt",
    "ddpm_250":                   CKPT / "ddpm_250_ensemble_seed42_modelB.pt",
    "ddpm_500":                   CKPT / "ddpm_500_ensemble_seed42_modelB.pt",
    "ddpm_heavy_500":             CKPT / "ddpm_heavy_500_ensemble" / "model_b_seed42.pth",
    "combined_ddpm500_heavy_aug": CKPT / "combined_ddpm500_heavy_aug" / "model_b_seed42.pth",
}

CKPT_MODEL_B_PRETRAINED = config.PRETRAINED_MODEL_B

OUT_DIR = Path("mcnemar_results")
UH_IDX  = config.CLASSES.index("unhealthy")


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
# McNemar helper
# ---------------------------------------------------------------------------
def run_mcnemar(y_true: np.ndarray,
                ref_pred: np.ndarray,
                other_pred: np.ndarray,
                mask: np.ndarray | None = None) -> dict:
    """
    Build the 2x2 contingency table and run McNemar's exact test.
    mask: optional boolean array to restrict to a subset of samples.
    Returns dict with b, c, statistic, p_value.
    """
    if mask is not None:
        y_true    = y_true[mask]
        ref_pred  = ref_pred[mask]
        other_pred = other_pred[mask]

    ref_correct   = (ref_pred   == y_true)
    other_correct = (other_pred == y_true)

    # 2x2 table:
    #           other correct  other wrong
    # ref correct     a              b
    # ref wrong       c              d
    a = int(np.sum( ref_correct &  other_correct))
    b = int(np.sum( ref_correct & ~other_correct))
    c = int(np.sum(~ref_correct &  other_correct))
    d = int(np.sum(~ref_correct & ~other_correct))

    table = np.array([[a, b], [c, d]])
    result = mcnemar(table, exact=True)

    return {
        "b": b,
        "c": c,
        "statistic": float(result.statistic),
        "p_value":   float(result.pvalue),
        "significant_at_0.05": "Yes" if result.pvalue < 0.05 else "No",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Build test loader                                                 #
    # ------------------------------------------------------------------ #
    print("\n[1/4] Building test loader ...")
    _, class_to_idx = load_train_df()
    loader, _        = build_test_loader(class_to_idx)
    y_true           = get_labels(loader)
    N                = len(y_true)
    uh_mask          = (y_true == UH_IDX)
    print(f"  Test set   : {N} samples")
    print(f"  Unhealthy  : {uh_mask.sum()} samples")

    # ------------------------------------------------------------------ #
    # 2. Run inference — one model at a time                               #
    # ------------------------------------------------------------------ #
    print("\n[2/4] Running inference ...")
    store = {}

    print("  Model A ...")
    store["_model_a"] = get_probs(config.PRETRAINED_MODEL_A, loader, device)

    print("  Pretrained Model B ...")
    store["_model_b_pretrained"] = get_probs(CKPT_MODEL_B_PRETRAINED, loader, device)

    for cond, ckpt in CONDITIONS.items():
        print(f"  Model B [{cond}] ...")
        store[f"_b_{cond}"] = get_probs(ckpt, loader, device)

    # ------------------------------------------------------------------ #
    # 3. Build evaluation units                                            #
    # ------------------------------------------------------------------ #
    print("\n[3/4] Building evaluation units ...")
    units = {}

    units["Model A"] = store["_model_a"].argmax(1)

    for cond in CONDITIONS:
        b_key = f"_b_{cond}"
        units[f"Model B — {cond}"]  = store[b_key].argmax(1)
        ens = ensemble_probs(store["_model_a"], store[b_key])
        units[f"Ensemble — {cond}"] = ens.argmax(1)

    five_probs = ensemble_probs(
        store["_model_a"],
        store["_model_b_pretrained"],
        store["_b_gan_500"],
        store["_b_ddpm_500"],
        store["_b_ddpm_heavy_500"],
    )
    units["5-model Enlarged"] = five_probs.argmax(1)

    # ------------------------------------------------------------------ #
    # 4. McNemar tests                                                     #
    # ------------------------------------------------------------------ #
    print("\n[4/4] Running McNemar tests ...")
    REFERENCE = "Ensemble — baseline"
    ref_pred  = units[REFERENCE]

    overall_rows   = []
    unhealthy_rows = []

    for uname, other_pred in units.items():
        if uname == REFERENCE:
            continue

        res_overall = run_mcnemar(y_true, ref_pred, other_pred, mask=None)
        res_uh      = run_mcnemar(y_true, ref_pred, other_pred, mask=uh_mask)

        overall_rows.append({"condition": uname, **res_overall})
        unhealthy_rows.append({"condition": uname, **res_uh})

        print(f"  {uname:<42}  "
              f"overall p={res_overall['p_value']:.4f} ({res_overall['significant_at_0.05']})  "
              f"uh p={res_uh['p_value']:.4f} ({res_uh['significant_at_0.05']})")

    # ------------------------------------------------------------------ #
    # 5. Save CSVs                                                         #
    # ------------------------------------------------------------------ #
    fields = ["condition", "b_baseline_correct_other_wrong",
              "c_baseline_wrong_other_correct", "statistic", "p_value",
              "significant_at_0.05"]

    def rename_bc(rows):
        out = []
        for r in rows:
            out.append({
                "condition":                       r["condition"],
                "b_baseline_correct_other_wrong":  r["b"],
                "c_baseline_wrong_other_correct":  r["c"],
                "statistic":                       round(r["statistic"], 6),
                "p_value":                         round(r["p_value"], 6),
                "significant_at_0.05":             r["significant_at_0.05"],
            })
        return out

    overall_out   = rename_bc(overall_rows)
    unhealthy_out = rename_bc(unhealthy_rows)

    csv_overall = OUT_DIR / "mcnemar_overall.csv"
    with open(csv_overall, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(overall_out)
    print(f"\n-> {csv_overall}")

    csv_uh = OUT_DIR / "mcnemar_unhealthy.csv"
    with open(csv_uh, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(unhealthy_out)
    print(f"-> {csv_uh}")

    # ------------------------------------------------------------------ #
    # 6. Summary text                                                      #
    # ------------------------------------------------------------------ #
    lines = []
    lines.append("McNemar's Exact Test — reference: Ensemble — baseline")
    lines.append(f"Overall N={N}   Unhealthy N={int(uh_mask.sum())}")
    lines.append("=" * 80)
    lines.append(
        f"  {'Condition':<42}  "
        f"{'Overall p':>10}  {'Sig':>4}  "
        f"{'Unhealthy p':>11}  {'Sig':>4}"
    )
    lines.append("  " + "-" * 76)

    for r_ov, r_uh in zip(overall_out, unhealthy_out):
        sig_ov = "*" if r_ov["significant_at_0.05"] == "Yes" else " "
        sig_uh = "*" if r_uh["significant_at_0.05"] == "Yes" else " "
        lines.append(
            f"  {r_ov['condition']:<42}  "
            f"{r_ov['p_value']:>10.4f}  {sig_ov:>4}  "
            f"{r_uh['p_value']:>11.4f}  {sig_uh:>4}"
        )

    lines.append("")
    lines.append("* = significant at p < 0.05 (raw, no correction)")
    lines.append("b = baseline correct, other wrong")
    lines.append("c = baseline wrong, other correct")

    txt_path = OUT_DIR / "summary.txt"
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"-> {txt_path}")

    # Print summary to terminal
    print()
    for ln in lines:
        print(ln)


if __name__ == "__main__":
    main()
