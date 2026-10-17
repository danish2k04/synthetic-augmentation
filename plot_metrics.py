"""
Comprehensive metrics visualisation — all conditions, all metrics.

Reads every *_metrics.json (including pretrained_ensemble_metrics.json)
and saves a 4-panel comparison chart.

Usage:
    python plot_metrics.py

Output:
    outputs/plots/metrics_full.png
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

import config

OUT_PATH = config.PLOTS_DIR / "metrics_full.png"

# Display order + short axis labels
ORDER = [
    ("pretrained_ensemble",       "Pretrained\n(original)"),
    ("baseline_ensemble",         "Baseline"),
    ("heavy_aug_ensemble",        "Heavy Aug"),
    ("gan_250_ensemble",          "GAN-250"),
    ("gan_500_ensemble",          "GAN-500"),
    ("ddpm_250_ensemble",         "DDPM-250"),
    ("ddpm_500_ensemble",         "DDPM-500"),
    ("ddpm_heavy_500_ensemble",   "DDPM-H\n500"),
]

CLASSES = config.CLASSES  # ["healthy", "rubbish", "unhealthy"]


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
BAR_COLOURS = {
    "weighted_f1": "#1565C0",
    "macro_f1":    "#0288D1",
    "accuracy":    "#00897B",
    "healthy":     "#43A047",
    "rubbish":     "#FB8C00",
    "unhealthy":   "#E53935",
}


# ---------------------------------------------------------------------------
# Data loading & normalisation
# ---------------------------------------------------------------------------

def _parse_report(report: str) -> dict:
    """Extract per-class precision/recall from a sklearn classification_report string."""
    out = {}
    for cls in CLASSES:
        m = re.search(
            rf"\s{cls}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
            report
        )
        if m:
            out[cls] = {"precision": float(m.group(1)),
                        "recall":    float(m.group(2)),
                        "f1":        float(m.group(3))}
    return out


def _get(m: dict, cls: str, metric: str) -> float:
    """Unified getter that handles both nested and flat JSON layouts."""
    # flat layout: unhealthy_f1 / healthy_f1 etc.
    flat = m.get(f"{cls}_{metric}")
    if flat is not None:
        return float(flat)
    # nested layout: per_class_f1 / per_class_precision / per_class_recall
    key_map = {"f1": "per_class_f1",
               "precision": "per_class_precision",
               "recall": "per_class_recall"}
    nested_key = key_map.get(metric)
    if nested_key:
        val = m.get(nested_key, {}).get(cls)
        if val is not None:
            return float(val)
    # fall back to parsed classification_report
    parsed = m.get("_parsed", {})
    val = parsed.get(cls, {}).get(metric)
    return float(val) if val is not None else 0.0


def load_metrics(metrics_dir: Path) -> dict:
    """Load all *_metrics.json, normalise keys, return {condition: data}."""
    results = {}

    # pretrained has no _seed_ in the name, handle separately
    pretrained = metrics_dir / "pretrained_ensemble_metrics.json"
    if pretrained.exists():
        data = json.loads(pretrained.read_text())
        if "classification_report" in data:
            data["_parsed"] = _parse_report(data["classification_report"])
        results["pretrained_ensemble"] = data

    for path in sorted(metrics_dir.glob("*_seed*_metrics.json")):
        stem  = path.stem[: -len("_metrics")]   # strip _metrics
        parts = stem.rsplit("_seed", 1)
        if len(parts) != 2:
            continue
        cond = parts[0]
        data = json.loads(path.read_text())
        if "classification_report" in data:
            data["_parsed"] = _parse_report(data["classification_report"])
        results[cond] = data

    return results


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _bar_group(ax, x, values_list, labels_list, colours, width=0.22,
               show_val=True, val_fmt="{:.3f}", fontsize=7):
    n = len(values_list)
    offsets = np.linspace(-(n - 1) * width / 2, (n - 1) * width / 2, n)
    bars_all = []
    for off, vals, col in zip(offsets, values_list, colours):
        bars = ax.bar(x + off, vals, width, color=col, alpha=0.88)
        bars_all.append(bars)
        if show_val:
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            h + 0.004, val_fmt.format(h),
                            ha="center", va="bottom", fontsize=fontsize,
                            rotation=90)
    return bars_all


def _setup_ax(ax, x, xlabels, title, ylim=(0, 1.12)):
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=8.5)
    ax.set_ylim(*ylim)
    ax.set_ylabel("Score", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.grid(axis="y", alpha=0.3, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------

def make_plot(results: dict, out: Path):
    # Build ordered condition list (only those present in results)
    ordered = [(k, lbl) for k, lbl in ORDER if k in results]
    # Append any extra conditions not in ORDER
    known = {k for k, _ in ORDER}
    for k in sorted(results):
        if k not in known:
            ordered.append((k, k.replace("_", "\n")))

    keys   = [k   for k, _ in ordered]
    xlbls  = [lbl for _, lbl in ordered]
    x      = np.arange(len(keys))

    # ---- Collect all values ----
    wf1      = [results[k].get("weighted_f1", 0) for k in keys]
    macro_f1 = [results[k].get("macro_f1",    0) for k in keys]
    acc      = [results[k].get("accuracy",     0) for k in keys]

    prec  = {c: [_get(results[k], c, "precision") for k in keys] for c in CLASSES}
    rec   = {c: [_get(results[k], c, "recall")    for k in keys] for c in CLASSES}
    f1    = {c: [_get(results[k], c, "f1")        for k in keys] for c in CLASSES}

    # ---- Figure: 2×2 ----
    fig, axes = plt.subplots(2, 2, figsize=(20, 13))
    fig.suptitle("Ensemble Classifier — Full Metrics Comparison (seed 42)",
                 fontsize=14, fontweight="bold", y=1.01)

    # ── Panel 1: Overall metrics ──────────────────────────────────────────
    ax = axes[0, 0]
    _bar_group(ax, x,
               [wf1, macro_f1, acc],
               ["Weighted F1", "Macro F1", "Accuracy"],
               [BAR_COLOURS["weighted_f1"], BAR_COLOURS["macro_f1"],
                BAR_COLOURS["accuracy"]])
    _setup_ax(ax, x, xlbls, "Overall Metrics")
    ax.legend(handles=[
        mpatches.Patch(color=BAR_COLOURS["weighted_f1"], label="Weighted F1"),
        mpatches.Patch(color=BAR_COLOURS["macro_f1"],    label="Macro F1"),
        mpatches.Patch(color=BAR_COLOURS["accuracy"],    label="Accuracy"),
    ], fontsize=8, loc="lower right")

    # ── Panel 2: Unhealthy — Precision / Recall / F1 ────────────────────
    ax = axes[0, 1]
    _bar_group(ax, x,
               [prec["unhealthy"], rec["unhealthy"], f1["unhealthy"]],
               ["Precision", "Recall", "F1"],
               ["#EF5350", "#E53935", "#B71C1C"])
    _setup_ax(ax, x, xlbls, "Unhealthy Class — Precision / Recall / F1",
              ylim=(0, 0.85))
    ax.legend(handles=[
        mpatches.Patch(color="#EF5350", label="Precision"),
        mpatches.Patch(color="#E53935", label="Recall"),
        mpatches.Patch(color="#B71C1C", label="F1"),
    ], fontsize=8, loc="lower right")

    # ── Panel 3: Per-class F1 ─────────────────────────────────────────────
    ax = axes[1, 0]
    _bar_group(ax, x,
               [f1["healthy"], f1["rubbish"], f1["unhealthy"]],
               ["Healthy F1", "Rubbish F1", "Unhealthy F1"],
               [BAR_COLOURS["healthy"], BAR_COLOURS["rubbish"],
                BAR_COLOURS["unhealthy"]])
    _setup_ax(ax, x, xlbls, "Per-class F1")
    ax.legend(handles=[
        mpatches.Patch(color=BAR_COLOURS["healthy"],   label="Healthy"),
        mpatches.Patch(color=BAR_COLOURS["rubbish"],   label="Rubbish"),
        mpatches.Patch(color=BAR_COLOURS["unhealthy"], label="Unhealthy"),
    ], fontsize=8, loc="lower right")

    # ── Panel 4: Per-class Recall ─────────────────────────────────────────
    ax = axes[1, 1]
    _bar_group(ax, x,
               [rec["healthy"], rec["rubbish"], rec["unhealthy"]],
               ["Healthy Recall", "Rubbish Recall", "Unhealthy Recall"],
               [BAR_COLOURS["healthy"], BAR_COLOURS["rubbish"],
                BAR_COLOURS["unhealthy"]])
    _setup_ax(ax, x, xlbls, "Per-class Recall")
    ax.legend(handles=[
        mpatches.Patch(color=BAR_COLOURS["healthy"],   label="Healthy"),
        mpatches.Patch(color=BAR_COLOURS["rubbish"],   label="Rubbish"),
        mpatches.Patch(color=BAR_COLOURS["unhealthy"], label="Unhealthy"),
    ], fontsize=8, loc="lower right")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    results = load_metrics(config.METRICS_DIR)
    if not results:
        print("No metrics files found in", config.METRICS_DIR)
        return
    print(f"Loaded {len(results)} condition(s): {', '.join(sorted(results))}")
    make_plot(results, OUT_PATH)


if __name__ == "__main__":
    main()
