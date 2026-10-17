"""
Scan outputs/metrics/ for *_seed*_metrics.json files, print a formatted
comparison table, and save a bar-chart PNG.

Usage:
    python compare_results.py
    python compare_results.py --metrics_dir outputs/metrics

Outputs:
    Prints table to stdout.
    Saves outputs/metrics/comparison_table.csv.
    Saves outputs/plots/results_comparison.png.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config

DISPLAY_ORDER = [
    "baseline_ensemble",
    "heavy_aug_ensemble",
    "gan_250_ensemble",
    "gan_500_ensemble",
    "ddpm_250_ensemble",
    "ddpm_500_ensemble",
    "ddpm_heavy_500_ensemble",
]

# (header, dotted-key, right-align width)
COLUMNS = [
    ("Condition",    "condition",                        28),
    ("wF1",          "weighted_f1",                       7),
    ("mF1",          "macro_f1",                          7),
    ("Acc",          "accuracy",                          7),
    ("Unhealthy P",  "per_class_precision.unhealthy",    11),
    ("Unhealthy R",  "per_class_recall.unhealthy",       11),
    ("Unhealthy F1", "per_class_f1.unhealthy",           11),
    ("Time(min)",    "training_time_minutes",              9),
]


def get_field(m: dict, key: str):
    if "." in key:
        outer, inner = key.split(".", 1)
        return m.get(outer, {}).get(inner, "")
    return m.get(key, "")


def fmt(val) -> str:
    if isinstance(val, float):
        return f"{val:.4f}"
    if val == "" or val is None:
        return "—"
    return str(val)


def load_all_metrics(metrics_dir: Path) -> dict[str, dict]:
    results = {}
    for path in sorted(metrics_dir.glob("*_metrics.json")):
        stem = path.stem  # e.g. baseline_ensemble_seed42_metrics
        if not stem.endswith("_metrics"):
            continue
        stem = stem[: -len("_metrics")]          # strip _metrics suffix
        parts = stem.rsplit("_seed", 1)
        if len(parts) != 2:
            continue                              # skip pretrained_ensemble_metrics etc.
        condition = parts[0]
        try:
            data = json.loads(path.read_text())
            results[condition] = data
        except Exception as e:
            print(f"  Warning: could not load {path.name}: {e}")
    return results


def build_rows(results: dict) -> list[list[str]]:
    ordered = [k for k in DISPLAY_ORDER if k in results]
    ordered += sorted(k for k in results if k not in DISPLAY_ORDER)
    rows = []
    for key in ordered:
        m = results[key]
        row = []
        for _, col_key, _ in COLUMNS:
            val = get_field(m, col_key)
            row.append(fmt(val) if col_key != "condition" else key)
        rows.append(row)
    return rows


def print_table(headers: list[str], rows: list[list[str]]):
    widths = [
        max(len(h), max((len(r[i]) for r in rows), default=0))
        for i, h in enumerate(headers)
    ]
    sep = " | "
    header_line = sep.join(h.ljust(widths[i]) for i, h in enumerate(headers))
    divider     = "-" * len(header_line)
    print()
    print(header_line)
    print(divider)
    for row in rows:
        print(sep.join(v.ljust(widths[i]) for i, v in enumerate(row)))
    print()


# Short display labels for the chart x-axis
SHORT_LABELS = {
    "baseline_ensemble":       "Baseline",
    "heavy_aug_ensemble":      "Heavy Aug",
    "gan_250_ensemble":        "GAN-250",
    "gan_500_ensemble":        "GAN-500",
    "ddpm_250_ensemble":       "DDPM-250",
    "ddpm_500_ensemble":       "DDPM-500",
    "ddpm_heavy_500_ensemble": "DDPM-H-500",
}


def _get_f1(m: dict, cls: str) -> float:
    """Handle both nested per_class_f1.X and flat X_f1 key layouts."""
    nested = m.get("per_class_f1", {}).get(cls)
    if nested is not None:
        return float(nested)
    flat = m.get(f"{cls}_f1")
    if flat is not None:
        return float(flat)
    return 0.0


def save_plot(results: dict, out_path: Path):
    ordered = [k for k in DISPLAY_ORDER if k in results]
    ordered += sorted(k for k in results if k not in DISPLAY_ORDER)

    labels    = [SHORT_LABELS.get(k, k) for k in ordered]
    wf1       = [results[k].get("weighted_f1",  0.0) for k in ordered]
    healthy   = [_get_f1(results[k], "healthy")      for k in ordered]
    rubbish   = [_get_f1(results[k], "rubbish")      for k in ordered]
    unhealthy = [_get_f1(results[k], "unhealthy")    for k in ordered]

    x   = np.arange(len(labels))
    w   = 0.20
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ---- Left: weighted F1 + unhealthy F1 ----
    ax = axes[0]
    b1 = ax.bar(x - w/2, wf1,       w, label="Weighted F1", color="#1976D2")
    b2 = ax.bar(x + w/2, unhealthy, w, label="Unhealthy F1", color="#E53935")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 Score")
    ax.set_title("Weighted F1 vs Unhealthy F1\nper condition", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.35)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    # ---- Right: per-class F1 breakdown ----
    ax = axes[1]
    offsets = [-w, 0, w]
    colours = ["#43A047", "#FB8C00", "#E53935"]
    class_names = ["Healthy", "Rubbish", "Unhealthy"]
    for off, col, name, vals in zip(offsets, colours, class_names,
                                    [healthy, rubbish, unhealthy]):
        bars = ax.bar(x + off, vals, w, label=name, color=col)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-class F1 breakdown\nper condition", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.35)

    fig.suptitle("Ensemble Results — All Conditions (seed 42)", fontsize=13, y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved  -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir", type=Path, default=config.METRICS_DIR)
    args = parser.parse_args()

    metrics_dir = args.metrics_dir
    if not metrics_dir.exists():
        print(f"Metrics directory not found: {metrics_dir}")
        return

    results = load_all_metrics(metrics_dir)
    if not results:
        print("No standardised *_seed*_metrics.json files found yet.")
        return

    headers = [col[0] for col in COLUMNS]
    rows    = build_rows(results)

    print_table(headers, rows)
    print(f"  {len(rows)} condition(s) loaded from {metrics_dir}")

    csv_path = metrics_dir / "comparison_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"  Table saved -> {csv_path}")

    plot_path = config.PLOTS_DIR / "results_comparison.png"
    save_plot(results, plot_path)


if __name__ == "__main__":
    main()
