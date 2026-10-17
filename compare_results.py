"""
Scan outputs/metrics/ for *_seed*_metrics.json files and print a formatted
comparison table. Works with however many conditions have finished so far.

Usage:
    python compare_results.py
    python compare_results.py --metrics_dir outputs/metrics

Outputs:
    Prints table to stdout.
    Saves outputs/metrics/comparison_table.csv.
"""
import argparse
import csv
import json
from pathlib import Path

import config

DISPLAY_ORDER = [
    "baseline_ensemble",
    "heavy_aug_ensemble",
    "gan_250_ensemble",
    "gan_500_ensemble",
    "ddpm_250_ensemble",
    "ddpm_500_ensemble",
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


if __name__ == "__main__":
    main()
