"""
Read all *_modelB_log.csv files from outputs/logs/ and save loss + F1 plots
to outputs/plots/.

One PNG per condition:
  outputs/plots/{condition}_seed{N}_curves.png

Each PNG contains two side-by-side subplots:
  Left  — Train loss & Val loss vs epoch (phase boundary marked)
  Right — Val weighted F1 vs epoch (best point marked)

Usage:
    python plot_curves.py
    python plot_curves.py --logs_dir outputs/logs --out_dir outputs/plots
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

import config


PHASE_COLOR  = "#7c3aed"
TRAIN_COLOR  = "#2563eb"
VAL_COLOR    = "#ea580c"
F1_COLOR     = "#059669"
BEST_COLOR   = "#dc2626"


def plot_condition(df: pd.DataFrame, condition: str, out_path: Path):
    phase_boundary = df.loc[df["phase"] == "phase2", "epoch"].iloc[0] - 0.5 \
                     if "phase2" in df["phase"].values else None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor("#0e1117")
    for ax in axes:
        ax.set_facecolor("#141828")
        ax.tick_params(colors="#8892a4", labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#252d3d")
        ax.xaxis.label.set_color("#8892a4")
        ax.yaxis.label.set_color("#8892a4")
        ax.title.set_color("#dce3f5")
        ax.grid(color="#252d3d", linewidth=0.6, linestyle="--")
        ax.set_axisbelow(True)

    epochs = df["epoch"].values

    # ── subplot 1: losses ────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(epochs, df["train_loss"], color=TRAIN_COLOR, linewidth=1.8,
            label="Train loss", zorder=3)
    ax.plot(epochs, df["val_loss"],   color=VAL_COLOR,   linewidth=1.8,
            label="Val loss",   zorder=3)
    if phase_boundary:
        ax.axvline(phase_boundary, color=PHASE_COLOR, linewidth=1.2,
                   linestyle="--", alpha=0.8, zorder=2)
        ax.text(phase_boundary + 0.4, ax.get_ylim()[1] * 0.97,
                "P2", color=PHASE_COLOR, fontsize=8, va="top",
                fontfamily="monospace")
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("Loss",  fontsize=9)
    ax.set_title("Loss Curves", fontsize=10, fontweight="semibold", pad=8)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    leg = ax.legend(fontsize=8, framealpha=0.2,
                    labelcolor="linecolor", facecolor="#141828",
                    edgecolor="#252d3d")
    for text in leg.get_texts():
        text.set_color("#dce3f5")

    # ── subplot 2: val wF1 ───────────────────────────────────────────────────
    ax = axes[1]
    f1_vals  = df["val_wf1"].values
    best_idx = int(f1_vals.argmax())
    ax.plot(epochs, f1_vals, color=F1_COLOR, linewidth=1.8,
            label="Val wF1", zorder=3)
    ax.fill_between(epochs, f1_vals,
                    alpha=0.12, color=F1_COLOR, zorder=2)
    ax.scatter([epochs[best_idx]], [f1_vals[best_idx]],
               color=BEST_COLOR, s=55, zorder=5,
               label=f"Best: {f1_vals[best_idx]:.4f} (ep {epochs[best_idx]})")
    if phase_boundary:
        ax.axvline(phase_boundary, color=PHASE_COLOR, linewidth=1.2,
                   linestyle="--", alpha=0.8, zorder=2)
        ax.text(phase_boundary + 0.4,
                ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.97,
                "P2", color=PHASE_COLOR, fontsize=8, va="top",
                fontfamily="monospace")
    ax.set_xlabel("Epoch",     fontsize=9)
    ax.set_ylabel("wF1",       fontsize=9)
    ax.set_title("Validation Weighted F1", fontsize=10,
                 fontweight="semibold", pad=8)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    leg2 = ax.legend(fontsize=8, framealpha=0.2,
                     facecolor="#141828", edgecolor="#252d3d")
    for text in leg2.get_texts():
        text.set_color("#dce3f5")

    status = "" if len(df) >= 35 else f"  [{len(df)} epochs — in progress]"
    fig.suptitle(f"{condition}{status}",
                 color="#dce3f5", fontsize=11, fontweight="semibold", y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_dir", type=Path, default=config.LOG_DIR)
    parser.add_argument("--out_dir",  type=Path, default=config.PLOTS_DIR)
    args = parser.parse_args()

    log_files = sorted(args.logs_dir.glob("*_modelB_log.csv"))
    if not log_files:
        print(f"No *_modelB_log.csv files found in {args.logs_dir}")
        return

    print(f"Found {len(log_files)} log file(s):\n")
    for log_path in log_files:
        # parse condition + seed from filename: {condition}_seed{N}_modelB_log.csv
        stem  = log_path.stem                        # e.g. baseline_ensemble_seed42_modelB_log
        stem  = stem.replace("_modelB_log", "")      # -> baseline_ensemble_seed42
        parts = stem.rsplit("_seed", 1)
        condition = parts[0]
        seed      = parts[1] if len(parts) == 2 else "?"

        df = pd.read_csv(log_path)
        out_path = args.out_dir / f"{condition}_seed{seed}_curves.png"
        print(f"  Plotting {condition} (seed={seed}, {len(df)} epochs) ...")
        plot_condition(df, condition, out_path)

    print(f"\nAll plots saved to {args.out_dir}")


if __name__ == "__main__":
    main()
