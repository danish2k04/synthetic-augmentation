"""
Resume heavy_aug_ensemble training from the existing checkpoint.

Reads the training log to determine the last completed phase-2 epoch,
fast-forwards the LR scheduler to that point, then continues until either
max_epochs or early stopping. Appends to the existing log. When done,
evaluates the ensemble and saves the standardised metrics JSON.

Usage:
    python finish_heavy_aug.py [--seed 42]
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score, confusion_matrix,
)
from tqdm import tqdm

import config
from dataset import (
    load_train_df, split_train_val, compute_class_weights, make_sampler,
    build_model_b_loaders_heavy, build_test_loader,
)
from models.classifier import (
    build_model, _make_cosine_warmup_fn, _run_epoch, set_seed, load_checkpoint,
)
from models.ensemble import ensemble_predict

CONDITION = "heavy_aug_ensemble"


def evaluate_ensemble(model_a, model_b, test_loader, device) -> dict:
    preds, labels = ensemble_predict(model_a, model_b, test_loader, device)
    wf1      = f1_score(labels, preds, average="weighted", zero_division=0)
    mf1      = f1_score(labels, preds, average="macro",    zero_division=0)
    acc      = accuracy_score(labels, preds)
    prec_cls = precision_score(labels, preds, average=None, zero_division=0).tolist()
    rec_cls  = recall_score(labels, preds,    average=None, zero_division=0).tolist()
    f1_cls   = f1_score(labels, preds,        average=None, zero_division=0).tolist()
    cm       = confusion_matrix(labels, preds).tolist()
    return {
        "weighted_f1":         round(float(wf1), 6),
        "macro_f1":            round(float(mf1), 6),
        "accuracy":            round(float(acc), 6),
        "per_class_precision": {c: round(prec_cls[i], 6) for i, c in enumerate(config.CLASSES)},
        "per_class_recall":    {c: round(rec_cls[i],  6) for i, c in enumerate(config.CLASSES)},
        "per_class_f1":        {c: round(f1_cls[i],   6) for i, c in enumerate(config.CLASSES)},
        "confusion_matrix":    cm,
        "unhealthy_recall":    round(float(rec_cls[config.UNHEALTHY_IDX]), 6),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed      = args.seed
    ckpt_path = config.CHECKPOINT_DIR / f"{CONDITION}_seed{seed}_modelB.pt"
    log_path  = config.LOG_DIR        / f"{CONDITION}_seed{seed}_modelB_log.csv"
    out_path  = config.METRICS_DIR    / f"{CONDITION}_seed{seed}_metrics.json"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device    : {device}")
    print(f"Condition : {CONDITION}  seed={seed}")

    # ── Read log (may be empty if a prior run was killed before writing) ──────
    import pandas as pd
    log_df    = pd.read_csv(log_path)
    phase2_df = log_df[log_df["phase"] == "phase2"] if "phase" in log_df.columns else pd.DataFrame()

    ckpt_data     = torch.load(ckpt_path, map_location=device)
    ckpt_epoch    = ckpt_data.get("epoch", 0)          # global epoch saved
    ckpt_val_wf1  = ckpt_data.get("val_wf1", 0.0)

    cfg = config.MODEL_B

    if phase2_df.empty:
        # Log was wiped by a restart — derive state from checkpoint alone
        last_p2_epoch = max(0, ckpt_epoch - cfg["phase1_epochs"])
        best_val_wf1  = ckpt_val_wf1
        no_improve    = 0
        print(f"Log is empty — using checkpoint metadata.")
    else:
        last_p2_epoch = len(phase2_df)
        best_val_wf1  = float(phase2_df["val_wf1"].max())
        wf1_series    = phase2_df["val_wf1"].values
        no_improve    = 0
        for v in reversed(wf1_series):
            if v >= best_val_wf1:
                break
            no_improve += 1

    remaining = cfg["phase2_max_epochs"] - last_p2_epoch
    print(f"Checkpoint epoch    : {ckpt_epoch}  (phase-2 ep {last_p2_epoch})")
    print(f"Best val wF1        : {best_val_wf1:.6f}")
    print(f"Patience counter    : {no_improve} / {cfg['phase2_patience']}")
    print(f"Epochs remaining    : {remaining}")

    if remaining <= 0 or no_improve >= cfg["phase2_patience"]:
        print("Training already complete (max epochs or patience reached). Skipping to eval.")
    else:
        # ── Data ──────────────────────────────────────────────────────────────
        set_seed(seed)
        df, class_to_idx = load_train_df()
        df_train, df_val = split_train_val(df, random_state=seed)
        cw        = compute_class_weights(df_train)
        cw_tensor = torch.tensor(cw)
        sampler   = make_sampler(df_train, cw)
        train_loader, val_loader = build_model_b_loaders_heavy(df_train, df_val, sampler)

        # ── Load checkpoint into model ────────────────────────────────────────
        model     = build_model().to(device)
        ckpt_data = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt_data.get("model_state", ckpt_data))
        print(f"\nLoaded checkpoint (epoch {ckpt_data.get('epoch','?')}, "
              f"val_wf1={ckpt_data.get('val_wf1',0):.6f})")

        criterion = nn.CrossEntropyLoss(
            weight=cw_tensor.to(device),
            label_smoothing=config.LABEL_SMOOTHING,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["phase2_lr"],
            weight_decay=cfg["phase2_weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=_make_cosine_warmup_fn(cfg["warmup_epochs"], cfg["phase2_max_epochs"]),
        )
        scaler = GradScaler('cuda')

        # Fast-forward scheduler to the correct LR for the next epoch
        for _ in range(last_p2_epoch):
            scheduler.step()

        # ── Append to existing log ────────────────────────────────────────────
        fh = open(log_path, "a", newline="")
        writer = csv.writer(fh)

        print(f"\n[B] Resuming phase-2 from epoch {last_p2_epoch + 1} ...")
        t0 = time.time()

        for epoch in range(last_p2_epoch + 1, cfg["phase2_max_epochs"] + 1):
            lr = optimizer.param_groups[0]["lr"]
            tr_loss, _ = _run_epoch(
                model, train_loader, criterion, optimizer, scaler, device, train=True,
                desc=f"[B] P2 ep{epoch:03d}/{cfg['phase2_max_epochs']} train",
            )
            val_loss, val_wf1 = _run_epoch(
                model, val_loader, criterion, None, None, device, train=False,
                desc=f"[B] P2 ep{epoch:03d}/{cfg['phase2_max_epochs']} val  ",
            )
            scheduler.step()

            global_epoch = cfg["phase1_epochs"] + epoch
            writer.writerow([global_epoch, "phase2",
                             f"{tr_loss:.6f}", f"{val_loss:.6f}",
                             f"{val_wf1:.6f}", f"{lr:.2e}"])
            fh.flush()
            print(f"[B] P2 ep{epoch:03d} lr={lr:.2e} "
                  f"train_loss={tr_loss:.4f} val_wF1={val_wf1:.4f}")

            if val_wf1 > best_val_wf1:
                best_val_wf1, no_improve = val_wf1, 0
                torch.save({"epoch": global_epoch, "model_state": model.state_dict(),
                            "val_wf1": val_wf1}, ckpt_path)
                print(f"  ** New best: {best_val_wf1:.6f} — checkpoint updated")
            else:
                no_improve += 1
                if no_improve >= cfg["phase2_patience"]:
                    print(f"[B] Early stopping at phase-2 epoch {epoch}. "
                          f"Best val wF1={best_val_wf1:.4f}")
                    break

        fh.close()
        training_minutes = (time.time() - t0) / 60.0
        print(f"\nResume training done. Time: {training_minutes:.1f} min")

    # ── Evaluate ensemble ─────────────────────────────────────────────────────
    print(f"\nEvaluating ensemble (Model A + {CONDITION} Model B) ...")
    _, class_to_idx = load_train_df()
    test_loader, _  = build_test_loader(class_to_idx)

    model_a = load_checkpoint(config.PRETRAINED_MODEL_A, device)
    model_b = load_checkpoint(ckpt_path, device)
    metrics = evaluate_ensemble(model_a, model_b, test_loader, device)

    full_log  = pd.read_csv(log_path)
    p2_epochs = len(full_log[full_log["phase"] == "phase2"]) if "phase" in full_log.columns else last_p2_epoch
    metrics["training_time_minutes"] = round(p2_epochs * 25.0, 1)  # ~25 min/epoch estimate
    metrics["condition"] = CONDITION
    metrics["seed"]      = seed

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))

    p, r, f = metrics["per_class_precision"], metrics["per_class_recall"], metrics["per_class_f1"]
    print(f"\n{'='*55}")
    print(f"  Weighted F1      : {metrics['weighted_f1']:.4f}")
    print(f"  Macro F1         : {metrics['macro_f1']:.4f}")
    print(f"  Accuracy         : {metrics['accuracy']:.4f}")
    print(f"  Unhealthy Recall : {metrics['unhealthy_recall']:.4f}")
    print(f"  Per-class  P / R / F1:")
    for cls in config.CLASSES:
        print(f"    {cls:<10}  {p[cls]:.4f} / {r[cls]:.4f} / {f[cls]:.4f}")
    print(f"{'='*55}")
    print(f"  Metrics -> {out_path}")


if __name__ == "__main__":
    main()
