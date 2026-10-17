"""
EfficientNetV2-S classifier with two training procedures:
  train_model_a — standard aug, no head warmup, CosineAnnealing
  train_model_b — dual aug, 5-epoch head warmup then full fine-tuning with warmup+cosine LR
"""
import csv
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from sklearn.metrics import f1_score
from tqdm import tqdm

import config


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(num_classes: int = config.NUM_CLASSES) -> nn.Module:
    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def freeze_backbone(model: nn.Module):
    for name, param in model.named_parameters():
        if not name.startswith("classifier"):
            param.requires_grad = False


def unfreeze_all(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = True


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# One epoch: train / eval
# ---------------------------------------------------------------------------

def _run_epoch(model, loader, criterion, optimizer, scaler, device, train: bool):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    phase = "train" if train else "val"
    with ctx:
        for imgs, labels in tqdm(loader, desc=f"  {phase}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            with autocast():
                logits = model(imgs)
                loss   = criterion(logits, labels)
            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            total_loss += loss.item() * len(labels)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    n    = len(all_labels)
    wf1  = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    return total_loss / n, wf1


# ---------------------------------------------------------------------------
# CSV log helpers
# ---------------------------------------------------------------------------

def _open_log(path: Path) -> tuple:
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "w", newline="")
    writer = csv.writer(fh)
    writer.writerow(["epoch", "phase", "train_loss", "val_loss", "val_wf1", "lr"])
    return fh, writer


# ---------------------------------------------------------------------------
# train_model_a
# ---------------------------------------------------------------------------

def train_model_a(train_loader, val_loader, class_weights: torch.Tensor,
                  ckpt_path: Path, log_path: Path, seed: int) -> Path:
    """
    Model A: no head warmup, end-to-end training with standard aug.
    Returns path to best checkpoint.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg    = config.MODEL_A

    model     = build_model().to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=config.LABEL_SMOOTHING,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["t_max"]
    )
    scaler = GradScaler()

    best_wf1, best_epoch, no_improve = 0.0, 0, 0
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    fh, writer = _open_log(log_path)

    for epoch in range(1, cfg["max_epochs"] + 1):
        lr = optimizer.param_groups[0]["lr"]
        tr_loss, _  = _run_epoch(model, train_loader, criterion, optimizer, scaler, device, train=True)
        val_loss, val_wf1 = _run_epoch(model, val_loader, criterion, None, None, device, train=False)
        scheduler.step()

        writer.writerow([epoch, "train+val", f"{tr_loss:.6f}", f"{val_loss:.6f}", f"{val_wf1:.6f}", f"{lr:.2e}"])
        fh.flush()
        print(f"[A] ep{epoch:03d} lr={lr:.2e} train_loss={tr_loss:.4f} val_loss={val_loss:.4f} val_wF1={val_wf1:.4f}")

        if val_wf1 > best_wf1:
            best_wf1, best_epoch, no_improve = val_wf1, epoch, 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_wf1": val_wf1}, ckpt_path)
        else:
            no_improve += 1
            if no_improve >= cfg["patience"]:
                print(f"[A] Early stopping at epoch {epoch}. Best val wF1={best_wf1:.4f} (epoch {best_epoch})")
                break

    fh.close()
    print(f"[A] Best val wF1={best_wf1:.4f} — checkpoint saved to {ckpt_path}")
    return ckpt_path


# ---------------------------------------------------------------------------
# train_model_b
# ---------------------------------------------------------------------------

def _make_cosine_warmup_fn(warmup_epochs: int, total_epochs: int):
    """LambdaLR multiplier: linear warmup then cosine decay."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


def train_model_b(train_loader, val_loader, class_weights: torch.Tensor,
                  ckpt_path: Path, log_path: Path, seed: int) -> Path:
    """
    Model B: 5-epoch head warmup then full fine-tuning with warmup+cosine LR.
    Returns path to best checkpoint.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg    = config.MODEL_B

    model     = build_model().to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=config.LABEL_SMOOTHING,
    )
    scaler = GradScaler()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    fh, writer = _open_log(log_path)

    # --- Phase 1: freeze backbone, train head only ---
    freeze_backbone(model)
    head_params = [p for p in model.parameters() if p.requires_grad]
    opt1 = torch.optim.AdamW(head_params, lr=cfg["phase1_lr"], weight_decay=cfg["phase1_weight_decay"])

    print(f"[B] Phase 1 — training head for {cfg['phase1_epochs']} epochs")
    for epoch in range(1, cfg["phase1_epochs"] + 1):
        lr = opt1.param_groups[0]["lr"]
        tr_loss, _  = _run_epoch(model, train_loader, criterion, opt1, scaler, device, train=True)
        val_loss, val_wf1 = _run_epoch(model, val_loader, criterion, None, None, device, train=False)
        writer.writerow([epoch, "phase1", f"{tr_loss:.6f}", f"{val_loss:.6f}", f"{val_wf1:.6f}", f"{lr:.2e}"])
        fh.flush()
        print(f"[B] P1 ep{epoch:02d} lr={lr:.2e} train_loss={tr_loss:.4f} val_wF1={val_wf1:.4f}")

    # --- Phase 2: unfreeze all, full fine-tuning ---
    unfreeze_all(model)
    opt2 = torch.optim.AdamW(
        model.parameters(), lr=cfg["phase2_lr"], weight_decay=cfg["phase2_weight_decay"]
    )
    sched2 = torch.optim.lr_scheduler.LambdaLR(
        opt2, lr_lambda=_make_cosine_warmup_fn(cfg["warmup_epochs"], cfg["phase2_max_epochs"])
    )

    best_wf1, best_epoch, no_improve = 0.0, 0, 0

    print(f"[B] Phase 2 — full fine-tuning for up to {cfg['phase2_max_epochs']} epochs")
    for epoch in range(1, cfg["phase2_max_epochs"] + 1):
        lr = opt2.param_groups[0]["lr"]
        tr_loss, _  = _run_epoch(model, train_loader, criterion, opt2, scaler, device, train=True)
        val_loss, val_wf1 = _run_epoch(model, val_loader, criterion, None, None, device, train=False)
        sched2.step()

        global_epoch = cfg["phase1_epochs"] + epoch
        writer.writerow([global_epoch, "phase2", f"{tr_loss:.6f}", f"{val_loss:.6f}", f"{val_wf1:.6f}", f"{lr:.2e}"])
        fh.flush()
        print(f"[B] P2 ep{epoch:03d} lr={lr:.2e} train_loss={tr_loss:.4f} val_wF1={val_wf1:.4f}")

        if val_wf1 > best_wf1:
            best_wf1, best_epoch, no_improve = val_wf1, epoch, 0
            torch.save({"epoch": global_epoch, "model_state": model.state_dict(),
                        "val_wf1": val_wf1}, ckpt_path)
        else:
            no_improve += 1
            if no_improve >= cfg["phase2_patience"]:
                print(f"[B] Early stopping at phase-2 epoch {epoch}. Best val wF1={best_wf1:.4f} (epoch {best_epoch})")
                break

    fh.close()
    print(f"[B] Best val wF1={best_wf1:.4f} — checkpoint saved to {ckpt_path}")
    return ckpt_path


# ---------------------------------------------------------------------------
# Load a saved checkpoint back into a model
# ---------------------------------------------------------------------------

def load_checkpoint(ckpt_path: Path, device=None) -> nn.Module:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    # Support both raw state-dict and wrapped {"model_state": ...} checkpoints
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model
