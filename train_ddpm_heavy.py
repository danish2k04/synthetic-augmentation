"""
Train the heavy conditional DDPM on unhealthy-class Pap-smear cells.

Architecture: 4 stages (64,128,256,512 ch), attention at 16×16 and 8×8,
              3 res-blocks per stage, T=1000 timesteps.

Usage:
    python train_ddpm_heavy.py [--epochs N] [--seed S]

Checkpoints  → outputs/checkpoints/ddpm_heavy/ddpm_heavy_ep<NNNN>.pth
Sample grids → outputs/generated_images/ddpm_heavy/samples/grid_ep<NNNN>.png
Loss log     → outputs/logs/ddpm_heavy_train_log.csv
"""
import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

import config
from dataset import UnlabelledFolderDataset
from models.ddpm import UNet
from train_ddpm import NoiseSchedule, ddim_sample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def gan_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(epochs: int, seed: int):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = config.DDPM_HEAVY

    ckpt_dir   = config.CHECKPOINT_DIR / "ddpm_heavy"
    sample_dir = config.GEN_IMAGE_DIR  / "ddpm_heavy" / "samples"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    unhealthy_dir = config.TRAIN_DIR / "unhealthy"
    dataset = UnlabelledFolderDataset(unhealthy_dir,
                                      gan_transform(cfg["image_size"]))
    loader  = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True,
    )
    print(f"Unhealthy training images : {len(dataset)}")
    print(f"Batches per epoch         : {len(loader)}")

    # ------------------------------------------------------------------
    # Noise schedule, model, optimiser
    # ------------------------------------------------------------------
    sched = NoiseSchedule(cfg["timesteps"], cfg["beta_start"],
                          cfg["beta_end"], device)

    model = UNet(cfg=cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Heavy UNet parameters     : {n_params:.1f}M")

    opt    = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    scaler = GradScaler()

    unhealthy_cls = config.UNHEALTHY_IDX   # 2

    # Fixed noise + labels for reproducible sample grids
    fixed_z      = torch.randn(64, 3, cfg["image_size"], cfg["image_size"],
                               device=device)
    fixed_labels = torch.full((64,), unhealthy_cls, dtype=torch.long,
                              device=device)

    log_rows: list[dict] = []

    print(f"\nStarting heavy DDPM training for {epochs} epochs …\n")

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        n_batches  = 0

        for x0 in tqdm(loader, desc=f"Ep {epoch:03d}/{epochs}", leave=False):
            x0  = x0.to(device)
            bsz = x0.size(0)

            t      = torch.randint(0, cfg["timesteps"], (bsz,), device=device)
            noise  = torch.randn_like(x0)
            xt     = sched.q_sample(x0, t, noise)
            labels = torch.full((bsz,), unhealthy_cls, dtype=torch.long,
                                device=device)

            opt.zero_grad()
            with autocast():
                eps_pred = model(xt, t, labels)
                loss     = F.mse_loss(eps_pred, noise)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / n_batches
        log_rows.append({"epoch": epoch, "loss": avg_loss})
        print(f"[DDPM-heavy] ep{epoch:03d}  loss={avg_loss:.6f}")

        if epoch % cfg["save_every"] == 0 or epoch == epochs:
            ckpt_path = ckpt_dir / f"ddpm_heavy_ep{epoch:04d}.pth"
            torch.save({"epoch": epoch,
                        "model_state": model.state_dict()}, ckpt_path)

            imgs = ddim_sample(model, sched, fixed_z, fixed_labels,
                               cfg["ddim_steps"])
            imgs = (imgs * 0.5 + 0.5).clamp(0, 1)
            grid = make_grid(imgs, nrow=8, padding=2)
            grid_path = sample_dir / f"grid_ep{epoch:04d}.png"
            save_image(grid, grid_path)
            print(f"  → ckpt + grid saved  ({grid_path.name})")

    # ------------------------------------------------------------------
    # Final grid + loss log
    # ------------------------------------------------------------------
    final_grid_path = sample_dir / "grid_final.png"
    imgs = ddim_sample(model, sched, fixed_z, fixed_labels, cfg["ddim_steps"])
    imgs = (imgs * 0.5 + 0.5).clamp(0, 1)
    save_image(make_grid(imgs, nrow=8, padding=2), final_grid_path)
    print(f"\nFinal sample grid → {final_grid_path}")

    log_path = config.LOG_DIR / "ddpm_heavy_train_log.csv"
    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "loss"])
        w.writeheader()
        w.writerows(log_rows)
    print(f"Loss log           → {log_path}")

    return model, final_grid_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int,
                        default=config.DDPM_HEAVY["max_epochs"])
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()
    train(args.epochs, args.seed)


if __name__ == "__main__":
    main()
