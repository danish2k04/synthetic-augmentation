"""
Train a conditional DCGAN on unhealthy-class Pap-smear cells.

Usage:
    python train_dcgan.py [--epochs N] [--seed S]

Checkpoints  → outputs/checkpoints/dcgan/dcgan_ep<NNNN>.pth
Sample grids → outputs/generated_images/dcgan/samples/grid_ep<NNNN>.png
Loss log     → outputs/logs/dcgan_train_log.csv
"""
import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

import config
from dataset import UnlabelledFolderDataset
from models.dcgan import Generator, Discriminator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def weights_init(m: nn.Module):
    """DCGAN weight init from the original paper."""
    cls = m.__class__.__name__
    if "Conv" in cls:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in cls:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
    elif "Linear" in cls and hasattr(m, "weight"):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


def gan_transform(img_size: int) -> transforms.Compose:
    """[-1, 1] normalisation so Tanh output and real images live in the same range."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


@torch.no_grad()
def save_grid(G: nn.Module, fixed_z: torch.Tensor,
              fixed_labels: torch.Tensor, path: Path) -> torch.Tensor:
    G.eval()
    imgs = G(fixed_z, fixed_labels)            # [-1, 1]
    imgs = (imgs * 0.5 + 0.5).clamp(0, 1)     # [0, 1]
    grid = make_grid(imgs, nrow=8, padding=2)
    save_image(grid, path)
    G.train()
    return grid


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(epochs: int, seed: int):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = config.DCGAN

    ckpt_dir   = config.CHECKPOINT_DIR / "dcgan"
    sample_dir = config.GEN_IMAGE_DIR  / "dcgan" / "samples"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Data — only unhealthy images; normalise to [-1, 1] for DCGAN
    # ------------------------------------------------------------------
    unhealthy_dir = config.TRAIN_DIR / "unhealthy"
    dataset = UnlabelledFolderDataset(unhealthy_dir, gan_transform(cfg["image_size"]))
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
    # Models
    # ------------------------------------------------------------------
    G = Generator().to(device)
    D = Discriminator().to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    opt_G = torch.optim.Adam(G.parameters(), lr=cfg["lr"], betas=cfg["betas"])
    opt_D = torch.optim.Adam(D.parameters(), lr=cfg["lr"], betas=cfg["betas"])
    criterion = nn.BCEWithLogitsLoss()

    unhealthy_cls = config.UNHEALTHY_IDX  # 2

    # Fixed noise for reproducible sample grids (64 samples → 8×8 grid)
    fixed_z      = torch.randn(64, cfg["latent_dim"], device=device)
    fixed_labels = torch.full((64,), unhealthy_cls, dtype=torch.long, device=device)

    log_rows: list[dict] = []

    print(f"\nStarting DCGAN training for {epochs} epochs …\n")

    for epoch in range(1, epochs + 1):
        g_losses: list[float] = []
        d_losses: list[float] = []

        for real_imgs in tqdm(loader, desc=f"Ep {epoch:03d}/{epochs}", leave=False):
            real_imgs = real_imgs.to(device)
            bsz = real_imgs.size(0)

            cls_labels = torch.full((bsz,), unhealthy_cls, dtype=torch.long, device=device)
            real_y = torch.ones(bsz,  1, device=device)
            fake_y = torch.zeros(bsz, 1, device=device)

            # ---- Train Discriminator ----
            opt_D.zero_grad()
            z    = torch.randn(bsz, cfg["latent_dim"], device=device)
            fake = G(z, cls_labels).detach()
            loss_D = (criterion(D(real_imgs, cls_labels), real_y) +
                      criterion(D(fake,      cls_labels), fake_y))
            loss_D.backward()
            opt_D.step()

            # ---- Train Generator ----
            opt_G.zero_grad()
            z    = torch.randn(bsz, cfg["latent_dim"], device=device)
            fake = G(z, cls_labels)
            loss_G = criterion(D(fake, cls_labels), real_y)   # fool D
            loss_G.backward()
            opt_G.step()

            g_losses.append(loss_G.item())
            d_losses.append(loss_D.item())

        g_mean = float(np.mean(g_losses))
        d_mean = float(np.mean(d_losses))
        log_rows.append({"epoch": epoch, "g_loss": g_mean, "d_loss": d_mean})
        print(f"[DCGAN] ep{epoch:03d}  D={d_mean:.4f}  G={g_mean:.4f}")

        if epoch % cfg["save_every"] == 0 or epoch == epochs:
            ckpt_path = ckpt_dir / f"dcgan_ep{epoch:04d}.pth"
            torch.save({"epoch": epoch,
                        "G_state": G.state_dict(),
                        "D_state": D.state_dict()}, ckpt_path)
            grid_path = sample_dir / f"grid_ep{epoch:04d}.png"
            save_grid(G, fixed_z, fixed_labels, grid_path)
            print(f"  → ckpt + grid saved  ({grid_path.name})")

    # ------------------------------------------------------------------
    # Final grid + loss log
    # ------------------------------------------------------------------
    final_grid_path = sample_dir / "grid_final.png"
    save_grid(G, fixed_z, fixed_labels, final_grid_path)
    print(f"\nFinal sample grid → {final_grid_path}")

    log_path = config.LOG_DIR / "dcgan_train_log.csv"
    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "g_loss", "d_loss"])
        w.writeheader()
        w.writerows(log_rows)
    print(f"Loss log           → {log_path}")

    return G, final_grid_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.DCGAN["max_epochs"])
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()
    train(args.epochs, args.seed)


if __name__ == "__main__":
    main()
