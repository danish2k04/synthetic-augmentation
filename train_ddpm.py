"""
Train a lightweight conditional DDPM on unhealthy-class Pap-smear cells.

Usage:
    python train_ddpm.py [--epochs N] [--seed S]

Checkpoints  → outputs/checkpoints/ddpm/ddpm_ep<NNNN>.pth
Sample grids → outputs/generated_images/ddpm/samples/grid_ep<NNNN>.png
Loss log     → outputs/logs/ddpm_train_log.csv
"""
import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

import config
from dataset import UnlabelledFolderDataset
from models.ddpm import UNet


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
    """Normalise real images to [-1, 1] to match the diffusion convention."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


# ---------------------------------------------------------------------------
# Noise schedule
# ---------------------------------------------------------------------------

class NoiseSchedule:
    """Linear beta schedule with all derived quantities pre-computed."""
    def __init__(self, timesteps: int, beta_start: float, beta_end: float,
                 device: torch.device):
        self.T = timesteps
        betas        = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas       = 1.0 - betas
        alpha_bar    = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = torch.cat([torch.ones(1, device=device), alpha_bar[:-1]])

        self.register = lambda name, val: setattr(self, name, val)
        self.betas         = betas
        self.alphas        = alphas
        self.alpha_bar     = alpha_bar
        self.alpha_bar_prev= alpha_bar_prev
        self.sqrt_ab       = alpha_bar.sqrt()
        self.sqrt_one_m_ab = (1 - alpha_bar).sqrt()

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor) -> torch.Tensor:
        """Forward diffusion: x_t = sqrt(ā_t)*x0 + sqrt(1-ā_t)*eps."""
        s  = self.sqrt_ab[t][:, None, None, None]
        sm = self.sqrt_one_m_ab[t][:, None, None, None]
        return s * x0 + sm * noise


# ---------------------------------------------------------------------------
# DDIM sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def ddim_sample(model: nn.Module, sched: NoiseSchedule,
                z: torch.Tensor, labels: torch.Tensor,
                ddim_steps: int, eta: float = 0.0) -> torch.Tensor:
    """
    Deterministic DDIM sampler (eta=0).
    z      : (B, 3, H, W) — pure Gaussian noise
    labels : (B,) — class indices
    """
    model.eval()
    T       = sched.T
    # Sub-sample timestep sequence (uniform)
    step_indices = torch.linspace(0, T - 1, ddim_steps + 1, dtype=torch.long)
    timesteps    = step_indices.flip(0)[:-1]   # T-1 ... 0, length=ddim_steps

    x = z
    for i, t_val in enumerate(timesteps):
        t_batch = torch.full((x.size(0),), t_val.item(), dtype=torch.long,
                             device=x.device)
        ab_t    = sched.alpha_bar[t_val]
        ab_prev = sched.alpha_bar[timesteps[i + 1]] if i + 1 < len(timesteps) \
                  else torch.tensor(1.0, device=x.device)

        eps_pred = model(x, t_batch, labels)
        x0_pred  = (x - (1 - ab_t).sqrt() * eps_pred) / ab_t.sqrt()
        x0_pred  = x0_pred.clamp(-1, 1)

        # Direction pointing to x_t
        dir_xt = (1 - ab_prev).sqrt() * eps_pred
        x      = ab_prev.sqrt() * x0_pred + dir_xt

    model.train()
    return x


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(epochs: int, seed: int):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = config.DDPM

    ckpt_dir   = config.CHECKPOINT_DIR / "ddpm"
    sample_dir = config.GEN_IMAGE_DIR  / "ddpm" / "samples"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Data — only unhealthy images; normalised to [-1, 1]
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

    model = UNet().to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    scaler = GradScaler()

    unhealthy_cls = config.UNHEALTHY_IDX   # 2

    # Fixed noise + labels for reproducible sample grids
    fixed_z      = torch.randn(64, 3, cfg["image_size"], cfg["image_size"],
                               device=device)
    fixed_labels = torch.full((64,), unhealthy_cls, dtype=torch.long,
                              device=device)

    log_rows: list[dict] = []

    print(f"\nStarting DDPM training for {epochs} epochs …\n")

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        n_batches  = 0

        for x0 in tqdm(loader, desc=f"Ep {epoch:03d}/{epochs}", leave=False):
            x0  = x0.to(device)
            bsz = x0.size(0)

            t     = torch.randint(0, cfg["timesteps"], (bsz,), device=device)
            noise = torch.randn_like(x0)
            xt    = sched.q_sample(x0, t, noise)
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
        print(f"[DDPM] ep{epoch:03d}  loss={avg_loss:.6f}")

        if epoch % cfg["save_every"] == 0 or epoch == epochs:
            ckpt_path = ckpt_dir / f"ddpm_ep{epoch:04d}.pth"
            torch.save({"epoch": epoch,
                        "model_state": model.state_dict()}, ckpt_path)

            # Generate sample grid via DDIM
            imgs = ddim_sample(model, sched, fixed_z, fixed_labels,
                               cfg["ddim_steps"])
            imgs  = (imgs * 0.5 + 0.5).clamp(0, 1)
            grid  = make_grid(imgs, nrow=8, padding=2)
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

    log_path = config.LOG_DIR / "ddpm_train_log.csv"
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
    parser.add_argument("--epochs", type=int, default=config.DDPM["max_epochs"])
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()
    train(args.epochs, args.seed)


if __name__ == "__main__":
    main()
