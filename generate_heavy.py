"""
Generate 500 unhealthy-class images from the heavy DDPM checkpoint
and save them to outputs/generated_images/ddpm_heavy/unhealthy_500/.

Usage:
    python generate_heavy.py
"""
from pathlib import Path

import torch
from torchvision import transforms
from tqdm import tqdm

import config
from models.ddpm import UNet
from train_ddpm import NoiseSchedule, ddim_sample

CKPT  = config.CHECKPOINT_DIR / "ddpm_heavy" / "ddpm_heavy_ep0300.pth"
OUT   = config.GEN_IMAGE_DIR  / "ddpm_heavy" / "unhealthy_500"
N     = 500
BATCH = 32
SEED  = 42


def main():
    import random, numpy as np
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Checkpoint : {CKPT}")

    cfg   = config.DDPM_HEAVY
    model = UNet(cfg=cfg).to(device)
    ckpt  = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Model loaded (epoch {ckpt['epoch']})")

    sched = NoiseSchedule(cfg["timesteps"], cfg["beta_start"], cfg["beta_end"], device)

    OUT.mkdir(parents=True, exist_ok=True)
    to_pil = transforms.ToPILImage()
    cls    = config.UNHEALTHY_IDX
    count  = 0

    for start in tqdm(range(0, N, BATCH), desc="DDIM sampling"):
        bsz = min(BATCH, N - start)
        z   = torch.randn(bsz, 3, cfg["image_size"], cfg["image_size"], device=device)
        lbl = torch.full((bsz,), cls, dtype=torch.long, device=device)

        with torch.no_grad():
            imgs = ddim_sample(model, sched, z, lbl, cfg["ddim_steps"])

        imgs = (imgs * 0.5 + 0.5).clamp(0, 1).cpu()
        for t in imgs:
            to_pil(t).save(OUT / f"{count:05d}.png")
            count += 1

    print(f"\nSaved {count} images → {OUT}")


if __name__ == "__main__":
    main()
