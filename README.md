# Lightweight Diffusion Models for Synthetic Minority-Class Generation in Medical Image Classification

TDK (Scientific Students' Associations Conference) research project exploring whether lightweight conditional diffusion models can generate useful synthetic minority-class images to improve classification on severely imbalanced medical datasets.

## Problem

The [ISBI 2025 PS3C](https://ps3c.grand-challenge.org/) pap smear cell classification dataset contains ~82,000 images across three classes:

| Class | Train Samples | Share |
|-------|--------------|-------|
| Healthy | 28,895 | 35.4% |
| Rubbish | 50,371 | 61.7% |
| Unhealthy | 2,366 | **2.9%** |

The clinically important class (unhealthy) is severely underrepresented at under 3% of the dataset.

## Approach

A two-model EfficientNetV2-S ensemble serves as the downstream classifier. Model A is fixed across all experiments — only Model B's training data changes, isolating the effect of each augmentation strategy.

Three generative approaches are compared:

- **Lightweight DDPM** — U-Net with 3 channel stages (64, 128, 256), attention at 16x16 only, 500 diffusion steps, DDIM 50-step sampling
- **Full-size DDPM** — 4 channel stages (64, 128, 256, 512), multi-resolution attention, 1000 steps (ablation)
- **Conditional DCGAN** — latent dim 128, hinge loss, conditional batch norm (baseline generative model)

All generative models are trained at 128x128 on only the ~2,366 unhealthy-class images.

## Experimental Conditions

| Condition | wF1 | Unhealthy F1 | Description |
|-----------|-----|-------------|-------------|
| Baseline | 0.8493 | 0.4269 | Standard augmentation only |
| Heavy Aug | 0.8458 | 0.3659 | Class-specific heavy augmentation |
| GAN-250 | 0.8486 | 0.4129 | + 250 DCGAN synthetic images |
| GAN-500 | 0.8485 | 0.4179 | + 500 DCGAN synthetic images |
| DDPM-light-250 | 0.8484 | 0.4116 | + 250 lightweight DDPM images |
| DDPM-light-500 | 0.8492 | 0.4193 | + 500 lightweight DDPM images |
| DDPM-heavy-500 | 0.8513 | 0.4377 | + 500 full-size DDPM images |

## FID Scores (Synthetic Image Quality)

| Model | FID (500 images) |
|-------|-----------------|
| DCGAN | 55.97 |
| DDPM-light | 63.77 |
| DDPM-heavy | **82.99** |

The full-size DDPM produces *worse* FID than the lightweight version — it overfits on the small (~2.4K) training set. This is one of the key findings.

## Repository Structure

```
├── config.py                    # All hyperparameters and paths
├── dataset.py                   # Dataset classes, transforms, data loaders
├── models/
│   ├── classifier.py            # EfficientNetV2-S training (Model A & B)
│   ├── ensemble.py              # Two-model softmax ensemble
│   ├── dcgan.py                 # Conditional DCGAN
│   └── ddpm.py                  # Conditional DDPM (lightweight & full-size)
├── train_dcgan.py               # DCGAN training script
├── train_ddpm.py                # Lightweight DDPM training script
├── train_ddpm_heavy.py          # Full-size DDPM training script
├── generate_eval.py             # Generate synthetic images + FID + t-SNE
├── run_downstream_f1.py         # Train Model B + evaluate ensemble
├── run_all_conditions.py        # Run all experimental conditions
├── evaluate_ensemble.py         # Standalone ensemble evaluation
├── compare_results.py           # Generate comparison tables and plots
├── compare_ddpm.py              # DDPM-light vs DDPM-heavy comparison
├── plot_curves.py               # Training curve plots
├── plot_metrics.py              # Metrics visualization
├── outputs/
│   ├── generated_images/        # Synthetic samples (DCGAN, DDPM, DDPM-heavy)
│   ├── metrics/                 # Per-condition JSON results + FID scores
│   ├── logs/                    # Training logs (CSV)
│   └── plots/                   # Comparison grids, t-SNE, training curves
```

## Key Design Decisions

- **Controlled experiment**: Model A checkpoint is frozen across all conditions. Only Model B's training data varies, isolating the effect of each synthesis strategy.
- **Consistent augmentation**: Synthetic images receive the same standard augmentation (`get_std_tf_b()`) as real images during classifier training — no information leakage through transform differences.
- **WeightedRandomSampler** with inverse class frequency + CrossEntropyLoss with label smoothing (0.05) in all conditions.
- **PadToSquareWhite**: Custom transform that pads non-square cell images to square with white fill before resizing, preserving aspect ratio.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- torchvision
- timm
- scikit-learn
- matplotlib
- pytorch-fid
- PIL / Pillow

Trained on a single 8 GB consumer GPU (NVIDIA RTX 4060 Laptop).

## Author

Danish Kumar — BSc Computer Science, University of Debrecen
