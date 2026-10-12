"""
Central configuration for the TDK synthetic-augmentation experiment.
All hyperparameters live here. Nothing is hardcoded elsewhere.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_ROOT   = Path(r"C:\Users\danish\Desktop\pap_cell_classification\data")
TRAIN_CSV   = DATA_ROOT / "isbi2025-ps3c-train-dataset.csv"
TRAIN_DIR   = DATA_ROOT / "isbi2025-ps3c-train-dataset"   # contains label/ subdirs
TEST_CSV    = DATA_ROOT / "isbi2025-ps3c-test-dataset-annotated.csv"
TEST_DIR    = DATA_ROOT / "isbi2025-ps3c-test-dataset"     # flat folder

# Pre-trained checkpoints from prior work (seed=42, heavy_aug_ensemble)
PRETRAINED_MODEL_A = Path(r"C:\Users\danish\Desktop\model_1\data\best_model_3.pth")
PRETRAINED_MODEL_B = Path(r"C:\Users\danish\Desktop\model_2\data\best_model_v2_6.pth")

OUTPUT_DIR      = Path("outputs")
CHECKPOINT_DIR  = OUTPUT_DIR / "checkpoints"
GEN_IMAGE_DIR   = OUTPUT_DIR / "generated_images"
METRICS_DIR     = OUTPUT_DIR / "metrics"
PLOTS_DIR       = OUTPUT_DIR / "plots"
LOG_DIR         = OUTPUT_DIR / "logs"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
CLASSES           = ["healthy", "rubbish", "unhealthy"]   # sorted alphabetically
NUM_CLASSES       = 3
DROP_BOTHCELLS    = True
UNHEALTHY_IDX     = CLASSES.index("unhealthy")            # 2

# Train / val split  (StratifiedShuffleSplit)
SPLIT_TEST_SIZE    = 0.15
SPLIT_RANDOM_STATE = 42

# Image sizes
IMG_SIZE     = 300    # classifier
GEN_IMG_SIZE = 128    # generative models

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------
BATCH_SIZE   = 32
NUM_WORKERS  = 0       # must be 0 on Windows to avoid deadlocks
PIN_MEMORY   = True

# ---------------------------------------------------------------------------
# Loss / sampling
# ---------------------------------------------------------------------------
LABEL_SMOOTHING = 0.05   # CrossEntropyLoss label_smoothing
# Weighted random sampler: inverse class frequency, computed at runtime

# ---------------------------------------------------------------------------
# Model A — standard augmentation, no head warmup
# ---------------------------------------------------------------------------
MODEL_A = dict(
    lr           = 2e-4,
    weight_decay = 1e-4,
    t_max        = 35,    # CosineAnnealingLR
    max_epochs   = 35,
    patience     = 6,     # early stopping on val weighted F1
)

# Model A train augmentation pipeline
MODEL_A_AUG = dict(
    rotation      = 20,
    jitter_b      = 0.08,
    jitter_c      = 0.08,
    jitter_s      = 0.08,
    jitter_h      = 0.0,
)

# ---------------------------------------------------------------------------
# Model B — dual augmentation, with head warmup
#
# Uses PapCellDatasetDualAug: heavy aug for unhealthy, standard aug for others.
# "Standard for Model B" = get_standard_tf_m2 pipeline below.
# "Heavy for Model B"    = get_heavy_tf_m2 pipeline below.
# This dual-aug strategy matches the code that produced the 0.8514 result.
# ---------------------------------------------------------------------------
MODEL_B = dict(
    # Phase 1: head-only warmup
    phase1_epochs       = 5,
    phase1_lr           = 1e-3,
    phase1_weight_decay = 1e-4,
    # Phase 2: full fine-tuning
    phase2_lr           = 3e-4,
    phase2_weight_decay = 1e-4,
    phase2_max_epochs   = 35,
    phase2_patience     = 7,
    warmup_epochs       = 3,    # linear LR warmup at start of phase 2
)

# Model B standard augmentation (applied to healthy + rubbish)
MODEL_B_STD_AUG = dict(
    rotation      = 20,
    translate     = (0.1, 0.1),
    jitter_b      = 0.2,
    jitter_c      = 0.2,
    jitter_s      = 0.15,
    jitter_h      = 0.05,
    erasing_p     = 0.1,
    erasing_scale = (0.02, 0.1),
)

# Model B heavy augmentation (applied to unhealthy only)
MODEL_B_HEAVY_AUG = dict(
    rotation      = 40,
    translate     = (0.15, 0.15),
    shear         = 15,
    jitter_b      = 0.4,
    jitter_c      = 0.4,
    jitter_s      = 0.3,
    jitter_h      = 0.1,
    grayscale_p   = 0.1,
    erasing_p     = 0.2,
    erasing_scale = (0.02, 0.15),
)

# ---------------------------------------------------------------------------
# DCGAN (GAN baseline)
# ---------------------------------------------------------------------------
DCGAN = dict(
    latent_dim    = 128,
    image_size    = GEN_IMG_SIZE,
    g_channels    = [512, 256, 128, 64],
    d_channels    = [64, 128, 256, 512],
    lr            = 2e-4,
    betas         = (0.5, 0.999),
    max_epochs    = 200,
    save_every    = 10,
    num_classes   = NUM_CLASSES,
    label_emb_dim = 50,
)

# ---------------------------------------------------------------------------
# DDPM — lightweight (our contribution)
# ---------------------------------------------------------------------------
DDPM = dict(
    image_size        = GEN_IMG_SIZE,          # 128×128
    timesteps         = 500,
    noise_schedule    = "linear",
    beta_start        = 1e-4,
    beta_end          = 2e-2,
    unet_channels     = (64, 128, 256),        # 3 stages, not 4
    attn_resolutions  = (16,),                 # attention only at 16×16
    res_blocks        = 2,
    num_classes       = NUM_CLASSES,
    lr                = 2e-4,
    max_epochs        = 300,
    save_every        = 20,
    ddim_steps        = 50,
)

# ---------------------------------------------------------------------------
# DDPM Heavy — wider, deeper, longer (ablation vs lightweight)
# ---------------------------------------------------------------------------
DDPM_HEAVY = dict(
    image_size        = GEN_IMG_SIZE,          # 128×128
    timesteps         = 1000,
    noise_schedule    = "linear",
    beta_start        = 1e-4,
    beta_end          = 2e-2,
    unet_channels     = (64, 128, 256, 512),   # 4 stages
    attn_resolutions  = (8, 16),               # attention at 16×16 and 8×8
    res_blocks        = 3,
    num_classes       = NUM_CLASSES,
    lr                = 2e-4,
    max_epochs        = 300,
    save_every        = 20,
    ddim_steps        = 50,
)

# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
SYNTH_COUNTS = [250, 500]
SEEDS        = [42, 123, 456]

CONDITIONS = [
    "baseline_ensemble",   # Model A: std aug; Model B: std aug (no dual aug)
    "heavy_aug_ensemble",  # Model A: std aug; Model B: dual aug (reproduces 0.8514)
    "gan_ensemble",        # Model A: std aug; Model B: std aug + DCGAN synthetic
    "ddpm_ensemble",       # Model A: std aug; Model B: std aug + DDPM synthetic
    "ddpm_500_heavy",      # Model A: std aug; Model B: dual aug + 500 DDPM synthetic
                           #   tests whether synthetic data helps ON TOP OF heavy aug
]

# ddpm_500_heavy always uses 500 synthetic images (fixed count, not in SYNTH_COUNTS loop)
DDPM_HEAVY_SYNTH_COUNT = 500
