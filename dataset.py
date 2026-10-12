"""
Dataset classes, transforms, and data loading utilities.
Reproduces the exact augmentation strategy from the 0.8514-result code.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

import config


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

class PadToSquareWhite:
    """Pad image to square with white background before resizing."""
    def __call__(self, img):
        w, h = img.size
        m = max(w, h)
        l = (m - w) // 2
        t = (m - h) // 2
        return ImageOps.expand(img, border=(l, t, m - w - l, m - h - t), fill=(255, 255, 255))


def get_train_tf_a(img_size: int = config.IMG_SIZE) -> transforms.Compose:
    """Model A standard augmentation."""
    c = config.MODEL_A_AUG
    return transforms.Compose([
        PadToSquareWhite(),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(c["rotation"], fill=255),
        transforms.ColorJitter(
            brightness=c["jitter_b"],
            contrast=c["jitter_c"],
            saturation=c["jitter_s"],
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


def get_std_tf_b(img_size: int = config.IMG_SIZE) -> transforms.Compose:
    """Model B standard augmentation — applied to healthy and rubbish."""
    c = config.MODEL_B_STD_AUG
    return transforms.Compose([
        PadToSquareWhite(),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(c["rotation"], fill=255),
        transforms.RandomAffine(degrees=0, translate=c["translate"], fill=255),
        transforms.ColorJitter(
            brightness=c["jitter_b"],
            contrast=c["jitter_c"],
            saturation=c["jitter_s"],
            hue=c["jitter_h"],
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        transforms.RandomErasing(p=c["erasing_p"], scale=c["erasing_scale"]),
    ])


def get_heavy_tf_b(img_size: int = config.IMG_SIZE) -> transforms.Compose:
    """Model B heavy augmentation — applied to unhealthy only."""
    c = config.MODEL_B_HEAVY_AUG
    return transforms.Compose([
        PadToSquareWhite(),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(c["rotation"], fill=255),
        transforms.RandomAffine(
            degrees=0, translate=c["translate"], shear=c["shear"], fill=255
        ),
        transforms.ColorJitter(
            brightness=c["jitter_b"],
            contrast=c["jitter_c"],
            saturation=c["jitter_s"],
            hue=c["jitter_h"],
        ),
        transforms.RandomGrayscale(p=c["grayscale_p"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        transforms.RandomErasing(p=c["erasing_p"], scale=c["erasing_scale"]),
    ])


def get_eval_tf(img_size: int = config.IMG_SIZE) -> transforms.Compose:
    """Evaluation / inference transform — no augmentation."""
    return transforms.Compose([
        PadToSquareWhite(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


def get_gen_eval_tf(img_size: int = config.GEN_IMG_SIZE) -> transforms.Compose:
    """Transform used when reading raw generated 128×128 images for FID/t-SNE."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------

class PapCellDataset(Dataset):
    """Train / val dataset — images in label subdirectories."""
    def __init__(self, df: pd.DataFrame, img_root, tfm):
        self.df       = df.reset_index(drop=True)
        self.img_root = str(img_root)
        self.tfm      = tfm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        path = os.path.join(self.img_root, row["label"], row["image_name"])
        img  = Image.open(path).convert("RGB")
        return self.tfm(img), int(row["target"])


class PapCellDatasetDualAug(Dataset):
    """Model B train dataset — heavy aug for unhealthy, standard for others."""
    def __init__(self, df: pd.DataFrame, img_root, std_tfm, heavy_tfm):
        self.df        = df.reset_index(drop=True)
        self.img_root  = str(img_root)
        self.std_tfm   = std_tfm
        self.heavy_tfm = heavy_tfm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        path = os.path.join(self.img_root, row["label"], row["image_name"])
        img  = Image.open(path).convert("RGB")
        tfm  = self.heavy_tfm if int(row["target"]) == config.UNHEALTHY_IDX else self.std_tfm
        return tfm(img), int(row["target"])


class SyntheticAugDataset(Dataset):
    """
    Wraps PapCellDataset and appends synthetic images for the minority class.
    Used in gan_ensemble and ddpm_ensemble conditions.
    Synthetic images are already 128×128 PNGs; they are resized to IMG_SIZE
    and put through the standard Model B augmentation pipeline.
    """
    def __init__(self, base_df: pd.DataFrame, img_root, tfm,
                 synth_paths: list, synth_label: int, synth_tfm):
        """
        base_df     — real training dataframe
        img_root    — root for real images (label subdirs)
        tfm         — transform for real images
        synth_paths — list of absolute paths to synthetic PNG files
        synth_label — integer class index for all synthetic images
        synth_tfm   — transform for synthetic images
        """
        self.base   = PapCellDataset(base_df, img_root, tfm)
        self.synth_paths = synth_paths
        self.synth_label = synth_label
        self.synth_tfm   = synth_tfm
        self.n_base  = len(self.base)
        self.n_synth = len(synth_paths)

    def __len__(self):
        return self.n_base + self.n_synth

    def __getitem__(self, idx):
        if idx < self.n_base:
            return self.base[idx]
        img = Image.open(self.synth_paths[idx - self.n_base]).convert("RGB")
        return self.synth_tfm(img), self.synth_label


class PapCellTestDataset(Dataset):
    """Test dataset — images in a single flat directory."""
    def __init__(self, df: pd.DataFrame, img_root, tfm):
        self.df       = df.reset_index(drop=True)
        self.img_root = str(img_root)
        self.tfm      = tfm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        path = os.path.join(self.img_root, row["image_name"])
        img  = Image.open(path).convert("RGB")
        return self.tfm(img), int(row["target"])


class UnlabelledFolderDataset(Dataset):
    """Load all PNGs from a directory; used to read real unhealthy images for GAN/DDPM training."""
    def __init__(self, folder, tfm):
        self.paths = sorted(Path(folder).glob("*.png")) + sorted(Path(folder).glob("*.jpg"))
        self.tfm   = tfm

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.tfm(img)


# ---------------------------------------------------------------------------
# Data-loading helpers
# ---------------------------------------------------------------------------

def load_train_df() -> tuple[pd.DataFrame, dict]:
    """Load train CSV, drop bothcells, add numeric target column."""
    df = pd.read_csv(config.TRAIN_CSV)
    if config.DROP_BOTHCELLS:
        df = df[df["label"] != "bothcells"].copy()
    class_to_idx = {c: i for i, c in enumerate(config.CLASSES)}
    df["target"] = df["label"].map(class_to_idx)
    return df, class_to_idx


def load_test_df(class_to_idx: dict) -> pd.DataFrame:
    """Load test CSV, drop bothcells, add numeric target column."""
    df = pd.read_csv(config.TEST_CSV)
    if config.DROP_BOTHCELLS:
        df = df[df["label"] != "bothcells"].copy()
    df["target"] = df["label"].map(class_to_idx)
    return df.reset_index(drop=True)


def split_train_val(df: pd.DataFrame, random_state: int = config.SPLIT_RANDOM_STATE):
    """Stratified train / val split."""
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=config.SPLIT_TEST_SIZE, random_state=random_state
    )
    train_idx, val_idx = next(sss.split(df["image_name"], df["target"]))
    return df.iloc[train_idx].copy().reset_index(drop=True), \
           df.iloc[val_idx].copy().reset_index(drop=True)


def compute_class_weights(df_train: pd.DataFrame) -> np.ndarray:
    """Inverse-frequency class weights, normalized to mean=1."""
    counts = df_train["target"].value_counts().sort_index().values.astype("float32")
    w = 1.0 / counts
    return (w / w.mean()).astype("float32")


def make_sampler(df_train: pd.DataFrame, class_weights: np.ndarray) -> WeightedRandomSampler:
    """WeightedRandomSampler with inverse class frequency."""
    sample_w = np.array([class_weights[t] for t in df_train["target"].values])
    return WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)


def make_loader(dataset: Dataset, sampler=None, shuffle: bool = False) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        sampler=sampler,
        shuffle=(shuffle and sampler is None),
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=False,  # must be False when num_workers=0
    )


# ---------------------------------------------------------------------------
# Convenience: build all loaders for a given condition
# ---------------------------------------------------------------------------

def build_model_a_loaders(df_train: pd.DataFrame, df_val: pd.DataFrame, sampler):
    """Standard aug datasets + loaders for Model A."""
    train_ds = PapCellDataset(df_train, config.TRAIN_DIR, get_train_tf_a())
    val_ds   = PapCellDataset(df_val,   config.TRAIN_DIR, get_eval_tf())
    return make_loader(train_ds, sampler=sampler), make_loader(val_ds)


def build_model_b_loaders_heavy(df_train: pd.DataFrame, df_val: pd.DataFrame, sampler):
    """Dual-aug datasets + loaders for Model B (heavy_aug_ensemble condition)."""
    train_ds = PapCellDatasetDualAug(
        df_train, config.TRAIN_DIR,
        std_tfm=get_std_tf_b(),
        heavy_tfm=get_heavy_tf_b(),
    )
    val_ds = PapCellDataset(df_val, config.TRAIN_DIR, get_eval_tf())
    return make_loader(train_ds, sampler=sampler), make_loader(val_ds)


def build_model_b_loaders_std(df_train: pd.DataFrame, df_val: pd.DataFrame, sampler):
    """Uniform standard aug + loaders for Model B (baseline_ensemble condition)."""
    train_ds = PapCellDataset(df_train, config.TRAIN_DIR, get_std_tf_b())
    val_ds   = PapCellDataset(df_val,   config.TRAIN_DIR, get_eval_tf())
    return make_loader(train_ds, sampler=sampler), make_loader(val_ds)


def build_model_b_loaders_synth(
    df_train: pd.DataFrame, df_val: pd.DataFrame,
    sampler, synth_paths: list
):
    """Standard aug + synthetic images for Model B (gan/ddpm_ensemble conditions).

    Synthetic images receive the same standard aug pipeline as real majority-class
    images so the model sees consistent augmentation across all training samples.
    """
    synth_tfm = get_std_tf_b()   # full standard aug (resize, flips, jitter, erasing)
    train_ds  = SyntheticAugDataset(
        df_train, config.TRAIN_DIR,
        tfm=get_std_tf_b(),
        synth_paths=synth_paths,
        synth_label=config.UNHEALTHY_IDX,
        synth_tfm=synth_tfm,
    )
    val_ds = PapCellDataset(df_val, config.TRAIN_DIR, get_eval_tf())

    # Rebuild sampler to include synthetic samples
    n_synth  = len(synth_paths)
    ext_targets = list(df_train["target"].values) + [config.UNHEALTHY_IDX] * n_synth
    counts   = np.bincount(ext_targets, minlength=config.NUM_CLASSES).astype("float32")
    w        = 1.0 / np.where(counts > 0, counts, 1.0)
    w        = (w / w.mean()).astype("float32")
    sample_w = np.array([w[t] for t in ext_targets])
    synth_sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)

    return make_loader(train_ds, sampler=synth_sampler), make_loader(val_ds)


def build_model_b_loaders_synth_heavy(
    df_train: pd.DataFrame, df_val: pd.DataFrame,
    synth_paths: list
):
    """Dual aug on real images + synthetic images for ddpm_500_heavy condition.

    Real images: heavy aug for unhealthy, standard for healthy/rubbish (same as
    heavy_aug_ensemble).  Synthetic images: standard aug (get_std_tf_b).
    Sampler rebuilt to account for added synthetic samples.
    """
    synth_tfm = get_std_tf_b()
    train_ds  = SyntheticAugDataset(
        df_train, config.TRAIN_DIR,
        tfm=get_std_tf_b(),           # used for real healthy + rubbish via PapCellDataset inside
        synth_paths=synth_paths,
        synth_label=config.UNHEALTHY_IDX,
        synth_tfm=synth_tfm,
    )
    # Override the real-image portion with dual aug by swapping the base dataset
    dual_base = PapCellDatasetDualAug(
        df_train, config.TRAIN_DIR,
        std_tfm=get_std_tf_b(),
        heavy_tfm=get_heavy_tf_b(),
    )
    train_ds.base   = dual_base
    train_ds.n_base = len(dual_base)

    val_ds = PapCellDataset(df_val, config.TRAIN_DIR, get_eval_tf())

    n_synth  = len(synth_paths)
    ext_targets = list(df_train["target"].values) + [config.UNHEALTHY_IDX] * n_synth
    counts   = np.bincount(ext_targets, minlength=config.NUM_CLASSES).astype("float32")
    w        = 1.0 / np.where(counts > 0, counts, 1.0)
    w        = (w / w.mean()).astype("float32")
    sample_w = np.array([w[t] for t in ext_targets])
    synth_sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)

    return make_loader(train_ds, sampler=synth_sampler), make_loader(val_ds)


def build_test_loader(class_to_idx: dict) -> tuple[DataLoader, pd.DataFrame]:
    df_test  = load_test_df(class_to_idx)
    test_ds  = PapCellTestDataset(df_test, config.TEST_DIR, get_eval_tf())
    return make_loader(test_ds), df_test
