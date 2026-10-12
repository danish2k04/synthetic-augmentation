"""
Ensemble inference: average softmax outputs of Model A and Model B.
"""
import torch
import torch.nn.functional as F

from models.classifier import load_checkpoint
import config


def ensemble_predict(model_a, model_b, loader, device=None):
    """
    Returns (preds, labels) numpy arrays for the full loader.
    final_pred = argmax( (softmax(logits_a) + softmax(logits_b)) / 2 )
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_a.eval()
    model_b.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            sa   = F.softmax(model_a(imgs), dim=1)
            sb   = F.softmax(model_b(imgs), dim=1)
            preds = ((sa + sb) / 2).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    import numpy as np
    return np.array(all_preds), np.array(all_labels)


def load_ensemble(ckpt_a, ckpt_b, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_checkpoint(ckpt_a, device), load_checkpoint(ckpt_b, device)
