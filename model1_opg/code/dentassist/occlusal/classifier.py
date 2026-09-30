"""EfficientNet-B0 / MobileNetV3 classifiers for tooth type and caries severity (shared by train + inference)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

ARCHS = ("efficientnet_b0", "mobilenet_v3")
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def build_model(arch: str, n_classes: int, pretrained: bool = True) -> nn.Module:
    import torchvision.models as tvm
    if arch == "efficientnet_b0":
        ctor, w = tvm.efficientnet_b0, "IMAGENET1K_V1"
    elif arch == "mobilenet_v3":
        ctor, w = tvm.mobilenet_v3_large, "IMAGENET1K_V2"
    else:
        raise ValueError(arch)
    import os
    if not pretrained or os.getenv("DENTASSIST_ALLOW_SCRATCH") == "1":   # tests only
        m = ctor(weights=None)
    else:
        try:
            m = ctor(weights=w)
        except Exception as e:
            raise RuntimeError(
                f"Could not download ImageNet weights for {arch} ({e.__class__.__name__}: {e}). "
                "Turn Internet ON (Kaggle: Settings > Internet). Training from scratch on a few hundred "
                "teeth does not work - it was tested and stays at chance level.") from e
    last = m.classifier[-1]
    m.classifier[-1] = nn.Linear(last.in_features, n_classes)
    return m


def train_transform(size: int):
    from torchvision.transforms import v2 as T
    return T.Compose([
        T.ToImage(),
        T.RandomResizedCrop(size, scale=(0.75, 1.0), ratio=(0.9, 1.1), antialias=True),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(20),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),   # small hue: lesion colour matters
        T.ToDtype(torch.float32, scale=True), T.Normalize(MEAN, STD)])


def eval_transform(size: int):
    from torchvision.transforms import v2 as T
    return T.Compose([T.ToImage(), T.Resize((size, size), antialias=True),
                      T.ToDtype(torch.float32, scale=True), T.Normalize(MEAN, STD)])


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class Classifier:
    """Loads a checkpoint saved by occ_train.py; predict(list_of_BGR_crops) -> probs (N, C)."""

    def __init__(self, ckpt_path: str | Path, device: str | None = None):
        ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        self.classes, self.size, self.arch = ck["classes"], ck["img_size"], ck["arch"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(self.arch, len(self.classes), pretrained=False)
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval().to(self.device)
        self.tf = eval_transform(self.size)

    @torch.no_grad()
    def predict(self, crops_bgr: list[np.ndarray], tta: bool = True) -> np.ndarray:
        if not crops_bgr:
            return np.zeros((0, len(self.classes)))
        x = torch.stack([self.tf(bgr_to_rgb(c)) for c in crops_bgr]).to(self.device)
        p = self.model(x).softmax(1)
        if tta:
            p = (p + self.model(torch.flip(x, dims=[3])).softmax(1)) / 2
        return p.cpu().numpy()
