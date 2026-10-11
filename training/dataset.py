"""PyTorch Dataset for (image, mask) folders.

Expected layout (produced by build_synthetic_dataset.py, and also usable for
CASIA / CoMoFoD after conversion - see data/README.md):

    <split_dir>/images/<id>.jpg|png
    <split_dir>/masks/<id>.png          (0 = authentic pixel, 255 = tampered)
    <split_dir>/metadata.csv            (optional: id,file,type,label,...)

Augmentation is deliberately limited to random crops and flips: resizing,
blurring or re-compressing would destroy or fake the forensic traces the
network must learn from.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing.image_preprocessing import normalize, pad_to_multiple


def read_records(split_dir: Path) -> list[dict]:
    split_dir = Path(split_dir)
    meta = split_dir / "metadata.csv"
    if meta.exists():
        with open(meta, newline="") as f:
            return list(csv.DictReader(f))
    records = []
    for img in sorted((split_dir / "images").iterdir()):
        if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".bmp"}:
            records.append({"id": img.stem, "file": img.name, "type": "unknown", "label": ""})
    return records


def load_pair(split_dir: Path, rec: dict) -> tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(str(Path(split_dir) / "images" / rec["file"]), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(rec["file"])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mpath = Path(split_dir) / "masks" / f"{rec['id']}.png"
    mask = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE) if mpath.exists() else None
    if mask is None:          # authentic image without a mask file
        mask = np.zeros(img.shape[:2], np.uint8)
    if mask.shape != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    return img, (mask > 127).astype(np.float32)


class TamperDataset(Dataset):
    def __init__(self, split_dir: Path, crop: int | None = None, train: bool = False):
        self.split_dir = Path(split_dir)
        self.records = read_records(self.split_dir)
        self.crop = crop
        self.train = train

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i):
        img, mask = load_pair(self.split_dir, self.records[i])
        if self.crop:
            h, w = mask.shape
            c = self.crop
            if h < c or w < c:
                img = cv2.copyMakeBorder(img, 0, max(0, c - h), 0, max(0, c - w), cv2.BORDER_REFLECT_101)
                mask = cv2.copyMakeBorder(mask, 0, max(0, c - h), 0, max(0, c - w), cv2.BORDER_CONSTANT, value=0)
                h, w = mask.shape
            y = random.randint(0, h - c) if self.train else (h - c) // 2
            x = random.randint(0, w - c) if self.train else (w - c) // 2
            img, mask = img[y:y + c, x:x + c], mask[y:y + c, x:x + c]
        else:
            img, _ = pad_to_multiple(img)
            mask, _ = pad_to_multiple(mask)
        if self.train and random.random() < 0.5:
            img, mask = img[:, ::-1], mask[:, ::-1]
        x = torch.from_numpy(normalize(np.ascontiguousarray(img)))
        y = torch.from_numpy(np.ascontiguousarray(mask))[None]
        return x, y
