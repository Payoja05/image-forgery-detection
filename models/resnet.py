"""Minimal ResNet-18 (He et al., 2016) whose parameter names match
torchvision's, so the official PyTorch ImageNet weights load directly.

torchvision itself is not used because its compiled extension is blocked by
Windows Smart App Control on the development machine. The weights are the
official file published by the PyTorch team:
    https://download.pytorch.org/models/resnet18-f37072fd.pth
(identical to torchvision's ResNet18_Weights.IMAGENET1K_V1; BSD-3 licence).
The download is verified with the SHA-256 prefix embedded in the file name.
"""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn

RESNET18_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
RESNET18_SHA256_PREFIX = "f37072fd"


class BasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.downsample = None
        if stride != 1 or cin != cout:
            self.downsample = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        idt = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + idt)


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        self.layer2 = nn.Sequential(BasicBlock(64, 128, 2), BasicBlock(128, 128))
        self.layer3 = nn.Sequential(BasicBlock(128, 256, 2), BasicBlock(256, 256))
        self.layer4 = nn.Sequential(BasicBlock(256, 512, 2), BasicBlock(512, 512))
        # classification head kept only so the official state_dict loads strictly
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 1000)


def imagenet_resnet18(cache_dir: Path | None = None) -> ResNet18:
    """ResNet-18 with official ImageNet weights (downloaded once, then cached)."""
    cache_dir = cache_dir or Path(torch.hub.get_dir()) / "checkpoints"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / RESNET18_URL.rsplit("/", 1)[1]
    if not path.exists():
        print(f"Downloading ImageNet ResNet-18 weights -> {path}")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(RESNET18_URL, tmp)
        tmp.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if not digest.startswith(RESNET18_SHA256_PREFIX):
        raise RuntimeError(f"Checksum mismatch for {path.name}; delete it and retry.")
    model = ResNet18()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return model
