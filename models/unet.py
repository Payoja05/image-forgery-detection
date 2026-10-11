"""ForensicUNet: a U-Net for pixel-level tampering localisation.

Architecture
------------
input RGB (ImageNet-normalised, 3 x H x W)
   |-- SRM filters (fixed, 3 high-pass 5x5 kernels) -> 3-channel noise residual
   |
   concat -> 6 channels
   |
ResNet-18 encoder (official PyTorch ImageNet weights; first conv widened to 6 ch)
   1/2 (64) -> 1/4 (64) -> 1/8 (128) -> 1/16 (256) -> 1/32 (512)
   |
U-Net decoder with skip connections (bilinear upsampling + 2x conv-bn-relu)
   |
1 x H x W logits  ->  sigmoid  ->  tampering probability per pixel

Why SRM? The rich-model high-pass filters (Fridrich & Kodovsky, 2012) suppress
image content and expose local noise patterns, which are often inconsistent
between a pasted region and its host image. RGB-N (Zhou et al., CVPR 2018)
showed they help tampering localisation. RGB alone tends to learn *semantic*
cues ("there is an animal here") instead of *forensic* ones.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet import ResNet18, imagenet_resnet18


def _srm_kernels() -> torch.Tensor:
    k1 = torch.tensor([[0, 0, 0, 0, 0],
                       [0, -1, 2, -1, 0],
                       [0, 2, -4, 2, 0],
                       [0, -1, 2, -1, 0],
                       [0, 0, 0, 0, 0]], dtype=torch.float32) / 4.0
    k2 = torch.tensor([[-1, 2, -2, 2, -1],
                       [2, -6, 8, -6, 2],
                       [-2, 8, -12, 8, -2],
                       [2, -6, 8, -6, 2],
                       [-1, 2, -2, 2, -1]], dtype=torch.float32) / 12.0
    k3 = torch.tensor([[0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 1, -2, 1, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0]], dtype=torch.float32) / 2.0
    # each output channel = same kernel applied to R, G and B, averaged
    w = torch.stack([k1, k2, k3])[:, None].repeat(1, 3, 1, 1) / 3.0
    return w  # 3 x 3 x 5 x 5


class SRMLayer(nn.Module):
    def __init__(self, clip: float = 3.0):
        super().__init__()
        self.register_buffer("weight", _srm_kernels())
        self.clip = clip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(F.conv2d(x, self.weight, padding=2), -self.clip, self.clip)


class ConvBlock(nn.Sequential):
    def __init__(self, cin: int, cout: int):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )


class UpBlock(nn.Module):
    def __init__(self, cin: int, cskip: int, cout: int):
        super().__init__()
        self.conv = ConvBlock(cin + cskip, cout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class ForensicUNet(nn.Module):
    ARCH_NAME = "ForensicUNet-ResNet18-SRM"

    def __init__(self, pretrained_encoder: bool = False):
        super().__init__()
        self.srm = SRMLayer()
        enc = imagenet_resnet18() if pretrained_encoder else ResNet18()

        # widen the stem from 3 to 6 input channels (RGB + SRM residual)
        old = enc.conv1
        self.conv1 = nn.Conv2d(6, 64, 7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.conv1.weight[:, :3] = old.weight
            self.conv1.weight[:, 3:] = old.weight * 0.5
        self.bn1, self.relu, self.maxpool = enc.bn1, enc.relu, enc.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = enc.layer1, enc.layer2, enc.layer3, enc.layer4

        self.up4 = UpBlock(512, 256, 256)
        self.up3 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.up1 = UpBlock(64, 64, 64)
        self.head = nn.Sequential(ConvBlock(64, 32), nn.Conv2d(32, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        x = torch.cat([x, self.srm(x)], dim=1)
        x0 = self.relu(self.bn1(self.conv1(x)))      # 1/2
        x1 = self.layer1(self.maxpool(x0))           # 1/4
        x2 = self.layer2(x1)                         # 1/8
        x3 = self.layer3(x2)                         # 1/16
        x4 = self.layer4(x3)                         # 1/32
        d = self.up4(x4, x3)
        d = self.up3(d, x2)
        d = self.up2(d, x1)
        d = self.up1(d, x0)
        d = F.interpolate(d, size=size, mode="bilinear", align_corners=False)
        return self.head(d)                          # logits, 1 x H x W
