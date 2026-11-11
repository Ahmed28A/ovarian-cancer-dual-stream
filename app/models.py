"""
Model definitions for the Ovarian Cancer Dual-Stream system.

Stream A : ConvNeXt-Base + CBAM (after stage 4) + dual pooling (GAP + GMP) + MLP head
           -> 5 histological subtypes (HGSC, EC, CC, MC, LGSC) from H&E patches.
Stream B : CatBoost on a 16-biomarker clinical blood panel (see app.py).

Architecture code is identical to notebooks/01_stream_a_histopathology_convnext_cbam.ipynb
(class ConvNeXtSingleCBAM), so the checkpoint trained there loads directly.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

SUBTYPES = ["HGSC", "EC", "CC", "MC", "LGSC"]
SUBTYPE_NAMES = {
    "HGSC": "High-Grade Serous Carcinoma",
    "EC": "Endometrioid Carcinoma",
    "CC": "Clear Cell Carcinoma",
    "MC": "Mucinous Carcinoma",
    "LGSC": "Low-Grade Serous Carcinoma",
}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x):
        gate = torch.sigmoid(self.mlp(self.avg(x)) + self.mlp(self.max(x)))
        return x * gate.view(x.size(0), x.size(1), 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx = x.max(dim=1, keepdim=True)[0]
        return x * torch.sigmoid(self.bn(self.conv(torch.cat([avg, mx], dim=1))))


class CBAM(nn.Module):
    """Convolutional Block Attention Module: channel attention then spatial attention."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention()

    def forward(self, x):
        return self.spatial(self.channel(x))


class ConvNeXtSingleCBAM(nn.Module):
    """ConvNeXt-Base + single CBAM after the last stage + GAP/GMP dual pooling + MLP head."""

    def __init__(self, num_classes=5, drop=0.3):
        super().__init__()
        base = models.convnext_base(weights=None)  # trained weights come from the checkpoint
        self.backbone = base.features
        self.cbam = CBAM(1024)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))
        self.head = nn.Sequential(
            nn.LayerNorm(2048),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.cbam(self.backbone(x))
        return self.head(torch.cat([self.gap(x).flatten(1), self.gmp(x).flatten(1)], dim=1))


def load_stream_a(checkpoint_path, device="cpu"):
    """Load a checkpoint saved by the notebook (plain state_dict or a dict wrapping it)."""
    model = ConvNeXtSingleCBAM()
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model_state", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt)
    return model.to(device).eval()


class GradCAM:
    """Hook-based Grad-CAM on the last ConvNeXt stage."""

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self._fwd = target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, "activations", o.detach())
        )
        self._bwd = target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "gradients", go[0].detach())
        )

    def generate(self, tensor, cls_idx):
        self.model.zero_grad()
        out = self.model(tensor)
        out[0, cls_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1).squeeze()).cpu().numpy()
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    def remove(self):
        self._fwd.remove()
        self._bwd.remove()
