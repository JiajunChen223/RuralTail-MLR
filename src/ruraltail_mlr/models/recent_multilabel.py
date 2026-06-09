from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def _build_resnet_features(backbone_name: str, pretrained: bool = True) -> nn.Module:
    from torchvision import models

    if backbone_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        net = models.resnet18(weights=weights)
    elif backbone_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        net = models.resnet50(weights=weights)
    else:
        raise ValueError(f"Unsupported recent-method backbone: {backbone_name}")
    return net


class ResNetFeaturePyramid(nn.Module):
    def __init__(self, backbone_name: str = "resnet18", pretrained: bool = True) -> None:
        super().__init__()
        net = _build_resnet_features(backbone_name, pretrained=pretrained)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        mid = self.layer3(x)
        high = self.layer4(mid)
        return mid, high


class SemanticSpatialBlock(nn.Module):
    def __init__(self, dim: int, num_classes: int, num_heads: int = 4) -> None:
        super().__init__()
        self.classifier = nn.Conv2d(dim, num_classes, kernel_size=1)
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.num_heads = num_heads
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim * 4, kernel_size=1),
            nn.BatchNorm2d(dim * 4),
            nn.ReLU6(inplace=True),
            nn.Conv2d(dim * 4, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
        )

    def _semantic_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits_map = self.classifier(x)
        aux_logits = logits_map.mean(dim=(2, 3))
        weights = torch.sigmoid(logits_map).unsqueeze(2)
        semantic = (weights * x.unsqueeze(1)).mean(dim=(3, 4))
        return aux_logits, semantic

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        aux_logits, semantic = self._semantic_features(x)
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)

        q = self.query(tokens)
        k = self.key(semantic)
        v = self.value(semantic)
        head_dim = c // self.num_heads
        if c % self.num_heads != 0:
            raise ValueError("SFIN attention dimension must be divisible by num_heads")
        q = q.view(b, h * w, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(b, -1, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(b, -1, self.num_heads, head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        attn = attn.softmax(dim=-1)
        fused = (attn @ v).transpose(1, 2).reshape(b, h * w, c)
        fused = self.proj(fused).transpose(1, 2).reshape(b, c, h, w)
        x = x + self.norm1(fused)
        x = x + self.norm2(self.ffn(x))
        return aux_logits, semantic, x


class CrossInteractionAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_high = nn.Conv2d(2, 1, kernel_size=1)
        self.conv_mid = nn.Conv2d(2, 1, kernel_size=1)

    @staticmethod
    def _spatial_summary(x: torch.Tensor) -> torch.Tensor:
        max_map = torch.max(x, dim=1, keepdim=True).values
        avg_map = torch.mean(x, dim=1, keepdim=True)
        return torch.cat([max_map, avg_map], dim=1)

    def forward(self, high: torch.Tensor, mid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        high_attn = self.conv_high(self._spatial_summary(high))
        mid_attn = self.conv_mid(self._spatial_summary(mid))
        mid_to_high = F.adaptive_avg_pool2d(mid_attn, high.shape[-2:])
        high_to_mid = F.interpolate(high_attn, size=mid.shape[-2:], mode="bilinear", align_corners=False)
        high_gate = torch.sigmoid(high_attn) + torch.sigmoid(mid_to_high)
        mid_gate = torch.sigmoid(mid_attn) + torch.sigmoid(high_to_mid)
        return high * high_gate, mid * mid_gate


class SFINClassifier(nn.Module):
    """Stable SFIN-style adapter for the common RuralTail-MLR train/eval pipeline."""

    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        heads: int = 4,
        high_dim: int = 512,
        mid_dim: int = 256,
    ) -> None:
        super().__init__()
        self.backbone = ResNetFeaturePyramid(backbone_name, pretrained=pretrained)
        self.high_block = SemanticSpatialBlock(high_dim, num_classes, num_heads=heads)
        self.mid_block = SemanticSpatialBlock(mid_dim, num_classes, num_heads=heads)
        self.ciam = CrossInteractionAttention()
        self.high_classifier = nn.Conv2d(high_dim, num_classes, kernel_size=1)
        self.mid_classifier = nn.Conv2d(mid_dim, num_classes, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        mid, high = self.backbone(image)
        _, _, high = self.high_block(high)
        _, _, mid = self.mid_block(mid)
        high, mid = self.ciam(high, mid)
        high_logits = self.high_classifier(high).mean(dim=(2, 3))
        mid_logits = self.mid_classifier(mid).mean(dim=(2, 3))
        return 0.5 * (high_logits + mid_logits)


class GroupWiseLinear(nn.Module):
    def __init__(self, num_classes: int, hidden_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, hidden_dim))
        self.bias = nn.Parameter(torch.empty(num_classes)) if bias else None
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = (x * self.weight.unsqueeze(0)).sum(dim=-1)
        if self.bias is not None:
            out = out + self.bias.unsqueeze(0)
        return out


class MLMambaClassifier(nn.Module):
    """MLMamba-style adapter using mamba-ssm when available."""

    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        hidden_dim: int = 128,
        backbone_dim: int = 512,
    ) -> None:
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except ImportError as exc:
            raise ImportError(
                "MLMambaClassifier requires mamba-ssm and causal-conv1d. "
                "Install them before running the MLMamba experiment."
            ) from exc
        self.backbone = ResNetFeaturePyramid(backbone_name, pretrained=pretrained)
        self.visual_proj = nn.Linear(backbone_dim, hidden_dim)
        self.visual_mamba = Mamba(d_model=hidden_dim, d_state=16, d_conv=4, expand=2)
        self.query_embed = nn.Parameter(torch.zeros(1, num_classes, hidden_dim))
        nn.init.normal_(self.query_embed, std=0.02)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.label_mamba = Mamba(d_model=hidden_dim, d_state=16, d_conv=4, expand=2)
        self.classifier = GroupWiseLinear(num_classes, hidden_dim, bias=True)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        _, high = self.backbone(image)
        tokens = high.flatten(2).transpose(1, 2)
        tokens = self.visual_proj(tokens)
        tokens = self.visual_mamba(tokens)
        queries = self.query_embed.expand(image.shape[0], -1, -1)
        label_tokens, _ = self.cross_attn(queries, tokens, tokens, need_weights=False)
        label_tokens = self.label_mamba(label_tokens)
        return self.classifier(label_tokens)
