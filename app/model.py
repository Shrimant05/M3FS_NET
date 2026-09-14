from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, expand: int):
        super().__init__()
        hidden = in_channels * expand
        if expand == 1:
            layers = [
                nn.Sequential(
                    nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
                    nn.BatchNorm2d(hidden),
                    nn.ReLU6(inplace=True),
                ),
                nn.Conv2d(hidden, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        else:
            layers = [
                nn.Sequential(
                    nn.Conv2d(in_channels, hidden, 1, bias=False),
                    nn.BatchNorm2d(hidden),
                    nn.ReLU6(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
                    nn.BatchNorm2d(hidden),
                    nn.ReLU6(inplace=True),
                ),
                nn.Conv2d(hidden, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        self.conv = nn.Sequential(*layers)
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, inputs: Tensor) -> Tensor:
        output = self.conv(inputs)
        return inputs + output if self.use_residual else output


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, reduction),
            nn.ReLU6(inplace=True),
            nn.Linear(reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        weights = self.fc(inputs.mean(dim=(-2, -1))).unsqueeze(-1).unsqueeze(-1)
        return inputs * weights


class ThreeBranchDetail(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = self._branch(1)
        self.b2 = self._branch(2)
        self.b3 = self._branch(4)
        self.pool1 = nn.AdaptiveAvgPool2d(14)
        self.pool2 = nn.AdaptiveAvgPool2d(28)
        self.pool3 = nn.AdaptiveAvgPool2d(56)
        self.out_pool = nn.AdaptiveAvgPool2d(7)
        self.fuse = nn.Sequential(nn.Conv2d(384, 256, 1), nn.BatchNorm2d(256), nn.ReLU6(inplace=True))
        self.se = SqueezeExcitation(256, 64)

    @staticmethod
    def _branch(dilation: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(3, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU6(inplace=True),
            nn.Conv2d(128, 128, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(128),
            nn.ReLU6(inplace=True),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        branch1 = self.out_pool(self.b1(self.pool1(inputs)))
        branch2 = self.out_pool(self.b2(self.pool2(inputs)))
        branch3 = self.out_pool(self.b3(self.pool3(inputs)))
        return self.se(self.fuse(torch.cat([branch1, branch2, branch3], dim=1)))


class SoftSBFS(nn.Module):
    def __init__(self, in_channels: int, select_count: int):
        super().__init__()
        self.select_count = select_count
        self.se = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(),
            nn.Linear(in_channels // 4, in_channels),
            nn.Sigmoid(),
        )

    @staticmethod
    def _normalize(values: Tensor) -> Tensor:
        return (values - values.min()) / (values.max() - values.min() + 1e-8)

    def _stats(self, inputs: Tensor) -> Tensor:
        batch, channels, height, width = inputs.shape
        flattened = inputs.reshape(batch, channels, -1)
        mean = flattened.mean(-1, keepdim=True)
        std = flattened.std(-1, keepdim=True) + 1e-6
        coefficient = (std / (mean.abs() + 1e-6)).squeeze(-1).mean(0)
        difference = flattened - mean
        kurtosis = (difference ** 4).mean(-1) / (std.squeeze(-1) ** 4 + 1e-6) - 3
        kurtosis = kurtosis.clamp(-10, 10).mean(0)
        probability = F.softmax(flattened.abs(), dim=-1)
        entropy = -(probability * torch.log(probability + 1e-8)).sum(-1).mean(0)
        q25 = flattened.quantile(0.25, dim=-1).mean(0)
        q75 = flattened.quantile(0.75, dim=-1).mean(0)
        iqr = q75 - q25
        return (
            self._normalize(coefficient)
            + self._normalize(kurtosis)
            + self._normalize(entropy)
            + self._normalize(iqr)
        ) / 4

    def forward(self, inputs: Tensor) -> Tensor:
        score = self._stats(inputs) * self.se(inputs.mean(dim=(-2, -1))).mean(0)
        _, indexes = score.topk(self.select_count)
        indexes, _ = indexes.sort()
        selected = inputs[:, indexes, :, :]
        weights = F.softmax(score[indexes], dim=0)
        return selected * weights.view(1, -1, 1, 1) * self.select_count


class PVDSClassifier(nn.Module):
    def __init__(self, class_count: int, alpha: int = 128):
        super().__init__()
        settings = [
            (1, 16, 1), (6, 24, 2), (6, 24, 1), (6, 32, 2),
            (6, 32, 1), (6, 32, 1), (6, 64, 2), (6, 64, 1),
            (6, 64, 1), (6, 64, 1), (6, 96, 1), (6, 96, 1),
            (6, 96, 1), (6, 160, 2), (6, 160, 1), (6, 160, 1),
            (6, 320, 1),
        ]
        layers: list[nn.Module] = [ConvBNAct(3, 32, 2)]
        in_channels = 32
        for expand, out_channels, stride in settings:
            layers.append(InvertedResidual(in_channels, out_channels, stride, expand))
            in_channels = out_channels
        layers.append(nn.Sequential(
            nn.Conv2d(in_channels, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True),
        ))
        self.backbone = nn.Sequential(*layers)
        self.sbfs1 = SoftSBFS(1280, alpha)
        self.sbfs2 = SoftSBFS(256, 96)
        self.tdb = ThreeBranchDetail()
        self.fusion = nn.Sequential(nn.Conv2d(224, 128, 3, padding=2, dilation=2), nn.BatchNorm2d(128), nn.ReLU6(inplace=True))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(alpha, 256),
            nn.ReLU6(),
            nn.Dropout(0.2),
            nn.Linear(256, class_count),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        backbone_features = self.sbfs1(self.backbone(inputs))
        detail_features = self.sbfs2(self.tdb(inputs))
        fused = self.fusion(torch.cat([backbone_features, detail_features], dim=1))
        return self.head(fused)


class DiseaseModel:
    def __init__(self, checkpoint_path: Path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.class_names = checkpoint["class_names"]
        self.image_size = checkpoint.get("img_size", 224)
        config = checkpoint.get("model_config", {})
        self.network = PVDSClassifier(len(self.class_names), config.get("alpha", 128))
        self.network.load_state_dict(checkpoint["state_dict"], strict=True)
        self.network.eval()
        self.mean = torch.tensor(checkpoint.get("normalize_mean", [0.485, 0.456, 0.406])).view(3, 1, 1)
        self.std = torch.tensor(checkpoint.get("normalize_std", [0.229, 0.224, 0.225])).view(3, 1, 1)

    def _transform(self, image: Image.Image) -> Tensor:
        resized = image.convert("RGB").resize((self.image_size, self.image_size))
        pixels = torch.frombuffer(bytearray(resized.tobytes()), dtype=torch.uint8)
        pixels = pixels.reshape(self.image_size, self.image_size, 3).permute(2, 0, 1).float() / 255
        return (pixels - self.mean) / self.std

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> tuple[str, float]:
        logits = self.network(self._transform(image).unsqueeze(0))
        probabilities = torch.softmax(logits, dim=1)[0]
        index = int(torch.argmax(probabilities))
        return self.class_names[index], float(probabilities[index])
