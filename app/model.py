from __future__ import annotations

from pathlib import Path

import torch
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
                ),
                nn.Conv2d(hidden, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        else:
            layers = [
                nn.Sequential(
                    nn.Conv2d(in_channels, hidden, 1, bias=False),
                    nn.BatchNorm2d(hidden),
                ),
                nn.Sequential(
                    nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
                    nn.BatchNorm2d(hidden),
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
        )


class ThreeBranchDetail(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = self._branch()
        self.b2 = self._branch()
        self.b3 = self._branch()
        self.fuse = nn.Sequential(nn.Conv2d(384, 256, 1), nn.BatchNorm2d(256), nn.ReLU6(inplace=True))
        self.se = SqueezeExcitation(256, 64)

    @staticmethod
    def _branch() -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(3, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU6(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU6(inplace=True),
        )


class PVDSClassifier(nn.Module):
    def __init__(self, class_count: int, alpha: int = 128):
        super().__init__()
        settings = [
            (1, 16, 1), (6, 24, 2), (6, 24, 1), (6, 32, 2),
            (6, 32, 1), (6, 32, 1), (6, 64, 2), (6, 64, 1),
            (6, 64, 1), (6, 64, 1), (6, 96, 1), (6, 96, 1),
            (6, 96, 1), (6, 160, 2), (6, 160, 1), (6, 160, 1),
            (6, 320, 2),
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
        self.sbfs1 = nn.Module()
        self.sbfs1.se = nn.Sequential(nn.Linear(1280, 320), nn.ReLU6(inplace=True), nn.Linear(320, 1280))
        self.sbfs2 = nn.Module()
        self.sbfs2.se = nn.Sequential(nn.Linear(256, 64), nn.ReLU6(inplace=True), nn.Linear(64, 256))
        self.tdb = ThreeBranchDetail()
        self.fusion = nn.Sequential(nn.Conv2d(224, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU6(inplace=True))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(alpha, 256),
            nn.ReLU6(),
            nn.Dropout(0.2),
            nn.Linear(256, class_count),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.backbone(inputs)
        pooled = torch.flatten(torch.mean(features, dim=(2, 3)), 1)
        return self.head[5](self.head[3](self.head[2](pooled[:, : self.head[2].in_features])))


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
