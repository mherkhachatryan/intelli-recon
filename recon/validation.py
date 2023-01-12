from typing import List

import torch
from dataclasses import dataclass, field
from torch import nn


@dataclass
class TrainParameters:
    model: nn.Module = None
    _loss: str = "BCEWithLogitsLoss"
    _optimizer: str = "adam"
    epochs: int = 5
    segmentation_threshold: float = 0.55

    @property
    def loss(self):
        if self._loss == "BCEWithLogitsLoss":
            return nn.BCEWithLogitsLoss()
        elif self._loss == "BCELoss":
            return nn.BCELoss()
        else:
            raise ValueError(f"Invalid loss function: {self._loss}")

    @property
    def optimizer(self):
        if self._optimizer.lower() == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=0.0001)


@dataclass
class ModelParameters:
    model_name: str = "resnet18"
    encoder_depth: int = 3
    decoder_channels: List[int] = field(default_factory=lambda: [64, 64, 16])  # TODO LENGTH MUST BE ENCODER_DEPTH
