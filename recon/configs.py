import torch
from dataclasses import dataclass, field
from torch import nn
import os
from pathlib import Path

from typing import List

# path setting
data_path = Path("/Users/mher/Codes/ASDS21-CV/intelli-recon/data")
log_path = Path("/Users/mher/Codes/ASDS21-CV/intelli-recon/logs/")
os.makedirs(log_path, exist_ok=True)

MODE = "train"  # valid, test, train
# is not used in training
MODEL_PATH = log_path / "model/kaggle_experiment_depth_3/cd_20230110_163308_1.pth"
show_examples = True
OUTPUT_SHAPE = (512, 512)  # do not change for now

# training parameters
BATCH_SIZE = 64
EPOCHS = 1
VALID_SIZE = 0.2
LOSS = "BCELoss"  # do not change for now
OPTIMIZER = "adam"  # do not change for now

# MODEL PARAMETERS
MODEL_NAME = "resnet18"
ENCODER_DEPTH = 5
DECODER_CHANNELS = [256, 128, 64, 32, 16]

# logging params
experiment_name = "kaggle_experiment_depth_5_deep"


# configurations that do not change


@dataclass
class TrainParameters:
    model: nn.Module = None
    _loss: str = "BCEWithLogitsLoss"
    _optimizer: str = "adam"
    epochs: int = 5

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
