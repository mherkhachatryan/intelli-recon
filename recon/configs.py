import neptune.new as neptune
import torch
from dataclasses import dataclass, field
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import os
from pathlib import Path

from typing import List

MODE = "train"  # valid, test, train
show_examples = True
OUTPUT_SHAPE = (512, 512)  # do not change for now

# training parameters
BATCH_SIZE = 64
EPOCHS = 15
VALID_SIZE = 0.2
LOSS = "BCEWithLogitsLoss"  # do not change for now
OPTIMIZER = "adam"  # do not change for now

# MODEL PARAMETERS
MODEL_NAME = "resnet18"
ENCODER_DEPTH = 5
DECODER_CHANNELS = [256, 128, 64, 32, 16]

data_path = Path("/Users/mher/Codes/ASDS21-CV/intelli-recon/data")
log_path = Path("/Users/mher/Codes/ASDS21-CV/intelli-recon/logs/")
os.makedirs(log_path, exist_ok=True)

neptune_project_name = "mherkhachatryan/intelli-recon"
neptune_config = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5MzM0ZThlNS1hZGUxLTRkOTQtYmQyYy1hYzEzM2U5MWUzODAifQ=="

# init neptune
neptune_logger = neptune.init_run(
    project=neptune_project_name,
    api_token=neptune_config,
)

neptune_logger["config/dataset/path"] = data_path

experiment_name = "kaggle_experiment_depth_5"
tb_writer = SummaryWriter(f'logs/runs/{experiment_name}')


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
