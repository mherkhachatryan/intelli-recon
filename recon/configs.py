import neptune.new as neptune
from torch.utils.tensorboard import SummaryWriter
import os
from pathlib import Path

MODE = "train"
show_examples = True
OUTPUT_SHAPE = (512, 512)  # do not change for now
MODEL_NAME = "resnet18"
BATCH_SIZE = 64
EPOCHS = 15
VALID_SIZE = 0.2
LOSS = "BCEWithLogitsLoss"  # do not change for now
OPTIMIZER = "adam"  # do not change for now

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

tb_writer = SummaryWriter('logs/runs/recon_kaggle_experiment')
