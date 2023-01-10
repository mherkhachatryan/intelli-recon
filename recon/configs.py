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

tb_writer = SummaryWriter('logs/runs/recon_kaggle_experiment')
