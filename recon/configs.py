import os
from pathlib import Path

# path setting
experiment_name = "depth_3"
data_path = Path("/content/drive/MyDrive/recon_training/data_light")
log_path = Path("/content/drive/MyDrive/recon_training/logs/")
os.makedirs(log_path, exist_ok=True)

MODE = "train"  # valid, test, train
# is not used in training
MODEL_PATH = log_path / "model/kaggle_experiment_depth_3/cd_20230110_163308_1.pth"
show_examples = True
OUTPUT_SHAPE = (512, 512)  # do not change for now

# training parameters
BATCH_SIZE = 64
EPOCHS = 80
VALID_SIZE = 0.2
LOSS = "BCELoss"  # do not change for now
OPTIMIZER = "adam"  # do not change for now

# MODEL PARAMETERS
MODEL_NAME = "resnet18"
ENCODER_DEPTH = 3
DECODER_CHANNELS = [64, 64, 16]
