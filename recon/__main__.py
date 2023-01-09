from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
import torchvision

from configs import *
from preprocessing import SentinelDataset, Preprocess
from model import ChangeNet
from training import TrainChangeDetection, TrainParameters

data_path = Path(data_path)
proc = Preprocess(data_path)
patches_df = proc.image_patches(mode=MODE)

patches_df_train, patches_df_test = train_test_split(patches_df, test_size=VALID_SIZE)

neptune_logger["config/dataset/train_size"] = len(patches_df_train)
neptune_logger["config/dataset/valid_size"] = len(patches_df_test)

# resetting index here so Dataset length and indexing is done correctly
train_ds = SentinelDataset(patches_df_train.reset_index(drop=True), output_shape=OUTPUT_SHAPE)
val_ds = SentinelDataset(patches_df_test.reset_index(drop=True), output_shape=OUTPUT_SHAPE)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

model = ChangeNet(model_name=MODEL_NAME)

# pass sample to model to log in tensorboard
dataiter = iter(train_loader)
image_1, image_2, label = next(dataiter)


tb_writer.add_graph(model, (image_1, image_2))
tb_writer.close()
neptune_logger["config/model"] = type(model).__name__

if MODE == "train":
    train_params = TrainParameters(model=model, _loss=LOSS, _optimizer=OPTIMIZER, epochs=EPOCHS)
    training = TrainChangeDetection(train_params, train_loader, val_loader)

    training.train()
