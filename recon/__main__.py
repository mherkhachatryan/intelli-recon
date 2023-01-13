from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from configs import *
from experiment_tracking import tb_writer, neptune_logger
from preprocessing import SentinelDataset, Preprocess
from model import ChangeNet
from training import TrainChangeDetection
from validation import TrainParameters, ModelParameters
from utils import show_tensor
import random

# Getting ready the data
proc = Preprocess(data_path)
patches_df = proc.image_patches(mode=MODE)

# getting ready the model
model_params = ModelParameters(model_name=MODEL_NAME, encoder_depth=ENCODER_DEPTH, decoder_channels=DECODER_CHANNELS)
model = ChangeNet(model_params=model_params)

# getting ready training
train_params = TrainParameters(model=model, _loss=LOSS, _optimizer=OPTIMIZER, epochs=EPOCHS,
                               segmentation_threshold=SEGMENTATION_THRESHOLD)
training = TrainChangeDetection(train_params)
neptune_logger["config/model_params"] = model_params.__dict__
neptune_logger["config/training_params"] = train_params.__dict__
neptune_logger["config/mode"] = MODE

if MODE in ["valid", "train"]:
    patches_df_train, patches_df_valid = train_test_split(patches_df, test_size=VALID_SIZE)

    # resetting index here so Dataset length and indexing is done correctly
    train_ds = SentinelDataset(patches_df_train.reset_index(drop=True), output_shape=OUTPUT_SHAPE)
    val_ds = SentinelDataset(patches_df_valid.reset_index(drop=True), output_shape=OUTPUT_SHAPE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # pass sample to model to log in tensorboard
    dataiter = iter(train_loader)
    image_1, image_2, label = next(dataiter)

    tb_writer.add_graph(model, (image_1, image_2))
    tb_writer.close()

    if MODE == "train":
        training.fit(train_loader=train_loader, val_loader=val_loader)

    elif MODE == "valid":
        if show_examples:  # TODO add tensorboard
            
            for _ in range(5):
                sample = random.randint(0, len(val_ds))
                image_1 = val_ds[sample][0]
                image_2 = val_ds[sample][1]
                gd = val_ds[sample][2]
                training.load_model(MODEL_PATH)
                mask = training.predict(val_ds, sample=sample)

                show_tensor([image_1, image_2, gd, mask], grid=True)


elif MODE == "test":
    test_ds = SentinelDataset(patches_df.reset_index(drop=True), output_shape=OUTPUT_SHAPE)
    training.load_model(MODEL_PATH)
    for sample in range(len(test_ds)):
        image_1 = test_ds[sample][0]
        image_2 = test_ds[sample][1]

        mask = training.predict(test_ds, sample=sample)
        show_tensor([image_1, image_2, mask, mask], grid=True)
