import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from datetime import datetime
from torchmetrics.classification import BinaryJaccardIndex, BinaryF1Score

from typing import Tuple
import os

from configs import log_path, experiment_name
from validation import TrainParameters
from experiment_tracking import tb_writer, neptune_logger
from utils import tensor_to_numpy


class TrainChangeDetection:
    def __init__(self, train_params: TrainParameters):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = train_params.model
        self.model.to(self.device)
        self.loss = train_params.loss
        self.optimizer = train_params.optimizer
        self.epochs = train_params.epochs
        self.segmentation_threshold = train_params.segmentation_threshold

        self.iou_metric = BinaryJaccardIndex(threshold=self.segmentation_threshold).to(self.device)
        self.f1_metric = BinaryF1Score(threshold=self.segmentation_threshold).to(self.device)

        self.model_save_path = log_path / "model" / experiment_name

        os.makedirs(self.model_save_path, exist_ok=True)

    def _train_one_epoch(self, epoch_idx: int, train_loader: DataLoader) -> Tuple[float, float, float]:
        running_loss = 0.
        last_loss = 0.

        running_iou = 0.
        last_iou = 0.

        running_f1 = 0.
        last_f1 = 0.

        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            image_1, image_2, target = data
            image_1, image_2, target = image_1.to(self.device), image_2.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(image_1, image_2).squeeze()
            target = target.float().squeeze()

            loss = self.loss(outputs, target)
            loss.backward()

            self.optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            running_iou += self.iou_metric(outputs, target.int())
            running_f1 += self.f1_metric(outputs, target.int())

            reporting_batch_size = len(train_loader) // 4
            if i % reporting_batch_size == reporting_batch_size - 1:
                last_loss = running_loss / reporting_batch_size  # gather data every reporting_batch_size mini-batch
                last_iou = running_iou / reporting_batch_size
                last_f1 = running_f1 / reporting_batch_size

                tb_writer.add_scalar('batch training loss',
                                     last_loss,
                                     epoch_idx * len(train_loader) + i)
                tb_writer.add_scalar('batch training iou',
                                     last_iou,
                                     epoch_idx * len(train_loader) + i)

                tb_writer.add_scalar('batch training f1',
                                     last_f1,
                                     epoch_idx * len(train_loader) + i)

                neptune_logger["train/batch/loss"].log(last_loss)
                neptune_logger["train/batch/iou"].log(last_iou)
                neptune_logger["train/batch/f1"].log(last_f1)

            running_loss = 0.
            running_iou = 0.
            running_f1 = 0.

        return last_loss, last_iou, last_f1

    def fit(self, train_loader: DataLoader = None,
            val_loader: DataLoader = None):  # TODO add option to reload training
        best_val_loss = 1e6

        for epoch in tqdm(range(self.epochs)):
            self.model.train(True)

            avg_train_loss, avg_train_iou, avg_train_f1 = self._train_one_epoch(epoch, train_loader)
            self.model.train(False)

            running_val_loss = []
            running_val_iou = []
            running_val_f1 = []

            for val_data in tqdm(val_loader, total=len(val_loader)):
                image_1_val, image_2_val, target_val = val_data
                image_1_val, image_2_val, target_val = image_1_val.to(self.device), image_2_val.to(
                    self.device), target_val.to(self.device)

                val_output = self.model(image_1_val, image_2_val).squeeze()
                target_val = target_val.float().squeeze()

                val_loss = self.loss(val_output, target_val)
                iou = self.iou_metric(val_output, target_val.int())
                f1 = self.f1_metric(val_output, target_val.int())

                running_val_loss.append(tensor_to_numpy(val_loss))  # to keep array without gradient
                running_val_iou.append(tensor_to_numpy(iou))
                running_val_f1.append(tensor_to_numpy(f1))

            avg_val_loss = np.mean(running_val_loss)
            avg_val_iou = np.mean(running_val_iou)
            avg_val_f1 = np.mean(running_val_f1)

            # reporting loss
            tb_writer.add_scalar('validation loss',
                                 avg_val_loss,
                                 epoch)
            tb_writer.add_scalar('training loss',
                                 avg_train_loss,
                                 epoch)

            neptune_logger["train/loss/"].log(avg_train_loss)
            neptune_logger["valid/loss/"].log(avg_val_loss)
            # reporting metric
            tb_writer.add_scalar('training iou',
                                 avg_train_iou,
                                 epoch)
            tb_writer.add_scalar('training f1',
                                 avg_train_f1,
                                 epoch)
            tb_writer.add_scalar('validation iou',
                                 avg_val_iou,
                                 epoch)
            tb_writer.add_scalar('validation f1',
                                 avg_val_f1,
                                 epoch)
            neptune_logger["train/iou/"].log(avg_train_iou)
            neptune_logger["valid/iou/"].log(avg_val_iou)
            neptune_logger["train/f1/"].log(avg_train_f1)
            neptune_logger["valid/f1/"].log(avg_val_f1)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self._save_model(epoch)

    def _save_model(self, epoch_idx):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f'cd_{timestamp}_{epoch_idx}.pth'
        model_full_path = self.model_save_path / model_name
        torch.save(self.model.state_dict(), model_full_path)
        neptune_logger["model_weights"].upload(str(model_full_path))

    def predict(self, dataset: Dataset = None, sample: int = 42) -> torch.tensor:
        """Evaluate model on given dataset for single given dataset.

        Parameters
        ----------
        dataset
        sample

        Returns
        -------
            Binary segmentation mask
        """
        self.model.eval()
        with torch.no_grad():
            input_image_1 = dataset[sample][0].unsqueeze(dim=0).to(self.device)
            input_image_2 = dataset[sample][1].unsqueeze(dim=0).to(self.device)
            mask = self.model(input_image_1, input_image_2).squeeze()

            binary_mask = torch.where(mask >= self.segmentation_threshold, torch.tensor(1, dtype=torch.uint8),
                                      torch.tensor(0, dtype=torch.uint8))
            return binary_mask

    def load_model(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
