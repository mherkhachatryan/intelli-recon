import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
from datetime import datetime
from torchmetrics.classification import BinaryJaccardIndex, BinaryF1Score

from typing import Tuple
import os

from config.configs import log_path, TrainParameters, experiment_name
from config.experiment_tracking import tb_writer


class TrainChangeDetection:
    def __init__(self, train_params: TrainParameters, train_loader, val_loader):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = train_params.model
        self.loss = train_params.loss
        self.optimizer = train_params.optimizer
        self.epochs = train_params.epochs

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.iou_metric = BinaryJaccardIndex(threshold=0.5)
        self.f1_metric = BinaryF1Score(threshold=0.5)

        self.model_save_path = log_path / "model" / experiment_name

        os.makedirs(self.model_save_path, exist_ok=True)

    def _train_one_epoch(self, epoch_idx: int) -> Tuple[float, float, float]:
        running_loss = 0.
        last_loss = 0.

        running_iou = 0.
        last_iou = 0.

        running_f1 = 0.
        last_f1 = 0.

        for i, data in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
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

            reporting_batch_size = len(self.train_loader) // 4
            if i % reporting_batch_size == reporting_batch_size - 1:
                last_loss = running_loss / reporting_batch_size  # gather data every reporting_batch_size mini-batch
                last_iou = running_iou / reporting_batch_size
                last_f1 = running_f1 / reporting_batch_size

                tb_writer.add_scalar('batch training loss',
                                     last_loss,
                                     epoch_idx * len(self.train_loader) + i)
                tb_writer.add_scalar('batch training iou',
                                     last_iou,
                                     epoch_idx * len(self.train_loader) + i)

                tb_writer.add_scalar('batch training f1',
                                     last_f1,
                                     epoch_idx * len(self.train_loader) + i)
            running_loss = 0.
            running_iou = 0.
            running_f1 = 0.

        return last_loss, last_iou, last_f1

    def train(self):  # TODO add option to reload training
        best_val_loss = 1e6

        for epoch in tqdm(range(self.epochs)):
            self.model.train(True)

            avg_train_loss, avg_train_iou, avg_train_f1 = self._train_one_epoch(epoch)
            self.model.train(False)

            running_val_loss = []
            running_val_iou = []
            running_val_f1 = []

            for val_data in tqdm(self.val_loader, total=len(self.val_loader)):
                image_1_val, image_2_val, target_val = val_data
                image_1_val, image_2_val, target_val = image_1_val.to(self.device), image_2_val.to(
                    self.device), target_val.to(self.device)

                val_output = self.model(image_1_val, image_2_val).squeeze()
                target_val = target_val.float().squeeze()

                val_loss = self.loss(val_output, target_val)
                iou = self.iou_metric(val_output, target_val.int())
                f1 = self.f1_metric(val_output, target_val.int())

                running_val_loss.append(val_loss.detach().numpy())  # to keep array without gradient
                running_val_iou.append(iou.detach().numpy())
                running_val_f1.append(f1.detach().numpy())

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

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self._save_model(epoch)

    def _save_model(self, epoch_idx):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f'cd_{timestamp}_{epoch_idx}.pth'
        model_full_path = self.model_save_path / model_name
        torch.save(self.model.state_dict(), model_full_path)

    def predict(self, dataset: Dataset = None, sample: int = 42):
        self.model.eval()
        with torch.no_grad():
            input_image_1 = dataset[sample][0].unsqueeze(dim=0).to(self.device)
            input_image_2 = dataset[sample][1].unsqueeze(dim=0).to(self.device)
            logits = self.model(input_image_1, input_image_2)

            prediction_mask = torch.sigmoid(logits).squeeze()

            return prediction_mask

    def load_model(self, path: str):
        self.model.load_state_dict(torch.load(path))
