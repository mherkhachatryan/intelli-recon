from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from typing import List, Tuple


def _read_image(img_path: Path, label=False) -> Image:
    if label:
        image = Image.open(img_path).convert("RGB")
    else:
        image = Image.open(img_path)

    return image  # noqa


class Preprocess:
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.images_path = data_path / "images"
        self.train_label_path = data_path / "train_labels"

        self.__train_file_path = self.images_path / "train.txt"
        self.__test_files_path = self.images_path / "test.txt"
        self.train_city_names = list(pd.read_csv(self.__train_file_path).columns)

    def _get_image_pairs_paths(self, city_name: str) -> Tuple[Path,
                                                              Path,
                                                              Path]:
        city_path = self.images_path / city_name / "pair"
        image_1 = city_path / "img1.png"
        image_2 = city_path / "img2.png"

        label = self.train_label_path / city_name / "cm" / "cm.png"

        return image_1, image_2, label

    def get_all_image_pairs_paths(self, cities: List[str]) -> Tuple[List[Path],
                                                                    List[Path],
                                                                    List[Path]]:
        images_1 = []
        images_2 = []
        labels = []
        for city in cities:
            image_1, image_2, label = self._get_image_pairs_paths(city)
            images_1.append(image_1)
            images_2.append(image_2)
            labels.append(label)

        return images_1, images_2, labels

    @staticmethod
    def _divide_into_patches(img1: List[Path], img2: List[Path], label: List[Path]) -> pd.DataFrame:
        path1 = []
        path2 = []
        label_path = []
        start_row = []
        end_row = []
        start_col = []
        end_col = []

        for im1, im2, lab in zip(img1, img2, label):
            for row in [0, 64, 128, 192, 256, 320, 384, 448]:  # TODO do for any patches and for any output shape
                for col in [0, 64, 128, 192, 256, 320, 384, 448]:
                    row_start = row
                    row_end = row + 64
                    col_start = col
                    col_end = col + 64

                    path1.append(im1)
                    path2.append(im2)
                    label_path.append(lab)

                    start_row.append(row_start)
                    end_row.append(row_end)

                    start_col.append(col_start)
                    end_col.append(col_end)

        patches_df = pd.DataFrame(
            {"path1": path1, "path2": path2, "label": label_path, "start_row": start_row, "end_row": end_row,
             "start_col": start_col, "end_col": end_col})
        return patches_df

    def image_patches(self, mode: str = "train") -> pd.DataFrame:
        """Divide image into patches and store information into dataframe.
        Dataframe stores info about first and second image filepath, label filepath, and coordinates to cut patches.

        Parameters
        ----------
        mode : ["train", "test"]
        # patch_size : even number to divide patches

        Returns
        -------
        DataFrame of patch information
        """
        if mode.lower() in ["train", "valid"]:
            image_1, image_2, label = self.get_all_image_pairs_paths(self.train_city_names)
        elif mode.lower() == "test":
            raise NotImplemented("Waiting for train")
        else:
            raise ValueError("Wrong mode")

        return self._divide_into_patches(image_1, image_2, label)


class SentinelDataset(Dataset):
    def __init__(self, image_patches: pd.DataFrame, output_shape=(512, 512), transform=None):
        self.image_patches = image_patches
        self.transform = transform
        self.output_shape = output_shape

    def __len__(self):
        return len(self.image_patches)

    def __getitem__(self, item_idx):
        path1 = self.image_patches.loc[item_idx, "path1"]
        path2 = self.image_patches.loc[item_idx, "path2"]
        label_path = self.image_patches.loc[item_idx, "label"]

        start_row = self.image_patches.loc[item_idx, "start_row"]
        end_row = self.image_patches.loc[item_idx, "end_row"]
        start_col = self.image_patches.loc[item_idx, "start_col"]
        end_col = self.image_patches.loc[item_idx, "end_col"]

        image_1 = _read_image(path1)
        image_1 = self._transform_image(image_1, output_shape=self.output_shape)
        image_1 = image_1[start_row:end_row, start_col:end_col]

        image_2 = _read_image(path2)
        image_2 = self._transform_image(image_2, output_shape=self.output_shape)
        image_2 = image_2[start_row:end_row, start_col:end_col]

        label = _read_image(label_path, label=True)
        label = self._transform_image(label, output_shape=self.output_shape, label=True)
        label = label[start_row:end_row, start_col:end_col]

        image_1 = torch.tensor(image_1.transpose((2, 1, 0)), dtype=torch.float)
        image_2 = torch.tensor(image_2.transpose((2, 1, 0)), dtype=torch.float)

        label = torch.tensor(label, dtype=torch.long)

        return image_1, image_2, label

    @staticmethod
    def _transform_image(image: Image, output_shape: Tuple[int, int] = (512, 512), label=False) -> np.ndarray:
        if label:
            image = np.array(image.resize(output_shape))
            image = (np.mean(image, axis=-1) > 0.5).astype(np.uint8)
        else:
            image = np.array(image.resize(output_shape)) / 255  # resize and normalize

        return image  # TODO output torch
