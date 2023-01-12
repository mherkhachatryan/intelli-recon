from PIL import Image
from pathlib import Path

import matplotlib.pyplot as plt
from typing import List
import torch
import numpy as np


def show_tensor(tensors: List[torch.Tensor], grid: bool = False, cmap: str = "gray"):
    """
    Displays PyTorch tensors as images using matplotlib.

    Parameters:
        tensors (List[torch.Tensor]): A list of tensors to be displayed.
        grid (bool, optional): Whether to display the images in a grid. Default is False.
        cmap (str, optional): Colormap to use for grayscale images. Default is "gray"

    Returns:
        None
    """
    if isinstance(tensors, list) and tensors:
        num_images = len(tensors)
        if grid:
            fig, axes = plt.subplots(int(num_images ** 0.5), int(num_images ** 0.5), figsize=(10, 10))
            axes = axes.ravel()
            for i in range(num_images):
                image = tensors[i].numpy()
                if image.ndim == 2:
                    image = image
                else:
                    image = image.transpose((1, 2, 0))
                if image.ndim == 2:
                    axes[i].imshow(image, cmap=cmap)
                else:
                    axes[i].imshow(image)
                axes[i].axis("off")
        else:
            fig, axes = plt.subplots(1, num_images, figsize=(10, 10))
            axes = axes.ravel()
            for i in range(num_images):
                image = tensors[i].numpy()
                if image.ndim == 2:
                    image = image[0]
                else:
                    image = image.transpose((1, 2, 0))
                if image.ndim == 2:
                    axes[i].imshow(image, cmap=cmap)
                else:
                    axes[i].imshow(image)
                axes[i].axis("off")
    else:
        raise ValueError("Input is not a list of tensors or empty")

    plt.show()


def tensor_to_numpy(tensor: torch.tensor) -> np.ndarray:
    """Works for both cpu  and gpu tensors"""
    if tensor.is_cuda:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.detach().numpy()


def read_image(img_path: Path, label=False) -> Image:
    if label:
        if img_path == "":
            return Image.fromarray(np.zeros(shape=((512, 512, 3))), 'RGB')  # dummy array
        image = Image.open(img_path).convert("RGB")
    else:
        image = Image.open(img_path)

    return image  # noqa
