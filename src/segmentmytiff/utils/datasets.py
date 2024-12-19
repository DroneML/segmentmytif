import logging
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import torch


class MonochromeFlairDataset(Dataset):
    def __init__(self, root_path, limit=None, split="train"):
        self.root_path = root_path
        self.limit = limit
        self.images = sorted([str(p) for p in (Path(root_path) / split / "input").glob("*.tif")])[:self.limit]

        def image_path_to_mask_path(image_path: Path) -> Path:
            return image_path.parent.parent / "labels" / f"MSK{image_path.stem[3:-2]}_0{image_path.suffix}"  # -2 for "_b" where b is band#

        self.masks = [str(image_path_to_mask_path(Path(p))) for p in self.images][:self.limit]
        non_existing_masks = [p for p in self.masks if Path(p).exists() == False]
        if non_existing_masks:
            logging.getLogger().warning(f"{len(non_existing_masks)} of a total of {len(self.masks)} masks not found.")

        if self.limit is None:
            self.limit = len(self.images)

    def __getitem__(self, index):
        img = transforms.ToTensor()(Image.open(self.images[index]).convert("L"))
        mask = load_and_one_hot_encode(self.masks[index])
        return img, mask

    def __len__(self):
        return min(len(self.images), self.limit)


def load_and_one_hot_encode(image_path, num_classes=19):
    """
    Loads a greyscale (labels) image from the specified path and performs one-hot encoding.

    Args:
        image_path (str): The file path to the image.
        num_classes (int, optional): The number of classes for one-hot encoding. Default is 20.

    Returns:
        torch.Tensor: A one-hot encoded tensor with shape (num_classes, height, width).
    """
    image = Image.open(image_path).convert("L")  # Load as grayscale

    image_array = np.array(image, dtype=np.int64) -1  # -1 to convert to zero-based
    image_tensor = torch.from_numpy(image_array)

    one_hot = torch.nn.functional.one_hot(image_tensor, num_classes=num_classes)

    return one_hot.permute(2, 0, 1).float()
