from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


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
            print(f"{len(non_existing_masks)} of a total of {len(self.masks)} masks not found.")

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()])

        if self.limit is None:
            self.limit = len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("L")
        mask = Image.open(self.masks[index]).convert("L")

        return self.transform(img), self.transform(mask)

    def __len__(self):
        return min(len(self.images), self.limit)
