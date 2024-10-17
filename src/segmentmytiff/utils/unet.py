"""Large parts copied from https://medium.com/@fernandopalominocobo/mastering-u-net-a-step-by-step-guide-to-segmentation-from-scratch-with-pytorch-6a17c5916114"""
import argparse
import time
from pathlib import Path

import torch
from PIL import Image
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from tqdm import tqdm
import mlflow


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_op(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)

        return down, p


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], 1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.down_convolution_1 = DownSample(in_channels, 64)
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)
        self.down_convolution_4 = DownSample(256, 512)

        self.bottle_neck = DoubleConv(512, 1024)

        self.up_convolution_1 = UpSample(1024, 512)
        self.up_convolution_2 = UpSample(512, 256)
        self.up_convolution_3 = UpSample(256, 128)
        self.up_convolution_4 = UpSample(128, 64)

        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        down_4, p4 = self.down_convolution_4(p3)

        b = self.bottle_neck(p4)

        up_1 = self.up_convolution_1(b, down_4)
        up_2 = self.up_convolution_2(up_1, down_3)
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)

        out = self.out(up_4)
        return out


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


def parse_args():
    parser = argparse.ArgumentParser(description="Process input and output TIFF files.")
    parser.add_argument('-r', '--root', type=Path, required=True, help='Root to the dataset')
    args = parser.parse_args()
    return args


def dice_coefficient(prediction, target, epsilon=1e-07):
    prediction_copy = prediction.clone()

    prediction_copy[prediction_copy < 0] = 0
    prediction_copy[prediction_copy > 0] = 1

    intersection = abs(torch.sum(prediction_copy * target))
    union = abs(torch.sum(prediction_copy) + torch.sum(target))
    dice = (2. * intersection + epsilon) / (union + epsilon)

    return dice


def main(root_path):
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

    mlflow.set_experiment("MLflow Quickstart")
    with mlflow.start_run():
        mlflow.set_tag("Training Info", "Segmentation model for ortho photos.")
        train_val_dataset = MonochromeFlairDataset(root_path, limit=5)
        test_dataset = MonochromeFlairDataset(root_path, split="test")
        generator = torch.Generator().manual_seed(0)
        train_dataset, validation_dataset = random_split(train_val_dataset, [0.8, 0.2], generator=generator)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cuda":
            num_workers = torch.cuda.device_count() * 4
        else:
            num_workers = 1

        LEARNING_RATE = 3e-4
        BATCH_SIZE = 8

        mlflow.log_param("LEARNING_RATE", LEARNING_RATE)
        mlflow.log_param("BATCH_SIZE", BATCH_SIZE)
        mlflow.log_param("device", device)
        mlflow.log_param("num_workers", num_workers)

        train_dataloader = DataLoader(dataset=train_dataset,
                                      num_workers=num_workers, pin_memory=False,
                                      batch_size=BATCH_SIZE,
                                      shuffle=True)
        validation_dataloader = DataLoader(dataset=validation_dataset,
                                           num_workers=num_workers, pin_memory=False,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)

        test_dataloader = DataLoader(dataset=test_dataset,
                                     num_workers=num_workers, pin_memory=False,
                                     batch_size=BATCH_SIZE,
                                     shuffle=True)

        model = UNet(in_channels=1, num_classes=1).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.BCEWithLogitsLoss()

        torch.cuda.empty_cache()

        train(model, train_dataloader, validation_dataloader, criterion, optimizer, device)
    mlflow.end_run()


def train(model, train_dataloader, validation_dataloader, criterion, optimizer, device):
    epochs = 10
    train_losses = []
    train_dcs = []
    val_losses = []
    val_dcs = []
    for epoch in tqdm(range(epochs)):
        model.train()
        train_running_loss = 0
        train_running_dc = 0

        for idx, img_mask in enumerate(tqdm(train_dataloader, position=0, leave=True)):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)

            y_pred = model(img)
            optimizer.zero_grad()

            dc = dice_coefficient(y_pred, mask)
            loss = criterion(y_pred, mask)

            train_running_loss += loss.item()
            train_running_dc += dc.item()

            loss.backward()
            optimizer.step()

        train_loss = train_running_loss / (idx + 1)
        train_dc = train_running_dc / (idx + 1)

        train_losses.append(train_loss)
        train_dcs.append(train_dc)

        model.eval()
        val_running_loss = 0
        val_running_dc = 0

        with torch.no_grad():
            for idx, img_mask in enumerate(tqdm(validation_dataloader, position=0, leave=True)):
                img = img_mask[0].float().to(device)
                mask = img_mask[1].float().to(device)

                y_pred = model(img)
                loss = criterion(y_pred, mask)
                dc = dice_coefficient(y_pred, mask)

                val_running_loss += loss.item()
                val_running_dc += dc.item()

            val_loss = val_running_loss / (idx + 1)
            val_dc = val_running_dc / (idx + 1)

        val_losses.append(val_loss)
        val_dcs.append(val_dc)

        print("-" * 30)
        print(f"Training Loss EPOCH {epoch}: {train_loss:.4f}")
        print(f"Training DICE EPOCH {epoch}: {train_dc:.4f}")
        print("\n")
        print(f"Validation Loss EPOCH {epoch}: {val_loss:.4f}")
        print(f"Validation DICE EPOCH {epoch}: {val_dc:.4f}")
        print("-" * 30)

        metrics = {
            "train_loss": train_loss,
            "train_dc": train_dc,
            "val_loss": val_loss,
            "val_dc": val_dc,
        }

        mlflow.log_metrics(metrics, step=epoch, timestamp=int(round(time.time())))
    # Saving the model
    torch.save(model.state_dict(), 'my_checkpoint.pth')


if __name__ == '__main__':
    args = parse_args()
    root_path = args.root
    main(root_path)
