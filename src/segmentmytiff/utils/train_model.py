"""Based on, and parts copied from https://medium.com/@fernandopalominocobo/mastering-u-net-a-step-by-step-guide-to-segmentation-from-scratch-with-pytorch-6a17c5916114"""
import argparse
import time
from pathlib import Path

import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

try:
    import mlflow
    mlflow_installed = True
except ImportError:
    mlflow_installed = False

from segmentmytiff.utils.datasets import MonochromeFlairDataset
from segmentmytiff.utils.models import UNet
from segmentmytiff.utils.performance_metrics import dice_coefficient


def main(root_path, use_mlflow=True, train_set_limit=None, epochs=None):
    if use_mlflow and not mlflow_installed:
        raise Exception("Please install mlflow first or specify to run without mlflow.")

    if use_mlflow:
        mlflow.set_tracking_uri(uri="http://127.0.0.1:80")
        mlflow.set_experiment("MLflow Quickstart")

    if use_mlflow:
        mlflow.start_run()
        mlflow.set_tag("Training Info", "Segmentation model for ortho photos.")
    train_val_dataset = MonochromeFlairDataset(root_path, limit=train_set_limit)
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

    if use_mlflow:
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

    _test_dataloader = DataLoader(dataset=test_dataset,
                                 num_workers=num_workers, pin_memory=False,
                                 batch_size=BATCH_SIZE,
                                 shuffle=True)

    model = UNet(in_channels=1, num_classes=19).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    torch.cuda.empty_cache()

    train(model, train_dataloader, validation_dataloader, criterion, optimizer, device, epochs)

    if use_mlflow:
        mlflow.end_run()


def train(model, train_dataloader, validation_dataloader, criterion, optimizer, device, epochs):
    for epoch in tqdm(range(epochs)):
        train_loss, train_dice = train_one_step(model, train_dataloader, optimizer, criterion, device)
        val_loss, val_dice = validate(model, validation_dataloader, criterion, device)

        print("-" * 30)
        print(f"Training Loss EPOCH {epoch}: {train_loss:.4f}")
        print(f"Training DICE EPOCH {epoch}: {train_dice:.4f}")
        print("\n")
        print(f"Validation Loss EPOCH {epoch}: {val_loss:.4f}")
        print(f"Validation DICE EPOCH {epoch}: {val_dice:.4f}")
        print("-" * 30)

        metrics = {
            "train_loss": train_loss,
            "train_dc": train_dice,
            "val_loss": val_loss,
            "val_dc": val_dice,
        }

        mlflow.log_metrics(metrics, step=epoch, timestamp=int(round(time.time())))
    # Saving the model
    torch.save(model.state_dict(), 'my_checkpoint.pth')


def validate(model, validation_dataloader, criterion, device):
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
    return val_dc, val_loss


def train_one_step(model, train_dataloader, optimizer, criterion, device):
    model.train()
    train_running_loss = 0
    train_running_dice = 0
    for idx, img_mask in enumerate(tqdm(train_dataloader, position=0, leave=True)):
        img = img_mask[0].float().to(device)
        mask = img_mask[1].float().to(device)

        y_pred = model(img)
        optimizer.zero_grad()

        dice = dice_coefficient(y_pred, mask)
        loss = criterion(y_pred, mask)

        train_running_loss += loss.item()
        train_running_dice += dice.item()

        loss.backward()
        optimizer.step()
    train_loss = train_running_loss / (idx + 1)
    train_dice = train_running_dice / (idx + 1)
    return train_loss, train_dice,


def parse_args():
    parser = argparse.ArgumentParser(description="Train a semantic segmentation model given a dataset of TIF images.")
    parser.add_argument('-r', '--root', type=Path, required=True, help='Root to the dataset')
    parser.add_argument('--no_mlflow', action='store_true', help='Flag for enabling or disabling MLflow')
    parser.add_argument('--train_set_limit', type=int, default=None, help='Limit for the size of the train set (default: no limit)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train (default: 10)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    root_path = args.root
    main(root_path, use_mlflow=not args.no_mlflow, train_set_limit=args.train_set_limit, epochs=args.epochs)
