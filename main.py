import argparse
from pathlib import Path
from typing import Any

import numpy as np
import rasterio


def read_input_and_labels_and_save_predictions(input_path: Path, labels_path: Path, output_path: Path) -> None:
    data, profile = read_geotiff(input_path)

    labels, _ = read_geotiff(labels_path)

    save(labels, output_path, profile)


def generate_train_labels(data: np.ndarray) -> np.ndarray:
    train_input = np.zeros((1, *data.shape[1:]), dtype=np.int32) - 1
    train_input[0, 250:, 450:] = 1  # grass
    train_input[0, 200:220, 0:100] = 0  # non-grass
    return train_input


def read_geotiff(input_path: Path) -> (np.ndarray, Any):
    with rasterio.open(input_path) as src:
        data = src.read()
        profile = src.profile
    return data, profile


def save(data: np.ndarray, output_path: Path, profile) -> None:
    profile.update(count=data.shape[0])  # set number of channels
    profile.update(compress=None)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(data)


def parse_args():
    parser = argparse.ArgumentParser(description="Process input and output TIFF files.")

    parser.add_argument('input', type=Path, help='Path to the input TIFF file')
    parser.add_argument('labels', type=Path, help='Path to the training labels TIFF file')
    parser.add_argument('predictions', type=Path, help='Path to the predictions output TIFF file')

    args = parser.parse_args()

    # Validate arguments
    if not args.input.exists() or not args.input.is_file():
        parser.error(f"The input file {args.input} does not exist or is not a file.")

    return args


if __name__ == '__main__':
    args = parse_args()
    input_path = args.input
    labels_path = args.labels
    predictions_path = args.predictions

    if False:  # Generate some dummy labels
        save(generate_train_labels(data), labels_path)

    read_and_save_geotiff(input_path, labels, predictions_path)
