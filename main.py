import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from numpy import ndarray
from sklearn.ensemble import RandomForestClassifier


def read_input_and_labels_and_save_predictions(input_path: Path, labels_path: Path, output_path: Path) -> None:
    input_data, profile = read_geotiff(input_path)

    labels, _ = read_geotiff(labels_path)
    prediction_map = make_predictions(input_data, labels)

    save(prediction_map, output_path, profile)


def make_predictions(input_data:ndarray, labels:ndarray) -> ndarray:
    """Makes predictions by training a classifier and using it for inference."""
    print(labels.shape)
    class1_labels = labels[0]
    flattened = class1_labels.flatten()
    positive_instances = input_data.reshape((input_data.shape[0], -1))[:, flattened == 1].transpose()
    negative_instances = input_data.reshape((input_data.shape[0], -1))[:, flattened == 0].transpose()
    print('instances:', positive_instances.shape, negative_instances.shape)
    train_data = np.concatenate((positive_instances, negative_instances))
    train_labels = np.concatenate(((flattened[flattened == 1]), (flattened[flattened == 0])))
    print('train data', train_labels.shape, train_data.shape)
    classifier = RandomForestClassifier(n_estimators=100, random_state=0)
    classifier.fit(train_data, train_labels)
    predictions = classifier.predict_proba(input_data.reshape((input_data.shape[0], -1)).transpose())
    print('predictions', predictions.shape, pd.DataFrame(predictions).value_counts())
    prediction_map = predictions.transpose().reshape((predictions.shape[1], *input_data.shape[1:]))
    print('prediction_map shape', prediction_map.shape)
    return prediction_map


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

    parser.add_argument('-i', '--input', type=Path, help='Path to the input TIFF file')
    parser.add_argument('-l', '--labels', type=Path, help='Path to the training labels TIFF file')
    parser.add_argument('-p', '--predictions', type=Path, help='Path to the predictions output TIFF file')

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

    read_input_and_labels_and_save_predictions(input_path, labels_path, predictions_path)
