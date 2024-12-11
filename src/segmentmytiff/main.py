import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.ensemble import RandomForestClassifier

from segmentmytiff.features import get_features, FeatureType
from segmentmytiff.utils.io import read_geotiff, save_tiff


def read_input_and_labels_and_save_predictions(input_path: Path, labels_path: Path, output_path: Path,
                                               feature_type=FeatureType.IDENTITY, features_path:Path=None) -> None:
    input_data, profile = read_geotiff(input_path)

    features = get_features(input_data, input_path, feature_type, features_path, profile)

    labels, _ = read_geotiff(labels_path)
    prediction_map = make_predictions(features, labels)

    save_tiff(prediction_map, output_path, profile)


def make_predictions(input_data: ndarray, labels: ndarray) -> ndarray:
    """Makes predictions by training a classifier and using it for inference.

    Expects input data with shape of [channels, width, height] and labels of shape [classes, width, height]
        :param input_data: input data with shape of [channels, width, height]
        :param labels: labels with shape [1, width, height]
    :return: probabilities with shape [class_values, width, height]
    """
    print(labels.shape)
    class1_labels = labels[0]  # Only single class is supported
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


def parse_args():
    parser = argparse.ArgumentParser(description="Process input and output TIFF files.")

    parser.add_argument('-i', '--input', type=Path, help='Path to the input TIFF file')
    parser.add_argument('-l', '--labels', type=Path, help='Path to the training labels TIFF file')
    parser.add_argument('-p', '--predictions', type=Path, help='Path to the predictions output TIFF file')
    parser.add_argument('-f', '--feature_type', type=FeatureType.from_string, choices=list(FeatureType),
                        default=FeatureType.IDENTITY,
                        help='Type of feature being used. "Identity" means the raw input is directly used as features.')

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
    feature_type = args.feature_type

    read_input_and_labels_and_save_predictions(input_path, labels_path, predictions_path, feature_type=feature_type)
