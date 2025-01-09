import argparse
from pathlib import Path

import numpy as np
import dask.array as da
from numpy import ndarray
from sklearn.ensemble import RandomForestClassifier

from segmentmytif.features import get_features, FeatureType
from segmentmytif.logging_config import setup_logger, log_duration, log_array
from segmentmytif.utils.io import read_geotiff, save_tiff

logger = setup_logger(__name__)


def read_input_and_labels_and_save_predictions(input_path: Path, labels_path: Path, output_path: Path,
                                               feature_type=FeatureType.IDENTITY, features_path: Path = None,
                                               **extractor_kwargs) -> None:
    logger.info("read_input_and_labels_and_save_predictions called with the following arguments:")
    for k, v in locals().items():
        logger.info(f"{k}: {v}")

    input_data, profile = read_geotiff(input_path)

    features = get_features(input_data, input_path, feature_type, features_path, profile, **extractor_kwargs)

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
    with log_duration("Prepare train data", logger):
        train_data, train_labels = prepare_training_data(input_data, labels)

    classifier = RandomForestClassifier(n_estimators=100, random_state=0)

    with log_duration("Train model", logger):
        classifier.fit(train_data, train_labels)

    with log_duration("Make predictions", logger):
        predictions = classifier.predict_proba(input_data.reshape((input_data.shape[0], -1)).transpose())
        prediction_map = predictions.transpose().reshape((predictions.shape[1], *input_data.shape[1:]))
        log_array(prediction_map, logger, array_name="Predictions")

    return prediction_map


def prepare_training_data(input_data, labels):
    """
        Prepares training data for a binary classification task.

        Parameters:
        - input_data: A 3D array-like object (e.g., NumPy or Dask array) where the first dimension represents
          instances, and the last two dimensions represent spatial data.
        - labels: A 3D array-like object (e.g., NumPy or Dask array) where the first dimension is the class index
          (only single-class supported), and the last two dimensions represent spatial labels.

        Process:
        1. Flattens the label array for the first class to a 1D array.
        2. Separates the positive and negative instances from the input data based on the labels.
        3. Computes the number of labeled and unlabeled instances and logs the statistics.
        4. Concatenates positive and negative instances into training data and corresponding labels.

        Returns:
        - train_data: A 2D array where each row is a training instance and each column is a feature.
        - train_labels: A 1D array containing the labels corresponding to the rows in `train_data`.
        """
    # Reshape input data to [n_instances, n_features]
    class1_labels = labels[0]  # Only single class is supported
    flattened = class1_labels.flatten()
    positive_instances = input_data.reshape((input_data.shape[0], -1))[:, flattened == 1].transpose()
    negative_instances = input_data.reshape((input_data.shape[0], -1))[:, flattened == 0].transpose()
    n_labeled = flattened.shape[0]
    n_unlabeled = np.prod(labels.shape[-2:]) - n_labeled
    logger.info(
        f"Dataset contains {n_labeled} ({round(100 * n_labeled / (n_labeled + n_unlabeled), 2)}%) labeled instances of a total of {n_labeled + n_unlabeled} instances.")
    # Subsample training data
    sampled_positive_instances = subsample(positive_instances, 10000)
    sampled_negative_instances = subsample(negative_instances, 10000)
    n_sampled_positive = sampled_positive_instances.shape[0]
    n_sampled_negative = sampled_negative_instances.shape[0]
    n_sampled_labeled = n_sampled_negative + n_sampled_positive
    logger.info(
        f"Training on {n_sampled_positive} ({round(100 * n_sampled_positive / n_sampled_labeled, 2)}%) positive labels and {n_sampled_negative} ({round(100 * n_sampled_negative / n_sampled_labeled, 2)}%) negative labels ")
    # Shuffle training data
    total_sample_size = sampled_positive_instances.shape[0] + sampled_negative_instances.shape[0]
    order = np.arange(total_sample_size)
    np.random.shuffle(order)
    train_data = np.concatenate((sampled_positive_instances, sampled_negative_instances))[order]
    train_labels = np.concatenate(((np.ones(shape=[sampled_positive_instances.shape[0]])), (np.zeros(shape=[sampled_negative_instances.shape[0]]))))[
        order]
    log_array(train_labels, logger, array_name="Train labels")
    log_array(train_data, logger, array_name="Train data")

    return train_data, train_labels


def subsample(instances, sample_size):
    if isinstance(instances, da.Array):
        instances.compute_chunk_sizes()
    n_instances = instances.shape[0]
    indices = np.arange(n_instances)
    sample_indices = np.random.choice(indices, size=min(sample_size, n_instances), replace=False)
    return instances[sample_indices]


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
