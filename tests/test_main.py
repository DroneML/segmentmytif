import logging
import time
from contextlib import contextmanager
from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import rasterio

from segmentmytiff.features import FeatureType, get_features_path
from segmentmytiff.logging_config import log_array
from segmentmytiff.main import read_input_and_labels_and_save_predictions, prepare_numpy_style, prepare_dask_style, \
    prepare_numpy_style_with_subsampling
from segmentmytiff.utils.io import save_tiff
from .utils import TEST_DATA_FOLDER


@pytest.mark.parametrize("test_image, test_labels, feature_type",
                         [
                             ("test_image.tif", "test_image_labels.tif", FeatureType.IDENTITY),
                             pytest.param("test_image.tif", "test_image_labels.tif", FeatureType.FLAIR,
                                          marks=pytest.mark.xfail(reason="model can only handle 512x512")),
                             ("test_image_512x512.tif", "test_image_labels_512x512.tif", FeatureType.IDENTITY),
                             ("test_image_512x512.tif", "test_image_labels_512x512.tif", FeatureType.FLAIR),
                         ])
def test_integration(tmpdir, test_image, test_labels, feature_type):
    input_path = TEST_DATA_FOLDER / test_image
    labels_path = TEST_DATA_FOLDER / test_labels
    predictions_path = Path(tmpdir) / f"{test_image}_predictions_{str(feature_type)}.tif"

    read_input_and_labels_and_save_predictions(input_path, labels_path, predictions_path, feature_type=feature_type,
                                               model_scale=0.125)  # scale down feature-extraction-model for testing

    assert predictions_path.exists()


@pytest.mark.parametrize("input_path, feature_type, expected_path", [
    ("input.tiff", FeatureType.FLAIR, "input_FLAIR.tiff"),
    ("../path/to/input.tiff", FeatureType.FLAIR, "../path/to/input_FLAIR.tiff"),
    ("../path/to/input.tiff", FeatureType.IDENTITY, "../path/to/input.tiff"),
])
def test_get_features_path(input_path, feature_type, expected_path):
    features_path = get_features_path(Path(input_path), feature_type)
    assert features_path == Path(expected_path)


def test_save(tmpdir):
    predictions_path = Path(tmpdir) / "test_image_predictions.tif"
    width = 100
    data = np.zeros((3, width, width))
    profile = {"width": width, "height": width, "dtype": rasterio.uint8}

    save_tiff(data, predictions_path, profile=profile)

    assert predictions_path.exists()


def test_golden_data_preparation():
    # prepare_training_data(input_data, labels)
    random = np.random.default_rng(0)
    length = 5000
    input_data = random.integers(low=0, high=256, size=(5, length, length))
    labels = random.choice([0,1,2], size=(1, length, length), replace=True)
    with time_code("numpy"):
        numpy_output = prepare_numpy_style(input_data, labels)
    with time_code("npsub"):
        _numpy_output = prepare_numpy_style_with_subsampling(input_data, labels)
    with time_code("dask "):
        dask_output = prepare_dask_style(input_data, labels)
    with time_code("daskd"):
        daskd_output = prepare_dask_style(da.from_array(input_data), da.from_array(labels))


    print_results(numpy_output)
    print_results(dask_output)
    print_results(daskd_output)

    for n in zip(numpy_output, dask_output):
        assert n == n
    for n in zip(dask_output, daskd_output):
        assert n == n


def print_results(numpy_output):
    n_labeled, n_negative, n_positive, n_unlabeled, train_data, train_labels = numpy_output
    print(
        f"{n_labeled} ({round(100 * n_labeled / (n_labeled + n_unlabeled), 2)}%) labeled instances  of a total of {n_labeled + n_unlabeled} instances.")
    print(
        f"Training on {n_positive} ({round(100 * n_positive / n_labeled, 2)}%) positive labels and {n_negative} ({round(100 * n_negative / n_labeled, 2)}%) negative labels ")
    print(f"train_data shape: {train_data.shape}")
    print(f"train_labels shape: {train_labels.shape}")


@contextmanager
def time_code(task_name: str):
    start_time = time.perf_counter()
    yield
    duration = time.perf_counter() - start_time
    print(f"{task_name} finished in {duration:.4f} seconds")