import time
from contextlib import contextmanager
from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import rasterio

from segmentmytif.features import FeatureType, get_features_path
from segmentmytif.main import read_input_and_labels_and_save_predictions, prepare_training_data
from segmentmytif.utils.io import save_tiff
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


@pytest.mark.parametrize("array_type", ["numpy", "dask"])
def test_prepare_training_data(array_type):
    random = np.random.default_rng(0)
    length = 200
    random_data = random.integers(low=0, high=256, size=(5, length, length))
    labels = random.choice([0, 1, 2], size=(1, length, length), replace=True)
    if array_type == "numpy":
        input_data = random_data
    elif array_type == "dask":
        input_data = da.from_array(random_data)

    prepare_training_data(input_data, labels)
