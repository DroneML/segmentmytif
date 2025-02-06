import shutil
from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import rasterio
import rioxarray
from scipy.spatial.distance import dice
import itertools

from segmentmytif.features import FeatureType, get_features_path
from segmentmytif.main import read_input_and_labels_and_save_predictions, prepare_training_data
from segmentmytif.utils.io import save_tiff
from .utils import TEST_DATA_FOLDER

@pytest.mark.parametrize("test_image, test_labels, feature_type, model_scale, dice_similarity_threshold",
                         [
                             pytest.param("test_image.tif", "test_image_labels.tif", FeatureType.IDENTITY, None, None, marks=pytest.mark.slow),
                             pytest.param("test_image.tif", "test_image_labels.tif", FeatureType.FLAIR, 0.125, None), # also slow, but necessary to test on each run
                             pytest.param("test_image_512x512.tif", "test_image_labels_512x512.tif", FeatureType.IDENTITY,None, 0.90, marks=pytest.mark.slow),
                             pytest.param("test_image_512x512.tif", "test_image_labels_512x512.tif", FeatureType.FLAIR,0.125,  0.84, marks=pytest.mark.slow),
                             pytest.param("test_image_512x512.tif", "test_image_labels_512x512.tif",  FeatureType.FLAIR,1.0, 0.98, marks=pytest.mark.slow),
                         ])
def test_integration(tmpdir, test_image, test_labels, feature_type, model_scale, dice_similarity_threshold):
    input_path = copy_file_and_get_new_path(test_image, tmpdir)
    labels_path = copy_file_and_get_new_path(test_labels, tmpdir)
    predictions_path = Path(tmpdir) / f"{test_image}_predictions_{str(feature_type)}.tif"
# Test integration with a cmbination of inputs
test_images_and_labels = [
    ("test_image.tif", "test_image_labels_positive.gpkg", "test_image_labels_negative.gpkg"),
    ("test_image_512x512.tif", "test_image_labels_positive_512x512.gpkg", "test_image_labels_negative_512x512.gpkg"),
]
feature_types = [FeatureType.IDENTITY, FeatureType.FLAIR]
compute_modes = ["normal", "parallel", "safe"]
combinations = list(itertools.product(test_images_and_labels, feature_types, compute_modes))


    read_input_and_labels_and_save_predictions(input_path, labels_path,
                                               predictions_path,
                                               feature_type=feature_type,
                                               model_scale=model_scale)
@pytest.mark.parametrize(
    ("test_images_and_labels", "feature_type", "compute_modes"),
    combinations,
)
def test_integration(tmpdir, test_images_and_labels, feature_type, compute_modes):
    input_path = copy_file_and_get_new_path(test_images_and_labels[0], tmpdir)
    labels_pos_path = copy_file_and_get_new_path(test_images_and_labels[1], tmpdir)
    labels_neg_path = copy_file_and_get_new_path(test_images_and_labels[2], tmpdir)
    predictions_path = Path(tmpdir) / f"{test_images_and_labels[0]}_predictions_{str(feature_type)}.tif"

    read_input_and_labels_and_save_predictions(
        input_path, labels_pos_path, labels_neg_path, predictions_path, feature_type=feature_type, model_scale=0.125
    )

    assert predictions_path.exists()

    # Check DICE similarity if threshold is provided
    if dice_similarity_threshold is None:
        return

    truth = rioxarray.open_rasterio(TEST_DATA_FOLDER / "test_image_512x512_out_ground_truth.tif").astype(np.int16)
    predictions = rioxarray.open_rasterio(predictions_path).astype(np.int16)
    dice_similarity = 1 - dice(truth.data.flatten(), predictions.data.flatten())
    print(f"DICE similarity index: {dice_similarity}")
    assert dice_similarity > dice_similarity_threshold


def copy_file_and_get_new_path(test_image, tmpdir):
    input_path = Path(tmpdir) / test_image
    shutil.copy(TEST_DATA_FOLDER / test_image, input_path)
    return input_path


@pytest.mark.parametrize(
    "input_path, feature_type, expected_path",
    [
        ("input.tiff", FeatureType.FLAIR, "input_FLAIR.tiff"),
        ("../path/to/input.tiff", FeatureType.FLAIR, "../path/to/input_FLAIR.tiff"),
        ("../path/to/input.tiff", FeatureType.IDENTITY, "../path/to/input.tiff"),
    ],
)
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
