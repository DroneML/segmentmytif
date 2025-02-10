import shutil
from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import rasterio
import rioxarray
from scipy.spatial.distance import dice

from segmentmytif.features import FeatureType, get_features_path
from segmentmytif.main import read_input_and_labels_and_save_predictions, prepare_training_data
from segmentmytif.utils.io import save_tiff
from .test_cases import TestCase, test_case1210, test_case512
from .utils import TEST_DATA_FOLDER


@pytest.mark.parametrize("test_case, feature_type, model_scale, dice_similarity_threshold, compute_mode",
                         [
                             pytest.param(test_case1210, FeatureType.IDENTITY, None, None, "normal", marks=pytest.mark.slow),
                             pytest.param(test_case1210, FeatureType.FLAIR, 0.125, None, "normal"), # also slow, but necessary to test on each run
                             pytest.param(test_case512, FeatureType.IDENTITY, None, 0.90, "normal", marks=pytest.mark.slow),
                             pytest.param(test_case512, FeatureType.FLAIR, 0.125, 0.84, "normal", marks=pytest.mark.slow),
                             pytest.param(test_case512, FeatureType.FLAIR, 1.0, 0.98, "normal", marks=pytest.mark.slow),
                             pytest.param(test_case1210, FeatureType.IDENTITY, None, None, "parallel", marks=pytest.mark.slow),
                             pytest.param(test_case1210, FeatureType.FLAIR, 0.125, None, "parallel", marks=pytest.mark.slow),
                             pytest.param(test_case1210, FeatureType.IDENTITY, None, None, "safe", marks=pytest.mark.slow),
                             pytest.param(test_case1210, FeatureType.FLAIR, 0.125, None, "safe", marks=pytest.mark.slow),

                         ], ids=lambda e : str(e))
def test_integration(tmpdir, test_case: TestCase, feature_type, model_scale, dice_similarity_threshold, compute_mode):
    input_path = copy_file_and_get_new_path(test_case.image_filename, tmpdir)
    labels_pos_path = copy_file_and_get_new_path(test_case.labels_pos_filename, tmpdir)
    labels_neg_path = copy_file_and_get_new_path(test_case.labels_neg_filename, tmpdir)
    predictions_path = Path(tmpdir) / f"{test_case.image_filename}_predictions_{str(feature_type)}.tif"

    read_input_and_labels_and_save_predictions(
        input_path, labels_pos_path, labels_neg_path, predictions_path, feature_type=feature_type, model_scale=model_scale
    )

    assert predictions_path.exists()

    # Check DICE similarity if threshold is provided
    if dice_similarity_threshold is None and test_case.ground_truth_filename is None:
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
