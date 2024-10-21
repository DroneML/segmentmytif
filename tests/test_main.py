from pathlib import Path

from utils import TEST_DATA_FOLDER
import numpy as np
import pytest
import rasterio

from segmentmytiff.main import read_input_and_labels_and_save_predictions, FeatureType, get_features_path, save_tiff


def test_integration(tmpdir):
    input_path = TEST_DATA_FOLDER / "test_image.tif"
    labels_path = TEST_DATA_FOLDER / "test_image_labels.tif"
    predictions_path = Path(tmpdir) / "test_image_predictions.tif"

    read_input_and_labels_and_save_predictions(input_path, labels_path, predictions_path)

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