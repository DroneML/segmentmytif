import shutil
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import rasterio
import rioxarray
from scipy.spatial.distance import dice
import geopandas as gpd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import xarray as xr
from segmentmytif.features import FeatureType, get_features_path, get_features
from segmentmytif.main import read_input_and_labels_and_save_predictions, prepare_training_data, make_predictions
from segmentmytif.utils.geospatial import get_label_array
from segmentmytif.utils.io import save_tiff
from .test_cases import TestCase, test_case1210, test_case512
from .utils import TEST_DATA_FOLDER


@pytest.mark.parametrize("test_case, feature_type, model_scale, dice_similarity_threshold, compute_mode",
                         [
                             pytest.param(test_case1210, FeatureType.IDENTITY, None, None, "normal", marks=pytest.mark.slow),
                             pytest.param(test_case1210, FeatureType.FLAIR, 0.125, None, "normal"), # also slow, but necessary to test on each run
                             pytest.param(test_case512, FeatureType.IDENTITY, None, 0.90, "normal", marks=pytest.mark.slow),
                             pytest.param(test_case512, FeatureType.FLAIR, 0.125, 0.82, "normal", marks=pytest.mark.slow),
                             pytest.param(test_case512, FeatureType.FLAIR, 1.0, 0.95, "normal", marks=pytest.mark.slow),
                             pytest.param(test_case1210, FeatureType.IDENTITY, None, None, "parallel", marks=pytest.mark.slow),
                             pytest.param(test_case1210, FeatureType.FLAIR, 0.125, None, "parallel", marks=pytest.mark.slow),
                             pytest.param(test_case1210, FeatureType.IDENTITY, None, None, "safe", marks=pytest.mark.slow),
                             pytest.param(test_case1210, FeatureType.FLAIR, 0.125, None, "safe", marks=pytest.mark.slow),
                         ], ids=lambda e : str(e))
def test_integration(tmpdir, test_case: TestCase, feature_type, model_scale, dice_similarity_threshold, compute_mode):
    tmpdir = Path(tmpdir)

    input_path = copy_file_and_get_new_path(test_case.image_filename, tmpdir)
    labels_pos_path = copy_file_and_get_new_path(test_case.labels_pos_filename, tmpdir)
    labels_neg_path = copy_file_and_get_new_path(test_case.labels_neg_filename, tmpdir)
    predictions_path = Path(tmpdir) / f"{test_case.image_filename}_predictions_{str(feature_type)}_model_{model_scale}.tif"

    read_input_and_labels_and_save_predictions(
        input_path, labels_pos_path, labels_neg_path, predictions_path, feature_type=feature_type, model_scale=model_scale
    )

    assert predictions_path.exists()

    # Check DICE similarity if threshold is provided
    if dice_similarity_threshold is None and test_case.ground_truth_filename is None:
        return

    truth = rioxarray.open_rasterio(TEST_DATA_FOLDER / "test_image_512x512_out_ground_truth.tif").astype(np.int16)
    predictions = rioxarray.open_rasterio(predictions_path).astype(np.float64)
    dice_similarity = 1 - dice(truth.data.flatten() > 0.5, predictions.data.flatten() > 0.999)
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

def test_ml_performance_test(tmpdir):
    use_case = test_case512
    raster = rioxarray.open_rasterio(TEST_DATA_FOLDER / use_case.image_filename)
    features = get_features(
        raster,
        TEST_DATA_FOLDER / use_case.image_filename,
        FeatureType.FLAIR,
        Path(tmpdir) / "features.tif",
    )
    squashed = rioxarray.raster_array((features - features.min(axis=1)) / features.max(axis=1))
    squashed_raster = raster.isel(band=0).drop_vars(["band"]).expand_dims(band=features.shape[0])
    squashed_raster.data = squashed
    normalized = (features - features.mean(axis=1)) / features.std(axis=1)
    normalized_raster = raster.isel(band=0).drop_vars(["band"]).expand_dims(band=features.shape[0])
    normalized_raster.data = normalized

    scaling = StandardScaler()
    scaling.fit(features.data.reshape(features.shape[0], -1))
    standardized = (scaling.transform(features.data.reshape(features.shape[0], -1))).reshape(features.shape)
    standardized_raster = raster.isel(band=0).drop_vars(["band"]).expand_dims(band=features.shape[0])
    standardized_raster.data = standardized

    principal = PCA()
    principal.fit(standardized.reshape(standardized.shape[0], -1))
    pcad = standardized #principal.transform(standardized.reshape(standardized.shape[0], -1)).reshape(standardized.shape)

    for d in [features, squashed_raster, normalized_raster, standardized_raster]:
        print(pd.DataFrame([(np.min(band), np.max(band), np.mean(band), np.std(band)) for band in d],
                     columns=["min", "max", "mean", "std"]))

    scores = [train_predict_score(f, raster, tmpdir, use_case) for f in [features, squashed, normalized, standardized]]
    results = pd.DataFrame(scores, columns=["DICE similarity index"], index=["Original", "Squashed", "Normalized", "Standardized"])
    print(results)


def train_predict_score(features, raster, tmpdir, use_case):
    pos_gdf = gpd.read_file(TEST_DATA_FOLDER / use_case.labels_pos_filename).to_crs(raster.rio.crs)
    neg_gdf = gpd.read_file(TEST_DATA_FOLDER / use_case.labels_neg_filename).to_crs(raster.rio.crs)
    predictions_path = Path(tmpdir) / "predictions.tif"
    # Get label arrays
    labels = get_label_array(features, pos_gdf, neg_gdf, compute_mode="normal")
    # Make predictions
    prediction_map = make_predictions(features.data, labels.data)
    # Use raster as the template and assign data
    prediction_raster = raster.isel(band=0).drop_vars(["band"]).expand_dims(band=prediction_map.shape[0])
    prediction_raster.data = prediction_map
    # Save predictions
    prediction_raster.rio.to_raster(predictions_path)
    truth = rioxarray.open_rasterio(TEST_DATA_FOLDER / use_case.ground_truth_filename).astype(np.int16)
    predictions = rioxarray.open_rasterio(predictions_path).astype(np.int16)
    dice_similarity = 1 - dice(truth.data.flatten(), predictions.data.flatten())
    return dice_similarity
