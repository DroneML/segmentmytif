import pytest
import numpy as np
from segmentmytif.features import (
    get_features,
    extract_features,
    FeatureType,
    NUM_FLAIR_CLASSES,
    get_flair_model_file_name,
    load_model,
)


class TestExtractFeatures:
    def test_get_identity_features(self):
        input_data = np.array(get_generated_multiband_image())
        result = get_features(input_data, None, FeatureType.IDENTITY, None)
        assert np.array_equal(result.data, input_data.data)

    @pytest.mark.parametrize(
        ["n_bands", "width", "height"],
        [
            (3, 1, 1),  # too small to be processed by the model, requires padding
            (3, 8, 8),  # too small to be processed by the model, requires padding
            (3, 16, 16),  # smallest size that can natively be processed by the model
            (3, 61, 39),  # not divisible by 16 so requires padding in both directions
            (3, 64, 48),  # smallest dimensions, > line above, that don't require padding
            (1, 512, 512),  # size of the model's training data (easiest case)
            (3, 1210, 718),  # not divisible by 16 so requires padding in both directions
        ],
    )
    def test_extract_flair_features(self, n_bands, width, height):
        input_data = np.array(get_generated_multiband_image(n_bands=n_bands, width=width, height=height))
        result = extract_features(input_data, FeatureType.FLAIR, model_scale=0.125)
        assert np.array_equal(result.shape, [n_bands * NUM_FLAIR_CLASSES] + list(input_data.shape[1:]))

    def test_extract_features_unsupported_type(self):
        input_data = np.array([[1, 2], [3, 4]])
        with pytest.raises(KeyError):
            extract_features(input_data, "UNSUPPORTED_TYPE")


def get_generated_multiband_image(n_bands=3, width=512, height=512):
    return np.random.random(size=[n_bands, width, height])


@pytest.mark.parametrize(
    ["model_scale", "file_name"],
    [
        (1.0, "flair_toy_ep10_scale1_0.pth"),
        (0.5, "flair_toy_ep10_scale0_5.pth"),
        (0.25, "flair_toy_ep10_scale0_25.pth"),
        (0.125, "flair_toy_ep10_scale0_125.pth"),
    ],
)
def test_get_flair_model_file_name_with_valid_scales(model_scale, file_name):
    assert get_flair_model_file_name(model_scale) == file_name


@pytest.mark.parametrize(["model_scale"], [(2,), (0.001,)])
def test_get_flair_model_file_name_with_invalid_scales(model_scale):
    with pytest.raises(ValueError):
        get_flair_model_file_name(model_scale)


@pytest.mark.downloader
@pytest.mark.parametrize(["model_scale"], [(0.125,), (0.25,), (0.5,), (1.0,)])
def test_load_model_downloads_model_from_hugging_face(model_scale, tmpdir):
    load_model(model_scale, models_dir=tmpdir)
