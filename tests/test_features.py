import pytest
import numpy as np
from segmentmytiff.features import extract_features, FeatureType, NUM_FLAIR_CLASSES


class TestExtractFeatures:
    def test_extract_identity_features(self):
        input_data = np.array(get_generated_3band_image())
        result = extract_features(input_data, FeatureType.IDENTITY)
        assert np.array_equal(result, input_data)

    def test_extract_flair_features(self):
        input_data = np.array(get_generated_3band_image())
        result = extract_features(input_data, FeatureType.FLAIR)
        assert np.array_equal(result.shape[1:], [NUM_FLAIR_CLASSES] + list(input_data.shape[1:]))

    def test_extract_features_unsupported_type(self):
        input_data = np.array([[1, 2], [3, 4]])
        with pytest.raises(KeyError):
            extract_features(input_data, "UNSUPPORTED_TYPE")

def get_generated_3band_image():
    return np.random.random(size=[3, 512, 512])