import logging
from enum import Enum
from pathlib import Path

import numpy as np
import torch
from numpy import ndarray
from segmentmytiff.logging_config import log_duration, log_array
from segmentmytiff.utils.io import save_tiff, read_geotiff
from segmentmytiff.utils.models import UNet
from torchinfo import summary

NUM_FLAIR_CLASSES = 19
logger = logging.getLogger(__name__)

class FeatureType(Enum):
    IDENTITY = 1
    FLAIR = 2

    @staticmethod
    def from_string(s):
        try:
            return FeatureType[s]
        except KeyError:
            raise ValueError()

def get_features(input_data: np.ndarray, input_path: Path, feature_type: FeatureType, features_path: Path, profile):
    """

    :param input_data: 'Raw' input data as stored in TIFs by a GIS user. Shape: [n_bands, height, width]
    :param input_path:
    :param feature_type: See FeatureType enum for options.
    :param features_path: Path used for caching features
    :param profile:
    :return:
    """
    if feature_type == FeatureType.IDENTITY:
        return input_data
    else:
        if features_path is None:
            features_path = get_features_path(input_path, feature_type)
        if not features_path.exists():
            logger.info(f"No existing {feature_type.name} found")
            with log_duration(f"Extracting {feature_type.name} features", logger):
                features = extract_features(input_data, feature_type)
            log_array(features,logger,array_name=f"{feature_type.name} features")
            logger.info(f"Saving {feature_type.name} features (shape {features.shape}) to {features_path}")
            save_tiff(features, features_path, profile)
        loaded_features, _ = read_geotiff(features_path)
        logger.info(f"Loading {feature_type.name} features (shape {loaded_features.shape}) to {features_path}")
        return loaded_features


def extract_features(input_data, feature_type):
    extractor = {
        FeatureType.IDENTITY: extract_identity_features,
        FeatureType.FLAIR: extract_flair_features,
    }[feature_type]

    return extractor(input_data)


def extract_identity_features(input_data: ndarray) -> ndarray:
    return input_data


def extract_flair_features(input_data: ndarray) -> ndarray:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=1, num_classes=NUM_FLAIR_CLASSES, model_scale=0.125)
    state = torch.load(Path("models") / "flair_toy_ep10_scale0_125.pth",
                       map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    input_data = torch.from_numpy(input_data[None, 1:2, :, :]).float().to(device)

    output = model(input_data)
    return output.detach().numpy()[0,:,:,:]


def get_features_path(input_path: Path, features_type: FeatureType) -> Path:
    if features_type == FeatureType.IDENTITY:
        return input_path
    return input_path.parent / f"{input_path.stem}_{features_type.name}{input_path.suffix}"
