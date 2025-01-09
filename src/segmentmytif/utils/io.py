import logging
from pathlib import Path
from typing import Any, Union

import numpy as np
import rasterio

from segmentmytif.logging_config import log_array

logger = logging.getLogger(__name__)


def read_geotiff(input_path: Path) -> (np.ndarray, Any):
    with rasterio.open(input_path) as src:
        data = src.read()
        profile = src.profile
    log_array(data, logger, array_name=input_path)
    return data, profile


def save_tiff(data: np.ndarray, output_path: Union[Path, str], profile) -> None:
    profile.update(count=data.shape[0])  # set number of channels
    profile.update(compress=None)
    with rasterio.open(str(output_path), 'w', **profile) as dst:
        dst.write(data)
