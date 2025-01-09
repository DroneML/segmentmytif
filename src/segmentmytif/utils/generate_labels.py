from pathlib import Path

import numpy as np

from segmentmytif.utils.io import read_geotiff, save_tiff


def generate_train_labels(data: np.ndarray) -> np.ndarray:
    train_input = np.zeros((1, *data.shape[1:]), dtype=np.int32) - 1
    train_input[0, 540:, 1090:] = 1  # water
    train_input[0, 678:, 516:560] = 0  # grass
    return train_input

def main():
    input_image_path = Path("tests/test_data/test_image.tif")
    labels_path = Path("tests/test_data/test_image_labels.tif")
    data, profile = read_geotiff(input_image_path)
    save_tiff(generate_train_labels(data), labels_path, profile)


if __name__ == '__main__':
    main()