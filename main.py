import argparse
from pathlib import Path

import rasterio


def read_and_save_geotiff(input_path, output_path):
    with rasterio.open(input_path) as src:
        data = src.read()
        profile = src.profile
        profile.update(compress=None)

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(data)


def parse_args():
    parser = argparse.ArgumentParser(description="Process input and output TIFF files.")

    parser.add_argument('input', type=Path, help='Path to the input TIFF file')
    parser.add_argument('output_tif', type=Path, help='Path to the output TIFF file')

    args = parser.parse_args()

    # Validate arguments
    if not args.input.exists() or not args.input.is_file():
        parser.error(f"The input file {args.input} does not exist or is not a file.")

    return args


if __name__ == '__main__':
    args = parse_args()
    input_path = args.input
    output_path = args.output_tif
    read_and_save_geotiff(input_path, output_path)
