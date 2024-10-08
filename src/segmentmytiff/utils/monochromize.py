import argparse
from pathlib import Path

from segmentmytiff.main import read_geotiff, save


def parse_args():
    parser = argparse.ArgumentParser(description="Process input and output TIFF files.")

    parser.add_argument('-i', '--input_folder', type=Path, help='Path to a folder of input tiff files', required=True)
    parser.add_argument('-o', '--output_folder', type=Path, help='Path to the output folder', required=True)

    args = parser.parse_args()

    if not args.input_folder.exists():
        parser.error(f"The input folder {args.input} does not exist.")

    return args


def monochromize(input_file_path: Path, output_folder_path: Path):
    data, profile = read_geotiff(input_file_path)
    for i_channel in range(data.shape[0]):
        output_file_name = f"{input_file_path.stem}_{i_channel}{input_file_path.suffix}"
        channel = data[i_channel:i_channel + 1]
        save(channel, output_folder_path / output_file_name, profile)


def monochromize_folder(input_folder: Path, output_folder: Path):
    output_folder.mkdir(parents=True, exist_ok=True)
    for input_file in input_folder.rglob('*.tif'):
        if input_file.is_file():
            monochromize(input_file, output_folder)


if __name__ == "__main__":
    args = parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder
    monochromize_folder(input_folder, output_folder)
