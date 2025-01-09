import argparse
from pathlib import Path

from tqdm import tqdm

from segmentmytif.utils.io import read_geotiff, save_tiff


def parse_args():
    parser = argparse.ArgumentParser(description="Process input and output TIFF files.")

    parser.add_argument('-i', '--input_folder', type=Path, help='Path to a folder of input tiff files', required=True)
    parser.add_argument('-o', '--output_folder', type=Path, help='Path to the output folder', required=True)

    args = parser.parse_args()

    if not args.input_folder.exists():
        parser.error(f"The input folder {args.input} does not exist.")

    return args


def monochromize_image(input_file_path: Path, output_folder_path: Path):
    data, profile = read_geotiff(input_file_path)
    profile['count'] = 1  # Number of bands
    for i_channel in range(data.shape[0]):
        output_file_name = f"{input_file_path.stem}_{i_channel}{input_file_path.suffix}"
        channel = data[i_channel:i_channel + 1]
        save_tiff(channel, output_folder_path / output_file_name, profile)


def monochromize_folder(input_folder: Path, output_folder: Path):
    output_folder.mkdir(parents=True, exist_ok=True)
    for input_file in tqdm(list(input_folder.rglob('*.tif')), desc="Processing images"):
        relative_path = input_file.relative_to(input_folder)
        if input_file.is_file():
            path = output_folder / relative_path.parent
            path.mkdir(parents=True, exist_ok=True)
            monochromize_image(input_file, path)


if __name__ == "__main__":
    args = parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder
    monochromize_folder(input_folder, output_folder)
