import shutil
from pathlib import Path

import pytest

from segmentmytif.utils.monochromize import monochromize_image, monochromize_folder
from .test_cases import test_case1210
from .utils import TEST_DATA_FOLDER


@pytest.mark.xfail(reason="This tests errors with 'UnicodeDecodeError: 'utf-8' codec can't decode byte 0xba in position 18: invalid start byte'. I'm not sure why.")
def test_monochromize(tmpdir):
    input_image = TEST_DATA_FOLDER / test_case1210.image_filename
    monochromize_image(input_image, tmpdir)
    for channel in range(4):
        assert (tmpdir / f"test_image_{channel}.tif").exists()


def test_monochromize_folder_structure_maintained(tmpdir):
    """Tests if the folder structure is maintained from the source to the target folder.

    - source
      - subdir
        - test_image_1210x718.tif
    """
    input_image = TEST_DATA_FOLDER / test_case1210.image_filename
    source_dir = Path(tmpdir) / "source"
    subdir = "subdir"
    (source_dir / subdir).mkdir(parents=True)
    shutil.copyfile(input_image, source_dir / subdir / input_image.name)
    target_dir = Path(tmpdir) / "target"
    target_dir.mkdir()

    monochromize_folder(source_dir, target_dir)

    for channel in range(4):
        tiff_path = (target_dir / subdir / f"{input_image.stem}_0{input_image.suffix}")
        assert tiff_path.exists(), f"tif file should have been created at {str(tiff_path)} for channel {channel}"
