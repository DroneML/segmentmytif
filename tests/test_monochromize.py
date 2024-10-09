from pathlib import Path

from segmentmytiff.utils.monochromize import monochromize


def test_monochromize(tmpdir):
    input_image = Path("test_data/test_image.tif")
    monochromize(input_image, tmpdir)
    for channel in range(3):
        assert (tmpdir / f"test_image_{channel}.tif").exists()

