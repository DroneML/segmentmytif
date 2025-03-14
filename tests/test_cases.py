from dataclasses import dataclass
from typing import Optional


@dataclass
class TestCase:
    __test__ = False  # This is to prevent pytest from running this class as a test case
    image_filename: str
    labels_pos_filename: str
    labels_neg_filename: str
    ground_truth_filename: Optional[str]

    def __repr__(self):
        return self.image_filename

    def __str__(self):
        return self.image_filename


test_case1210=TestCase(
    image_filename="test_image_1210x718.tif",
    labels_pos_filename="test_image_labels_positive_1210x718.gpkg",
    labels_neg_filename="test_image_labels_negative_1210x718.gpkg",
    ground_truth_filename=None,
)
test_case512=TestCase(
    image_filename="test_image_512x512.tif",
    labels_pos_filename="test_image_labels_positive_512x512.gpkg",
    labels_neg_filename="test_image_labels_negative_512x512.gpkg",
    ground_truth_filename="test_image_512x512_out_ground_truth.tif",
)
