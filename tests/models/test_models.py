import json
from pathlib import Path

import torch
from torchinfo import summary

from segmentmytif.utils.models import UNet
from tests.utils import TEST_DATA_FOLDER


class TestUNet:
    def test_initialize_unet_with_valid_parameters(self):
        """Initialize UNet with valid in_channels and num_classes"""
        in_channels = 3
        num_classes = 19

        model = UNet(in_channels, num_classes)

        assert isinstance(model, UNet)
        assert model.out.out_channels == num_classes

    def test_forward_with_minimum_valid_dimensions(self):
        """Handle input tensor with minimum valid dimensions"""
        in_channels = 3
        num_classes = 19
        width = height = 64
        input_tensor = torch.randn(1, in_channels, width, height)  # Minimum size for U-Net to work

        model = UNet(in_channels, num_classes)
        output = model(input_tensor)

        assert output.shape == (1, num_classes, width, height)

    def test_summary(self):
        """Summary of model is exactly as tested."""
        in_channels = 3
        num_classes = 19
        width = height = 64
        input_tensor = torch.randn(1, in_channels, width, height)  # Minimum size for U-Net to work
        expected_sum = load_summary('test_model_summary.json')

        model = UNet(in_channels, num_classes)
        sum = get_summary(model, input_tensor)

        assert sum == expected_sum

    def test_summary_model_scaled_down(self):
        """Summary of model is exactly as tested."""
        model_scale = 0.5
        in_channels = 3
        num_classes = 19
        width = height = 64
        input_tensor = torch.randn(1, in_channels, width, height)  # Minimum size for U-Net to work
        expected_sum = load_summary('test_model_summary_half.json')

        model = UNet(in_channels, num_classes, model_scale=model_scale)
        sum = get_summary(model, input_tensor)

        assert sum == expected_sum


def load_summary(file_name):
    expected_sum = (TEST_DATA_FOLDER / file_name).read_text(encoding='utf-8')
    return normalize_model_summary(expected_sum)


def get_summary(model, input_tensor):
    return normalize_model_summary(summary(model, input_data=input_tensor, verbose=0).__repr__())


def normalize_model_summary(text):
    return (text
            .replace("=", "")
            .replace("(G)", "(Units.GIGABYTES)")
            .replace("(M)", "(Units.MEGABYTES)")
            )
