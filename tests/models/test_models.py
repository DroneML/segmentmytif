import json
from pathlib import Path

import torch
from torchinfo import summary

from segmentmytiff.utils.models import UNet
from utils import TEST_DATA_FOLDER


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
        expected_sum = (TEST_DATA_FOLDER / 'test_model_summary.json').read_text(encoding='utf-8')

        model = UNet(in_channels, num_classes)
        sum = summary(model, input_data=input_tensor).__repr__()

        assert sum == expected_sum
