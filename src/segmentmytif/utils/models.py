from typing import Union

import torch
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_op(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)

        return down, p


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], 1)
        return self.conv(x)

def multiply_to_int(a:Union[int, float], b:Union[int, float]) -> int:
    return int(round(a*b))

class UNet(nn.Module):
    def __init__(self, in_channels, num_classes, model_scale=1):
        super().__init__()
        self.down_convolution_1 = DownSample(in_channels, multiply_to_int(64, model_scale))
        self.down_convolution_2 = DownSample(multiply_to_int(64, model_scale), multiply_to_int(128, model_scale))
        self.down_convolution_3 = DownSample(multiply_to_int(128, model_scale), multiply_to_int(256, model_scale))
        self.down_convolution_4 = DownSample(multiply_to_int(256, model_scale), multiply_to_int(512, model_scale))

        self.bottle_neck = DoubleConv(multiply_to_int(512, model_scale), multiply_to_int(1024, model_scale))

        self.up_convolution_1 = UpSample(multiply_to_int(1024, model_scale), multiply_to_int(512, model_scale))
        self.up_convolution_2 = UpSample(multiply_to_int(512, model_scale), multiply_to_int(256, model_scale))
        self.up_convolution_3 = UpSample(multiply_to_int(256, model_scale), multiply_to_int(128, model_scale))
        self.up_convolution_4 = UpSample(multiply_to_int(128, model_scale), multiply_to_int(64, model_scale))

        self.out = nn.Conv2d(in_channels=multiply_to_int(64, model_scale), out_channels=num_classes, kernel_size=1)



    def forward(self, x):
        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        down_4, p4 = self.down_convolution_4(p3)

        b = self.bottle_neck(p4)

        up_1 = self.up_convolution_1(b, down_4)
        up_2 = self.up_convolution_2(up_1, down_3)
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)


        out = self.out(up_4)
        return out
