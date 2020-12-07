# _*_coding:utf-8_*_
# Author      : JacquesdeH
# Create Time : 2020/11/29 19:40
# Project Name: MLCourse-FER
# File        : Encoder.py
# --------------------------------------------------

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, add_noise=None):
        super(Encoder, self).__init__()
        self.add_noise = add_noise
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )
        # conv1_ -> (batch, 32, 25, 25)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )
        # conv2_ -> (batch, 64, 13, 13)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )
        # conv3_ -> (batch, 128, 7, 7)

    def forward(self, x: torch.Tensor):
        if self.add_noise is not None:
            noise = torch.randn(x.shape) * self.add_noise
            noise = noise.to(x.device)
            x += noise
        conv1_ = self.conv1(x)
        conv2_ = self.conv2(conv1_)
        conv3_ = self.conv3(conv2_)
        ret = conv3_.reshape(-1, 128 * 7 * 7)
        return ret


if __name__ == '__main__':
    encoder = Encoder()

    input_ = torch.randn([16, 1, 48, 48])
    output_ = encoder(input_)
