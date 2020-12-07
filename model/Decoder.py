# _*_coding:utf-8_*_
# Author      : JacquesdeH
# Create Time : 2020/11/29 19:40
# Project Name: MLCourse-FER
# File        : Decoder.py
# --------------------------------------------------

import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, upsample_mode='nearest'):
        super(Decoder, self).__init__()
        self.upsample_mode = upsample_mode
        self.align_corners = True if upsample_mode in ['linear', 'bilinear', 'bicubic', 'trilinear'] else None
        self.deconv1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )
        # deconv1_ -> (batch, 64, 13, 13)
        self.deconv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU()
        )
        # deconv2_ -> (batch, 32, 25, 25)
        self.deconv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(num_features=1),
            nn.Sigmoid()
        )
        # deconv3_ -> (batch, 1, 48, 48)

    def forward(self, x: torch.FloatTensor):
        x = x.reshape(-1, 128, 7, 7)
        upsampled1_ = torch.nn.functional.interpolate(x, scale_factor=2, mode=self.upsample_mode, align_corners=self.align_corners)
        deconv1_ = self.deconv1(upsampled1_)
        upsampled2_ = torch.nn.functional.interpolate(deconv1_, scale_factor=2, mode=self.upsample_mode, align_corners=self.align_corners)
        deconv2_ = self.deconv2(upsampled2_)
        upsampled3_ = torch.nn.functional.interpolate(deconv2_, scale_factor=2, mode=self.upsample_mode, align_corners=self.align_corners)
        deconv3_ = self.deconv3(upsampled3_)
        ret = deconv3_
        return ret


if __name__ == '__main__':
    decoder = Decoder()
    embed_ = torch.randn([3, 64*7*7])
    output_ = decoder(embed_)
