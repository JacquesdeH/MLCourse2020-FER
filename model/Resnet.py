# _*_coding:utf-8_*_
# Author      : JacquesdeH
# Create Time : 2020/12/4 13:47
# Project Name: MLCourse-FER
# File        : Resnet.py
# --------------------------------------------------

import torch
import torch.nn as nn
from torchvision.models import resnet50, resnet101, resnet34, resnet152

class Resnet(nn.Module):
    def __init__(self, use_pretrained=False, num_classes=2, norm_layer=None, resnet_depth=50, dropout=0.5):
        super(Resnet, self).__init__()
        self.use_pretrained = use_pretrained
        self.num_classes = num_classes
        self.resnet_fn = resnet152 if resnet_depth == 152 \
            else resnet101 if resnet_depth == 101 \
            else resnet50 if resnet_depth == 50 \
            else resnet34 if resnet_depth == 34 \
            else None
        self.base = self.resnet_fn(pretrained=use_pretrained, norm_layer=norm_layer)
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features=1000, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=4096, out_features=1024),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=1024, out_features=num_classes)
        )

    def forward(self, x):
        base_ = self.base(x)
        reshape_ = base_
        fc_ = self.fc(reshape_)
        return fc_

    def baseParameters(self):
        return self.base.parameters()

    def finetuneParameters(self):
        return self.fc.parameters()

