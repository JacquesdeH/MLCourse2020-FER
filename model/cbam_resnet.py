# _*_coding:utf-8_*_
# Author      : JacquesdeH
# Create Time : 2020/12/05 0:32
# Project Name: MLCourse-FER
# File        : cbam_resnet.py
# --------------------------------------------------

import torch.nn as nn
from pytorchcv.model_provider import get_model as ptcv_get_model

def cbam_resnet50(in_channels, num_classes, pretrained=True):
    model = ptcv_get_model("cbam_resnet50", pretrained=True)
    model.output = nn.Linear(2048, 7)
    return model

