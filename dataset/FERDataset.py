# _*_coding:utf-8_*_
# Author      : JacquesdeH
# Create Time : 2020/11/30 16:58
# Project Name: MLCourse-FER
# File        : TrainDataset.py
# --------------------------------------------------

import os
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, ToTensor, RandomResizedCrop, Normalize, RandomHorizontalFlip, \
    RandomRotation, ToPILImage
import cv2


class FERDataset(Dataset):
    def __init__(self, images, labels=None, filenames=None, use_da=True, args=None):
        super(FERDataset, self).__init__()
        self.args = args
        self.images = images
        self.labels = labels
        self.filenames = filenames
        self.length = len(images)
        self.trans = Compose([ToTensor(),
                              # Normalize(mean=0.5, std=0.5),
                              ToPILImage(),
                              RandomRotation(45),
                              RandomResizedCrop([48, 48], scale=(0.75, 1.0)),
                              RandomHorizontalFlip(),
                              ToTensor()
                              ]) \
            if use_da else Compose([ToTensor()])
        pass

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image = self.images[index]
        image = np.array(image, dtype=np.uint8)
        image = cv2.resize(image, (224, 224))
        image = self.trans(image)
        if self.labels is not None:
            label = self.labels[index]
            label = torch.tensor([label], dtype=torch.long)
            return image, label
        elif self.filenames is not None:
            filename = self.filenames[index]
            return image, filename
        else:
            return image
