# _*_coding:utf-8_*_
# Author      : JacquesdeH
# Create Time : 2020/12/05 0:32
# Project Name: MLCourse-FER
# File        : run_cbam_resnet.py
# --------------------------------------------------

import os
import json
import random
import imgaug
import torch
import torch.multiprocessing as mp
import numpy as np
import cv2
import pandas as pd
from matplotlib.pyplot import imread
from torch.utils.data import DataLoader

from trainers.trainer import FER2013Trainer
from dataset.fer2013dataset import fer2013
from model.cbam_resnet import cbam_resnet50
from dataset.FERDataset import FERDataset


def run_cbam_resnet(config_path):
    # load configs and set random seed
    configs = json.load(open(config_path))
    configs["cwd"] = os.getcwd()
    # load model and data_loader
    model = cbam_resnet50
    train_set, val_set, test_set = get_dataset(configs)
    trainer = FER2013Trainer(model, train_set, val_set, test_set, configs)
    trainer.train()
    return trainer.queryCheckpointPath()


def get_dataset(configs):
    train_set = fer2013("train", configs)
    val_set = fer2013("val", configs)
    test_set = fer2013("test", configs, tta=True, tta_size=10)
    return train_set, val_set, test_set


def _load_test_data(args):
    file_map = pd.read_csv(os.path.join(args.RAW_PATH, 'submission.csv'))
    test_data = []
    img_names = []
    for file_name in file_map['file_name']:
        file_path = os.path.join(args.TEST_PATH, file_name)
        img_np = imread(file_path)
        img = img_np.copy()
        img = img.tolist()
        test_data.append(img)
        img_names.append(file_name)
    return test_data, img_names


def genTestResult(config_path, ckpt_path, args):
    with open(config_path) as f:
        configs = json.load(f)
    state = torch.load(ckpt_path)
    EMOTION_DICT = {
        0: "angry",
        1: "disgust",
        2: "fear",
        3: "happy",
        4: "sad",
        5: "surprise",
        6: "neutral",
    }

    model = cbam_resnet50(in_channels=3, num_classes=7).cuda()
    model.load_state_dict(state["net"])
    model.eval()

    print()
    print('-------------------------------------------------------')
    print('  -> Generating test result for {:} ...'.format('cbam_resnet50'))
    test_data, img_names = _load_test_data(args)
    test_length = len(test_data)
    testDataset = FERDataset(test_data, filenames=img_names, use_da=False, args=args)
    testDataloader = DataLoader(dataset=testDataset,
                                batch_size=16,
                                shuffle=False,
                                num_workers=args.num_workers)
    str_preds = []
    for images, filenames in testDataloader:
        model.eval()
        outs = model(images.repeat(1, 3, 1, 1).to(args.device))
        preds = outs.max(-1)[1].cpu().tolist()
        str_preds.extend([EMOTION_DICT[pred] for pred in preds])
    # generate submission
    assert len(str_preds) == len(img_names)
    submission = pd.DataFrame({'file_name': img_names, 'class': str_preds})
    submission.to_csv(os.path.join(args.DATA_PATH, 'submission.csv'), index=False, index_label=False)
    print('  -> Done generation of submission.csv with model {:}'.format(ckpt_path))


if __name__ == "__main__":
    from main import args
    run_cbam_resnet("config.json")
    genTestResult("config.json", "saved/checkpoints/cbam_resnet50__n_2020Dec14_15.13", args=args)
