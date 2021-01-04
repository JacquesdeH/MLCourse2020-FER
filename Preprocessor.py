# _*_coding:utf-8_*_
# Author      : JacquesdeH
# Create Time : 2020/12/14 11:14
# Project Name: MLCourse-FER
# File        : Preprocessor.py
# --------------------------------------------------

import os
from main import args
import matplotlib.pyplot as plt
from dataset.LabelEnum import LabelEnum

def calc_distribution(data_path: str):
    ret = {}
    for emotion in LabelEnum.__members__.keys():
        LABEL_PATH = os.path.join(data_path, emotion)
        cur_cnt = 0
        for dir_path, _, file_list in os.walk(LABEL_PATH, topdown=False):
            for file_name in file_list:
                file_path = os.path.join(dir_path, file_name)
                cur_cnt += 1
        ret[emotion] = cur_cnt
    return ret


if __name__ == '__main__':
    dist = calc_distribution(args.TRAIN_PATH)
    print(dist)
    plt.bar(dist.keys(), dist.values())
    plt.show()

