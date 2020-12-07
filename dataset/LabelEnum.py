# _*_coding:utf-8_*_
# Author      : JacquesdeH
# Create Time : 2020/11/30 17:13
# Project Name: MLCourse-FER
# File        : LabelEnum.py
# --------------------------------------------------

from enum import IntEnum

class LabelEnum(IntEnum):
    angry = 0
    disgust = 1
    fear = 2
    happy = 3
    neutral = 4
    sad = 5
    surprise = 6
