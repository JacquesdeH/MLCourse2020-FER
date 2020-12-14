# _*_coding:utf-8_*_
# Author      : JacquesdeH
# Create Time : 2020/12/12 22:56
# Project Name: MLCourse-FER
# File        : Ensembler.py
# --------------------------------------------------

import os
import pandas as pd
from dataset.LabelEnum import LabelEnum

RAW_SUBMISSION_PATH = os.path.join(os.path.join("data", "raw"), "submission.csv")
ENSEMBLE_PATH = os.path.join("data", "ensemble")
DST_SUBMISSION = os.path.join(ENSEMBLE_PATH, "submission.csv")

submissions = {
    "20201210-164100-Cbam-Resnet50-Version-26--73112": 73.112,
    "20201210-164100-Cbam-Resnet50-Version-27--72221": 72.221,
    "20201210-164100-Cbam-Resnet50-Version-28--71691": 71.691,
    "20201210-164100-Cbam-Resnet50-Version-29--72137": 72.137,
    "20201210-164100-Cbam-Resnet50-Version-30--71134": 71.134,
    "20201210-164100-Cbam-Resnet50-Version-31--72889": 72.889,
    "20201210-164100-Cbam-Resnet50-Version-32--71663": 71.663,
    "20201210-164100-Cbam-Resnet50-Version-33--71307": 71.307,
    "20201210-164100-Cbam-Resnet50-Version-34--73307": 73.307,
    "20201210-164100-Cbam-Resnet50-Version-35--72276": 72.226,
}

emotions = [emotion.name for emotion in LabelEnum]
raw = pd.read_csv(RAW_SUBMISSION_PATH)
files = raw['file_name'].tolist()
ensemble_dict = dict([(filename, dict([(emotion, 0.0) for emotion in emotions])) for filename in files])

for submission, score in submissions.items():
    csv_path = os.path.join(ENSEMBLE_PATH, os.path.join(submission, "submission.csv"))
    df = pd.read_csv(csv_path)
    for filename, emotion in zip(df['file_name'], df['class']):
        ensemble_dict[filename][emotion] += score

ensembled_pairs = dict([(filename, max(emotion_vec.items(), key=lambda x: x[1])[0])
                        for (filename, emotion_vec) in ensemble_dict.items()])

ensembled_df = pd.DataFrame(data={"file_name": list(ensembled_pairs.keys()),
                                  "class": list(ensembled_pairs.values())})
ensembled_df.to_csv(DST_SUBMISSION, index=False, index_label=False)
