import argparse
import os
import random
import numpy as np
import datetime
import torch

from Instructor import Instructor

RANDOM_SEED = 7

# instructor
CLASSES = 7
PIC_LEN = 48
EMBED_DIM = 128*7*7
USE_SPARSE = True
RHO = 0.02
REGULIZER_WEIGHT = 1
EPOCHS = 50
BATCH_SIZE = 32
PRETRAIN_LR = 3e-5
RESNET_BASE_LR = 3e-5
RESNET_FT_LR = 1e-3
RESNET_WEIGHT_DECAY = 1e-5
MAX_NORM = 1.0
NUM_WORKERS = 1
CUMUL_BATCH = 1
DISP_PERIOD = 50
USE_DA = True
SVM_C = 0.9
SVM_KER = 'rbf'
SVM_MAX_ITER = -1
WARMUP_RATE = 0.1
UPSAMPLE_MODE = 'nearest'
ADD_NOISE = 0.01
RESNET_DEPTH = 50

# paths
DATA_PATH = os.path.join("data")
RAW_PATH = os.path.join(DATA_PATH, "raw")
CKPT_PATH = os.path.join("ckpt")
LOG_PATH = os.path.join("log")
TRAIN_PATH = os.path.join(RAW_PATH, "train")
TEST_PATH = os.path.join(RAW_PATH, "test")
SAMPLE_PATH = os.path.join(DATA_PATH, 'sample')

parser = argparse.ArgumentParser()

parser.add_argument('--seed', default=RANDOM_SEED, type=int)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--pic_len', default=PIC_LEN, type=int)
parser.add_argument('--embed_dim', default=EMBED_DIM, type=int)
parser.add_argument('--use_sparse', default=USE_SPARSE, type=bool)
parser.add_argument('--rho', default=RHO, type=float)
parser.add_argument('--classes', default=CLASSES, type=int)
parser.add_argument('--regulizer_weight', default=REGULIZER_WEIGHT, type=float)
parser.add_argument('--epochs', default=EPOCHS, type=int)
parser.add_argument('--batch_size', default=BATCH_SIZE, type=int)
parser.add_argument('--pretrain_lr', default=PRETRAIN_LR, type=float)
parser.add_argument('--resnet_base_lr', default=RESNET_BASE_LR, type=float)
parser.add_argument('--resnet_ft_lr', default=RESNET_FT_LR, type=float)
parser.add_argument('--max_norm', default=MAX_NORM, type=float)
parser.add_argument('--num_workers', default=NUM_WORKERS, type=int)
parser.add_argument('--cumul_batch', default=CUMUL_BATCH, type=int)
parser.add_argument('--disp_period', default=DISP_PERIOD, type=int)
parser.add_argument('--use_da', default=USE_DA, type=bool)
parser.add_argument('--svm_c', default=SVM_C, type=float)
parser.add_argument('--svm_ker', default=SVM_KER, type=str)
parser.add_argument('--svm_max_iter', default=SVM_MAX_ITER, type=int)
parser.add_argument('--warmup_rate', default=WARMUP_RATE, type=int)
parser.add_argument('--upsample_mode', default=UPSAMPLE_MODE, type=str)
parser.add_argument('--add_noise', default=ADD_NOISE, type=float)
parser.add_argument('--resnet_depth', default=RESNET_DEPTH, type=int)
parser.add_argument('--weight_decay', default=RESNET_WEIGHT_DECAY, type=float)


args = parser.parse_args()

args.DATA_PATH = DATA_PATH
args.RAW_PATH = RAW_PATH
args.CKPT_PATH = CKPT_PATH
args.LOG_PATH = LOG_PATH
args.TRAIN_PATH = TRAIN_PATH
args.TEST_PATH = TEST_PATH
args.SAMPLE_PATH = SAMPLE_PATH

args.device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

if __name__ == '__main__':
    # timestamp = "20201205-030029"
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = "Resnet50-WD1e5-Noised-Epoch50-Version-17"
    model_name = timestamp + '-' + model_name
    use_model = 'resnet'  # 'svm' or 'resnet' or 'fc'
    instructor = Instructor(model_name, args)
    if use_model == 'svm':
        # instructor.trainAutoEncoder()
        # instructor.generateAutoEncoderTestResultSamples(sample_cnt=20)
        instructor.trainSVM(load=False)
        instructor.genTestResult(from_svm=True)
    elif use_model == 'fc':
        pass
    elif use_model == 'resnet':
        instructor.trainResnet()
        instructor.loadResnet(epoch=args.epochs)
        instructor.genTestResult(from_svm=False)
