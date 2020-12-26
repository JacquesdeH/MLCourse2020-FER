import argparse
import os
import random
import numpy as np
import warnings
import imgaug
import datetime
import torch

from Instructor import Instructor
from run_cbam_resnet import run_cbam_resnet, genTestResult

warnings.simplefilter(action="ignore", category=FutureWarning)

RANDOM_SEED = 521

# instructor
CLASSES = 7
PIC_LEN = 48
EMBED_DIM = 128 * 7 * 7
USE_SPARSE = True
RHO = 0.02
REGULIZER_WEIGHT = 1
EPOCHS = 50
BATCH_SIZE = 32
PRETRAIN_LR = 3e-5
RESNET_BASE_LR = 3e-4
RESNET_FT_LR = 1e-3
RESNET_WEIGHT_DECAY = 0
RESNET_DROPOUT = 0.5
RESNET_MOMENTUM = 0.9
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
RESNET_OPTIM = 'Adam'  # 'Adam' or 'SGD'

# paths
DATA_PATH = os.path.join("data")
RAW_PATH = os.path.join(DATA_PATH, "raw")
CKPT_PATH = os.path.join("ckpt")
LOG_PATH = os.path.join("log")
TRAIN_PATH = os.path.join(RAW_PATH, "train")
TEST_PATH = os.path.join(RAW_PATH, "test")
SAMPLE_PATH = os.path.join(DATA_PATH, 'sample')
CBAM_CONFIG_PATH = os.path.join("config.json")

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
parser.add_argument('--resnet_dropout', default=RESNET_DROPOUT, type=float)
parser.add_argument('--resnet_momentum', default=RESNET_MOMENTUM, type=float)
parser.add_argument('--resnet_optim', default=RESNET_OPTIM, type=str)

args = parser.parse_args()

args.DATA_PATH = DATA_PATH
args.RAW_PATH = RAW_PATH
args.CKPT_PATH = CKPT_PATH
args.LOG_PATH = LOG_PATH
args.TRAIN_PATH = TRAIN_PATH
args.TEST_PATH = TEST_PATH
args.SAMPLE_PATH = SAMPLE_PATH
args.CBAM_CONFIG_PATH = CBAM_CONFIG_PATH

args.device = torch.device('cuda:2' if args.cuda and torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(args.device)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
imgaug.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    # timestamp = "20201205-030029"
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = "Resnet50-BS32-BigLR-MoreFC-Dropout-Noised-WeightedLoss-Epoch50-Version-27"
    model_name = timestamp + '-' + model_name
    use_model = 'cbam_resnet'  # 'svm' or 'resnet' or 'cbam_resnet'

    instructor = Instructor(model_name, args)
    if use_model == 'svm':
        instructor.trainAutoEncoder()
        instructor.generateAutoEncoderTestResultSamples(sample_cnt=20)
        instructor.trainSVM(load=False)
        instructor.genTestResult(from_svm=True)
    elif use_model == 'cbam_resnet':
        cbam_resnet_ckpt_path = run_cbam_resnet(args.CBAM_CONFIG_PATH)
        # cbam_resnet_ckpt_path = "saved/checkpoints/cbam_resnet50__n_2020Dec15_01.31"
        genTestResult(args.CBAM_CONFIG_PATH, cbam_resnet_ckpt_path, args=args)
    elif use_model == 'resnet':
        instructor.trainResnet()
        instructor.loadResnet(epoch=args.epochs)
        instructor.genTestResult(from_svm=False)
