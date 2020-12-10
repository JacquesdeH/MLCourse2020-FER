# _*_coding:utf-8_*_
# Author      : JacquesdeH
# Create Time : 2020/11/29 18:32
# Project Name: MLCourse-FER
# File        : Instructor.py
# --------------------------------------------------

import os
import random
import math
import torch
import pandas as pd
from tqdm import tqdm
from matplotlib.pyplot import imread
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision.transforms import ToTensor, ToPILImage
from transformers import get_linear_schedule_with_warmup
from sklearn.svm import SVC
import joblib

from model.Encoder import Encoder
from model.Decoder import Decoder
from dataset.FERDataset import FERDataset
from dataset.LabelEnum import LabelEnum
from model.Resnet import Resnet


class Instructor:
    def __init__(self, model_name: str, args):
        self.model_name = model_name
        self.args = args
        self.encoder = Encoder(self.args.add_noise).to(self.args.device)
        self.decoder = Decoder(self.args.upsample_mode).to(self.args.device)
        self.pretrainDataset = None
        self.pretrainDataloader = None
        self.pretrainOptimizer = None
        self.pretrainScheduler = None
        self.RHO_tensor = None
        self.pretrain_batch_cnt = 0
        self.writer = None
        self.svmDataset = None
        self.svmDataloader = None
        self.testDataset = None
        self.testDataloader = None
        self.svm = SVC(C=self.args.svm_c, kernel=self.args.svm_ker, verbose=True, max_iter=self.args.svm_max_iter)
        self.resnet = Resnet(use_pretrained=True,
                             num_classes=self.args.classes,
                             resnet_depth=self.args.resnet_depth,
                             dropout=self.args.resnet_dropout).to(self.args.device)
        self.resnetOptimizer = None
        self.resnetScheduler = None
        self.resnetLossFn = None

    def _load_data_by_label(self, label: str) -> list:
        ret = []
        LABEL_PATH = os.path.join(self.args.TRAIN_PATH, label)
        for dir_path, _, file_list in os.walk(LABEL_PATH, topdown=False):
            for file_name in file_list:
                file_path = os.path.join(dir_path, file_name)
                img_np = imread(file_path)
                img = img_np.copy()
                img = img.tolist()
                ret.append(img)
        return ret

    def _load_all_data(self):
        all_data = []
        all_labels = []
        for label_id in range(0, self.args.classes):
            expression = LabelEnum(label_id)
            sub_data = self._load_data_by_label(expression.name)
            sub_labels = [label_id] * len(sub_data)
            all_data.extend(sub_data)
            all_labels.extend(sub_labels)
        return all_data, all_labels

    def _load_test_data(self):
        file_map = pd.read_csv(os.path.join(self.args.RAW_PATH, 'submission.csv'))
        test_data = []
        img_names = []
        for file_name in file_map['file_name']:
            file_path = os.path.join(self.args.TEST_PATH, file_name)
            img_np = imread(file_path)
            img = img_np.copy()
            img = img.tolist()
            test_data.append(img)
            img_names.append(file_name)
        return test_data, img_names

    def trainAutoEncoder(self):
        self.writer = SummaryWriter(os.path.join(self.args.LOG_PATH, self.model_name))
        all_data, all_labels = self._load_all_data()
        self.pretrainDataset = FERDataset(all_data, labels=all_labels, args=self.args)
        self.pretrainDataloader = DataLoader(dataset=self.pretrainDataset, batch_size=self.args.batch_size,
                                             shuffle=True, num_workers=self.args.num_workers)
        self.pretrainOptimizer = torch.optim.Adam([{'params': self.encoder.parameters(), 'lr': self.args.pretrain_lr},
                                                   {'params': self.decoder.parameters(), 'lr': self.args.pretrain_lr}])
        tot_steps = math.ceil(len(self.pretrainDataloader) / self.args.cumul_batch) * self.args.epochs
        self.pretrainScheduler = get_linear_schedule_with_warmup(self.pretrainOptimizer,
                                                                 num_warmup_steps=0, num_training_steps=tot_steps)
        self.RHO_tensor = torch.tensor([self.args.rho for _ in range(self.args.embed_dim)],
                                       dtype=torch.float).unsqueeze(0).to(self.args.device)
        epochs = self.args.epochs
        for epoch in range(1, epochs + 1):
            print()
            print("================ AutoEncoder Training Epoch {:}/{:} ================".format(epoch, epochs))
            print(" ---- Start training ------>")
            self.epochTrainAutoEncoder(epoch)
            print()
        self.writer.close()

    def epochTrainAutoEncoder(self, epoch):
        self.encoder.train()
        self.decoder.train()

        cumul_loss = 0
        cumul_steps = 0
        cumul_samples = 0

        self.pretrainOptimizer.zero_grad()
        cumulative_batch = 0

        for idx, (images, labels) in enumerate(tqdm(self.pretrainDataloader)):
            batch_size = images.shape[0]
            images, labels = images.to(self.args.device), labels.to(self.args.device)

            embeds = self.encoder(images)
            outputs = self.decoder(embeds)

            loss = torch.nn.functional.mse_loss(outputs, images)
            if self.args.use_sparse:
                rho_hat = torch.mean(embeds, dim=0, keepdim=True)
                sparse_penalty = self.args.regulizer_weight * torch.nn.functional.kl_div(
                    input=torch.nn.functional.log_softmax(rho_hat, dim=-1),
                    target=torch.nn.functional.softmax(self.RHO_tensor, dim=-1))
                loss = loss + sparse_penalty

            loss_each = loss / self.args.cumul_batch
            loss_each.backward()

            cumulative_batch += 1
            cumul_steps += 1
            cumul_loss += loss.detach().cpu().item() * batch_size
            cumul_samples += batch_size

            if cumulative_batch >= self.args.cumul_batch:
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=self.args.max_norm)
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=self.args.max_norm)
                self.pretrainOptimizer.step()
                self.pretrainScheduler.step()
                self.pretrainOptimizer.zero_grad()
                cumulative_batch = 0

            if cumul_steps >= self.args.disp_period or idx + 1 == len(self.pretrainDataloader):
                print(" -> cumul_steps={:} loss={:}".format(cumul_steps, cumul_loss / cumul_samples))
                self.pretrain_batch_cnt += 1
                self.writer.add_scalar('batch-loss', cumul_loss / cumul_samples, global_step=self.pretrain_batch_cnt)
                self.writer.add_scalar('encoder_lr', self.pretrainOptimizer.state_dict()['param_groups'][0]['lr'],
                                       global_step=self.pretrain_batch_cnt)
                self.writer.add_scalar('decoder_lr', self.pretrainOptimizer.state_dict()['param_groups'][1]['lr'],
                                       global_step=self.pretrain_batch_cnt)
                cumul_steps = 0
                cumul_loss = 0
                cumul_samples = 0

        self.saveAutoEncoder(epoch)

    def saveAutoEncoder(self, epoch):
        encoderPath = os.path.join(self.args.CKPT_PATH, self.model_name + "--Encoder" + "--EPOCH-{:}".format(epoch))
        decoderPath = os.path.join(self.args.CKPT_PATH, self.model_name + "--Decoder" + "--EPOCH-{:}".format(epoch))
        print("-----------------------------------------------")
        print("  -> Saving AutoEncoder {:} ......".format(self.model_name))
        torch.save(self.encoder.state_dict(), encoderPath)
        torch.save(self.decoder.state_dict(), decoderPath)
        print("  -> Successfully saved AutoEncoder.")
        print("-----------------------------------------------")

    def generateAutoEncoderTestResultSamples(self, sample_cnt):
        self.encoder.eval()
        self.decoder.eval()
        print('  -> Generating samples with AutoEncoder ...')
        save_path = os.path.join(self.args.SAMPLE_PATH, self.model_name)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        with torch.no_grad():
            for dir_path, _, file_list in os.walk(self.args.TEST_PATH, topdown=False):
                sample_file_list = random.choices(file_list, k=sample_cnt)
                for file_name in sample_file_list:
                    file_path = os.path.join(dir_path, file_name)
                    img_np = imread(file_path)
                    img = img_np.copy()
                    img = ToTensor()(img)
                    img = img.reshape(1, 1, 48, 48)
                    img = img.to(self.args.device)
                    embed = self.encoder(img)
                    out = self.decoder(embed).cpu()
                    out = out.reshape(1, 48, 48)
                    out_img = ToPILImage()(out)
                    out_img.save(os.path.join(save_path, file_name))
        print('  -> Done sampling from AutoEncoder with test pictures.')

    def loadAutoEncoder(self, epoch):
        encoderPath = os.path.join(self.args.CKPT_PATH, self.model_name + "--Encoder" + "--EPOCH-{:}".format(epoch))
        decoderPath = os.path.join(self.args.CKPT_PATH, self.model_name + "--Decoder" + "--EPOCH-{:}".format(epoch))
        print("-----------------------------------------------")
        print("  -> Loading AutoEncoder {:} ......".format(self.model_name))
        self.encoder.load_state_dict(torch.load(encoderPath))
        self.decoder.load_state_dict(torch.load(decoderPath))
        print("  -> Successfully loaded AutoEncoder.")
        print("-----------------------------------------------")

    def generateExtractedFeatures(self, data: torch.FloatTensor) -> torch.FloatTensor:
        """
        :param data: (batch, channel, l, w)
        :return: embed: (batch, embed_dim)
        """
        with torch.no_grad():
            data = data.to(self.args.device)
            embed = self.encoder(data)
            embed = embed.detach().cpu()
            return embed

    def trainSVM(self, load: bool):
        svm_path = os.path.join(self.args.CKPT_PATH, self.model_name + '--svm')
        self.loadAutoEncoder(self.args.epochs)
        self.encoder.eval()
        self.decoder.eval()
        if load:
            print('  -> Loaded from SVM trained model.')
            self.svm = joblib.load(svm_path)
            return
        print()
        print("================ SVM Training Starting ================")
        all_data, all_labels = self._load_all_data()
        all_length = len(all_data)
        self.svmDataset = FERDataset(all_data, labels=all_labels, use_da=False, args=self.args)
        self.svmDataloader = DataLoader(dataset=self.svmDataset,
                                        batch_size=self.args.batch_size,
                                        shuffle=False,
                                        num_workers=self.args.num_workers)
        print("  -> Converting to extracted features ...")
        cnt = 0
        all_embeds = []
        all_labels = []
        for images, labels in self.svmDataloader:
            cnt += 1
            embeds = self.generateExtractedFeatures(images)
            all_embeds.extend(embeds.tolist())
            all_labels.extend(labels.reshape(-1).tolist())
        print('  -> Start SVM fit ...')
        self.svm.fit(X=all_embeds, y=all_labels)
        # self.svm.fit(X=all_embeds[0:3], y=[0, 1, 2])
        joblib.dump(self.svm, svm_path)
        print("  -> Done training for SVM.")

    def genTestResult(self, from_svm=True):
        print()
        print('-------------------------------------------------------')
        print('  -> Generating test result for {:} ...'.format('SVM' if from_svm else 'Resnet'))
        test_data, img_names = self._load_test_data()
        test_length = len(test_data)
        self.testDataset = FERDataset(test_data, filenames=img_names, use_da=False, args=self.args)
        self.testDataloader = DataLoader(dataset=self.testDataset,
                                         batch_size=self.args.batch_size,
                                         shuffle=False,
                                         num_workers=self.args.num_workers)
        str_preds = []
        for images, filenames in self.testDataloader:
            if from_svm:
                embeds = self.generateExtractedFeatures(images)
                preds = self.svm.predict(X=embeds)
            else:
                self.resnet.eval()
                outs = self.resnet(images.repeat(1, 3, 1, 1).to(self.args.device))
                preds = outs.max(-1)[1].cpu().tolist()
            str_preds.extend([LabelEnum(pred).name for pred in preds])
        # generate submission
        assert len(str_preds) == len(img_names)
        submission = pd.DataFrame({'file_name': img_names, 'class': str_preds})
        submission.to_csv(os.path.join(self.args.DATA_PATH, 'submission.csv'), index=False, index_label=False)
        print('  -> Done generation of submission.csv with model {:}'.format(self.model_name))

    def epochTrainResnet(self, epoch):
        self.resnet.train()

        cumul_loss = 0
        cumul_acc = 0
        cumul_steps = 0
        cumul_samples = 0
        cumulative_batch = 0

        self.resnetOptimizer.zero_grad()

        for idx, (images, labels) in enumerate(tqdm(self.pretrainDataloader)):
            batch_size = images.shape[0]
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            images += torch.randn(images.shape).to(images.device) * self.args.add_noise
            images = images.repeat(1, 3, 1, 1)

            outs = self.resnet(images)
            preds = outs.max(-1)[1].unsqueeze(dim=1)
            cur_acc = (preds == labels).type(torch.int).sum().item()

            loss = self.resnetLossFn(outs, labels.squeeze(dim=1))

            loss_each = loss / self.args.cumul_batch
            loss_each.backward()

            cumulative_batch += 1
            cumul_steps += 1
            cumul_loss += loss.detach().cpu().item() * batch_size
            cumul_acc += cur_acc
            cumul_samples += batch_size

            if cumulative_batch >= self.args.cumul_batch:
                torch.nn.utils.clip_grad_norm_(self.resnet.parameters(), max_norm=self.args.max_norm)
                self.resnetOptimizer.step()
                self.resnetScheduler.step()
                self.resnetOptimizer.zero_grad()
                cumulative_batch = 0

            if cumul_steps >= self.args.disp_period or idx + 1 == len(self.pretrainDataloader):
                print(" -> cumul_steps={:} loss={:} acc={:}".format(
                    cumul_steps, cumul_loss / cumul_samples, cumul_acc / cumul_samples))
                self.pretrain_batch_cnt += 1
                self.writer.add_scalar('batch-loss', cumul_loss / cumul_samples, global_step=self.pretrain_batch_cnt)
                self.writer.add_scalar('batch-acc', cumul_acc / cumul_samples, global_step=self.pretrain_batch_cnt)
                self.writer.add_scalar('resnet_lr', self.resnetOptimizer.state_dict()['param_groups'][0]['lr'],
                                       global_step=self.pretrain_batch_cnt)
                cumul_steps = 0
                cumul_loss = 0
                cumul_acc = 0
                cumul_samples = 0

        if epoch % 10 == 0:
            self.saveResnet(epoch)

    def saveResnet(self, epoch):
        resnetPath = os.path.join(self.args.CKPT_PATH, self.model_name + "--Resnet" + "--EPOCH-{:}".format(epoch))
        print("-----------------------------------------------")
        print("  -> Saving Resnet {:} ......".format(self.model_name))
        torch.save(self.resnet.state_dict(), resnetPath)
        print("  -> Successfully saved Resnet.")
        print("-----------------------------------------------")

    def loadResnet(self, epoch):
        resnetPath = os.path.join(self.args.CKPT_PATH, self.model_name + "--Resnet" + "--EPOCH-{:}".format(epoch))
        print("-----------------------------------------------")
        print("  -> Loading Resnet {:} ......".format(self.model_name))
        self.resnet.load_state_dict(torch.load(resnetPath))
        print("  -> Successfully loaded Resnet.")
        print("-----------------------------------------------")

    def trainResnet(self):
        self.writer = SummaryWriter(os.path.join(self.args.LOG_PATH, self.model_name))
        all_data, all_labels = self._load_all_data()
        self.pretrainDataset = FERDataset(all_data, labels=all_labels, args=self.args)
        self.pretrainDataloader = DataLoader(dataset=self.pretrainDataset, batch_size=self.args.batch_size,
                                             shuffle=True, num_workers=self.args.num_workers)
        self.resnetOptimizer = self.getResnetOptimizer()
        tot_steps = math.ceil(len(self.pretrainDataloader) / self.args.cumul_batch) * self.args.epochs
        self.resnetScheduler = get_linear_schedule_with_warmup(
            self.resnetOptimizer, num_warmup_steps=tot_steps * self.args.warmup_rate, num_training_steps=tot_steps)
        self.resnetLossFn = torch.nn.CrossEntropyLoss(
            weight=torch.tensor([
                9.40661861,
                1.00104606,
                0.56843877,
                0.84912748,
                1.02660468,
                1.29337298,
                0.82603942,
            ], dtype=torch.float, device=self.args.device))
        epochs = self.args.epochs
        for epoch in range(1, epochs + 1):
            print()
            print("================ Resnet Training Epoch {:}/{:} ================".format(epoch, epochs))
            print(" ---- Start training ------>")
            self.epochTrainResnet(epoch)
            print()
        self.writer.close()

    def getResnetOptimizer(self):
        if self.args.resnet_optim == 'SGD':
            return torch.optim.SGD([
                {'params': self.resnet.baseParameters(), 'lr': self.args.resnet_base_lr,
                 'weight_decay': self.args.weight_decay, 'momentum': self.args.resnet_momentum},
                {'params': self.resnet.finetuneParameters(), 'lr': self.args.resnet_ft_lr,
                 'weight_decay': self.args.weight_decay, 'momentum': self.args.resnet_momentum}],
                lr=self.args.resnet_base_lr)
        elif self.args.resnet_optim == 'Adam':
            return torch.optim.Adam([
                {'params': self.resnet.baseParameters(), 'lr': self.args.resnet_base_lr},
                {'params': self.resnet.finetuneParameters(), 'lr': self.args.resnet_ft_lr,
                 'weight_decay': self.args.weight_decay}])


if __name__ == '__main__':
    from main import args

    instructor = Instructor('Baseline', args)
