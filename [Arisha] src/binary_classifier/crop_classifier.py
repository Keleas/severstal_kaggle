import numpy as np
from torchvision import models
import torch
import torch.nn as nn
from logger import Logger
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
import torch.optim as optim
import segmentation_models_pytorch as smp
import os
import re
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
import itertools
import model_network
import math
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import f1_score, hinge_loss, confusion_matrix
from albumentations import (
    OneOf,
    MotionBlur,
    MedianBlur,
    Blur,
    IAAAdditiveGaussianNoise,
    GaussNoise,
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ShiftScaleRotate,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,
    RandomGamma,
    Normalize
)
from albumentations.torch import ToTensor

PATH = 'D:\\My_Data\\_Desktop\\severstal'
num_crop = 4

def labels(mode):
    files, labels, cur_labels, masks, cur_masks = [], [], [], [], []
    if mode == 'train' or 'val':
        file_name = 'train'
    else:
        file_name = 'test'
    with open(os.path.join(PATH, file_name + '.csv')) as csv_file:
        next(csv_file)
        for num, line in enumerate(csv_file, 1):
            line_data = re.split(r'[_,\n]', line)
            file_name = line_data[0]
            if not file_name in files:
                files.append(line_data[0])
            if line_data[2]:
                cur_labels.append(1)
                cur_masks.append(np.array(list(map(int,line_data[2].split()))))
            else:
                cur_labels.append(0)
                cur_masks.append([0])
            if num % 4 == 0:
                # if cur_labels == [0, 0, 0, 0]:
                #     cur_labels.append(1)
                #     cur_masks.append([0])
                # else:
                #     cur_labels.append(0)
                #     cur_masks.append([0])
                labels.append(cur_labels)
                cur_labels = []
                masks.append(cur_masks)
                cur_masks = []
    return files, labels, masks

def rleToMask(segments):
    images = np.zeros((4, 256, 1600))
    for i, segment in enumerate(segments):
        if len(segment) != 1:
            rows, cols = 256, 1600
            rleNumbers = np.array(segment)
            rlePairs = rleNumbers.reshape(-1, 2)

            img = np.zeros(rows * cols, dtype=np.uint8)
            for index, length in rlePairs:
                index -= 1
                img[index:index + length] = 255
            img = img.reshape(cols, rows)
            img = img.T
            images[i] = img
    # images = images.reshape((4, 256, 1600))
    return images


class SteelDataset(Dataset):
    def __init__(self, mode):
        self.mode = mode
        self.files, self.labels, self.masks = labels(self.mode)
        split = len(self.files)*9 // 10
        if self.mode == 'train':
            self.files = self.files[:split]
        elif self.mode == 'val':
            self.files = self.files[split:]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        if self.mode == 'train' or 'val':
            file_name = 'train'
        else:
            file_name = 'test'
        file_path = os.path.join(os.path.join(PATH, file_name + '_images'), file)

        image = cv2.imread(file_path)
        mask = rleToMask(self.masks[idx])
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        train_aug = Compose([
            OneOf([
                ShiftScaleRotate(),
                VerticalFlip(p=0.8),
                HorizontalFlip(p=0.8),
            ], p=0.6),
            OneOf([
                RandomBrightnessContrast(),
                MotionBlur(p=0.5),
                MedianBlur(blur_limit=3, p=0.5),
                Blur(blur_limit=3, p=0.5),
            ], p=0.6),
            OneOf([
                IAAAdditiveGaussianNoise(),
                GaussNoise(),
            ], p=0.6),
            Normalize(mean=mean, std=std, p=1)
            # ToTensor()
        ])
        val_aug = Compose([
            Normalize(mean=mean, std=std, p=1),
            ToTensor()
        ])
        if self.mode == 'train':
            crop_images, crop_masks, new_crop_images = [], [], []
            crop_labels = np.zeros((num_crop, 4))
            for i in range(num_crop):
                crop_images.append(image[:, i*(1600//num_crop):(i+1)*(1600//num_crop), :])
                crop_masks.append(mask[:, :, i*(1600//num_crop):(i+1)*(1600//num_crop)])
            for i in range(num_crop):
                for ch in range(4):
                    if (crop_masks[i][ch, :, :] == np.zeros((crop_masks[i][ch, :, :].shape))).all():
                        crop_labels[i][ch] = 0
                    else:
                        crop_labels[i][ch] = 1
            for num in range(len(crop_images)):
                if self.mode == 'train':
                    augmented = train_aug(image=crop_images[num])
                    new_crop_images.append(augmented['image'])
                crop_labels[num] = np.array(crop_labels[num])
                crop_labels[num] = torch.tensor(crop_labels[num], dtype=torch.float32)
            new_crop_images = np.transpose(np.array(new_crop_images), (0, 3, 1, 2))
            return new_crop_images, crop_labels
        elif self.mode == 'val':
            augmented = val_aug(image=image)
            image = augmented['image']
            label = np.array(self.labels[idx])
            label = torch.tensor(label, dtype=torch.float32)
            return (image, label)




class Train_Model(object):
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.logger = Logger('logs/')

        self.train_loader = None
        self.val_loader = None
        self.train_data = None
        self.val_data = None

        self.num_epochs = 100
        self.num_classes = 4
        self.batch_size = 8
        self.learning_rate = 1e-2

    def crop_collate(self, batch):
        image = [np.array(item[0][i]) for item in batch for i in range(num_crop)]
        label = [np.array(item[1][i]) for item in batch for i in range(num_crop)]
        image, label = np.array(image), np.array(label)
        image, label = torch.from_numpy(image).float(), torch.from_numpy(label).float()
        return (image, label)



    def train(self):
        acc, losses = [], []
        self.model.train()
        for (image, label) in tqdm(self.train_loader):
            image = image.cuda()
            label = label.cuda()
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = self.model(image)
                outputs = nn.Sigmoid()(outputs)

                loss = nn.BCELoss()(outputs, label)
                losses.append(loss.item())
                loss.backward()
                self.optimizer.step()
                precision, _, _ = self.do_kaggle_metric(outputs.detach().cpu().numpy().squeeze()>0.5, label.detach().cpu().numpy())
                acc.append(np.mean(precision))
        print('TRAIN: Loss: {}'.format(np.mean(losses)))
        return np.mean(losses), np.mean(acc)


    def validation(self):
        losses, all_predicted, all_labels, acc = [], [], [], []
        self.model.eval()
        for (image, label) in tqdm(self.val_loader):
            image = image.cuda()
            label = label.cuda()
            with torch.set_grad_enabled(False):
                outputs = self.model(image)
                outputs = nn.Sigmoid()(outputs)
                loss = nn.BCELoss()(outputs, label)
                losses.append(loss.item())
                precision, _, _ = self.do_kaggle_metric(outputs.detach().cpu().numpy().squeeze() > 0.5,
                                                        label.detach().cpu().numpy())
                acc.append(np.mean(precision))
        print('VALIDATION: Loss: {}, Acc: {}'.format(np.mean(losses), np.mean(acc)))
        return np.mean(losses), np.mean(acc)

    def func_logger(self, epoch, val_loss, train_loss, val_acc):
        print("[Epoch {}: training loss: {}, val_loss: {}, acc: {}".format(epoch, train_loss, val_loss, val_acc))
        info = {'loss': train_loss,
                'val_loss': val_acc,
                'acc': val_acc}

        for tag, value in info.items():
            self.logger.scalar_summary(tag, value, epoch + 1)

    def do_kaggle_metric(self, predict, truth, threshold=0.5):

        N = len(predict)
        predict = predict.reshape(N, -1)
        truth = truth.reshape(N, -1)

        predict = predict > threshold
        truth = truth > 0.5
        intersection = truth & predict
        union = truth | predict
        iou = intersection.sum(1) / (union.sum(1) + 1e-8)

        # -------------------------------------------
        result = []
        precision = []
        is_empty_truth = (truth.sum(1) == 0)
        is_empty_predict = (predict.sum(1) == 0)

        threshold = np.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
        for t in threshold:
            p = iou >= t

            tp = (~is_empty_truth) & (~is_empty_predict) & (iou > t)
            fp = (~is_empty_truth) & (~is_empty_predict) & (iou <= t)
            fn = (~is_empty_truth) & (is_empty_predict)
            fp_empty = (is_empty_truth) & (~is_empty_predict)
            tn_empty = (is_empty_truth) & (is_empty_predict)

            p = (tp + tn_empty) / (tp + tn_empty + fp + fp_empty + fn)

            result.append(np.column_stack((tp, fp, fn, tn_empty, fp_empty)))
            precision.append(p)

        result = np.array(result).transpose(1, 2, 0)
        precision = np.column_stack(precision)
        precision = precision.mean(1)

        return precision, result, threshold

    def training(self):
        best_acc = 0
        for epoch in range(self.num_epochs):
            print("**************** Epoch :[{}/{}] ****************".format(epoch + 1, self.num_epochs))
            train_loss, train_acc = self.train()
            val_loss, val_acc= self.validation()
            self.func_logger(epoch, val_loss, train_loss, val_acc)
            self.scheduler.step(val_loss)
            if best_acc < val_acc:
                print('Validation accuracy increased {:.6f} ----> {:.6f}. Saving best model ...'.format(best_acc, val_acc))
                torch.save({
                    'epoch': self.num_epochs,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
                }, os.path.join(PATH, 'models\\crop_pretrained.pth'))
                best_acc = val_acc
        # self.confusion_matrix(predicted, all_labels)


    def main(self):
        self.model = model_network.Resnet34_classification()
        checkpoint = torch.load('D:\\My_Data\\_Desktop\\severstal\\models\\00007500_model.pth')
        self.model.load_state_dict(checkpoint)
        self.model = self.model.to('cuda:0')
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=15, eta_min=0)

        train_set = SteelDataset('train')
        val_set = SteelDataset('val')
        # validation_split = .1
        # shuffle_dataset = True
        # random_seed = 42
        # dataset_size = len(data_set)
        # indices = list(range(dataset_size))
        # split = int(np.floor(validation_split * dataset_size))
        # if shuffle_dataset:
        #     np.random.seed(random_seed)
        #     np.random.shuffle(indices)
        # train_indices, val_indices = indices[split:], indices[:split]
        #
        # train_sampler = SubsetRandomSampler(train_indices)
        # valid_sampler = SubsetRandomSampler(val_indices)
        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size//num_crop, shuffle=True, collate_fn=self.crop_collate)
        self.val_loader = torch.utils.data.DataLoader(val_set, batch_size=self.batch_size, shuffle=True)
        self.training()


if __name__ == '__main__':
    model = Train_Model()
    model.main()
