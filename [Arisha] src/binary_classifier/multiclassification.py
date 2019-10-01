import numpy as np
from torchvision import models
import torch
import torch.nn as nn
from logger import Logger
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
import torch.optim as optim
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

def labels(mode):
    files = []
    labels = []
    cur_labels = []
    with open(os.path.join(PATH, mode + '.csv')) as csv_file:
        next(csv_file)
        for num, line in enumerate(csv_file, 1):
            line_data = re.split(r'[_,\n]', line)
            file_name = line_data[0]
            if not file_name in files:
                files.append(line_data[0])
            if line_data[2]:
                cur_labels.append(1)
                # cur_labels.append(int(line_data[1]))
            else:
                cur_labels.append(0)
            if num % 4 == 0:
                labels.append(cur_labels)
                cur_labels = []
    return files, labels


class SteelDataset(Dataset):
    def __init__(self):
        self.mode = 'train'
        self.files, self.labels = labels(self.mode)


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        file_path = os.path.join(os.path.join(PATH, self.mode + '_images'), file)
        image = cv2.imread(file_path)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        train_aug = Compose([
            OneOf([
                VerticalFlip(),
                HorizontalFlip(),
            ], p=0.5),
            OneOf([
                MotionBlur(p=0.4),
                MedianBlur(p=0.4, blur_limit=3),
                Blur(p=0.5, blur_limit=3)
            ], p=0.4),
            OneOf([
                IAAAdditiveGaussianNoise(),
                GaussNoise()
            ], p=0.4),
            Normalize(mean=mean, std=std, p=1),
            # CLAHE(p=0.5),
            ToTensor()])
        augmented = train_aug(image=image)
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
        self.learning_rate = 1e-3

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
                # acc.append(f1_score(y_pred=outputs.detach().cpu().numpy().squeeze()>0.5, y_true=label.detach().cpu().numpy().squeeze(), average='samples'))
                precision, _, _ = self.do_kaggle_metric(outputs.detach().cpu().numpy().squeeze()>0.5, label.detach().cpu().numpy())
                acc.append( np.mean(precision))
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


    def confusion_matrix(self, predicted, labels):
        predicted = list(itertools.chain.from_iterable(predicted))
        labels = list(itertools.chain.from_iterable(labels))
        all = len(labels)
        data = (dict(Counter(zip(predicted, labels))))
        mas = np.zeros((self.num_classes, self.num_classes))
        mas_prob = np.zeros((self.num_classes, self.num_classes))
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if (i, j) in data.keys():
                    mas[i, j] = data[(i, j)]
                    mas_prob[i, j] = data[(i, j)] / all
        plt.figure()
        plt.subplot(211)
        ax = sns.heatmap(mas, annot=True)
        plt.ylabel('predicted')
        plt.xlabel('label')
        plt.subplot(212)
        ax = sns.heatmap(mas_prob, annot=True)
        plt.ylabel('predicted')
        plt.xlabel('label')
        plt.savefig(os.path.join(PATH, 'results.png'))
        plt.show()


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
                }, os.path.join(PATH, 'models\\cl_model_hengs_tune.pth'))
                best_acc = val_acc
        # self.confusion_matrix(predicted, all_labels)


    def main(self):
        self.model = model_network.Resnet34_classification()
        checkpoint = torch.load('.\\00007500_model.pth')
        # self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.load_state_dict(checkpoint)
        self.model = self.model.to('cuda:0')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=10, verbose=True)

        data_set = SteelDataset()
        validation_split = .1
        shuffle_dataset = True
        random_seed = 42
        dataset_size = len(data_set)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        if shuffle_dataset:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        self.train_loader = torch.utils.data.DataLoader(data_set, sampler=train_sampler, batch_size=self.batch_size, shuffle=False)
        self.val_loader = torch.utils.data.DataLoader(data_set, sampler=valid_sampler, batch_size=self.batch_size, shuffle=False)
        self.training()


if __name__ == '__main__':
    model = Train_Model()
    model.main()
    
