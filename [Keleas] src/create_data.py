import os
import gc
import cv2
import time
import math
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from sklearn.model_selection import train_test_split
from albumentations import (
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
from sklearn.model_selection import KFold

from src.utils import rle_decode, rle_encode
from src.transform import *

import warnings
warnings.filterwarnings("ignore")


class SteelDatabase(Dataset):
    def __init__(self, image_list, df, mode, is_tta=False, fine_size=(256, 1600, 3)):
        self.image_list = image_list
        self.df = df
        self.mode = mode
        self.is_tta = is_tta
        self.root = 'input/severstal-steel-defect-detection/'
        self.fine_size = fine_size

        self.test_idx = os.listdir(self.root + 'test_images')

    def __len__(self):
        if self.mode != 'test':
            return len(self.image_list)
        else:
            return len(os.listdir('input/severstal-steel-defect-detection/test_images'))

    def __getitem__(self, idx):
        if self.mode == 'train':
            image_id = self.image_list[idx]
            image_id, mask, target = rle_encode(image_id, self.df)
            image_path = os.path.join(self.root + 'train_images', image_id)
            image = cv2.imread(image_path)

            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
            train_aug = Compose([
                # PadIfNeeded(min_height=256, min_width=1600, p=1),
                VerticalFlip(p=0.5),
                HorizontalFlip(p=0.5),
                Normalize(mean=mean, std=std, p=1),
                ToTensor()])

            augmented = train_aug(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            mask = mask[0].permute(2, 0, 1)

            return image, mask, target

        elif self.mode == 'val':
            image_id = self.image_list[idx]
            image_id, mask, target = rle_encode(image_id, self.df)
            image_path = os.path.join(self.root + 'train_images', image_id)
            image = cv2.imread(image_path)

            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
            test_aug = Compose([
                Normalize(mean=mean, std=std, p=1),
                ToTensor()])

            augmented = test_aug(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            mask = mask[0].permute(2, 0, 1)

            return image, mask, target

        elif self.mode == 'test':
            image_id = self.test_idx[idx]
            image_path = os.path.join(self.root, "test_images", image_id)
            image = cv2.imread(image_path)

            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
            if self.is_tta:
                test_aug = Compose([
                    Normalize(mean=mean, std=std, p=1),
                    HorizontalFlip(p=0.5),
                    ToTensor()])
            else:
                test_aug = Compose([
                    Normalize(mean=mean, std=std, p=1),
                    ToTensor()])

            augmented = test_aug(image=image)
            image = augmented['image']

            return image


def getDatabase(mode, image_idx):
    df = pd.read_csv('input/severstal-steel-defect-detection/train.csv')
    df['ImageId'], df['ClassId'] = zip(*df['ImageId_ClassId'].str.split('_'))
    df['ClassId'] = df['ClassId'].astype(int)
    df = df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')
    df['defects'] = df.count(axis=1)

    return SteelDatabase(image_list=image_idx, df=df, mode=mode)


if __name__ == '__main__':

    num_fold = 5
    num_train = len(os.listdir('input/severstal-steel-defect-detection/train_images'))
    num_train = 40
    indices = list(range(num_train))
    kf = KFold(n_splits=num_fold, random_state=42, shuffle=True)

    train_idx = []
    valid_idx = []
    for t, v in kf.split(indices):
        train_idx.append(t)
        valid_idx.append(v)

    print(train_idx[0])
    print(valid_idx[0])

    for fold in range(num_fold):
        train_data = getDatabase(mode='train', image_idx=train_idx[fold])
        train_loader = DataLoader(train_data,
                                  shuffle=RandomSampler(train_data),
                                  batch_size=30,
                                  num_workers=4,
                                  pin_memory=True)

        val_data = getDatabase(mode='val', image_idx=valid_idx[fold])
        val_loader = DataLoader(val_data,
                                shuffle=False,
                                batch_size=30,
                                num_workers=4,
                                pin_memory=True)

        test_data = getDatabase(mode='test', image_idx=None)
        test_loader = DataLoader(test_data,
                                 shuffle=False,
                                 batch_size=30,
                                 num_workers=4,
                                 pin_memory=True)

    import torch.nn.functional as F
    image, mask, target = next(iter(train_loader))
    out_f = F.softmax(mask, dim=1)
    from src.losses import lovasz_softmax
    from src.utils import do_kaggle_metric
    loss = lovasz_softmax(out_f.squeeze(1), target.squeeze(1))
    precision, _, _ = do_kaggle_metric(mask.cpu().numpy(), mask.cpu().numpy(), 0.5)
    precision = precision.mean()
    print(f'lovasz: {loss}, kaggle_metric: {precision}')
    print(out_f.cpu().numpy().shape, mask.cpu().numpy().shape)

    print(F.sigmoid(mask).cpu().numpy().shape, F.softmax(mask, dim=1).cpu().numpy().shape)





