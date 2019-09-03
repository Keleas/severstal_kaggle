from torch.utils.data import Dataset
import pandas as pd
import os
import torch
from skimage import io
import numpy as np
from torchvision.transforms import ToTensor, Resize
import cv2
import random
from matplotlib import pyplot as plt
import albumentations as albu

import preprocessing
CROP_LEN = 800

class SteelDataset(Dataset):

    def __init__(self, mode):
        self.mode = mode
        self.crop_im, self.images, self.files_names = preprocessing.images(mode)
        self.labels_dict, self.crop_mask = preprocessing.labels(mode)


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        aug = albu.Compose([
            albu.Flip(p=0.3),
            albu.Rotate(p=0.9),
            albu.Blur(p=0.4)
        ], p=0.9)
        # image = self.crop_im[idx]
        # segment = self.crop_mask[idx]
        image = self.images[idx]

        # mask = self.masks[idx]
        segment = self.labels_dict[self.files_names[idx]]['segment']
        segment = preprocessing.rleToMask(segment).reshape((4, 256, 1600))

        # mask = np.transpose(mask, (2, 1, 0))
        # if image.shape != (200, 256, 4):
        #     image = cv2.resize(image, dsize=(200, 256))
        #     mask = cv2.resize(mask, dsize=(200, 256))

        # image = preprocessing.one_augment(aug, image)

        # segment = np.transpose(segment, (1, 2, 0))
        image = ToTensor()(image).float()
        # segment = ToTensor()(segment).float()

        # mask = ToTensor()(mask).float()
        return (image, segment)
