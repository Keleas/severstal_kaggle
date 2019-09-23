import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from torch.utils.data.sampler import RandomSampler

from src.utils import rle_decode, rle_encode
from src.create_data import SteelDatabase, getDatabase


if __name__ == '__main__':

    num_train = len(os.listdir('input/severstal-steel-defect-detection/train_images'))
    # indices = list(range(num_train))
    indices = list(range(100))
    train_data = getDatabase(mode='train', image_idx=indices)
    train_loader = DataLoader(train_data,
                              shuffle=RandomSampler(train_data),
                              batch_size=30,
                              num_workers=6,
                              pin_memory=True)


    def show_mask_image(imgs, masks):
        for img, mask in zip(imgs, masks):
            img = img.reshape((256, 1600, 3)).astype(np.uint8)
            mask = mask.reshape((256, 1600, 4)).astype(np.uint8) * 255
            palet = [(249, 192, 12), (0, 185, 241), (114, 0, 218), (249, 50, 12)]
            fig, ax = plt.subplots(figsize=(15, 15))
            for ch in range(4):
                contours, _ = cv2.findContours(mask[:, :, ch], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                for i in range(0, len(contours)):
                    cv2.polylines(img, contours[i], True, palet[ch], 2)
            ax.imshow(img)
            plt.show()


    for imgs, masks, target in train_loader:
        show_mask_image(imgs.numpy(), masks.numpy())
