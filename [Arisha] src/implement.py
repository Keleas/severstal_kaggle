import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import models
from tqdm import tqdm
import segmentation_models_pytorch as smp

import dataset
import preprocessing
import training
import visualisation


from matplotlib import pyplot as plt
batch_size = 4

if __name__ == '__main__':

    data_set = dataset.SteelDataset(mode='train')

    test_set = dataset.SteelDataset(mode='test')
    validation_split = .2
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

    train_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                              shuffle=False)

    model = smp.Unet('resnet18', encoder_weights='imagenet', classes=4)
    model = model.to('cuda:0')
    # criterion = nn.BCEWithLogitsLoss()
    train_accuracy, val_predicted, val_labels, test_predicted, test_labels = \
        training.training(train_loader, val_loader, test_loader, model)
