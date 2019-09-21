import data_set
import numpy as np
from torchvision import models
import torch
import torch.nn as nn
import train
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
import torch.optim as optim
import train

batch_size = 8


if __name__ == '__main__':
    data_set = data_set.SteelDataset()
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

    train_loader = torch.utils.data.DataLoader(data_set, sampler=train_sampler, batch_size=batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(data_set, sampler=valid_sampler, batch_size=batch_size, shuffle=False)
    model = models.resnet18()
    model = model.to('cuda:0')

    # checkpoint = torch.load('G:\\SteelDetection\\new_model.pth')
    # model.load_state_dict(checkpoint)

    train.training(train_loader, val_loader, model)



