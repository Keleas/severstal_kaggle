import numpy as np
from torchvision import models
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
import torch.optim as optim
import os
import re
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import itertools
import math
import numpy as np
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
from collections import Counter

def labels(mode):
    files = []
    labels = []
    count_def = 0
    with open(os.path.join('G:\\SteelDetection', mode + '.csv')) as csv_file:
        next(csv_file)
        for num, line in enumerate(csv_file, 1):
            line_data = re.split(r'[_,\n]', line)
            file_name = line_data[0]

            if file_name not in files:
                files.append(file_name)
            if not not line_data[2]:
                count_def += 1
            if (num - 1) % 4 == 0:
                if count_def > 0:
                    labels.append(1)
                else:
                    labels.append(0)
                count_def = 0
    return files, labels



class SteelDataset(Dataset):
    def __init__(self):
        self.mode = 'train'
        self.files, self.labels = labels(self.mode)


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        file_path = os.path.join(('G:\\SteelDetection\\' + self.mode + '_images'), file)
        image = Image.open(file_path)
        image = transforms.ToTensor()(image).float()
        label = self.labels[idx]
        return (image, label)



class Train_Model(object):
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.scheduler = None

        self.train_loader = None
        self.val_loader = None
        self.train_data = None
        self.val_data = None

        self.num_epochs = 100
        self.num_classes = 2
        self.batch_size = 8
        self.learning_rate = 0.01

    def train(self):
        current_loss, current_acc = 0, 0
        self.model.train()
        for (image, label) in tqdm(self.train_loader):
            image = image.cuda()
            label = label.cuda()
            self.optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = self.model(image)
                loss = nn.CrossEntropyLoss()(outputs, label)
                loss.backward()
                self.optimizer.step()

                # print('gradients =', [x.grad.data for x in model.parameters()])
                _, predictions = torch.max(outputs, 1)
                # print('weights after backpropagation = ', list(model.parameters()))
            current_loss += loss.item() * image.size(0)
            current_acc += torch.sum(predictions == label.data)
        total_loss = current_loss / len(self.train_loader.dataset)
        total_acc = 100 * current_acc.double() / len(self.train_loader.dataset)
        print('TRAIN: Loss: {}, Accuracy: {}%'.format(total_loss, total_acc))
        return total_loss


    def validation(self):
        losses, all_predicted, all_labels = [], [], []
        total, correct = 0, 0
        self.model.eval()
        for (image, label) in tqdm(self.val_loader):
            image = image.cuda()
            label = label.cuda()
            with torch.set_grad_enabled(False):
                outputs = self.model(image)
                loss = nn.CrossEntropyLoss()(outputs, label)

                losses.append(loss.item())
                total += label.size(0)
                _, predicted = torch.max(outputs.data, 1)
                all_predicted.append(predicted.cpu().numpy())
                all_labels.append(label.cpu().numpy())
                correct += (predicted == label).sum().item()
        mean_loss = np.mean(losses)
        print('VALIDATION: Loss: {}, Accuracy: {}%'.format(mean_loss, (correct/total) * 100))
        return mean_loss, list(itertools.chain.from_iterable(all_predicted)), list(itertools.chain.from_iterable(all_labels))

    def training(self):
        for epoch in range(self.num_epochs):
            print("Epoch:[{}/{}]".format(epoch + 1, self.num_epochs))
            self.train()
            val_loss, predicted, labels = self.validation()
            self.scheduler.step(val_loss)
        torch.save(self.model.state_dict(), 'G:\\SteelDetection\\new_model.pth')

    def main(self):
        self.model = models.resnet18()
        self.model = self.model.to('cuda:0')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", patience=2, verbose=True)

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

        self.train_loader = torch.utils.data.DataLoader(data_set, sampler=train_sampler, batch_size=self.batch_size,
                                                   shuffle=False)
        self.val_loader = torch.utils.data.DataLoader(data_set, sampler=valid_sampler, batch_size=self.batch_size, shuffle=False)
        self.training()


if __name__ == '__main__':
    model = Train_Model()
    model.main()
    # checkpoint = torch.load('G:\\SteelDetection\\new_model.pth')
    # model.load_state_dict(checkpoint)




