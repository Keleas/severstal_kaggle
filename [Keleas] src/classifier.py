import numpy as np
import pandas as pd
import cv2
from torchvision import models
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import os
import re
from torch.utils.data import Dataset
from torchvision import transforms
import itertools
import torch
from tqdm import tqdm
from albumentations.pytorch.transforms import ToTensor
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
from torch.nn import functional as F
BatchNorm2d = nn.BatchNorm2d

###############################################################################
class ConvBn2d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1, stride=1):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.bn   = nn.BatchNorm2d(out_channel, eps=1e-5)

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        return x




#############  resnext50 pyramid feature net #######################################
# https://github.com/Hsuxu/ResNeXt/blob/master/models.py
# https://github.com/D-X-Y/ResNeXt-DenseNet/blob/master/models/resnext.py
# https://github.com/miraclewkf/ResNeXt-PyTorch/blob/master/resnext.py


# bottleneck type C
class BasicBlock(nn.Module):
    def __init__(self, in_channel, channel, out_channel, stride=1, is_shortcut=False):
        super(BasicBlock, self).__init__()
        self.is_shortcut = is_shortcut

        self.conv_bn1 = ConvBn2d(in_channel,    channel, kernel_size=3, padding=1, stride=stride)
        self.conv_bn2 = ConvBn2d(   channel,out_channel, kernel_size=3, padding=1, stride=1)

        if is_shortcut:
            self.shortcut = ConvBn2d(in_channel, out_channel, kernel_size=1, padding=0, stride=stride)


    def forward(self, x):
        z = F.relu(self.conv_bn1(x),inplace=True)
        z = self.conv_bn2(z)

        if self.is_shortcut:
            x = self.shortcut(x)

        z += x
        z = F.relu(z,inplace=True)
        return z




class ResNet34(nn.Module):

    def __init__(self, num_class=1000 ):
        super(ResNet34, self).__init__()


        self.block0  = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2, bias=False),
            BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.block1  = nn.Sequential(
             nn.MaxPool2d(kernel_size=3, padding=1, stride=2),
             BasicBlock( 64, 64, 64, stride=1, is_shortcut=False,),
          * [BasicBlock( 64, 64, 64, stride=1, is_shortcut=False,) for i in range(1,3)],
        )
        self.block2  = nn.Sequential(
             BasicBlock( 64,128,128, stride=2, is_shortcut=True, ),
          * [BasicBlock(128,128,128, stride=1, is_shortcut=False,) for i in range(1,4)],
        )
        self.block3  = nn.Sequential(
             BasicBlock(128,256,256, stride=2, is_shortcut=True, ),
          * [BasicBlock(256,256,256, stride=1, is_shortcut=False,) for i in range(1,6)],
        )
        self.block4 = nn.Sequential(
             BasicBlock(256,512,512, stride=2, is_shortcut=True, ),
          * [BasicBlock(512,512,512, stride=1, is_shortcut=False,) for i in range(1,3)],
        )
        self.logit = nn.Linear(512,num_class)



    def forward(self, x):
        batch_size = len(x)

        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = F.adaptive_avg_pool2d(x,1).reshape(batch_size,-1)
        logit = self.logit(x)
        return logit

class Resnet34_classification(nn.Module):
    def __init__(self,num_class=4):
        super(Resnet34_classification, self).__init__()
        e = ResNet34()
        self.block = nn.ModuleList([
            e.block0,
            e.block1,
            e.block2,
            e.block3,
            e.block4,
        ])
        e = None  #dropped
        self.feature = nn.Conv2d(512,32, kernel_size=1) #dummy conv for dim reduction
        self.logit = nn.Conv2d(32,num_class, kernel_size=1)

    def forward(self, x):
        batch_size,C,H,W = x.shape

        for i in range( len(self.block)):
            x = self.block[i](x)
            #print(i, x.shape)

        x = F.dropout(x,0.5,training=self.training)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.feature(x)
        logit = self.logit(x)
        return logit


class SteelDataset(Dataset):
    def __init__(self):
        self.root = 'input/severstal-steel-defect-detection/'

        def get_df():
            train_df = pd.read_csv('input/severstal-steel-defect-detection/train.csv')
            labels = []
            for i in range(len(train_df)):
                if type(train_df.EncodedPixels[i]) == str:
                    labels.append(1)
                else:
                    labels.append(0)
            labels = np.array(labels)
            labels = labels.reshape((int(len(train_df) / 4), 4))

            images_df = pd.DataFrame(train_df.iloc[::4, :].ImageId_ClassId.str[:-2].reset_index(drop=True))
            labels_df = pd.DataFrame(labels)
            proc_train_df = pd.concat((images_df, labels_df), 1)

            return proc_train_df
        self.train_df = get_df()

    def __len__(self):
        return self.train_df.shape[0]

    def __getitem__(self, idx):
        file = self.train_df.iloc[idx].values[0]
        file_path = os.path.join((self.root + 'train_images'), file)
        image = cv2.imread(file_path)

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        train_aug = Compose([
            # PadIfNeeded(min_height=256, min_width=1600, p=1),
            VerticalFlip(p=0.5),
            HorizontalFlip(p=0.5),
            Normalize(mean=mean, std=std, p=1),
            ToTensor()])

        augmented = train_aug(image=image)
        image = augmented['image']
        label = torch.tensor(np.array(self.train_df.iloc[idx].values[1:], dtype=np.float32))
        return image, label


class Train_Model(object):
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.scheduler = None

        self.train_loader = None
        self.val_loader = None
        self.train_data = None
        self.val_data = None

        self.save_path = 'output/weights'

        self.num_epochs = 100
        self.batch_size = 16
        self.learning_rate = 0.001
        self.weight_decay = 1e-3

    def train(self):
        current_loss, current_acc = 0, 0

        self.model.train()
        for image, label in tqdm(self.train_loader, total=len(self.train_loader), ascii=True, desc='train'):
            image = image.cuda()
            label = label.cuda()
            self.optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = self.model(image)
                batch_size, num_class, H, W = outputs.shape
                outputs = outputs.view(batch_size, num_class)
                label = label.view(batch_size, num_class)
                loss = F.binary_cross_entropy_with_logits(outputs, label, reduction='none')
                loss = loss.mean()

                loss.backward()
                self.optimizer.step()

                # print('gradients =', [x.grad.data for x in model.parameters()])
                # _, predictions = torch.max(outputs, 1)
                # print('weights after backpropagation = ', list(model.parameters()))
            current_loss += loss.item() * image.size(0)
            current_acc += torch.sum(outputs.data == label.data)
        total_loss = current_loss / len(self.train_loader.dataset)
        total_acc = 100 * current_acc.double() / len(self.train_loader.dataset)
        print('TRAIN: Loss: {}, Accuracy: {}%'.format(total_loss, total_acc))
        return total_loss

    def validation(self):
        losses, all_predicted, all_labels = [], [], []
        total, correct = 0, 0

        self.model.eval()
        for (image, label) in tqdm(self.val_loader, total=len(self.val_loader), ascii=True, desc='validation'):
            image = image.cuda()
            label = label.cuda()
            with torch.set_grad_enabled(False):
                outputs = self.model(image)
                batch_size, num_class, H, W = outputs.shape
                outputs = outputs.view(batch_size, num_class)
                label = label.view(batch_size, num_class)
                loss = F.binary_cross_entropy_with_logits(outputs, label, reduction='none')
                loss = loss.mean()

                losses.append(loss.item())
                total += label.size(0)
                all_predicted.append(outputs.cpu().numpy())
                all_labels.append(label.cpu().numpy())
                correct += (outputs == label).sum().item()
        mean_loss = np.mean(losses)
        print('VALIDATION: Loss: '
              '{}, Accuracy: {}%'.format(mean_loss, (correct/total) * 100))
        return mean_loss, list(itertools.chain.from_iterable(all_predicted)), list(itertools.chain.from_iterable(all_labels))

    def training(self):
        for epoch in range(self.num_epochs):
            print("Epoch:[{}/{}]".format(epoch + 1, self.num_epochs))
            self.train()
            val_loss, predicted, labels = self.validation()
            self.scheduler.step(val_loss)
        torch.save(self.model.state_dict(), self.save_path + '/cl2_resnet34.pth')

    def main(self):
        self.model = Resnet34_classification()
        self.model = self.model.to('cuda:0')
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.learning_rate,
                                          weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", patience=2, verbose=True)

        data_set = SteelDataset()
        validation_split = 0.2
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

        self.train_loader = torch.utils.data.DataLoader(data_set,
                                                        sampler=train_sampler,
                                                        batch_size=self.batch_size,
                                                        num_workers=6,
                                                        pin_memory=True,
                                                        shuffle=False)

        self.val_loader = torch.utils.data.DataLoader(data_set,
                                                      sampler=valid_sampler,
                                                      batch_size=self.batch_size,
                                                      num_workers=6,
                                                      pin_memory=True,
                                                      shuffle=False)
        self.training()


if __name__ == '__main__':
    model = Train_Model()
    model.main()
