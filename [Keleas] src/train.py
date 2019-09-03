import os
import gc
import time
import argparse
import itertools
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from sklearn.model_selection import KFold
from torch.utils.data.sampler import RandomSampler
from sklearn.model_selection import train_test_split

from src.logger import Logger
from src.utils import do_kaggle_metric
from src.create_data import getDatabase
from src.torchutils import EarlyStopping, AdamW, CyclicLRWithRestarts
from src.losses import lovasz_softmax, jaccard_loss, dice_loss, dice_channel_torch


class TrainModel(object):
    def __init__(self):
        self.logger = Logger('logs/')

        # model
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None

        # data
        self.train_loader = None
        self.val_loader = None
        self.train_data = None
        self.val_data = None

    def val_step(self):
        """ Validation step """
        cum_loss = 0
        # predicts = [np.zeros((args.batch_size, 4, 256, 1600))] * len(self.val_loader)
        # truths = [np.zeros((args.batch_size, 4, 256, 1600))] * len(self.val_loader)
        predicts = []
        truths = []

        self.model.eval()
        for inputs, masks, target in tqdm(self.val_loader, total=len(self.val_loader), ascii=True, desc='validation'):
            inputs, masks, target = inputs.to(device), masks.to(device), target.to(device)
            with torch.set_grad_enabled(False):
                out = self.model(inputs)
                loss = nn.BCEWithLogitsLoss()(out, masks)
                # loss1 = jaccard_loss(masks, out)
                # loss = lovasz_softmax(F.softmax(out, dim=1).squeeze(1), target.squeeze(1))  # tune
                # loss = loss1 + loss2

            predicts.append(F.sigmoid(out).detach().cpu().numpy())
            truths.append(masks.detach().cpu().numpy())

            cum_loss += loss.item() * inputs.size(0)
            gc.collect()

        start = time.time()
        predicts = np.concatenate(predicts).squeeze()
        truths = np.concatenate(truths).squeeze()
        precision, _, _ = do_kaggle_metric(predicts, truths, 0.5)
        precision = precision.mean()
        mean_dice = dice_channel_torch(predicts, truths, 0.5)
        val_loss = cum_loss / self.val_data.__len__()
        print(f"Val calculated: {(time.time() - start):.3f}s")
        return val_loss, precision, mean_dice

    def train_step(self):
        """ Training step """
        cum_loss = 0
        self.model.train()
        for inputs, masks, target in tqdm(self.train_loader, total=len(self.train_loader), ascii=True, desc='train'):
            inputs, masks, target = inputs.to(device), masks.to(device), target.to(device)
            self.optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                out = self.model(inputs)
                loss = nn.BCEWithLogitsLoss()(out, masks)
                # loss1 = jaccard_loss(masks, out)
                # loss2 = lovasz_softmax(F.softmax(out, dim=1).squeeze(1), target.squeeze(1))  # tune
                # loss = loss1 + loss2

                loss.backward()
                self.optimizer.step()

            cum_loss += loss.item() * inputs.size(0)

        epoch_loss = cum_loss / self.train_data.__len__()
        return epoch_loss

    def logger_step(self, cur_epoch, losses_train, losses_val, accuracy, dice):
        """ Log information """
        print(f"[Epoch {cur_epoch}] training loss: {losses_train[-1]:.6f} | val_loss: {losses_val[-1]:.6f} | "
              f"val_acc: {accuracy:.6f}, dice: {dice:.6f}")
        print(f"Learning rate: {self.lr_scheduler.get_lr()[0]:.6f}")

        # 1. Log scalar values (scalar summary)
        info = {'loss': losses_train[-1],
                'val_loss': losses_val[-1],
                'val_acc': accuracy.item(),
                'dice': dice}

        for tag, value in info.items():
            self.logger.scalar_summary(tag, value, cur_epoch + 1)

        # 2. Log values and gradients of the parameters (histogram summary)
        for tag, value in self.model.named_parameters():
            tag = tag.replace('.', '/')
            self.logger.histo_summary(tag, value.data.cpu().numpy(), cur_epoch + 1)
            self.logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), cur_epoch + 1)

        # 3. Log training images (image summary)
        # self.model.eval()
        # inputs, labels = next(iter(self.val_loader))
        # inputs, labels = inputs.to(device), labels.to(device)
        # with torch.set_grad_enabled(False):
        #     out = self.model(inputs)
        #
        # _, y_pred = torch.max(out, 1)
        # info = {'images': (inputs.view(-1, 256, 1600)[:10].cpu().numpy(), y_pred.data.tolist(), labels.data.tolist())}
        #
        # for tag, images in info.items():
        #     self.logger.image_summary(tag, images, cur_epoch + 1)

        return True

    def main(self):
        """ Main training loop """
        # Get Model
        self.model = smp.Unet(args.model, classes=4, encoder_weights='imagenet')
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        self.model.to(device)
        scheduler_step = args.epoch // args.snapshot

        num_train = len(os.listdir('input/severstal-steel-defect-detection/train_images'))
        # num_train = 1000
        indices = list(range(num_train))

        args.num_fold = 10
        if args.num_fold > 1:
            kf = KFold(n_splits=args.num_fold, random_state=42, shuffle=True)
            train_idx = []
            valid_idx = []
            for t, v in kf.split(indices):
                train_idx.append(t)
                valid_idx.append(v)
        elif args.num_fold == 1:
            train_idx, valid_idx, _, _ = train_test_split(indices, indices, test_size=0.2, random_state=42)
            train_idx, valid_idx = [train_idx], [valid_idx]
        else:
            raise Exception('Invalid number of args.num_fold')

        for fold in range(args.num_fold):
            print(f'************************'
                  f'**** [FOLD: {fold}] ****'
                  f'************************')
            self.train_data = getDatabase(mode='train', image_idx=train_idx[fold])
            self.train_loader = DataLoader(self.train_data,
                                           shuffle=RandomSampler(self.train_data),
                                           batch_size=args.batch_size,
                                           num_workers=5,
                                           pin_memory=True)
            self.val_data = getDatabase(mode='val', image_idx=valid_idx[fold])
            self.val_loader = DataLoader(self.val_data,
                                         shuffle=False,
                                         batch_size=args.batch_size,
                                         num_workers=5,
                                         pin_memory=True)

            num_snapshot = 0
            best_acc = 0
            # Setup optimizer
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.max_lr, momentum=args.momentum,
                                             weight_decay=args.weight_decay)
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                           scheduler_step, args.min_lr)
            early_stopping = EarlyStopping(patience=args.patience, verbose=True)

            # Service variables
            losses_train = []  # save training losses
            losses_val = []  # save validation losses

            for epoch in range(args.epoch):
                train_loss = self.train_step()
                # train_loss = 1
                val_loss, accuracy, dice = self.val_step()
                self.lr_scheduler.step()

                losses_train.append(train_loss)
                losses_val.append(val_loss)

                self.logger_step(epoch, losses_train, losses_val, accuracy, dice)

                # scheduler checkpoint
                if accuracy >= best_acc:
                    best_acc = accuracy
                    best_param = self.model.state_dict()

                if (epoch + 1) % scheduler_step == 0:
                    torch.save(best_param, args.save_weight + args.weight_name +
                               '_f' + str(fold) + '_s' + str(num_snapshot) + '.pth')

                    self.optimizer = torch.optim.SGD(self.model.parameters(),
                                                     lr=args.max_lr,
                                                     momentum=args.momentum,
                                                     weight_decay=args.weight_decay)
                    self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                                   scheduler_step,
                                                                                   args.min_lr)
                    num_snapshot += 1
                    best_acc = 0

                # If the model could not get better, then stop training
                early_stopping(losses_val[-1], self.model)

                if early_stopping.early_stop:
                    print("Early stopping")
                    print(f"Early stop at {epoch} epoch")
                    return

        return True


parser = argparse.ArgumentParser(description='')
parser.add_argument('--model', default='resnet18', type=str, help='Model version')
parser.add_argument('--fine_size', default=96, type=int, help='Resized image size')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
parser.add_argument('--epoch', default=100, type=int, help='Number of training epochs')
parser.add_argument('--num_fold', default=1, type=int, help='Number of folds')
parser.add_argument('--snapshot', default=20, type=int, help='Number of snapshots per fold')
parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
parser.add_argument('--save_weight', default='output/weights/', type=str, help='weight save space')
parser.add_argument('--max_lr', default=0.01, type=float, help='max learning rate')
parser.add_argument('--min_lr', default=0.001, type=float, help='min learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for SGD')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--patience', default=40, type=int, help='Number of epoch waiting for best score')

args = parser.parse_args()
fine_size = args.fine_size
args.weight_name = 'model_' + str(fine_size) + '_' + args.model

if not os.path.isdir(args.save_weight):
    os.mkdir(args.save_weight)

device = torch.device('cuda' if args.cuda else 'cpu')


if __name__ == '__main__':
    # ================================================================== #
    #                        Tensorboard Logging                         #
    # ================================================================== #
    # tensorboard --logdir=D:\Github\Wafer_maps\logs --port=6006
    # http://localhost:6006/

    model = TrainModel()
    model.main()
