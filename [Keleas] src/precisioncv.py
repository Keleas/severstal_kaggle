from tqdm import tqdm
import os
import argparse
import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.create_data import getDatabase

parser = argparse.ArgumentParser(description='')
parser.add_argument('--model', default='res34v5', type=str, help='Model version')
parser.add_argument('--fine_size', default=101, type=int, help='Resized image size')
parser.add_argument('--pad_left', default=13, type=int, help='Left padding size')
parser.add_argument('--pad_right', default=14, type=int, help='Right padding size')
parser.add_argument('--batch_size', default=18, type=int, help='Batch size for training')
parser.add_argument('--epoch', default=300, type=int, help='Number of training epochs')
parser.add_argument('--snapshot', default=5, type=int, help='Number of snapshots per fold')
parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
parser.add_argument('--save_weight', default='weights/', type=str, help='weight save space')
parser.add_argument('--save_pred', default='fold_predictions/', type=str, help='prediction save space')
parser.add_argument('--max_lr', default=0.01, type=float, help='max learning rate')
parser.add_argument('--min_lr', default=0.001, type=float, help='min learning rate')
parser.add_argument('--fold', default='0', type=str, help='number of split fold')
parser.add_argument('--start_snap', default=0, type=int)
parser.add_argument('--end_snap', default=3, type=int)
parser.add_argument('--test_folder', default='/test_data/', type=str, help='path to the folder with test images')

args = parser.parse_args()
fine_size = args.fine_size + args.pad_left + args.pad_right
args.weight_name = 'model_' + str(fine_size) + '_' + args.model
args.save_pred += args.model + '_'

device = torch.device('cuda' if args.cuda else 'cpu')

num_test = len(os.listdir('input/severstal-steel-defect-detection/test_images'))
test_indices = list(range(num_test))

if __name__ == '__main__':
    # Load test data
    # image_test = testImageFetch(test_id)
    #
    # overall_pred = np.zeros((num_test, 256, 1600), dtype=np.float32)

    # Get model
    model = smp.Unet(args.model, classes=4, encoder_weights='imagenet')
    salt = model.to(device)

    # Start prediction
    for step in range(args.start_snap, args.end_snap + 1):
        print('Predicting Snapshot', step)
        pred_null = []
        pred_flip = []
        # Load weight
        param = torch.load(args.save_weight + args.weight_name + args.fold + str(step) + '.pth')
        salt.load_state_dict(param)

        # Create DataLoader
        test_data = getDatabase(mode='test', image_idx=test_indices)
        test_loader = DataLoader(test_data,
                                 shuffle=False,
                                 batch_size=args.batch_size,
                                 num_workers=5,
                                 pin_memory=True)

        # Prediction with no TTA test data
        salt.eval()
        for images in tqdm(test_loader, total=len(test_loader)):
            images = images.to(device)
            with torch.set_grad_enabled(False):
                if args.model == 'res34v3':
                    pred, _, _ = salt(images)
                else:
                    pred = salt(images)
                pred = F.sigmoid(pred).squeeze(1).cpu().numpy()
            pred_null.append(pred)

        # Prediction with horizontal flip TTA test data
        test_data = getDatabase(mode='test', image_idx=test_indices, is_TTA=True)
        test_loader = DataLoader(test_data,
                                 shuffle=False,
                                 batch_size=args.batch_size,
                                 num_workers=5,
                                 pin_memory=True)

        salt.eval()
        for images in tqdm(test_loader, total=len(test_loader)):
            images = images.to(device)
            with torch.set_grad_enabled(False):
                if args.model == 'res34v3':
                    pred, _, _ = salt(images)
                else:
                    pred = salt(images)
                pred = F.sigmoid(pred).squeeze(1).cpu().numpy()
            for idx in range(len(pred)):
                pred[idx] = cv2.flip(pred[idx], 1)
            pred_flip.append(pred)

        pred_null = np.concatenate(pred_null).reshape(-1, args.fine_size, args.fine_size)
        pred_flip = np.concatenate(pred_flip).reshape(-1, args.fine_size, args.fine_size)
        overall_pred += (pred_null + pred_flip) / 2

    overall_pred /= (args.end_snap - args.start_snap + 1)

    # Save prediction
    if args.fine_size != 101:
        overall_pred_101 = np.zeros((len(test_id), 101, 101), dtype=np.float32)
        for idx in range(len(test_id)):
            overall_pred_101[idx] = cv2.resize(overall_pred[idx], dsize=(101, 101))
        np.save(args.save_pred + 'pred' + args.fold, overall_pred_101)
    else:
        np.save(args.save_pred + 'pred' + args.fold, overall_pred)