import os
import cv2
import time
import argparse
import numpy as np
from tqdm import tqdm
import segmentation_models_pytorch as smp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.create_data import SteelDatabase

parser = argparse.ArgumentParser(description='')
parser.add_argument('--model', default='resnet34', type=str, help='Model version')
parser.add_argument('--fine_size', default=96, type=int, help='Resized image size')
parser.add_argument('--batch_size', default=18, type=int, help='Batch size for training')
parser.add_argument('--snapshot', default=5, type=int, help='Number of snapshots per fold')
parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
parser.add_argument('--save_weight', default='output/weights/', type=str, help='weight save space')
parser.add_argument('--save_pred', default='output/fold_predictions/', type=str, help='prediction save space')
parser.add_argument('--fold', default='0', type=str, help='number of split fold')
parser.add_argument('--start_snap', default=0, type=int)
parser.add_argument('--end_snap', default=3, type=int)
parser.add_argument('--test_folder', default='/test_data/', type=str, help='path to the folder with test images')

args = parser.parse_args()
fine_size = args.fine_size
args.weight_name = 'model_' + str(fine_size) + '_' + args.model
args.save_pred += args.model + '_'

device = torch.device('cuda' if args.cuda else 'cpu')

num_test = len(os.listdir('input/severstal-steel-defect-detection/test_images'))
test_indices = list(range(num_test))

if __name__ == '__main__':
    # Load test data
    overall_pred = np.zeros((num_test, 4, 256, 1600), dtype=np.float32)

    # Get model
    model = smp.Unet(args.model, classes=4, encoder_weights='imagenet')
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model = model.to(device)

    # Start prediction
    for step in range(args.start_snap, args.end_snap + 1):
        print('Predicting Snapshot', step)
        pred_null = []
        pred_flip = []
        # Load weight
        param = torch.load(args.save_weight + args.weight_name + '_f' + args.fold + '_s' + str(step) + '.pth')
        model.load_state_dict(param)

        # Create DataLoader
        test_data = SteelDatabase(mode='test', is_tta=False)
        test_loader = DataLoader(test_data,
                                 shuffle=False,
                                 batch_size=args.batch_size,
                                 num_workers=5,
                                 pin_memory=True)

        # Prediction with no TTA test data
        model.eval()
        for batch in tqdm(test_loader, total=len(test_loader)):
            fnames, images = batch
            images = images.to(device)
            with torch.set_grad_enabled(False):
                pred = model(images)
                pred = F.sigmoid(pred).squeeze(1).cpu().numpy()
            pred_null.append(pred)

        # Prediction with horizontal flip TTA test data
        test_data = SteelDatabase(mode='test', is_tta=True)
        test_loader = DataLoader(test_data,
                                 shuffle=False,
                                 batch_size=args.batch_size,
                                 num_workers=5,
                                 pin_memory=True)

        model.eval()
        for batch in tqdm(test_loader, total=len(test_loader)):
            fnames, images = batch
            images = images.to(device)
            with torch.set_grad_enabled(False):
                pred = model(images)
                pred = F.sigmoid(pred).squeeze(1).cpu().numpy()
            for idx in range(len(pred)):
                reshape_pred = pred[idx].reshape((256, 1600, 4))
                reshape_pred = cv2.flip(reshape_pred, 0)
                pred[idx] = reshape_pred.reshape((4, 256, 1600))
            pred_flip.append(pred)

        start_time = time.time()
        pred_null = np.concatenate(pred_null)
        pred_flip = np.concatenate(pred_flip)
        overall_pred += (pred_null + pred_flip) / 2
        print(f'Concat reserved {(time.time()-start_time):.2f}[s]')
    overall_pred /= (args.end_snap - args.start_snap + 1)

    np.save(args.save_pred + 'pred' + args.fold, overall_pred)
