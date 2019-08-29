import cv2
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from src.utils import rle_decode
from src.create_data import getDatabase


def post_process(probability, threshold, min_size):
    '''Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored'''
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((256, 1600), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num


parser = argparse.ArgumentParser(description='')
parser.add_argument('--model', default='resnet18', type=str, help='Model version')
parser.add_argument('--fine_size', default=96, type=int, help='Resized image size')
parser.add_argument('--pad_left', default=0, type=int, help='Left padding size')
parser.add_argument('--pad_right', default=0, type=int, help='Right padding size')
parser.add_argument('--batch_size', default=6, type=int, help='Batch size for training')
parser.add_argument('--epoch', default=100, type=int, help='Number of training epochs')
parser.add_argument('--snapshot', default=20, type=int, help='Number of snapshots per fold')
parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
parser.add_argument('--save_weight', default='output/weights/', type=str, help='weight save space')
parser.add_argument('--max_lr', default=0.01, type=float, help='max learning rate')
parser.add_argument('--min_lr', default=0.001, type=float, help='min learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for SGD')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--patience', default=40, type=int, help='Number of epoch waiting for best score')

args = parser.parse_args()
fine_size = args.fine_size + args.pad_left + args.pad_right
args.weight_name = 'model_' + str(fine_size) + '_' + args.model

if not os.path.isdir(args.save_weight):
    os.mkdir(args.save_weight)

device = torch.device('cuda' if args.cuda else 'cpu')


if __name__ == '__main__':
    # initialize test dataloader
    best_threshold = 0.5
    print('best_threshold', best_threshold)
    min_size = 3500

    test_data = getDatabase(mode='test', image_idx=None)
    test_loader = DataLoader(test_data,
                             shuffle=False,
                             batch_size=args.batch_size,
                             num_workers=5,
                             pin_memory=True)

    # Initialize mode and load trained weights
    ckpt_path = "../output/weights/model_96_resnet18f2_0.pth"
    device = torch.device("cuda")
    model = smp.Unet(args.model, classes=4, encoder_weights='imagenet', activation=None)
    salt = model.to(device)
    model.eval()
    state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state["state_dict"])

    # start prediction
    predictions = []
    for i, batch in enumerate(tqdm(test_loader)):
        fnames, images = batch
        batch_preds = torch.sigmoid(model(images.to(device)))
        batch_preds = batch_preds.detach().cpu().numpy()
        for fname, preds in zip(fnames, batch_preds):
            for cls, pred in enumerate(preds):
                pred, num = post_process(pred, best_threshold, min_size)
                rle = rle_decode(pred)
                name = fname + f"_{cls+1}"
                predictions.append([name, rle])

    # save predictions to submission.csv
    df = pd.DataFrame(predictions, columns=['ImageId_ClassId', 'EncodedPixels'])
    df.to_csv("../output/submissions/submission.csv", index=False)


