import pdb
import os
import cv2
import argparse
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from albumentations import (Normalize, Compose)
from albumentations.torch import ToTensor
import torch.utils.data as data

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from sklearn.model_selection import KFold
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from src.logger import Logger
from src.create_data import getDatabase
from src.torchutils import EarlyStopping, AdamW, CyclicLRWithRestarts
from src.losses import lovasz_softmax, jaccard_loss, dice_loss
from src.utils import do_kaggle_metric, Meter


#https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode
def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


class TestDataset(Dataset):
    '''Dataset for test prediction'''
    def __init__(self, root, df, mean, std):
        self.root = root
        df['ImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
        self.fnames = df['ImageId'].unique().tolist()
        self.num_samples = len(self.fnames)
        self.transform = Compose(
            [
                Normalize(mean=mean, std=std, p=1),
                ToTensor(),
            ]
        )

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        path = os.path.join(self.root, fname)
        image = cv2.imread(path)
        images = self.transform(image=image)["image"]
        return fname, images

    def __len__(self):
        return self.num_samples


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


if __name__ == '__main__':

    num_test = len(os.listdir('input/severstal-steel-defect-detection/test_images'))
    test_indices = list(range(num_test))

    sample_submission_path = '../input/severstal-steel-defect-detection/sample_submission.csv'
    test_data_folder = "../input/severstal-steel-defect-detection/test_images"

    # initialize test dataloader
    best_threshold = 0.5
    num_workers = 2
    batch_size = 4
    print('best_threshold', best_threshold)
    min_size = 3500
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    df = pd.read_csv(sample_submission_path)
    testset = DataLoader(
        TestDataset(test_data_folder, df, mean, std),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Initialize mode and load trained weights
    ckpt_path = "../output/weights/model_96_resnet18f2_0.pth"
    device = torch.device("cuda")
    model = smp.Unet(args.model, classes=4, encoder_weights='imagenet')
    salt = model.to(device)
    model.eval()
    state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state["state_dict"])

    # start prediction
    predictions = []
    for i, batch in enumerate(tqdm(testset)):
        fnames, images = batch
        batch_preds = torch.sigmoid(model(images.to(device)))
        batch_preds = batch_preds.detach().cpu().numpy()
        for fname, preds in zip(fnames, batch_preds):
            for cls, pred in enumerate(preds):
                pred, num = post_process(pred, best_threshold, min_size)
                rle = mask2rle(pred)
                name = fname + f"_{cls+1}"
                predictions.append([name, rle])

    # save predictions to submission.csv
    df = pd.DataFrame(predictions, columns=['ImageId_ClassId', 'EncodedPixels'])
    df.to_csv("../output/submission.csv", index=False)


