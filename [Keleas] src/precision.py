import cv2
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from src.utils import rle_decode, rle_encode
from src.create_data import SteelDatabase


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
parser.add_argument('--model', default='resnet34', type=str, help='Model version')
parser.add_argument('--batch_size', default=6, type=int, help='Batch size for training')
parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
parser.add_argument('--is_plot', default=False, type=bool, help='Plot results of prediction')

args = parser.parse_args()


if __name__ == '__main__':
    # initialize test dataloader
    best_threshold = 0.5
    print('best_threshold', best_threshold)
    min_size = 1000

    test_data = SteelDatabase(mode='test', is_tta=False)
    test_loader = DataLoader(test_data,
                             shuffle=False,
                             batch_size=args.batch_size,
                             num_workers=5,
                             pin_memory=True)

    # Initialize mode and load trained weights
    ckpt_path = "output/weights/model_96_resnet34_f0_s0.pth"
    device = torch.device("cuda")
    model = smp.Unet(args.model, classes=4, encoder_weights='imagenet')
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model = model.to(device)
    state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state)
    model.eval()

    # start prediction
    predictions = []
    for i, batch in enumerate(tqdm(test_loader, total=len(test_loader))):
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
    df.to_csv("output/submissions/submission.csv", index=False)

    df['ImageId'], df['ClassId'] = zip(*df['ImageId_ClassId'].str.split('_'))
    df['ClassId'] = df['ClassId'].astype(int)
    df = df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')
    df['defects'] = df.count(axis=1)

    if args.is_plot:
        # def name_and_mask(start_idx):
        #     col = start_idx
        #     img_names = [str(i).split("_")[0] for i in df.iloc[col:col + 4, 0].values]
        #     print(img_names)
        #     if not (img_names[0] == img_names[1] == img_names[2] == img_names[3]):
        #         raise ValueError
        #
        #     labels = df.iloc[col:col + 4, 1]
        #     mask = np.zeros((256, 1600, 4), dtype=np.uint8)
        #
        #     for idx, label in enumerate(labels.values):
        #         if label is not np.nan:
        #             mask_label = np.zeros(1600 * 256, dtype=np.uint8)
        #             label = label.split(" ")
        #             positions = map(int, label[0::2])
        #             length = map(int, label[1::2])
        #             for pos, le in zip(positions, length):
        #                 mask_label[pos - 1:pos + le - 1] = 1
        #             mask[:, :, idx] = mask_label.reshape(256, 1600, order='F')
        #     return img_names[0], mask


        def show_mask_image(col):
            palet = [(249, 192, 12), (0, 185, 241), (114, 0, 218), (249, 50, 12)]
            name, mask, _ = rle_encode(col, df)
            img = cv2.imread('input/severstal-steel-defect-detection/test_images/' + name)
            fig, ax = plt.subplots(figsize=(15, 15))

            for ch in range(4):
                contours, _ = cv2.findContours(mask[:, :, ch], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                for i in range(0, len(contours)):
                    cv2.polylines(img, contours[i], True, palet[ch], 2)
            ax.set_title(name)
            ax.imshow(img)
            plt.show()


        num_train = len(os.listdir('input/severstal-steel-defect-detection/test_images'))
        indices = np.array(list(range(num_train)))
        sample_idx = None
        for idx in [0, 1]:
            show_mask_image(idx)
