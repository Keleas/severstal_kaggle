import cv2
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.utils import rle_decode, rle_encode

parser = argparse.ArgumentParser(description='')
parser.add_argument('--model', default='resnet34', type=str, help='Model version')
parser.add_argument('--fold', default='0', type=str, help='number of split fold')
parser.add_argument('--save_pred', default='output/fold_predictions/', type=str, help='prediction save space')
args = parser.parse_args()


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


if __name__ == '__main__':

    best_threshold = 0.5
    min_size = 100

    # start prediction
    predictions = []
    all_preds = np.load(args.save_pred + args.model + '_pred' + args.fold + '.npy')
    fnames = os.listdir('input/severstal-steel-defect-detection/test_images')


    def show_mask_image(col):
        palet = [(249, 192, 12), (0, 185, 241), (114, 0, 218), (249, 50, 12)]
        name, mask, = fnames[col], all_preds[col]
        mask = mask.reshape((256, 1600, 4))
        img = cv2.imread('input/severstal-steel-defect-detection/test_images/' + name)
        fig, ax = plt.subplots(figsize=(15, 15))

        for ch in range(4):
            contours, _ = cv2.findContours(cv2.convertScaleAbs(mask[:, :, ch]), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            for i in range(0, len(contours)):
                cv2.polylines(img, contours[i], True, palet[ch], 2)
        ax.set_title(name)
        ax.imshow(img)
        plt.show()


    # num_train = len(os.listdir('input/severstal-steel-defect-detection/test_images'))
    # indices = np.array(list(range(num_train)))
    # sample_idx = None
    for idx in range(10):
        show_mask_image(idx)

    # print(all_preds.shape)
    # print(fnames)
    # for fname, preds in tqdm(zip(fnames, all_preds), total=len(fnames)):
    #     for cls, pred in enumerate(preds):
    #         pred, num = post_process(pred, best_threshold, min_size)
    #         rle = rle_decode(pred)
    #         name = fname + f"_{cls + 1}"
    #         predictions.append([name, rle])
    #
    # # save predictions to submission.csv
    # df = pd.DataFrame(predictions, columns=['ImageId_ClassId', 'EncodedPixels'])
    # df.to_csv("output/submissions/submission_test.csv", index=False)
