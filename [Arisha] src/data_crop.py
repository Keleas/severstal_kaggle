import pandas as pd
import os
import re
from collections import Counter
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from torchvision import transforms
from tqdm import tqdm
import copy
import torch

import dataset


def labels(mode):
    data_dict = {}
    with open(os.path.join('G:\\SteelDetection', mode + '.csv')) as csv_file:
        next(csv_file)
        for num, line in enumerate(csv_file, 1):
            if num > 20:
                break
            line_data = re.split(r'[_,\n]', line)
            file_name = line_data[0]
            if (num - 1) % 4 == 0:
                data_dict[file_name] = {}
                data_dict[file_name]['label'] = [0,0,0,0]
                data_dict[file_name]['segment'] = [[0], [0], [0], [0]]
            if not not line_data[2]:
                data_dict[file_name]['label'][(num-1)%4] = int(line_data[1])
                pixels = line_data[2].split()
                data_dict[file_name]['segment'][(num-1)%4] = list(map(int,pixels))
    return data_dict


def labels_distribution(): #({3.0: 5150, 1.0: 897, 4.0: 801, 2.0: 247})
    list_labels = np.array([])
    dict = labels('train')
    for file in dict:
        list_labels = np.append(list_labels, dict[file]['label'])
    print(Counter(list_labels))


def images_distribution():
    height, weight, channels = [], [] , []
    _, list_images = images('train')
    for image in list_images:
        image_height, image_weight, image_channels = image.shape
        height.append(image_height)
        weight.append(image_weight)
        channels.append(image_channels)
    print('Height:', Counter(height))
    print('Weight:', Counter(weight))
    print('Channels:', Counter(channels))
    # Height: Counter({256: 12568})
    # Width: Counter({1600: 12568})
    # Channels: Counter({3: 12568})


def one_augment(aug, image):
    return aug(image=image)['image']


def first_step_images(mode):
    files, images = [], []
    dict_keys = labels(mode).keys()
    for file in  os.listdir('G:\\SteelDetection\\' + mode + '_images'):
        if file in dict_keys:
            file_path = os.path.join(('G:\\SteelDetection\\' + mode + '_images'), file)
            image = cv2.imread(file_path)
            files.append(file)
            if (dict_keys[file]['label'] == [0,0,0,0]).all():
                dict_keys[file]['label'] = []
                dict_keys[file]['segment'] = []
                for i in range(7):
                    if (image[i*200:(i+1)*200] != np.zeros((256, 200, 3))).any():
                        images.append(image[0:265, i*200:(i+1)*200])
                        dict_keys[file]['label'].append([0,0,0,0])
                        dict_keys[file]['segment'].append(preprocessing.rleToMask([[0], [0], [0], [0]]).reshape((4, 256, 1600)))
            else:
                mask = preprocessing.rleToMask(dict_keys[file]['segment']).reshape((4, 256, 1600))
                dict_keys, images = crop_with_defects(file, image, mask, dict_keys)

    return dict_keys, files, images


def crop_with_defects(file, img, mask, dict_keys):
    mask = (mask).astype('uint8')
    image = copy.deepcopy(img)
    all_x, all_w = [], []
    masks, all_type_contours = [], []
    for ch in range(4):
        contours, _ = cv2.findContours(mask[ch, :, :], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        all_type_contours.append(contours)
    contours = list(itertools.chain.from_iterable(all_type_contours))

    LENGTH = len(contours)
    status = np.zeros((LENGTH, 1))
    for i, cnt1 in enumerate(contours):
        x = i
        if i != LENGTH - 1:
            for j, cnt2 in enumerate(contours[i + 1:]):
                x = x + 1
                dist = find_if_close(cnt1, cnt2)
                if dist == True:
                    val = min(status[i], status[x])
                    status[x] = status[i] = val
                else:
                    if status[x] == status[i]:
                        status[x] = i + 1
    unified = []
    maximum = int(max(status)) + 1
    for i in range(maximum):
        pos = np.where(status == i)[0]
        if pos.size != 0:
            cont = np.vstack(contours[i] for i in pos)
            hull = cv2.convexHull(cont)
            unified.append(hull)
    for c in unified:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        all_x.append(x)
        all_w.append(x + w)

    all_x = np.array(all_x)
    all_w = np.array(all_w)
    limits = []
    for i in range(0, 1600, 200):
        limits.append(i)

    for i in range(len(limits)):
        for j in range(len(all_x)):
            if limits[i] > all_x[j] and limits[i] < all_w[j]:
                limits[i] = all_w[j]

    images = []
    dict_keys[file]['label'] = []
    dict_keys[file]['segment'] = []
    for i in range(len(limits) - 1):
        if (image[0:265, limits[i]:limits[i + 1]] != np.zeros((256, limits[i+1] - limits[i], 3))).any():
            images.append(image[0:265, limits[i]:limits[i + 1]])
            dict_keys[file]['label'].append([0, 0, 0, 0])
            dict_keys[file]['segment'].append(preprocessing.rleToMask([[0], [0], [0], [0]]).reshape((4, 256, 1600)))
            for ch in range(4):
                crop_mask = mask[ch, 0:265, limits[i]:limits[i + 1]]
                dict_keys[file]['label'][-1][ch] = ch 
                dict_keys[file]['segment'][-1][ch] = crop_mask 

    return dict_keys, images


def first_step_labels(mode):
    data_dict = {}
    count = 0
    with open(os.path.join('G:\\SteelDetection\\', mode + '.csv')) as csv_file:
        next(csv_file)
        for num, line in enumerate(csv_file, 1):
            line_data = re.split(r'[_,\n]', line)
            if not not line_data[2]:
                file_name = line_data[0]
                count += 1

                data_dict[file_name] = {}
                data_dict[file_name]['label']= int(line_data[1])
                pixels = line_data[2].split()
                data_dict[file_name]['segment'] = (list(map(int,pixels)))

            if count > 800:
                break
    return data_dict

def find_intersection(mode):
    data_dict = {}

    with open(os.path.join('G:\\SteelDetection', mode + '.csv')) as csv_file:
        next(csv_file)
        for num, line in tqdm(enumerate(csv_file, 1)):

            line_data = re.split(r'[_,\n]', line)
            file_name = line_data[0]
            if (num - 1) % 4 == 0:
                data_dict[file_name] = {}
                data_dict[file_name]['label'] = []
                data_dict[file_name]['segment'] = []
            if not not line_data[2]:
                data_dict[file_name]['label'].append(int(line_data[1]))
                pixels = line_data[2].split()
                data_dict[file_name]['segment'].append((list(map(int,pixels))))
            if num % 4 == 0 and (not data_dict[file_name]['label']): #or len(data_dict[file_name]['label'])< 2):
                data_dict.pop(file_name)
            if num > 20:
                break

    return data_dict


def rleToMask(segments):
    images = np.zeros((4, 256, 1600))
    for i, segment in enumerate(segments):
        if segment != [0]:
            rows, cols = 256, 1600
            rleNumbers = np.array(segment)
            rlePairs = rleNumbers.reshape(-1, 2)
            img = np.zeros(rows * cols, dtype=np.uint8)
            for index, length in rlePairs:
                index -= 1
                img[index:index + length] = 255
            img = img.reshape(cols, rows)
            img = img.T
            images[i] = img

    images = images.reshape((4, 256, 1600))
    return images


def paint_all_mask():
    data_set = dataset.SteelDataset(mode='train')
    fig = plt.figure(figsize=(20, 100))
    count = 0
    for i in data_set:
        file, image, label, segment = i
        fig.add_subplot(8, 5, count + 1)
        segm_image = rleToMask(segment)
        image = transforms.ToPILImage()(image)
        plt.imshow(image)
        plt.imshow(segm_image, alpha=0.3)
        plt.title(label)
        plt.axis('off')
        count += 1
    plt.show()


def paint_1_class_mask():
    data_set = dataset.SteelDataset(mode='train')
    fig = plt.figure()
    fig.suptitle('Label:4', fontsize=15)
    count = 0
    for i in data_set:
        file, image, label, segment = i
        if label == 4:
            fig.add_subplot(10, 5, count + 1)
            segm_image = rleToMask(segment)
            image = transforms.ToPILImage()(image)
            plt.imshow(image)
            plt.imshow(segm_image, alpha=0.3)
            count += 1
            plt.axis('off')
            if count >= 50:
                break
    plt.show()


def more_segments():
    data_set = dataset.SteelDataset('train')
    for i in data_set:
        file, image, label, segment = i

        fig = plt.figure()
        image = transforms.ToPILImage()(image)
        plt.imshow(image)
        for i in range(len(segment)):
            segm_image = rleToMask(segment[i])
            plt.imshow(segm_image, alpha=0.4 + i * 0.2)
            plt.axis('off')

        plt.savefig('G:\SteelDetection' + file)
        plt.close(fig)


def quantity():
    data_set = dataset.SteelDataset('train')
    quan = np.zeros((4, 4))
    files = []
    for i in tqdm(data_set):
        file, image, label, segment = i
        if len(label) == 2:
            quan[label[0] - 1, label[1] - 1] += 1
        elif len(label) == 3:
            files.append(file)
    print(quan)
    print(files)
    ax = sns.heatmap(quan, annot=True)
    plt.show()


def name_and_mask(start_idx):
    col = start_idx
    img_names = [str(i).split("_")[0] for i in train_df.iloc[col:col + 4, 0].values]
    if not (img_names[0] == img_names[1] == img_names[2] == img_names[3]):
        raise ValueError

    labels = train_df.iloc[col:col + 4, 1]
    mask = np.zeros((256, 1600, 4), dtype=np.uint8)

    for idx, label in enumerate(labels.values):
        if label is not np.nan:
            mask_label = np.zeros(1600 * 256, dtype=np.uint8)
            label = label.split(" ")
            positions = map(int, label[0::2])
            length = map(int, label[1::2])
            for pos, le in zip(positions, length):
                mask_label[pos - 1:pos + le - 1] = 1
            mask[:, :, idx] = mask_label.reshape(256, 1600, order='F')
    return img_names[0], mask


def find_if_close(cnt1, cnt2):
    row1, row2 = cnt1.shape[0], cnt2.shape[0]
    for i in range(row1):
        for j in range(row2):
            dist = np.linalg.norm(cnt1[i] - cnt2[j])
            if abs(dist) < 200:
                return True
            elif i == row1 - 1 and j == row2 - 1:
                return False


def check_if_less(new_x, new_w, new_all_x, new_all_w):
    if new_w - new_x < 100:
        mean = new_x + (new_w - new_x) / 2
        new_x = mean - 100
        new_w = mean + 100
        if new_w > 1600:
            new_w = 1600
            new_x = 1400
        if new_x < 0:
            new_x = 0
            new_w = 200
    new_all_x.append(int(new_x))
    new_all_w.append(int(new_w))


def rectangular(all_x, all_w, img, masks):
    crop_mas = []
    for j in range(len(all_x)):
        if all_w[j] - all_x[j] < 100:
            check_if_less(all_x[j], all_w[j], crop_mas, crop_mas)
        else:
            crop_mas.append(all_x[j])
            crop_mas.append(all_w[j])
    if crop_mas[-1] < 1400:
        for j in range(crop_mas[-1], 1600, 200):
            if abs(min(crop_mas, key=lambda a: abs(a - j)) - j) > 100:
                crop_mas.append(j)
    if crop_mas[0] > 400:
        for j in range(0, crop_mas[0], 200):
            if abs(min(crop_mas, key=lambda a: abs(a - j)) - j) > 100:
                crop_mas.append(j)
                new_mask.append(0)


    crop_mas.append(0)
    crop_mas.append(1600)
    crop_mas.sort()
    crop_mas = np.array(crop_mas)
    diff = np.diff(crop_mas)
    new_crop_mas = []
    for i in range(len(diff)):
        if diff[i] >= 100:
            new_crop_mas.append(crop_mas[i])

    for i in range(len(new_crop_mas)-1):
        cv2.rectangle(img, (int(new_crop_mas[i]), 0), (int(new_crop_mas[i]), 256), (0, 255, 0), 2)
    #     if new_mask[i] == 0:
    #         new_mask[i] = np.zeros((256, ))
    return new_crop_mas


def all_classes():
    idx_no_defect = []
    idx_class_1 = []
    idx_class_2 = []
    idx_class_3 = []
    idx_class_4 = []
    idx_class_multi = []
    idx_class_triple = []

    for col in range(0, len(train_df), 4):
        img_names = [str(i).split("_")[0] for i in train_df.iloc[col:col + 4, 0].values]
        if not (img_names[0] == img_names[1] == img_names[2] == img_names[3]):
            raise ValueError

        labels = train_df.iloc[col:col + 4, 1]
        if labels.isna().all():
            idx_no_defect.append(col)
        elif (labels.isna() == [False, True, True, True]).all():
            idx_class_1.append(col)
        elif (labels.isna() == [True, False, True, True]).all():
            idx_class_2.append(col)
        elif (labels.isna() == [True, True, False, True]).all():
            idx_class_3.append(col)
        elif (labels.isna() == [True, True, True, False]).all():
            idx_class_4.append(col)
        elif labels.isna().sum() == 1:
            idx_class_triple.append(col)
        else:
            idx_class_multi.append(col)
    return idx_no_defect, idx_class_1, idx_class_2, idx_class_3, \
           idx_class_4, idx_class_multi, idx_class_triple


