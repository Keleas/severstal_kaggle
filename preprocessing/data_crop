import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from torchvision import transforms
from tqdm import tqdm

import dataset

batch_size = 4


def rleToMask(segment):
    rows, cols = 256, 1600
    rleNumbers = np.array(segment)
    rlePairs = rleNumbers.reshape(-1, 2)
    img = np.zeros(rows * cols, dtype=np.uint8)
    for index, length in rlePairs:
        index -= 1
        img[index:index + length] = 255
    img = img.reshape(cols, rows)
    img = img.T
    return img


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

        plt.savefig('C:\\Users\\K-132-14\\Downloads\\severstal-steel-defect-detection\\pic\\' + file)
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


train_df = pd.read_csv("C:\\Users\\K-132-14\\Downloads\\severstal-steel-defect-detection\\train.csv")  # change here
sample_df = pd.read_csv("C:\\Users\\K-132-14\\Downloads\\severstal-steel-defect-detection\\test.csv")


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
            if abs(dist) < 200:  # <-- threshold
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


def rectangular(all_x, all_w, img):
    all_x.sort()
    all_w.sort()
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

    crop_mas.append(0)
    crop_mas.append(1600)
    crop_mas.sort()
    crop_mas = np.array(crop_mas)
    diff = np.diff(crop_mas)
    new_crop_mas = []
    for i in range(len(diff)):
        if diff[i] >= 100:
            new_crop_mas.append(crop_mas[i])

    for i in range(len(new_crop_mas)):
        cv2.rectangle(img, (int(new_crop_mas[i]), 0), (int(new_crop_mas[i]), 256), (0, 255, 0), 2)

    return new_crop_mas


def show_mask_image(col):
    name, mask = name_and_mask(col)
    img = cv2.imread(str('C:\\Users\\K-132-14\\Downloads\\severstal-steel-defect-detection\\train_images\\' + name))
    palet = [(249, 192, 12), (0, 185, 241), (114, 0, 218), (249, 50, 12)]
    all_x, all_w = [], []

    for ch in range(4):
        contours, _ = cv2.findContours(mask[:, :, ch], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if len(contours) > 0:
            for i in range(0, len(contours)):
                cv2.polylines(img, contours[i], True, palet[ch], 2)

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
    images = []
    if not not all_x:
        crop_mas = rectangular(all_x, all_w, img)
        for i in range(len(crop_mas) - 1):
            images.append(img[0:265, crop_mas[i]:crop_mas[i + 1]])
    else:
        images = img
    plt.imshow(img)
    plt.show()
    return images


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


if __name__ == '__main__':
    no_def, class_1, class_2, class_3, class_4, multi, triple = all_classes()
    classes = [class_1, class_2, class_3, class_4, multi, triple]
    for new_class in classes:
        for idx in tqdm(new_class):
            show_mask_image(idx)
