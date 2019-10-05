import os
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from albumentations import (
    MotionBlur,
    HorizontalFlip,
    VerticalFlip,
    MedianBlur,
    Blur,
    Compose,
    IAAAdditiveGaussianNoise,
    GaussNoise,
    OneOf,
    Normalize,
    RandomBrightnessContrast,
    ShiftScaleRotate,
    RandomSizedCrop,
)
from albumentations.pytorch.transforms import ToTensor
from sklearn.model_selection import KFold

from src.utils import rle_encode
from src.transform import *

import warnings
warnings.filterwarnings("ignore")


class SteelDatabase(Dataset):
    def __init__(self, mode, image_list=None, df=None, is_tta=False, fine_size=(256, 1600, 3)):
        self.image_list = image_list
        self.df = df
        self.mode = mode
        self.is_tta = is_tta
        self.root = 'input/severstal-steel-defect-detection/'
        self.fine_size = fine_size

        self.test_idx = os.listdir(self.root + 'test_images')

    def __len__(self):
        if self.mode != 'test':
            return len(self.image_list)
        else:
            return len(os.listdir('input/severstal-steel-defect-detection/test_images'))

    def __getitem__(self, idx):
        if self.mode == 'train':
            image_id = self.image_list[idx]
            image_id, mask, target = rle_encode(image_id, self.df)
            image_path = os.path.join(self.root + 'train_images', image_id)
            image = cv2.imread(image_path)

            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
            train_aug = Compose([
                OneOf([
                    ShiftScaleRotate(),
                    VerticalFlip(p=0.8),
                    HorizontalFlip(p=0.8),
                ], p=0.6),
                OneOf([
                    RandomBrightnessContrast(),
                    MotionBlur(p=0.5),
                    MedianBlur(blur_limit=3, p=0.5),
                    Blur(blur_limit=3, p=0.5),
                ], p=0.6),
                OneOf([
                    IAAAdditiveGaussianNoise(),
                    GaussNoise(),
                ], p=0.6),
                RandomSizedCrop(min_max_height=(250, 256), height=256, width=400, p=1),
                Normalize(mean=mean, std=std, p=1),
                ToTensor()])

            data = {"image": image, "mask": mask, "target": target}
            augmented = train_aug(**data)
            image = augmented['image']
            mask = augmented['mask']
            mask = mask[0].permute(2, 0, 1)

            target = augmented['target']

            return image, mask, target

        elif self.mode == 'val':
            image_id = self.image_list[idx]
            image_id, mask, target = rle_encode(image_id, self.df)
            image_path = os.path.join(self.root + 'train_images', image_id)
            image = cv2.imread(image_path)

            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
            test_aug = Compose([
                Normalize(mean=mean, std=std, p=1),
                ToTensor()])

            data = {"image": image, "mask": mask, "target": target}
            augmented = test_aug(**data)
            image = augmented['image']
            mask = augmented['mask']
            mask = mask[0].permute(2, 0, 1)

            target = augmented['target']

            return image, mask, target

        elif self.mode == 'test':
            image_id = self.test_idx[idx]
            image_path = os.path.join(self.root, "test_images", image_id)
            image = cv2.imread(image_path)

            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
            if self.is_tta:
                test_aug = Compose([
                    Normalize(mean=mean, std=std, p=1),
                    HorizontalFlip(p=1),
                    ToTensor()])
            else:
                test_aug = Compose([
                    Normalize(mean=mean, std=std, p=1),
                    ToTensor()])

            augmented = test_aug(image=image)
            image = augmented['image']

            return image_id, image


def getDatabase(mode, image_idx):
    df = pd.read_csv('input/severstal-steel-defect-detection/train.csv')
    df['ImageId'], df['ClassId'] = zip(*df['ImageId_ClassId'].str.split('_'))
    df['ClassId'] = df['ClassId'].astype(int)
    df = df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')
    df['defects'] = df.count(axis=1)

    return SteelDatabase(image_list=image_idx, df=df, mode=mode)


if __name__ == '__main__':

    num_fold = 5
    num_train = len(os.listdir('input/severstal-steel-defect-detection/train_images'))
    num_train = 40
    indices = list(range(num_train))
    kf = KFold(n_splits=num_fold, random_state=42, shuffle=True)

    train_idx = []
    valid_idx = []
    for t, v in kf.split(indices):
        train_idx.append(t)
        valid_idx.append(v)

    for fold in range(num_fold):
        train_data = getDatabase(mode='train', image_idx=train_idx[fold])
        train_loader = DataLoader(train_data,
                                  shuffle=RandomSampler(train_data),
                                  batch_size=30,
                                  num_workers=4,
                                  pin_memory=True)

    image, mask, target = next(iter(train_loader))

    import matplotlib.pyplot as plt
    for i in range(30):
        img = np.moveaxis(mask[i][0].numpy(), 0, -1)
        plt.imshow(img.astype('uint8'))
        plt.show()







