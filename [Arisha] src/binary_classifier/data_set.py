import os
import re
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms


def labels(mode):
    files = []
    labels = []
    count_def = 0
    with open(os.path.join('G:\\SteelDetection', mode + '.csv')) as csv_file:
        next(csv_file)
        for num, line in enumerate(csv_file, 1):
            line_data = re.split(r'[_,\n]', line)
            file_name = line_data[0]

            if file_name not in files:
                files.append(file_name)
            if not not line_data[2]:
                count_def += 1
            if (num - 1) % 4 == 0:
                if count_def > 0:
                    labels.append(1)
                else:
                    labels.append(0)
                count_def = 0

    return files, labels


class SteelDataset(Dataset):
    def __init__(self):
        self.mode = 'train'
        self.files, self.labels = labels(self.mode)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        file_path = os.path.join(('G:\\SteelDetection\\' + self.mode + '_images'), file)
        image = Image.open(file_path)
        image = transforms.ToTensor()(image).float()
        label = self.labels[idx]
        return (image, label)



