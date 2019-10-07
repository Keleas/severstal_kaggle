import random
import re
import pandas as pd
import numpy as np

train_df = pd.read_csv('input\\severstal\\train.csv')
def separate_classes():
    '''
    Из csv файла разбивает данные по классам
    :return: массивы индексов в csv 
    '''
    idx_no_defect = []
    idx_class_1 = []
    idx_class_2 = []
    idx_class_3 = []
    idx_class_4 = []

    idx_dict = {0: idx_class_1, 1: idx_class_2, 2: idx_class_3, 3:idx_class_4, 4: idx_no_defect}
    with_defects_df = train_df.dropna()
    for col in range(0, len(with_defects_df)):
        labels = with_defects_df.iloc[col, 0]
        num = train_df.ImageId_ClassId[train_df.ImageId_ClassId==labels].index.tolist()
        for idx in range(4):
            if int(re.split(r'[_,\n]', labels)[1]) == idx + 1:
                idx_dict[idx].append(num[-1])

    for col in range(0, len(train_df), 4):
        img_names = [str(i).split("_")[0] for i in train_df.iloc[col:col + 4, 0].values]
        if not (img_names[0] == img_names[1] == img_names[2] == img_names[3]):
            raise ValueError

        labels = train_df.iloc[col:col + 4, 1]
        if labels.isna().all():
            idx_no_defect.append(col)

    for one_class in [idx_no_defect, idx_class_1, idx_class_2, idx_class_3, idx_class_4]:
        random.shuffle(one_class)
        # total_len += len(one_class)

    return idx_no_defect, idx_class_1, idx_class_2, idx_class_3, idx_class_4

def create_fold(mode):
    '''
    
    :param mode:  simple - перемешивает индексы и выдает 5 массивов индексов 
    equal - перемешеивает индексы и выдает 5 массивов индексов, в которых каждого класса равное количество
    weighted - перемешивает индексы и выдает 5 массивов по распредлению весов  WEIGHTS
    :return: 5 массивов индексов 
    '''
    WEIGHTS = [6172, 128, 43, 741, 120]
    #num_fold = 5
    no_def, def_1, def_2, def_3, def_4 = separate_classes()
    total_len = 0
    min_len = len(no_def) * 1000
    sum_weights = sum(WEIGHTS)
    for one_class in [no_def, def_1, def_2, def_3, def_4]:

        total_len += len(one_class)
        if len(one_class) < min_len:
            min_len = len(one_class)
    if mode == 'simple':
        shake = no_def+ def_1+ def_2+ def_3+ def_4
        random.shuffle(shake)
        split = total_len//5
        return shake[:split], shake[split:2*split], shake[2*split:3*split], shake[3*split:4*split], shake[4*split:]
    elif mode == 'equal':
        shake =  no_def[:min_len] + def_1[:min_len]+ def_2[:min_len]+ def_3[:min_len]+ def_4[:min_len]
        random.shuffle(shake)
        split = total_len//5
        return shake[:split], shake[split:2*split], shake[2*split:3*split], shake[3*split:4*split], shake[4*split:]
    elif mode == 'weighted':
        step = total_len // sum_weights
        shake = no_def[:step*WEIGHTS[0]]+ def_1[:step*WEIGHTS[1]]+ def_2[:step*WEIGHTS[2]]+ def_3[:step*WEIGHTS[3]]+ def_4[:step*WEIGHTS[4]]
        random.shuffle(shake)
        split = total_len // 5
        return shake[:split], shake[split:2 * split], shake[2 * split:3 * split], shake[3 * split:4 * split], shake[4 * split:]


if __name__ == '__main__':
    for mode in ['simple', 'equal', 'weighted']:
        print(create_fold(mode))
