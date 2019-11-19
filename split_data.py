import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


import torch
"""
    3.1 Create train/validation/test splits

    This script will split the data/data.tsv into train/validation/test.tsv files.
"""



train_portion = 0.64
val_portion = 0.16
test_portion = 0.2
seed = 0

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


text = 'text'
label = 'label'
fpath = './data/data.tsv'
data = pd.read_table(fpath, header = 0)
# data['label'].dtype('int')
print('total: subjective: {}, objective: {}'.format(data['label'].value_counts()[0], data['label'].value_counts()[1]))

temp, test = train_test_split(data, test_size=0.2, random_state=seed, stratify=data['label'])

train, val = train_test_split(temp, test_size=val_portion/(1-test_portion), random_state=seed, stratify=temp['label'])

print('train: {}'.format(train.shape))
print(train['label'].value_counts(),'\n')

print('validation: {}'.format(val.shape))
print(val['label'].value_counts(), '\n')

print('test: {}'.format(test.shape))
print(test['label'].value_counts(),'\n')

train.to_csv('./data/train.tsv', sep='\t', index = False)
val.to_csv('./data/validation.tsv', sep='\t', index = False)
test.to_csv('./data/test.tsv', sep='\t', index = False)