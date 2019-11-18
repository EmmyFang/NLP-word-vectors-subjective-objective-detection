import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter

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
num = data['label'].value_counts()[0]
data1, data0 = np.split(data, [num],axis = 0)
data1.columns=['text','label']
data0.columns=['text','label']

data1_text, data1_label = np.split(data1, [1], axis = 1)
data0_text, data0_label = np.split(data0, [1], axis = 1)

text1_train, text1_left, label1_train, label1_left = train_test_split (data1_text, data1_label, test_size = 1-train_portion, random_state=seed)
text0_train, text0_left, label0_train, label0_left = train_test_split (data0_text, data0_label, test_size = 1-train_portion, random_state=seed)


text_train = [text1_train, text0_train]
text_train = pd.concat(text_train)

label_train = [label1_train, label0_train]
label_train = pd.concat(label_train)



text1_val, text1_test, label1_val, label1_test = train_test_split (text1_left, label1_left, test_size = test_portion/(1-train_portion), random_state=seed)
text0_val, text0_test, label0_val, label0_test = train_test_split (text0_left, label0_left, test_size = test_portion/(1-train_portion), random_state=seed)

text_val = [text1_val, text0_val]
text_val = pd.concat(text_val)

label_val = [label1_val, label0_val]
label_val = pd.concat(label_val)

text_test = [text1_test,text0_test]
text_test = pd.concat(text_test)

label_test = [label1_test, label0_test]
label_test = pd.concat(label_test)

text_test = text_test.values.squeeze()
label_test = label_test.values.squeeze()
test = pd.DataFrame({'text': text_test, 'label': label_test })
test.to_csv('./data/test.tsv', sep='\t', index = False)

text_val = text_val.values.squeeze()
label_val = label_val.values.squeeze()
val = pd.DataFrame({'text': text_val, 'label': label_val})
val.to_csv('./data/validation.tsv', sep='\t', index = False)


text_train = text_train.values.squeeze()
label_train = label_train.values.squeeze()
train = pd.DataFrame({'text': text_train, 'label': label_train})
train.to_csv('./data/train.tsv', sep='\t', index = False)


print('train: {}'.format(train.shape))
print(train['label'].value_counts())
print()

print('validation: {}'.format(val.shape))
print(val['label'].value_counts())
print()


print('test: {}'.format(test.shape))
print(test['label'].value_counts())
