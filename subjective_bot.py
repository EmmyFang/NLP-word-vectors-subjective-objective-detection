"""
    6. Testing on Your Own Sentence.

    You will write a python script that prompts the user for a sentence input on the command line, and prints the
    classification from each of the three models, as well as the probability that this sentence is subjective.

    An example console output:

        Enter a sentence
        What once seemed creepy now just seems campy

        Model baseline: subjective (0.964)
        Model rnn: subjective (0.999)
        Model cnn: subjective (1.000)

        Enter a sentence
"""
import torch
import torch.optim as optim

import torchtext
from torchtext import data
import spacy

import argparse
import os
import random
import numpy as np
from models import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd


seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed = seed
np.random.seed = seed
torch.backends.cudnn.deterministic=True

nlp = spacy.load('en')
def tokenizer(inp): # create a tokenizer function
    return [tok.text for tok in nlp.tokenizer(inp)]


text = data.Field(sequential=True, tokenize='spacy', include_lengths=True)
label = data.Field(sequential=False, use_vocab=False)
######
train, val, test = data.TabularDataset.splits(path='', train='train.tsv', validation='validation.tsv', test='test.tsv',
                                              format='tsv', skip_header=True, fields=[('Text', text), ('Label', label)])

text.build_vocab(train)
vocab = text.vocab
vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=100))
dic = vocab.stoi

while(True):
    inp = str(input('Enter a sentence: \n'))
    # inp = 'What once seemed creepy now just seems campy'
    toks = tokenizer(inp)
    list = []
    for t in toks:
        list.append(dic[t])
    x = torch.tensor(list).view(len(list),1)
    xl = torch.tensor(len(list)).view(1)

    model1 = torch.load('model_baseline.pt')
    model2 = torch.load('model_rnn.pt')
    model3 = torch.load('model_cnn.pt')

    y = model1(x,xl)

    if (float(y) > 0.5):
        type = 'subjective'
    else:
        type = 'objective'

    print('Model baseline: {} ({})'.format(type, format(np.around(y.detach().numpy(),decimals = 3), '.3f')))

    y = model2(x,xl)
    if (float(y) > 0.5):
        type = 'subjective'
    else:
        type = 'objective'

    print('Model rnn: {} ({})'.format(type, format(np.around(y.detach().numpy(), decimals=3), '.3f')))
    y = model3(x,xl)

    if (float(y) > 0.5):
        type = 'subjective'
    else:
        type = 'objective'

    print('Model cnn: {} ({})'.format(type, format(np.around(y.detach().numpy(),decimals = 3), '.3f')))
    print()


