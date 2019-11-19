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

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed = seed
np.random.seed = seed


def load_model(args, vocab, filter_size):
    if (args.model == 'cnn'):
        model = CNN(args.emb_dim, vocab, args.num_filt, filter_size)
    elif (args.model == 'rnn'):
        model = RNN(args.emb_dim, vocab, args.rnn_hidden_dim)
    else:
        model = Baseline(args.emb_dim, vocab)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fnc = torch.nn.BCELoss()
    return model, loss_fnc, optimizer

def evaluate(model, val_loader, loss_fnc):
    total_corr = 0
    total_loss = 0.0
    total_epoch = 0
    for i, d in enumerate (val_loader):
        (x, x_lengths), label = d.Text, d.Label
        prediction = model(x, x_lengths)
        corr = (prediction > 0.5).squeeze().long() == label
        total_corr += int(corr.sum())
        batch_loss = loss_fnc(input=prediction.squeeze(), target=label.float())
        total_loss += batch_loss.item()
        total_epoch += len(label)
        # print('val corr: {}, val size: {}'.format(total_corr, len(label)))

    # print('total epoch: {}'.format(total_epoch))
    # print('total corr: {}'.format(total_corr))
    # print('batch: {}'.format(i+1))
    acc = float((total_corr)/total_epoch)
    loss = float(total_loss) / (i + 1)
    return acc, loss

def main(args):
    ######
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed = seed
    np.random.seed = seed
    torch.backends.cudnn.deterministic = True
    spacy_en = spacy.load('en')
    # 3.2 Processing of the data
    text = data.Field(sequential=True, tokenize='spacy', include_lengths=True)
    label = data.Field(sequential=False, use_vocab=False)
    ######
    train, val, test = data.TabularDataset.splits(path='./data/', train='train.tsv', validation='validation.tsv', test='test.tsv', format='tsv', skip_header = True, fields=[('Text', text), ('Label', label)])
    train_iter, val_iter, test_iter = data.BucketIterator.splits((train, val, test), sort_key=lambda x: len(x.Text), batch_sizes=(args.batch_size,args.batch_size,args.batch_size),sort_within_batch=True, repeat = False)

    # train_iter, val_iter, test_iter = data.Iterator.splits((train, val, test), sort_key=lambda x: len(x.Text), batch_sizes=(args.batch_size, 1600, 2000),sort_within_batch=True, repeat = False)
    
    # to construct the Vocab object that represents the set of possible values for this field.
    text.build_vocab(train)
    # text.build_vocab(train, vectors="glove.6B.100d")
    
    # a vocabulary object that will be used to numericalize a field
    vocab = text.vocab
    # load one of or a list containing instantiations of the GloVe, CharNGram, or Vectors classes.
    vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=args.emb_dim))

    filter_size = (2, 4)

    ######
    model, loss_fnc, optimizer = load_model(args, vocab, filter_size)
    # 5 Training and Evaluation
    for epoch in range (args.epochs):
        accum_loss = 0
        tot_corr = 0
        train_corr = 0
        train_size_count = 0
        for i,d in enumerate (train_iter):
            (x, x_lengths), label = d.Text, d.Label
            # print(x_lengths)
            optimizer.zero_grad()
            predictions = model(x, x_lengths)
            batch_loss = loss_fnc(input=predictions.squeeze(), target=label.float())
            accum_loss += batch_loss.item()
            batch_loss.backward()
            optimizer.step()
            corr = (predictions > 0.5).squeeze().long() == label
            train_corr += int(corr.sum())
            train_size_count += len(label)
        valid_acc, valid_loss = evaluate(model, val_iter, loss_fnc)
        train_acc = train_corr / train_size_count
        train_loss = accum_loss / (i+1)
        # print('train corr: {}, train size: {}'.format(train_corr, train_size_count))

        print('epoch {} train acc: {}, train loss: {}, val acc: {}, val loss: {}'.format(epoch+1, train_acc, train_loss, valid_acc, valid_loss))

    print('no packed ')
    torch.save(model, 'model_{}_no_pack.pt'.format(args.model))

    test_acc, test_loss = evaluate(model, test_iter, loss_fnc)
    print('test acc: {}, test loss: {}'.format(test_acc, test_loss))
    print('model {}'.format(args.model))
    ######


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64) #
    parser.add_argument('--lr', type=float, default=0.001) #
    parser.add_argument('--epochs', type=int, default=25) #
    parser.add_argument('--model', type=str, default='rnn',
                        help="Model type: baseline,rnn,cnn (Default: baseline)") #
    parser.add_argument('--emb-dim', type=int, default=100) #
    parser.add_argument('--rnn-hidden-dim', type=int, default=100) #
    parser.add_argument('--num-filt', type=int, default=50)

    args = parser.parse_args()

    main(args)


