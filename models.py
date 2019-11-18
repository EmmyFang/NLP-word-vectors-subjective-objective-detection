import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Baseline(nn.Module):
    def __init__(self, embedding_dim, vocab):
        super(Baseline, self).__init__()
        self.embed = nn.Embedding.from_pretrained(vocab.vectors)
        self.emb_dim = embedding_dim
        self.fc1 = nn.Linear(self.emb_dim,1)
        ######

        # 4.1 YOUR CODE HERE

        ######

    def forward(self, x, lengths=None):

        ######

        # 4.1 YOUR CODE HERE
        embed_input = self.embed(x)
        mean = torch.mean(input = embed_input, dim = 0)
        y = self.fc1(mean)
        y = torch.sigmoid(y)
        y = y.squeeze()
        return y

        ######


class RNN(nn.Module):
    def __init__(self, embedding_dim, vocab, hidden_dim):
        super(RNN, self).__init__()
        self.embed = nn.Embedding.from_pretrained(vocab.vectors)
        self.emb_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(self.hidden_dim, 1)
        self.GRU = nn.GRU(input_size= self.emb_dim, hidden_size= self.hidden_dim)

    def forward(self, x, lengths=None):
        y = self.embed(x)
        y = torch.nn.utils.rnn.pack_padded_sequence(y, lengths)
        y = self.GRU(y)[1]
        y = y.squeeze()
        y = self.fc1(y)
        act = nn.Sigmoid()
        y = act(y)
        y = y.squeeze()
        return y

class CNN(nn.Module):
    def __init__(self, embedding_dim, vocab, n_filters, filter_sizes):
        super(CNN, self).__init__()
        self.embed = nn.Embedding.from_pretrained(vocab.vectors)
        self.emb_dim = embedding_dim
        self.n_filters = n_filters
        self.filter_sizes = filter_sizes
        ks1 = (filter_sizes[0], self.emb_dim)
        ks2 = (filter_sizes[1], self.emb_dim)
        self.conv1 = nn.Conv2d(1, self.n_filters, kernel_size=ks1)
        self.conv2 = nn.Conv2d(1, self.n_filters, kernel_size=ks2)

        self.fc = nn.Linear(self.emb_dim,1)



    def forward(self, x, lengths=None):
        ######
        x = self.embed(x)
        x = np.transpose(x, (1,0,2))
        x = x.view(x.shape[0],1,x.shape[1],x.shape[2])
        x1 = self.conv1(x)
        pool = nn.MaxPool2d((x1.shape[2],1))
        x1 = pool(x1)
        x1 = F.relu(x1)

        x2 = self.conv2(x)
        pool = nn.MaxPool2d((x2.shape[2], 1))
        x2 = pool(x2)
        x2 = F.relu(x2)

        x1 = x1.view(x.shape[0],-1)
        x2 = x2.view(x.shape[0],-1)
        x = torch.cat((x1, x2), 1)
        x = torch.sigmoid(self.fc(x))
        x = x.squeeze()
        return x


