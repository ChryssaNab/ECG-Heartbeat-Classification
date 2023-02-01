import os
import torch
from torch import nn
import torch.nn.functional as F

""" This class represents one block of two convolutional layers with a skip connection and is used as a building block 
for the 1D ResNet. """

NUM_BLOCKS = 4


def calc_padding(padding):
    pl = int(padding / 2)
    pr = int(padding / 2)
    # Take care of odd number of padding
    if pl + pr < padding:
        pr += 1
    return pl, pr


class Block(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super(Block, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, stride=1)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=kernel_size, stride=1)
        self.maxpool = nn.MaxPool1d(kernel_size=kernel_size, stride=2)

        # In the case where the number of input channels is different from the number of the output channels,
        # we use this pseudo-convolution to make them match in order to be able to perform the addition with the skip
        # connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels,
                                      kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        # Add padding to the output of the second convolution to be able to add the results with those of the first
        # (skip connection). Up for discussion.
        padding = x.size(2) - out.size(2)
        pl, pr = calc_padding(padding)
        out = F.pad(input=out, pad=(pl, pr), mode='replicate')
        out += self.shortcut(x)
        out = F.relu(out)
        out = self.maxpool(out)
        return out


class CNN1D(nn.Module):

    def __init__(self, block_channels, kernel_size):
        super(CNN1D, self).__init__()
        self.block_channels = block_channels
        self.conv1 = nn.Conv1d(1, block_channels, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.blocks = nn.Sequential()
        for _ in range(NUM_BLOCKS):
            self.blocks.append(Block(in_channels=block_channels, out_channels=block_channels, kernel_size=kernel_size))

        num_features = 160
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(num_features, num_features)
        self.linear2 = nn.Linear(num_features, 1)
        self.sigmoid = nn.Sigmoid()
        self.num_features = num_features

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.blocks(out)
        out = self.flatten(out)
        out = self.linear(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out


def cnn_1D(**kwargs):
    """ Constructs a 1D CNN model. """
    model = CNN1D(**kwargs)
    return model
