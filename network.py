#!/usr/bin/env python3
from torch.nn import \
  Conv2d, ReLU, Linear, MaxPool2d, Module, Flatten, Sequential, BatchNorm2d, Dropout2d, Dropout, \
  LeakyReLU

class Network(Module):

  def __init__(self):
    super(Network, self).__init__()
    self.__version__ = "1.7"

    # self.pool = MaxPool2d(2)  # 2*2 max pooling

    self.cnn_relu_stack = Sequential(

      # Conv Layer block 1 -- feature extraction
      Conv2d(3, 32, 3, 1),
      BatchNorm2d(32),
      ReLU(inplace=True),
      Conv2d(32, 128, 3, 1), # 64
      ReLU(inplace=True),
      MaxPool2d(2, 2),
      Dropout2d(p=0.1),

      # Conv Layer block 2
      Conv2d(128, 128, 3, 1),
      BatchNorm2d(128),
      ReLU(inplace=True),
      Dropout2d(p=0.16),
      Conv2d(128, 256, 3, 1),
      ReLU(inplace=True),
      MaxPool2d(3, 2),
      Dropout2d(p=0.2),

      # Conv Layer block 3
      Conv2d(256, 256, 3, 1),
      BatchNorm2d(256),
      ReLU(inplace=True),
      Dropout2d(p=0.2),
      Conv2d(256, 384, 3, 1),
      ReLU(inplace=True),
      MaxPool2d(3, 2),
      Dropout2d(p=0.16),

      Flatten(),

      Dropout(p=0.05),
      Linear(3456, 2560),
      LeakyReLU(inplace=True),
      Dropout(p=0.25),
      Linear(2560, 384),
      LeakyReLU(inplace=True),
      Dropout(p=0.2),
      Linear(384, 10),
      # softmax (?)
    )

  def forward(self, x):
    return self.cnn_relu_stack(x)
