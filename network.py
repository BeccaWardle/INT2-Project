#!/usr/bin/env python3
from torch.nn import \
  Conv2d, ReLU, Linear, MaxPool2d, Module, Flatten, Sequential, BatchNorm2d, Dropout2d, Dropout, \
  LeakyReLU

class Network(Module):

  def __init__(self):
    super(Network, self).__init__()
    self.__version__ = "1.22-optim-var-2"

    # self.pool = MaxPool2d(2)  # 2*2 max pooling

    self.cnn_relu_stack = Sequential(

      # Conv Layer block 1 -- feature extraction
      Conv2d(3, 16, 3, 1),
      LeakyReLU(inplace=True),
      BatchNorm2d(16),
      Conv2d(16, 16, 3, 1), # 64
      LeakyReLU(inplace=True),
      # MaxPool2d(2, 2),
      Dropout2d(p=0.25),

      # Conv Layer block 2
      Conv2d(16, 64, 3, 1),
      LeakyReLU(inplace=True),
      BatchNorm2d(64),
      Dropout(p=0.4),
      Conv2d(64, 64, 3, 1),
      LeakyReLU(inplace=True),
      MaxPool2d(2, 2),
      Dropout2d(p=0.35),

      # Conv Layer block 3
      Conv2d(64, 128, 3, 1),
      LeakyReLU(inplace=True),
      BatchNorm2d(128),
      Dropout(p=0.3),
      Conv2d(128, 128, 3, 1),
      LeakyReLU(inplace=True),
      MaxPool2d(3, 2),
      Dropout2d(p=0.375),

      Flatten(),

      Dropout(p=0.2),
      Linear(1152, 1024),
      LeakyReLU(inplace=True),
      Dropout(p=0.4),
      Linear(1024, 128),
      LeakyReLU(inplace=True),
      Dropout(p=0.25),
      Linear(128, 10),
      # softmax (?)
    )

  def forward(self, x):
    return self.cnn_relu_stack(x)
