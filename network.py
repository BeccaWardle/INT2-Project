#!/usr/bin/env python3
from torch.nn import \
  Conv2d, ReLU, Linear, MaxPool2d, Module, Flatten, Sequential, BatchNorm2d, Dropout2d, Dropout, \
  GELU

class Network(Module):

  def __init__(self):
    super(Network, self).__init__()

    # self.pool = MaxPool2d(2)  # 2*2 max pooling

    self.cnn_relu_stack = Sequential(

      # Conv Layer block 1
      Conv2d(3, 32, 3, 1),
      BatchNorm2d(32),
      ReLU(inplace=True),
      Conv2d(32, 128, 3, 1), # 64
      ReLU(inplace=True),
      MaxPool2d(2, 2),
      Dropout(p=0.1),

      # Conv Layer block 2
      Conv2d(128, 192, 3, 1),
      BatchNorm2d(192),
      GELU(),
      Dropout(p=0.16),
      Conv2d(192, 224, 3, 1),
      GELU(),
      MaxPool2d(2, 2),
      Dropout2d(p=0.05),
      # Dropout(p=0.2),

      # Conv Layer block 3
      Conv2d(224, 256, 3, 1),
      BatchNorm2d(256),
      ReLU(inplace=True),
      Conv2d(256, 288, 3, 1),
      ReLU(inplace=True),
      MaxPool2d(3, 2),
      Dropout2d(p=0.1),

      Flatten(),

      Dropout(p=0.1),
      Linear(4096, 2048),
      ReLU(inplace=True),
      Linear(2048, 512),
      ReLU(inplace=True),
      Dropout(p=0.05),
      Linear(512, 10),
    )

  def forward(self, x):
    return self.cnn_relu_stack(x)
