#!/usr/bin/env python3
from torch.nn import \
  Conv2d, ReLU, Linear, MaxPool2d, Module, Flatten, Sequential, BatchNorm2d

class Network(Module):

  def __init__(self):
    super(Network, self).__init__()

    # self.pool = MaxPool2d(2)  # 2*2 max pooling

    self.cnn_relu_stack = Sequential(
      Conv2d(3, 32, 3),  # 3 in-channel, 16 out-channel, number of kernel
      ReLU(),
      MaxPool2d(2),
      Conv2d(32, 56, 3),
      ReLU(),
      MaxPool2d(2),
      Conv2d(56, 112, 3),
      ReLU(),
      MaxPool2d(2),
      Conv2d(112, 256, 2),
      ReLU(),
      BatchNorm2d(256),
      Flatten(),
      Linear(256, 128),
      ReLU(),
      Linear(128, 64),
      ReLU(),
      Linear(64, 10),  # 10 classes, final output
    )

  def forward(self, x):
    return self.cnn_relu_stack(x)
