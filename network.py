#!/usr/bin/env python3
from torch.nn import \
  Conv2d, ReLU, Linear, MaxPool2d, Module, Flatten, Sequential, BatchNorm2d, Dropout2d

class Network(Module):

  def __init__(self):
    super(Network, self).__init__()

    # self.pool = MaxPool2d(2)  # 2*2 max pooling

    self.cnn_relu_stack = Sequential(
      Conv2d(3, 32, (3,3)),  # 3 in-channel, 16 out-channel, number of kernel
      ReLU(),
      MaxPool2d(2),
      Conv2d(32, 128, (3,3)),
      ReLU(),
      MaxPool2d(2),
      BatchNorm2d(128),
      Conv2d(128, 256, (3,3)),
      ReLU(),
      # MaxPool2d(2),
      Dropout2d(0.01),
      Conv2d(256, 384, (3,3)),
      ReLU(),
      Dropout2d(0.025),
      BatchNorm2d(384),
      Flatten(),
      Linear(1536, 512),
      ReLU(),
      Linear(512, 96),
      ReLU(),
      Linear(96, 10),  # 10 classes, final output
    )

  def forward(self, x):
    return self.cnn_relu_stack(x)
