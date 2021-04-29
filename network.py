#!/usr/bin/env python3
from torch.nn import \
  Conv2d, ReLU, Linear, MaxPool2d, Module, Flatten, Sequential

class Network(Module):

  def __init__(self):
    super(Network, self).__init__()

    self.pool = MaxPool2d(2)  # 2*2 max pooling

    self.cnn_relu_stack = Sequential(
      Conv2d(3, 16, 3),  # 3 in-channel, 16 out-channel, number of kernel
      ReLU(),
      self.pool,
      Conv2d(16, 24, 4),
      ReLU(),
      self.pool,
      Conv2d(24, 56, 3),
      ReLU(),
      self.pool,
      Flatten(),
      Linear(224, 64),
      ReLU(),
      Linear(64, 48),
      ReLU(),
      Linear(48, 10),  # 10 classes, final output
    )

  def forward(self, x):
    return self.cnn_relu_stack(x)
