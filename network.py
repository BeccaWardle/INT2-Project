#!/usr/bin/env python3
from torch.nn import \
  Conv2d, ReLU, Linear, MaxPool2d, Module, Flatten, Sequential, BatchNorm2d, Dropout2d, Dropout

class Network(Module):

  def __init__(self):
    super(Network, self).__init__()

    # self.pool = MaxPool2d(2)  # 2*2 max pooling

    self.cnn_relu_stack = Sequential(

      # 3 in-channel, 16 out-channel, number of kernel

      Conv2d(3, 32, 3, stride=2),ReLU(),# MaxPool2d(2),
      Conv2d(32, 32, 3), ReLU(),# MaxPool2d(2),
      BatchNorm2d(32), # re-normalise

      Conv2d(32, 128, 3), ReLU(), MaxPool2d(2),
      Conv2d(128, 224, 3), ReLU(),# MaxPool2d(2),
      Dropout2d(0.1), # regularise by drop-out
      BatchNorm2d(224),

      Conv2d(224, 896, 3), ReLU(),# MaxPool2d(2),
      BatchNorm2d(896),
      Dropout2d(0.2),

      Flatten(),

      Linear(896, 360),
      ReLU(),
      Linear(360, 64),
      ReLU(),
      Linear(64, 10),  # 10 classes, final output
    )

  def forward(self, x):
    return self.cnn_relu_stack(x)
