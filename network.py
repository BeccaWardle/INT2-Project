#!/usr/bin/env python3
from torch.nn import \
  Conv2d, ReLU, Linear, MaxPool2d, Module, Flatten, Sequential, Dropout2d, BatchNorm2d

class Network(Module):

  def __init__(self):
    super(Network, self).__init__()

    # self.pool = MaxPool2d(2)  # 2*2 max pooling

    self.cnn_relu_stack = Sequential(

      #32x32x3 --> 32x32x32
      Conv2d(3, 32, 3, padding = 1), # 3 in-channel, 32 out-channel, number of kernel
      #32x32x32 --> 30x30x64
      Conv2d(32, 64, 3),
      #30x30x64 --> 28x28x64
      Conv2d(64, 64, 3),
      ReLU(),
      #28x28x64 --> 14x14x64
      MaxPool2d(2, 2), 
      #14x14x64 --> 12x12x128
      Conv2d(64, 128, 3),
      #12x12x128 --> 10x10x128
      Conv2d(128, 128, 3),
      #Conv2d(128, 128, 1),
      ReLU(),
      #10x10x128 --> 5x5x128
      MaxPool2d(2, 2),
      Dropout2d(),
      ReLU(),
      Flatten(),
      Linear(3200, 400),
      ReLU(),
      Linear(400, 200),
      ReLU(),
      Linear(200, 10),  # 10 classes, final output
    )

  def forward(self, x):
    return self.cnn_relu_stack(x)
