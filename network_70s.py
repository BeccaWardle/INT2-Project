#!/usr/bin/env python3
from torch.nn import \
  Conv2d, ReLU, Linear, MaxPool2d, Module, Flatten, Sequential, BatchNorm2d, Dropout, Dropout2d


class Network(Module):
  def __init__(self):
    super(Network, self).__init__()

    self.conv_layer = Sequential(

      # Conv Layer block 1
      Conv2d(3, 32, 3, 1),
      BatchNorm2d(32),
      ReLU(inplace=True),
      Conv2d(32, 64, 3, 1),
      ReLU(inplace=True),
      MaxPool2d(2, 2),

      # Conv Layer block 2
      Conv2d(64, 128, 3, 1),
      BatchNorm2d(128),
      ReLU(inplace=True),
      Conv2d(128, 128, 3, 1),
      ReLU(inplace=True),
      MaxPool2d(2, 2),
      Dropout2d(p=0.05),

      # Conv Layer block 3
      Conv2d(128, 256, 3, 1),
      BatchNorm2d(256),
      ReLU(inplace=True),
      Conv2d(256, 256, 3, 1),
      ReLU(inplace=True),
      MaxPool2d(2, 2),
    )


    self.fc_layer = Sequential(
      Dropout(p=0.1),
      Linear(4096, 1024),
      ReLU(inplace=True),
      Linear(1024, 512),
      ReLU(inplace=True),
      Dropout(p=0.1),
      Linear(512, 10)
    )


  def forward(self, x):
    """Perform forward."""
    
    # conv layers
    x = self.conv_layer(x)
    
    # flatten
    x = x.view(x.size(0), -1)
    
    # fc layer
    x = self.fc_layer(x)

    return x
