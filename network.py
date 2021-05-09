#!/usr/bin/env python3
import torch
from torch.autograd import Variable
from torch.nn import \
  Conv2d, ReLU, Linear, MaxPool2d, Module, Flatten, Sequential, BatchNorm2d, Dropout, Dropout2d, LeakyReLU
from torchviz import make_dot, make_dot_from_trace

class Network(Module):
  def __init__(self):
    super(Network, self).__init__()

    self.cnn_relu_stack = Sequential(

      # Conv Layer block 1
      Conv2d(3, 32, 3, 1),
      BatchNorm2d(32),
      LeakyReLU(inplace=True),
      Conv2d(32, 128, 3, 1), # 64
      LeakyReLU(inplace=True),
      MaxPool2d(2, 2),
      Dropout2d(p=0.16),

      # Conv Layer block 2
      Conv2d(128, 192, 3, 1),
      BatchNorm2d(192),
      LeakyReLU(inplace=True),
      Dropout2d(p=0.16),
      Conv2d(192, 384, 3, 1),
      LeakyReLU(inplace=True),
      MaxPool2d(2, 2),
      Dropout2d(p=0.2),
      # Dropout(p=0.2),

      # Conv Layer block 3
      Conv2d(384, 512, 3, 1),
      BatchNorm2d(512),
      LeakyReLU(inplace=True),
      Dropout2d(p=0.2),
      Conv2d(512, 1024, 3, 1),
      LeakyReLU(inplace=True),
      BatchNorm2d(1024),
      Dropout2d(p=0.16),

      Conv2d(1024, 1024, 3, 1),
      LeakyReLU(inplace=True),
      MaxPool2d(3, 2),
      Dropout2d(p=0.16),

      Flatten(),

      Dropout(p=0.16),
      Linear(9216, 4096),
      LeakyReLU(inplace=True),
      Linear(4096, 2048),
      LeakyReLU(inplace=True),
      Dropout(p=0.1),
      Linear(2048, 512),
      LeakyReLU(inplace=True),
      Dropout(p=0.16),
      Linear(512, 10),
      # softmax (?)
    )
          ## Conv Layer block 1
  
  def draw(self, y):
    model = self.cnn_relu_stack
    make_dot(y.mean(), params=dict(model.named_parameters()))

  def forward(self, x):
    return self.cnn_relu_stack(x)