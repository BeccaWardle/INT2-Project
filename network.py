#!/usr/bin/env python3
from torch.nn import \
  Conv2d, ReLU, Linear, MaxPool2d, Module, Flatten, Sequential, BatchNorm2d, Dropout2d, Dropout, \
  LeakyReLU

class Network(Module):

  def __init__(self):

    super(Network, self).__init__()

    # 2.0: redesign entire network
    self.__version__ = "2.0"

    """
    Conv2d 3:32 -> LeakyReLU -> Conv2d 32:32 -> LeakyReLU -> Normalise
    Conv2d 32:128 -> LeakyReLU -> Conv2d 128:128 -> LeakyReLU -> Normalise -> MaxPool2d 4,2
      Dropout2d 0.48
    Conv2d 128:192 -> LeakyReLU -> Normalise -> MaxPool2d 4,2
      Dropout2d 0.48
    
    Flatten to 1D
    
    Reduction: 1728 -> 1024 -> 512 -> 10
    """

    # CNN
    self.cnn_relu_stack = Sequential(
      Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), stride=(1,1)),
      LeakyReLU(), # no inplace.

      Conv2d(32, 32, (3,3), (1,1)),
      LeakyReLU(),

      BatchNorm2d(32),

      Conv2d(32, 128, (3, 3), (1, 1)), # increase filter size
      LeakyReLU(),

      Conv2d(128, 128, (3, 3), (1, 1)),
      LeakyReLU(),

      BatchNorm2d(128),

      MaxPool2d(4,2), # subsampling, reduces parameter size, increase performance (128 filters)

      Dropout2d(0.48), # drop out entire filters, p=0.48

      Conv2d(128, 192, (3, 3), (1, 1)),
      LeakyReLU(),

      BatchNorm2d(192),

      MaxPool2d(4, 2),  # subsampling, reduces parameter size, increase performance (128 filters)

      Dropout2d(0.48),  # drop out entire filters, p=0.48
    )

    # FC reduction
    self.reduction_stack = Sequential(
      Linear(1728, 1024),
      LeakyReLU(),
      Linear(1024, 512),
      LeakyReLU(),
      Linear(512, 10),
    )

    self.composite = Sequential(
      self.cnn_relu_stack,
      Flatten(),
      self.reduction_stack
    )

  def forward(self, x): return self.composite(x)
