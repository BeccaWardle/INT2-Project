#!/usr/bin/env python3
from torch.nn import \
  Conv2d, ReLU, Linear, MaxPool2d, Module, Flatten, Sequential, BatchNorm2d, Dropout2d, Dropout, \
  LeakyReLU

class Network(Module):

  def __init__(self):

    super(Network, self).__init__()

    # 2.0: redesign entire network
    self.__version__ = "2.1"

    """
    Conv2d -> LeakyReLU -> Conv2d -> LeakyReLU -> Normalise
    Conv2d -> LeakyReLU -> Conv2d  -> LeakyReLU -> Normalise -> MaxPool2d 4,2
      Dropout2d 0.48
    Conv2d -> LeakyReLU -> Normalise -> MaxPool2d 4,2
      Dropout2d 0.48
    
    Flatten to 1D
    
    Reduction: _ -> 1024 -> 512 -> 10
    """

    k = lambda chan: 3 * (2**chan) # channel multiplier
    filter_dropout_p = 0.375
    linear_dropout_p = 0.4

    # CNN
    self.cnn_relu_stack = Sequential(

      #1
      Conv2d(in_channels=k(0), out_channels=k(3), kernel_size=(3,3), stride=(1,1)),
      LeakyReLU(), # no inplace.

      #2
      Conv2d(k(3), k(3), (3, 3), (1,1)),
      LeakyReLU(),

      BatchNorm2d(k(3)),

      #3
      Conv2d(k(3), k(5), (3, 3), (1, 1)), # increase filter size
      LeakyReLU(),

      #4
      Conv2d(k(5), k(5), (3, 3), (1, 1)),
      LeakyReLU(),

      BatchNorm2d(k(5)),

      MaxPool2d(2, 2), # subsampling, reduces parameter size, increase performance, halfs the size

      Dropout2d(filter_dropout_p), # drop out entire filters

      #5
      Conv2d(k(5), k(6), (3, 3), (1, 1)),
      LeakyReLU(),

      BatchNorm2d(k(6)),

      # MaxPool2d(2, 2),  # subsampling, reduces parameter size, increase performance

      #6
      Conv2d(k(6), k(6), (3, 3), (1, 1)),
      LeakyReLU(),

      #7
      Conv2d(k(6), k(6), (3, 3), (1, 1)),
      LeakyReLU(),
      BatchNorm2d(k(6)),
      #MaxPool2d(2, 2),  # subsampling, reduces parameter size, increase performance

      #8
      Conv2d(k(6), k(7), (3, 3), (1, 1)),
      LeakyReLU(),
      MaxPool2d(2, 2),  # subsampling, reduces parameter size, increase performance
      Dropout2d(filter_dropout_p),  # drop out entire filters

    )

    # FC reduction
    self.reduction_stack = Sequential(
      Linear(1536, 1024),
      LeakyReLU(),
      Dropout(linear_dropout_p),
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
