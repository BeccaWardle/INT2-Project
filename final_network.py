import torch
from torch.autograd import Variable
from torch.nn import \
  Conv2d, ReLU, Linear, MaxPool2d, Module, Flatten, Sequential, BatchNorm2d, Dropout, Dropout2d, LeakyReLU
from torchviz import make_dot, make_dot_from_trace

class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        k = lambda chan: 3 * (2 ** chan)  # channel multiplier
        filter_dropout_p = 0.48
        linear_dropout_p = 0.4

        self.cnn_relu_block_1 = Sequential(
            Conv2d(in_channels=k(0), out_channels=k(3), kernel_size=(3, 3), stride=(1, 1)), LeakyReLU(),  # no inplace.
            BatchNorm2d(k(3)),

            # 2
            Conv2d(k(3), k(5), (3, 3), (1, 1)), LeakyReLU(),
            BatchNorm2d(k(5)),

            # 3
            Conv2d(k(5), k(5), (3, 3), (1, 1)), LeakyReLU(),  # increase filter size
            BatchNorm2d(k(5)),

            Dropout(p=0.35),
        )

        self.cnn_relu_block_2 = Sequential(
            Conv2d(k(5), k(6), (3, 3), (1, 1)), LeakyReLU(),
            BatchNorm2d(k(6)),
            MaxPool2d(2, 2),  # subsampling, reduces parameter size, increase performance, halfs the size

            Dropout2d(filter_dropout_p),  # drop out entire filters

            # 5
            Conv2d(k(6), k(6), (3, 3), (1, 1)), LeakyReLU(),
            BatchNorm2d(k(6)),
            )

        self.cnn_relu_block_3 = Sequential(
            # Conv Layer block 3
            Conv2d(k(6), k(7), (3, 3), (1, 1)), LeakyReLU(),
            BatchNorm2d(k(7)),
            # MaxPool2d(2, 2),  # subsampling, reduces parameter size, increase performance

            # 7
            Conv2d(k(7), k(7), (3, 3), (1, 1)), LeakyReLU(),
            MaxPool2d(3, 2),  # subsampling, reduces parameter size, increase performance
            Dropout2d(filter_dropout_p),  # drop out entire filters
            )

        self.cnn_flatten = Sequential(
            Flatten(),
        )
        self.cnn_relu_block_linear = Sequential(
            Linear(1536, 1024),
            LeakyReLU(),
            Dropout(linear_dropout_p),
            Linear(1024, 128),
            LeakyReLU(),
            Linear(128, 10),
        )

    def forward(self, x):
        x = self.cnn_relu_block_1(x)
        x = self.cnn_relu_block_2(x)
        x = self.cnn_relu_block_3(x)
        #  print(f"{x.size()}")
        x = self.cnn_flatten(x)
        # print(f"{x.size()}")
        x = self.cnn_relu_block_linear(x)
        return x