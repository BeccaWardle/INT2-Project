#!/usr/bin/env python3
from torch.nn import (BatchNorm2d, Conv2d, Dropout, Dropout2d, Flatten,
                      ReLU, LeakyReLU, Linear, MaxPool2d, Module, Sequential)
from torchviz import make_dot, make_dot_from_trace


class Will(Module):
    def __init__(self):
        super(Will, self).__init__()
        self.name = "Will"
        self.__version__ = "1.1"
        # self.pool = MaxPool2d(2)  # 2*2 max pooling

        self.cnn_relu_stack = Sequential(
            Conv2d(3, 16, 3),  # 3 in-channel, 16 out-channel, number of kernel
            ReLU(),
            # MaxPool2d(2),
            Conv2d(16, 24, 4),
            ReLU(),
            MaxPool2d(2),
            Conv2d(24, 56, 5),
            ReLU(),
            MaxPool2d(2),
            Conv2d(56, 112, 3),
            ReLU(),
            Flatten(),
            Linear(448, 240),
            ReLU(),
            Linear(240, 120),
            ReLU(),
            Linear(120, 10),  # 10 classes, final output
        )

    def forward(self, x):
        return self.cnn_relu_stack(x)


class Lexffe(Module):
    def __init__(self):
        self.name = "Lexffe"
        super(Lexffe, self).__init__()
        self.__version__ = "1.25.1"

        # self.pool = MaxPool2d(2)  # 2*2 max pooling

        self.cnn_relu_stack = Sequential(

            # Conv Layer block 1 -- feature extraction
            Conv2d(3, 32, 3, 1),
            BatchNorm2d(32),
            LeakyReLU(inplace=True),
            Conv2d(32, 128, 3, 1), # 64
            LeakyReLU(inplace=True),
            MaxPool2d(2, 2),
            Dropout2d(p=0.25),

            # Conv Layer block 2
            Conv2d(128, 128, 3, 1),
            BatchNorm2d(128),
            LeakyReLU(inplace=True),
            Dropout(p=0.4),
            Conv2d(128, 128, 3, 1),
            LeakyReLU(inplace=True),
            MaxPool2d(3, 2),
            Dropout2d(p=0.35),

            # Conv Layer block 3
            # Conv2d(128, 256, 3, 1),
            # BatchNorm2d(256),
            # LeakyReLU(inplace=True),
            # Dropout(p=0.3),
            Conv2d(128, 256, 3, 1),
            BatchNorm2d(256),
            LeakyReLU(inplace=True),
            MaxPool2d(4, 2),
            Dropout2d(p=0.375),

            Flatten(),

            Dropout(p=0.2),
            Linear(4096, 1024),
            LeakyReLU(inplace=True),
            Dropout(p=0.375),
            Linear(1024, 128),
            LeakyReLU(inplace=True),
            Dropout(p=0.25),
            Linear(128, 10),
            # softmax (?)
        )

    def forward(self, x):
        return self.cnn_relu_stack(x)


class Becca(Module):
    def __init__(self):
        self.name = "Becca"
        self.__version__ = "1.41"
        super(Becca, self).__init__()

        self.cnn_relu_block_1 = Sequential(
            # Conv Layer block 1 -- feature extraction
            Conv2d(3, 32, 3, 1),
            BatchNorm2d(32),
            LeakyReLU(inplace=True),
            Conv2d(32, 128, 3, 1),
            LeakyReLU(inplace=True),
            MaxPool2d(4, 3),
            Dropout2d(p=0.25),

        )

        self.cnn_relu_block_2 = Sequential(
            # Conv Layer block 2
            Conv2d(128, 256, 3, 1),
            BatchNorm2d(256),
            LeakyReLU(inplace=True),
            Dropout(p=0.4),
            Conv2d(256, 256, 4, 1),
            LeakyReLU(inplace=True),
            MaxPool2d(5, 3),
            Dropout2d(p=0.5),
        )

        self.cnn_relu_block_3 = Sequential(
            # Conv Layer block 3
            Conv2d(256, 384, 4, 1),
            BatchNorm2d(384),
            LeakyReLU(inplace=True),
            Dropout(p=0.5),
            Conv2d(384, 384, 4, 1),
            LeakyReLU(inplace=True),
            MaxPool2d(5, 3),
            Dropout2d(p=0.375),
        )

        self.cnn_flatten = Sequential(
            Flatten(),
        )
        self.cnn_relu_block_linear = Sequential(
            Dropout(p=0.2),
            Linear(4096, 1024),
            LeakyReLU(inplace=True),
            Dropout(p=0.5),
            Linear(1024, 128),
            LeakyReLU(inplace=True),
            Dropout(p=0.4),
            Linear(128, 10),
            # softmax (?)
        )

    def forward(self, x):
        x = self.cnn_relu_block_1(x)
        x = self.cnn_relu_block_2(x)
        # x = self.cnn_relu_block_3(x)
        #  print(f"{x.size()}")
        x = self.cnn_flatten(x)
        # print(f"{x.size()}")
        x = self.cnn_relu_block_linear(x)
        return x


class Zijun(Module):
    def __init__(self):
        super(Zijun, self).__init__()
        self.name = "Zijun"
        self.__version__ = "1.1"
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
