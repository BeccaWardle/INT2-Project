#!/usr/bin/env python3
from torch.nn import \
    Conv2d, LeakyReLU, Linear, MaxPool2d, Module, Flatten, Sequential, BatchNorm2d, Dropout, Dropout2d


class Will_Network(Module):
    def __init__(self):
        self.name = "Will_net"
        self.version = 1.0
        super(Will_Network, self).__init__()

        self.conv_layer = Sequential(

            # Conv Layer block 1
            Conv2d(3, 32, 3, 1),
            BatchNorm2d(32),
            LeakyReLU(inplace=True),
            Conv2d(32, 64, 3, 1),
            LeakyReLU(inplace=True),
            MaxPool2d(2, 2),
            Dropout(p=0.2),

            # Conv Layer block 2
            Conv2d(64, 128, 3, 1),
            BatchNorm2d(128),
            LeakyReLU(inplace=True),
            Conv2d(128, 128, 3, 1),
            LeakyReLU(inplace=True),
            MaxPool2d(2, 2),
            Dropout2d(p=0.05),
            # Dropout(p=0.2),

            # Conv Layer block 3
            Conv2d(128, 256, 3, 1),
            BatchNorm2d(256),
            LeakyReLU(inplace=True),
            Conv2d(256, 256, 3, 1),
            LeakyReLU(inplace=True),
            MaxPool2d(2, 2),
            Dropout(p=0.2)
        )

        self.fc_layer = Sequential(
            Dropout(p=0.1),
            Linear(4096, 1024),
            LeakyReLU(inplace=True),
            Linear(1024, 512),
            LeakyReLU(inplace=True),
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


class Lexfee_Network(Module):
    def __init__(self):
        self.version = 1.0
        super(Lexfee_Network, self).__init__()

    # self.pool = MaxPool2d(2)  # 2*2 max pooling

        self.cnn_relu_stack = Sequential(

            # Conv Layer block 1
            Conv2d(3, 32, 3, 1),
            BatchNorm2d(32),
            LeakyReLU(inplace=True),
            Conv2d(32, 128, 3, 1),  # 64
            LeakyReLU(inplace=True),
            MaxPool2d(2, 2),
            Dropout(p=0.1),

            # Conv Layer block 2
            Conv2d(128, 192, 3, 1),
            BatchNorm2d(192),
            LeakyReLU(inplace=True),
            Dropout(p=0.16),
            Conv2d(192, 256, 3, 1),
            LeakyReLU(inplace=True),
            MaxPool2d(2, 2),
            Dropout2d(p=0.1),
            # Dropout(p=0.2),

            # Conv Layer block 3
            Conv2d(256, 512, 3, 1),
            BatchNorm2d(512),
            LeakyReLU(inplace=True),
            Conv2d(512, 512, 3, 1),
            LeakyReLU(inplace=True),
            MaxPool2d(3, 2),
            Dropout2d(p=0.2),

            Flatten(),

            Dropout(p=0.1),
            Linear(8192, 2048),
            LeakyReLU(inplace=True),
            Linear(2048, 512),
            LeakyReLU(inplace=True),
            Dropout(p=0.1),
            Linear(512, 10),
            # softmax (?)
        )

    def forward(self, x):
        return self.cnn_relu_stack(x)


class Zijun_Network(Module):
    def __init__(self):
        self.version = 1.0
        super(Zijun_Network, self).__init__()

        # self.pool = MaxPool2d(2)  # 2*2 max poolin
        self.cnn_relu_stack = Sequential(

            # 32x32x3 --> 32x32x32
            # 3 in-channel, 32 out-channel, number of kernel
            Conv2d(3, 32, 3, padding=1),
            # 32x32x32 --> 30x30x64
            Conv2d(32, 64, 3),
            # 30x30x64 --> 28x28x64
            Conv2d(64, 64, 3),
            LeakyReLU(),
            # 28x28x64 --> 14x14x64
            MaxPool2d(2, 2),
            # 14x14x64 --> 12x12x128
            Conv2d(64, 128, 3),
            # 12x12x128 --> 10x10x128
            Conv2d(128, 128, 3),
            # Conv2d(128, 128, 1),
            LeakyReLU(),
            # 10x10x128 --> 5x5x128
            MaxPool2d(2, 2),
            Dropout2d(),
            LeakyReLU(),
            Flatten(),
            Linear(3200, 400),
            LeakyReLU(),
            Linear(400, 200),
            LeakyReLU(),
            Linear(200, 10),  # 10 classes, final output
        )

    def forward(self, x):
        return self.cnn_relu_stack(x)
