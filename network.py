#!/usr/bin/env python3
from torch.nn import (BatchNorm2d, Conv2d, Dropout, Dropout2d, Flatten,
                      LeakyReLU, Linear, MaxPool2d, Module, Sequential)


class Will(Module):
    def __init__(self):
        self.name = "Will"
        self.__version__ = 1.0
        super(Will, self).__init__()

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


class Lexffe(Module):
    def __init__(self):
        self.name = "Lexffe"
        super(Lexffe, self).__init__()
        self.__version__ = "1.19"

        # self.pool = MaxPool2d(2)  # 2*2 max pooling

        self.cnn_relu_stack = Sequential(

        # Conv Layer block 1 -- feature extraction
        Conv2d(3, 32, 3, 1),
        BatchNorm2d(32),
        LeakyReLU(inplace=True),
        Conv2d(32, 128, 3, 1), # 64
        LeakyReLU(inplace=True),
        MaxPool2d(3, 2),
        Dropout2d(p=0.25),

        # Conv Layer block 2
        Conv2d(128, 256, 3, 1),
        BatchNorm2d(256),
        LeakyReLU(inplace=True),
        MaxPool2d(3, 2),
        Dropout(p=0.4),
        Conv2d(256, 256, 3, 1),
        LeakyReLU(inplace=True),
        MaxPool2d(4, 2),
        Dropout2d(p=0.25),

        # # Conv Layer block 3
        # Conv2d(128, 128, 3, 1),
        # BatchNorm2d(128),
        # LeakyReLU(inplace=True),
        # Dropout2d(p=0.3),
        # MaxPool2d(4, 2),

        Flatten(),

        Dropout(p=0.2),
        Linear(4096, 1024),
        LeakyReLU(inplace=True),
        Dropout(p=0.32),
        Linear(1024, 128),
        LeakyReLU(inplace=True),
        Dropout(p=0.2),
        Linear(128, 10),
        # softmax (?)
        )

    def forward(self, x):
        return self.cnn_relu_stack(x)


class Becca(Module):
    def __init__(self):
        self.name = "becca"
        self.__version__ = 1.32
        super(Becca, self).__init__()

        self.cnn_relu_block_1 = Sequential(
            # Conv Layer block 1 -- feature extraction
            Conv2d(3, 128, 3, 1),
            BatchNorm2d(128),
            MaxPool2d(2, 2),
            Conv2d(128, 256, 4, 1), # 64
            LeakyReLU(inplace=True),
            MaxPool2d(2, 2),
            Dropout2d(p=0.2),
        )

        self.cnn_relu_block_2 = Sequential(
            # Conv Layer block 2
            Conv2d(256, 256, 4, 1),
            BatchNorm2d(256),
            LeakyReLU(inplace=True),
            Dropout(p=0.3),
            Conv2d(256, 512, 7, 1),
            LeakyReLU(inplace=True),
            MaxPool2d(3, 2),
            Dropout2d(p=0.2),
        )

        self.cnn_relu_block_3 = Sequential(
            # Co # Conv Layer block 3
            Conv2d(512, 512, 6, 1),
            BatchNorm2d(512),
            LeakyReLU(inplace=True),
            Dropout2d(p=0.3),
            MaxPool2d(4, 2),
        )

        self.cnn_flatten = Sequential(
            Flatten(),
        )
        self.cnn_relu_block_linear = Sequential(
            Dropout(p=0.25),
            Linear(2048, 1024),
            LeakyReLU(inplace=True),
            Dropout(p=0.5),
            Linear(1024, 768    ),
            LeakyReLU(inplace=True),
            Dropout(p=0.3),
            Linear(768, 10),
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


class Becca_long(Module):
    def __init__(self):
        self.name = "becca_long"
        self.__version__ = 1.1
        super(Becca_long, self).__init__()

        self.cnn_relu_block_1 = Sequential(
            # Conv Layer block 1
            Conv2d(3, 64, 3, 1),
            BatchNorm2d(64),
            LeakyReLU(inplace=True),
            Conv2d(64, 128, 3, 1, padding=2),  # 64
            LeakyReLU(inplace=True),
            MaxPool2d(2, 2),
            Dropout2d(p=0.16),
        )

        self.cnn_relu_block_2 = Sequential(
            # Conv Layer block 2
            Conv2d(128, 256, 3, 1, padding=2),
            LeakyReLU(inplace=True),
            MaxPool2d(2, 2),
            Dropout2d(p=0.2),
            # Dropout(p=0.2),
        )

        self.cnn_relu_block_3 = Sequential(
            # Conv Layer block 3
            Conv2d(256, 256, 3, 1),
            BatchNorm2d(256),
            LeakyReLU(inplace=True),
            Dropout2d(p=0.2),
            Conv2d(256, 256, 3, 1, padding=2),
            LeakyReLU(inplace=True),
            MaxPool2d(3, 2),
            Dropout2d(p=0.25),
        )

        self.cnn_relu_block_4 = Sequential(
            # Conv Layer block 4
            Conv2d(256, 256, 3, 1),
            LeakyReLU(inplace=True),
            Conv2d(256, 512, 3, 1, padding=2),
            LeakyReLU(inplace=True),
            MaxPool2d(3, 2),
        )

        self.cnn_relu_block_5 = Sequential(
            # Conv Layer block 4
            Conv2d(512, 512, 3, 1, padding=2),
            LeakyReLU(inplace=True),
            MaxPool2d(3, 2),
        )

        self.cnn_flatten = Sequential(
            Flatten(),
        )

        self.cnn_relu_block_linear = Sequential(
            Linear(2048, 1024),
            LeakyReLU(inplace=True),
            Linear(1024, 512),
            LeakyReLU(inplace=True),
            Dropout(p=0.1),
            Linear(512, 384),
            LeakyReLU(inplace=True),
            Dropout(p=0.16),
            Linear(384, 256),
            LeakyReLU(inplace=True),
            Linear(256, 10),
        )

    def forward(self, x):
        x = self.cnn_relu_block_1(x)
        x = self.cnn_relu_block_2(x)
        x = self.cnn_relu_block_3(x)
        x = self.cnn_relu_block_4(x)
        x = self.cnn_relu_block_5(x)
        # print(f"{x.size()}")
        x = self.cnn_flatten(x)
        # print(f"{x.size()}")
        x = self.cnn_relu_block_linear(x)
        return x


class Zijun(Module):
    def __init__(self):
        self.name = "Zijun"
        self.__version__ = 1.0
        super(Zijun, self).__init__()

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
