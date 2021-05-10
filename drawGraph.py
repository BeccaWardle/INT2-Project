#!/usr/bin/env python3
# Dataset (in PyTorch)
# https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.CIFAR10

# Dataset homepage
# https://www.cs.toronto.edu/~kriz/cifar.html

# %%
# Imports

import csv
import datetime
from time import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

import torch
import torchvision
import torchvision.transforms as transforms

import network

script_start = time()
print(f"Started: {datetime.datetime.now()}")

# continue training
cont = False

# %%
# Hardware acceleration
## accuracy vs epoch recording
epoch_accuracy_pair = []

#%%
## Hardware acceleration

torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# normalise the data
transform = transforms.Compose(
    [transforms.Resize(64),
    transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# %%
# load data (download data from UToronto)
training_data = CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transform,
)

test_data = CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=transform,
)

# load the dataset, describe the shape
# %%

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# image is of 32x32 with 3 channel for colours
# batch size is 64, but can be modified

for X, y in test_dataloader:

    print("Shape of X [n, Channels, Height, Width]: ", X.shape, X.dtype)
    print("Shape of y: ", y.shape, y.dtype)  # classification
    break

# %%
# Visualisation

# train_features, train_labels = next(iter(train_dataloader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# img = train_features[0].squeeze().permute(1,2,0)
# label = train_labels[0]
# plt.imshow(img)
# plt.show()
# print(f"Label: {label}")

# %%

network_model = network.Network()

# continue training -> load previous model
if cont:
    print("continuing previous progress.")
    network_model.load_state_dict(torch.load("model.pth"))
    network_model.eval()

network_model.to(device)  # send tensors to CUDA cores
print(network_model)

# logits = network_model(train_dataloader)
# pred_probab = nn.Softmax(dim=1)(logits)

# define hyper-parameters

batch_size = 64
learning_rate = 0.03

cross_entropy_loss = nn.CrossEntropyLoss()
stochastic_GD = torch.optim.SGD(network_model.parameters(), lr=learning_rate)

epochs = 50
max_accuracy = 0
consecutive = 0
max_consecutive = 50

from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/fashion_mnist_experiment_1')

# get some random training images
dataiter = iter(train_dataloader)
images, labels = dataiter.next()

# create grid of images
img_grid = torchvision.utils.make_grid(images)


# write to tensorboard
writer.add_image('four_fashion_mnist_images', img_grid)
print("print successfully")
# tensorboard --logdir=runs

import network
net = network.Network()

writer.add_graph(net, images)
writer.close()

