# https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.CIFAR10


## Imports

#%%
import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Lambda, Compose
from torchvision import datasets

## load data (download data from UToronto)
#%%
training_data = CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

## Test run: load
#%%

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break
