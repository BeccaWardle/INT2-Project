#!/usr/bin/env python3

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch import utils
import matplotlib.pyplot as plt
import numpy as np

model_name = "1620682577"
version = "2.1"
tensorboard_log = f"tensorboard/model_{model_name}"
batch_size = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1):
    n, c, w, h = tensor.shape

    if allkernels:
        tensor = tensor.view(n * c, -1, w, h)
    elif c != 3:
        tensor = tensor[:, ch, :, :].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure(figsize=(nrow, rows))
    plt.imshow(grid.numpy().transpose((1, 2, 0)))


layer = 1

model = torch.load(f"result/network.{model_name}.{version}.pth", map_location=torch.device(device))
model.eval()

filter = model.features[layer].weight.data.clone()
visTensor(filter, ch=0, allkernels=False)

plt.axis('off')
plt.ioff()
plt.show()
