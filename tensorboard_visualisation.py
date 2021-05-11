#!/usr/bin/env python3
import network

model_name = "1620581746"
version = "1.24"
tensorboard_log = f"tensorboard/model_{model_name}"
batch_size = 64
use_state_dict = False

####

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.tensorboard import SummaryWriter

# normalise the data
transform = transforms.Compose(
    [transforms.Resize((64,64)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

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

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


network_model = network.Network()

if use_state_dict:
    network_model.load_state_dict(torch.load(f"result/model.{model_name}.pth"))
else:
    network_model = torch.load(f"result/network.{model_name}.{version}.pth")

network_model.eval()

writer = SummaryWriter(tensorboard_log)

dataiter = iter(train_dataloader)
images, labels = next(dataiter)

writer.add_graph(network_model, images)
