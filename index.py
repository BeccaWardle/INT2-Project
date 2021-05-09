#!/usr/bin/env python3
# Dataset (in PyTorch)
# https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.CIFAR10

# Dataset homepage
# https://www.cs.toronto.edu/~kriz/cifar.html

# Imports

import datetime
import signal
from os import system
from time import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

import network

script_start = time()
print(f"Started: {datetime.datetime.now()}")

# continue training
cont = "results/networks/1620517473-becca_1.41.pth"
pair = []


batch_size = 64

# Hardware acceleration

torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# normalise the data

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

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

train_dataloader = DataLoader(
    training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# image is of 32x32 with 3 channel for colours
# batch size is 64, but can be modified

for X, y in test_dataloader:

    print("Shape of X [n, Channels, Height, Width]: ", X.shape, X.dtype)
    print("Shape of y: ", y.shape, y.dtype)  # classification
    break

# Networks

# network_model = network.Will_Network()
# network_model = network.Lexffe()
# network_model = network.Becca()
# network_model = network.Zijun_Network()
network_model = network.Becca_long()

# continue training -> load previous model
if cont:
    print("continuing previous progress.")
    network_model.load_state_dict(torch.load(cont))
    network_model.eval()

network_model.to(device)  # send tensors to CUDA cores
print(network_model)

# logits = network_model(train_dataloader)
# pred_probab = nn.Softmax(dim=1)(logits)

# define hyper-parameters
learning_rate = 5e-3
patience = 10

cross_entropy_loss = nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(network_model.parameters(), momentum=0.9, lr=learning_rate)
# optimiser = torch.optim.AdamW(network_model.parameters(), lr=learning_rate)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', patience=patience)

# training
epoch_accuracy_pair = []
failed_write = False


def train_loop(dataloader, model: nn.Module, loss_fn, optimiser: torch.optim.Optimizer):
    model.train()
    iteration_start = time()
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):

        X, y = X.to(device), y.to(device)  # send data to device

        # predict
        pred = model(X)

        # compare prediction against actual
        loss = loss_fn(pred, y)

        # back_prop
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if batch % 50 == 0:
            loss, current = loss.item(), batch * len(X)
            print(
                f"Loss: {loss:>7f} [{current:>5d}/{size:>5d}]", sep="", end="\r", flush=True)

    print()
    print(
        f"time since start: {time() - script_start:>0.2f}s, time since iteration start: {time() - iteration_start:>0.2f}s \n")


def test_loop(dataloader, model: nn.Module, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size

    print(f"Accuracy: {(100 * correct):>0.2f}%, Avg loss: {test_loss:>8f}\n")

    return correct, test_loss


def save(t, correct):
    file_name = f"results/{int(script_start)}-{network_model.name}_{network_model.__version__}.csv"
    global failed_write
    global epoch_accuracy_pair
    try:
        with open(file_name, "a") as f:
            if failed_write:
                for line in epoch_accuracy_pair:
                    f.write(f"{line[0]}.{line[1]}\n")
                failed_write = False
            else:
                f.write(f"{t},{correct}\n")
    except PermissionError:
        epoch_accuracy_pair.append((t, correct))
        failed_write = True


def net_save(signum, frame):
    torch.save(network_model.state_dict(), f"results/networks/{int(script_start)}-{network_model.name}_{network_model.__version__}.pth")
    exit()


epochs = 120
max_accuracy = 0
consecutive = 0
max_consecutive = 15

signal.signal(signal.SIGINT, net_save)

for t in range(epochs):
    print(f"Epoch {t+1}/{epochs}, Learning rate: {optimiser.param_groups[0]['lr']}\n------------------------------------")
    train_loop(train_dataloader, network_model, cross_entropy_loss, optimiser)
    correct, loss = test_loop(test_dataloader, network_model, cross_entropy_loss)
    sched.step(loss)

    save(t, correct)

    if correct > max_accuracy:
        consecutive = 0
        max_accuracy = correct
    else:
        consecutive += 1
        print(f"no improvement: {consecutive}/{max_consecutive}, max accuracy: {(100 * max_accuracy):>0.2f}%\n")

    # decrease learning rate
    # if consecutive >= (max_consecutive/2) and learning_rate >= 1e-4:
    #     learning_rate /= 2
    #     stochastic_GD = torch.optim.SGD(network_model.parameters(),momentum=0.9, lr=learning_rate)
    #     print(f"Learning rate decreased by half to: {learning_rate}")
    #     # reset consecutive
    #     consecutive = 0

    if consecutive == max_consecutive:
        print("model reached max potential, stopping.")
        print(f"max accuracy: {(100 * max_accuracy):>0.2f}%")
        print(f"time since start: {time() - script_start:>0.2f}s")
        break

print("Done!")
system("vlc alert.ogg")
print(f"Average epoch length: {(time() - script_start)/epochs :>0.2f}s")

net_save(0, 0)
