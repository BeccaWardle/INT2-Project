#!/usr/bin/env python3
# Dataset (in PyTorch)
# https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.CIFAR10

# Dataset homepage
# https://www.cs.toronto.edu/~kriz/cifar.html

# parameters

batch_size = 64
learning_rate = 1e-2
epochs = 250
max_consecutive = 25

## feature: continue training
cont = False
cont_fname = "model.pth"

## Feature: notification service
SQS = True
queue = False
jit = True

## main code

if SQS is True:
    import boto3

    sqs = boto3.resource('sqs')

    # Create the queue. This returns an SQS.Queue instance
    queue = sqs.get_queue_by_name(QueueName='model.fifo')

# %%
# Imports

import csv
import datetime
import signal
from time import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

import network

script_start = time()
print(f"Started: {datetime.datetime.now()}")

# %%
## feature: accuracy vs epoch recording (CSV)
epoch_accuracy_pair = []

# %%
## feature: Hardware acceleration

torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"using {device}")

# normalise the data
transform = transforms.Compose(
    [transforms.Resize((64,64)),
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

if jit is True:
    network_model = torch.jit.script(network_model)

# continue training -> load previous model
if cont:
    print("continuing previous progress.")
    network_model.load_state_dict(torch.load(cont_fname))
    network_model.eval()

network_model.to(device)  # send tensors to CUDA cores
print(network_model)

# logits = network_model(train_dataloader)
# pred_probab = nn.Softmax(dim=1)(logits)

# define hyper-parameters

cross_entropy_loss = nn.CrossEntropyLoss()
stochastic_GD = torch.optim.SGD(network_model.parameters(), lr=learning_rate)


# training


def train_loop(dataloader, model: nn.Module, loss_fn, optimiser: torch.optim.Optimizer):
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
            print(f"Loss: {loss:>7f} [{current:>5d}/{size:>5d}]", sep="", end="\r", flush=True)

    print()
    print(
        f"time since start: {time() - script_start:>0.2f}s, time since iteration start: {time() - iteration_start:>0.2f}s \n")


def test_loop(dataloader, model: nn.Module, loss_fn):
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

    return correct


def save(signum, frame):
    ## save model state
    torch.save(network_model.state_dict(), f"result/model.{int(script_start)}.pth")

    # atexit doesn't work
    with open(f"result/{int(script_start)}.plot.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(epoch_accuracy_pair)
        f.close()

    exit()


signal.signal(signal.SIGINT, save)

max_accuracy = 0
consecutive = 0

## ------ main loop

for t in range(epochs):
    print(f"Epoch {t + 1}/{epochs}\n-------------------------------")
    train_loop(train_dataloader, network_model, cross_entropy_loss, stochastic_GD)
    correct = test_loop(test_dataloader, network_model, cross_entropy_loss)

    epoch_accuracy_pair.append((t, correct))

    if correct > max_accuracy:
        consecutive = 0  # reset counter
        max_accuracy = correct
    else:
        consecutive += 1
        print(f"no improvement: {consecutive}/{max_consecutive}, max accuracy: {(100 * max_accuracy):>0.2f}%")

    if consecutive == max_consecutive:
        print("model reached max potential, stopping.")
        print(f"max accuracy: {(100 * max_accuracy):>0.2f}%")
        print(f"time since start: {time() - script_start:>0.2f}s")
        break

    if SQS is True and queue:
        response = queue.send_message(MessageBody=f"{t},{correct},{max_accuracy}", MessageGroupId="model")

print("Done!")

save(0, 0)
