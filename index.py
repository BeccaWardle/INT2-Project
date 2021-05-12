#!/usr/bin/env python3
# Dataset (in PyTorch)
# https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.CIFAR10

# Dataset homepage
# https://www.cs.toronto.edu/~kriz/cifar.html

# parameters

batch_size = 64
learning_rate = 1e-2
momentum = 0.9
epochs = 2000
max_consecutive = 50

## feature: continue training
cont = ""

## Feature: notification service
SQS = False
queue = False # SQS

## Feature: torch-related configs
jit = False # JIT compiler

## NN: optimiser
adam = False # Adam optimiser

## Tensorboard
TBoard = True

## main code

if SQS is True:
    import boto3

    sqs = boto3.resource('sqs')

    # Create the queue. This returns an SQS.Queue instance
    queue = sqs.get_queue_by_name(QueueName='model.fifo')

if TBoard is True:
    from torch.utils.tensorboard import SummaryWriter

# %%
# Imports

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
    [
        # transforms.Resize((64,64)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

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

network_model = network.Lexffe()

if jit:
    network_model = torch.jit.script(network_model)

# continue training -> load previous model
if cont:
    print("continuing previous progress.")
    network_model = torch.load(cont)
    network_model = torch.jit.script(network_model)


network_model.to(device)  # send tensors to CUDA cores
print(network_model)

# logits = network_model(train_dataloader)
# pred_probab = nn.Softmax(dim=1)(logits)

# define hyper-parameters

cross_entropy_loss = nn.CrossEntropyLoss()
op = torch.optim.Adam(network_model.parameters()) if adam is True else \
    torch.optim.SGD(network_model.parameters(), lr=learning_rate, momentum=momentum)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(op, 'min', patience=12)


epoch_progress = []
failed_write = False
# training


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
            print(f"Loss: {loss:>7f} [{current:>5d}/{size:>5d}]", sep="", end="\r", flush=True)

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


def save_csv(t, correct):
    file_name = f"results/{int(script_start)}-{network_model.name}_{network_model.__version__}.csv"
    global failed_write
    global epoch_progress
    try:
        with open(file_name, "a") as f:
            if failed_write:
                for line in epoch_progress:
                    f.write(f"{line[0]}.{line[1]}\n")
                failed_write = False
            else:
                f.write(f"{t},{correct}\n")
    except PermissionError:
        epoch_progress.append((t, correct))
        failed_write = True


def save(signum, frame):
    # save model state

    timestamp = int(script_start)
    version = network.Lexffe().__version__

    torch.save(network_model, f"results/networks/{timestamp}-{network_model.name}-{network_model.__version__}.pth")

    torch.save(network_model.state_dict(), f"results/networks/{timestamp}-{network_model.name}-{network_model.__version__}.pth")

    # tensorboard subroutine

    if TBoard is True:
        tensorboard_log = f"tensorboard/{timestamp}-{network_model.name}-{network_model.__version__}"

        network_model.eval()
        writer = SummaryWriter(tensorboard_log)

        dataiter = iter(train_dataloader)
        images, labels = next(dataiter)
        images = images.to(device)

        writer.add_graph(network_model, images)
        writer.close()

    exit()


signal.signal(signal.SIGINT, save)

max_accuracy = 0
consecutive = 0

# ------ main loop

for t in range(epochs):
    print(f"Epoch {t + 1}/{epochs}, Learning rate: {op.param_groups[0]['lr']}\n-------------------------------")
    train_loop(train_dataloader, network_model, cross_entropy_loss, op)
    correct, loss = test_loop(test_dataloader, network_model, cross_entropy_loss)
    sched.step(loss)

    save_csv(t, correct)

    # epoch_accuracy_pair.append((t, correct))

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

# check number of parameters

params_accumulator = 0

# https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
for module_name, param in network_model.named_parameters():
    if not param.requires_grad: continue # ignore untrainable parameters

    n = param.numel()
    print(f"{module_name}\t{n}")

    params_accumulator += n

print(f"Trainable parameters: {params_accumulator}")

save(0, 0)
