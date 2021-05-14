#!/usr/bin/env python3

import os

import numpy as np
from matplotlib import pyplot as plt

dataframes = []
line_names = []
for file in os.listdir("results/"):
    if "Plotting" in file:
        continue
    if file.endswith(".csv"):
        line_names.append(file[:-4])
        dataframes.append(np.genfromtxt(os.path.join("results/", file), delimiter=','))

fig, ax = plt.subplots()
max_acc = 0
max_axi = 0
for i, data in enumerate(dataframes):
    print(line_names[i])
    max_axi = len(data[:, 0]) if len(data[:, 0]) > max_axi else max_axi
    max_acc = max(data[:, 1]) if max(data[:, 1]) > max_acc else max_acc
    # if i == len(dataframes) - 1:
    #     ax.plot(data[:, 0], data[:, 1], label=str(line_names[i]))
    if max(data[:, 0]) > 150:
        ax.plot(data[:150, 0], data[:150, 1], label=str(line_names[i]))
    else:
        ax.plot(data[:, 0], data[:, 1], label=str(line_names[i]))

ax.plot(max_acc, '-', label="max accuracy")
ax.axhline(max_acc, ls='--')
print(f"max_acc: {max_acc}")
ax.annotate(str(max_acc), xy=(0, max_acc+0.01))

ax.set(xlabel="epoch", ylabel="accuracy")
# plt.legend()
plt.show()
