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
        print(file)
        line_names.append(file[:-4])
        dataframes.append(np.genfromtxt(os.path.join("results/", file), delimiter=','))


fig, ax = plt.subplots()
counter = 0
for data in dataframes:
    ax.plot(data[:, 0], data[:, 1], label=str(line_names[counter]))
    counter += 1

ax.set(xlabel="epoch", ylabel="accuracy")
plt.legend()
plt.show()
