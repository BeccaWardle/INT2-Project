import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import cv2 as cv
from cv2 import cv2
import argparse
from torchvision import models, transforms
import new_network

from torchvision import utils

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,help='path to image')
args = vars(ap.parse_args())

model = new_network.Net()
model_weights = []     # we will save the conv layer weights in this list
conv_layers = []       # we will save the conv layers in this list
#get all the model children as list
model_children = list(model.children())
# type(model_children[0][0])
# kernels = model.cnn_relu_block_1.weight.detach()
# fig, axarr = plt.subplots(kernels.size(0))
# for idx in range(kernels.size(0)):
#     axarr[idx].imshow(kernels[idx].squeeze())

counter = 0 
for i in range(len(model_children)):
    if type(model_children[i]) == nn.Conv2d:
        counter += 1
        model_weights.append(model_children[i].weight)
        conv_layers.append(model_children[i])
    elif type(model_children[i]) == nn.Sequential:
        for j in range(len(model_children[i])):
            if type(model_children[i][j]) == nn.Conv2d:
                counter += 1
                model_weights.append(model_children[i][j].weight)
                conv_layers.append(model_children[i][j])

# read and visualize an image
img = cv2.imread(f"/Users/jg/Documents/GitHub/INT2-Project/documents/{args['image']}")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
#plt.show()
# define the transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
img = np.array(img)
# apply the transforms
img = transform(img)
#print(img.size())
# unsqueeze to add a batch dimension
img = img.unsqueeze(0)
#print(img.size())

# pass the image through all the layers
results = [conv_layers[0](img)]
for i in range(1, len(conv_layers)):
    # pass the result from the last layer to the next layer
    results.append(conv_layers[i](results[-1]))
# make a copy of the `results`
outputs = results

# visualize 64 features from each layer 
# (although there are more feature maps in the upper layers)
for num_layer in range(len(outputs)):
    plt.figure(figsize=(30, 30))
    layer_viz = outputs[num_layer][0, :, :, :]
    layer_viz = layer_viz.data
    #print(layer_viz.size())
    for i, filter in enumerate(layer_viz):
        if i == 64: # we will visualize only 8x8 blocks from each layer
            break
        plt.subplot(8, 8, i + 1)
        plt.imshow(filter, cmap='gray')
        plt.axis("off")
    #print(f"Saving layer {num_layer} feature maps...")
    #plt.savefig(f"../documents/layer_{num_layer}.png")
    plt.show()
    plt.close()

#using python feature_map.py --image cat.jpg to run