import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle


plt.rcdefaults()
NumDots = 4
NumConvMax = 8
NumFcMax = 20
White = 1.0
Light = 0.7
Medium = 0.5
Dark = 0.3
Darker = 0.15
Black = 0.0


def add_layer(patches, colors, size=(24, 24), num=5, top_left=[0, 0], loc_diff=[3, -3]):
    # add a rectangle
    top_left = np.array(top_left)
    loc_diff = np.array(loc_diff)
    loc_start = top_left - np.array([0, size[0]])
    for i in range(num):
        patches.append(Rectangle(loc_start + i * loc_diff, size[1], size[0]))
        if i % 2:
            colors.append(Medium)
        else:
            colors.append(Light)


def add_layer_with_omission(patches, colors, size=(24, 24), num=5, num_max=8, num_dots=4, top_left=[0, 0], loc_diff=[3, -3], neurons=False):
    # add a rectangle
    top_left = np.array(top_left)
    loc_diff = np.array(loc_diff)
    loc_start = top_left - np.array([0, size[0]])
    this_num = min(num, num_max)
    start_omit = (this_num - num_dots) // 2
    end_omit = this_num - start_omit
    start_omit -= 1
    for i in range(this_num):
        if (num > num_max) and (start_omit < i < end_omit):
            omit = True
        else:
            omit = False

        if omit:
            patches.append(Circle(loc_start + i * loc_diff + np.array(size) / 2, 0.2))
        else:
            if not neurons:
                patches.append(Rectangle(loc_start + i * loc_diff, size[1], size[0]))
            else:
                patches.append(Circle(loc_start + i * loc_diff + np.array(size) / 2, 2))

        if omit:
            colors.append(Black)
        elif i % 2:
            colors.append(Medium)
        else:
            colors.append(Light)


def add_mapping(patches, colors, start_ratio, end_ratio, patch_size, ind_bgn, top_left_list, loc_diff_list, num_show_list, size_list):

    start_loc = top_left_list[ind_bgn] + (num_show_list[ind_bgn] - 1) * np.array(loc_diff_list[ind_bgn]) + np.array([start_ratio[0] * (size_list[ind_bgn][1] - patch_size[1]), - start_ratio[1] * (size_list[ind_bgn][0] - patch_size[0])])
    
    end_loc = top_left_list[ind_bgn + 1]  + (num_show_list[ind_bgn + 1] - 1) * np.array(loc_diff_list[ind_bgn + 1]) + np.array([end_ratio[0] * size_list[ind_bgn + 1][1], - end_ratio[1] * size_list[ind_bgn + 1][0]])


    patches.append(Rectangle(start_loc, patch_size[1], -patch_size[0]))
    colors.append(Dark)
    
    patches.append(Line2D([start_loc[0], end_loc[0]], [start_loc[1], end_loc[1]]))
    colors.append(Darker)
    
    patches.append(Line2D([start_loc[0] + patch_size[1], end_loc[0]], [start_loc[1], end_loc[1]]))
    colors.append(Darker)
    
    patches.append(Line2D([start_loc[0], end_loc[0]], [start_loc[1] - patch_size[0], end_loc[1]]))
    colors.append(Darker)
    
    patches.append(Line2D([start_loc[0] + patch_size[1], end_loc[0]], [start_loc[1] - patch_size[0], end_loc[1]]))
    colors.append(Darker)



def label(xy, text, xy_off=[0, 4]):
    plt.text(xy[0] + xy_off[0], xy[1] + xy_off[1], text, family="sans-serif", size=8)


if __name__ == "__main__":

    fc_unit_size = 2
    # 
    layer_width = 40
    # simplify diagram with dots
    flag_omit = True

    patches = []
    colors = []

    fig, ax = plt.subplots()

    ############################
    # conv layers
    # height and width of tensors
    size_list = [(32, 32), (30, 30), (28, 28), (26, 26), (24, 24), (22, 22), (10, 10), (8, 8), (6, 6), (2, 2)]
    # depth of tensors
    num_list = [32, 24, 96, 96, 96, 192, 192, 384, 386, 384]
    # layer seperations
    x_diff_list = [0] +  [layer_width] * (len(size_list) - 1)
    # tensor names
    text_list = ["Inputs"] + ["Feature\nmaps"]
    # frame seperation
    loc_diff_list = [[2, -2]] * (len(size_list))

    num_show_list = list(map(min, num_list, [NumConvMax] * len(num_list)))
    top_left_list = np.c_[np.cumsum(x_diff_list), np.zeros(len(x_diff_list))]

    for i in range(len(size_list)-1,-1,-1):
        if flag_omit:
            add_layer_with_omission(patches, colors, size=size_list[i], num=num_list[i], num_max=NumConvMax, num_dots=NumDots, top_left=top_left_list[i], loc_diff=loc_diff_list[i])
        else:
            add_layer(patches, colors, size=size_list[i], num=num_show_list[i], top_left=top_left_list[i], loc_diff=loc_diff_list[i])
        label(top_left_list[i], text_list[i if i == 0 else -1] + f"\n{num_list[i]}@{size_list[i][0]}x{size_list[i][1]}")

    
    ############################
    # in between layers
    
    # kernel size
    patch_size_list = [(3, 3), (3, 3), (3, 3), (3, 3), (2, 2), (3, 3), (3, 3), (3, 3), (3, 3)]
    # kernel types
    text_list = ["Convolution"] * 5 + ["Max Pool", "Convolution", "Convolution", "Max Pool"]
    # kernel location on previous layer
    start_ratio_list = [[0.3, 0.4]] * (len(size_list))
    # kernel location on next layer
    end_ratio_list = [[0.6, 0.7]] * (len(size_list))

    for i in range(len(patch_size_list)):
        add_mapping(patches, colors, start_ratio_list[i], end_ratio_list[i], patch_size_list[i], i, top_left_list, loc_diff_list, num_show_list, size_list)
        label(top_left_list[i], f"{text_list[i]} \n{patch_size_list[i][0]}x{patch_size_list[i][1]} kernel", xy_off=[26, -65])


    ############################
    # Fully Connected layers
    
    # neuron number per layer
    num_list = [1536, 1024, 128, 10]
    # fully connected layers
    size_list = [(fc_unit_size, fc_unit_size)] * len(num_list)

    num_show_list = list(map(min, num_list, [NumFcMax] * len(num_list)))
    
    # offset of layers from one another
    x_diff_list = [sum(x_diff_list) + layer_width] + [layer_width] * (len(size_list) - 1)
    
    top_left_list = np.c_[np.cumsum(x_diff_list), np.zeros(len(x_diff_list))]
    loc_diff_list = [[fc_unit_size, -fc_unit_size]] * len(top_left_list)
    
    text_list = ["Hidden\nlayers"] * (len(size_list) - 1) + ["Outputs"]

    for i in range(len(size_list)):
        if flag_omit:
            add_layer_with_omission(patches, colors, size=size_list[i], num=num_list[i], num_max=NumFcMax, num_dots=NumDots, top_left=top_left_list[i], loc_diff=loc_diff_list[i], neurons=True)
        else:
            add_layer(patches, colors, size=size_list[i], num=num_show_list[i], top_left=top_left_list[i], loc_diff=loc_diff_list[i])
        label(top_left_list[i], f"{text_list[i]}\n{num_list[i]}")

    text_list = ["Flatten\n", "Fully\nconnected"]

    for i in range(len(size_list)):
        label(top_left_list[i], text_list[i if i == 0 else -1], xy_off=[-10, -65])

    ############################
    for patch, color in zip(patches, colors):
        patch.set_color(color * np.ones(3))
        if isinstance(patch, Line2D):
            ax.add_line(patch)
        else:
            patch.set_edgecolor(Black * np.ones(3))
            ax.add_patch(patch)

    # plt.tight_layout()
    plt.axis("equal")
    plt.axis("off")
    plt.show()
    # fig.set_size_inches(8, 2.5)

    # fig_dir = "./"
    # fig_ext = ".png"
    # fig.savefig(os.path.join(fig_dir, "convnet_fig" + fig_ext),
    #             bbox_inches="tight", pad_inches=0)