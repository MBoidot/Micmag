import torch
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn
import torchvision.utils
from torch.distributions import uniform
from mpl_toolkits.axes_grid1 import ImageGrid
import os


fig = plt.figure(figsize=(100, 100))


def show_images(images, index, label):
    ax = fig.add_subplot(21, 1, index + 1, xticks=[], yticks=[])
    plt.gca().set_title(label)
    # ax.set_title(label,fontsize = 40)
    plt.imshow(images.cpu(), cmap="gray")
    plt.show()


def show_grids(images, labels, n_epoch, label_dict):
    labels_title = []
    for i in labels:
        labels_title.append(label_dict[i.item()])
    fig = plt.figure(figsize=(40.0, 40.0))
    grid = ImageGrid(fig, 111, nrows_ncols=(2, 2), axes_pad=0.5)
    j = 0
    for ax, im in zip(grid, images.cpu().view(-1, image_size, image_size)):
        ax.imshow(im, cmap="gray")
        ax.title.set_text(labels_title[j])
        ax.title.set_size(28)
        # fig.gca().set_title(labelt[i])
        j += 1
    # plt.show()
    figname = str(current_dir) + "/Generated-Images/" + str(1200 + n_epoch)
    fig.savefig(figname, bbox_inches="tight")
    plt.close(fig)


def save_model(address):
    checkpoint = {
        "model_state": model.state_dict(),
        "ema_model_state": ema_model.state_dict(),
        "model_optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, address)


def load_model(address):
    checkpoint = torch.load(address)
    model.load_state_dict(checkpoint["model_state"])
    ema_model.load_state_dict(checkpoint["ema_model_state"])
    optimizer.load_state_dict(checkpoint["model_optimizer"])


def load_model2(address):
    checkpoint = torch.load(address)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["model_optimizer"])


# class maker (updated) contains sample information for updating the class_table :
# CT : Cast temperature
# WR : Wheel Roughness
# COMP : Composition (described as a string (comp_1, comp_2, ...) a table should list the compositions)
# MFR : Matter Flow Rate
# Mag : Image magnification


def class_maker(batch_size, labels, class_table):
    CT = torch.zeros([batch_size])
    WR = torch.zeros([batch_size])
    COMP = torch.zeros([batch_size])
    MFR = torch.zeros([batch_size])
    MAG = torch.zeros([batch_size])
    for i in range(batch_size):
        CT[i] = class_table[0, labels[i]]
        WR[i] = class_table[1, labels[i]]
        COMP[i] = class_table[2, labels[i]]
        MFR[i] = class_table[3, labels[i]]
        MAG[i] = class_table[4, labels[i]]

    return CT, WR, COMP, MFR, MAG


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find("Norm") != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)
