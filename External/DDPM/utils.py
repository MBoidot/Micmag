import torch
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn
import torchvision.utils
from torch.distributions import uniform
from mpl_toolkits.axes_grid1 import ImageGrid
import os
from torchvision import datasets
import torchvision.transforms.functional as TF

fig = plt.figure(figsize=(100, 100))


# -------------------------------------------------
# Encode physical values -> discrete indices
# -------------------------------------------------
def encode_levels(values):
    """
    values: 1D tensor
    returns:
        encoded tensor
        dict value -> index
    """
    unique_vals = torch.unique(values)
    unique_vals, _ = torch.sort(unique_vals)
    value_to_idx = {v.item(): i for i, v in enumerate(unique_vals)}
    encoded = torch.tensor([value_to_idx[v.item()] for v in values])
    return encoded, value_to_idx


def show_images(images, index, label):
    ax = fig.add_subplot(21, 1, index + 1, xticks=[], yticks=[])
    plt.gca().set_title(label)
    # ax.set_title(label,fontsize = 40)
    plt.imshow(images.cpu(), cmap="gray")
    plt.show()


def show_grids(images, n_epoch, current_dir, titles=None, suffix=None, image_size=512):
    """
    Display and save a grid of images.

    Args:
        images (Tensor): [B, H, W] ou [B, 1, H, W]
        n_epoch (int): epoch number, used for filename
        current_dir (str): path to save images
        titles (list of str, optional): titles for each subplot
        suffix (str, optional): additional string for filename
        image_size (int, optional): height/width of square images
    """
    n_images = images.size(0)
    ncols = 2
    nrows = (n_images + 1) // ncols

    fig = plt.figure(figsize=(ncols * 10, nrows * 10))
    grid = ImageGrid(fig, 111, nrows_ncols=(nrows, ncols), axes_pad=0.5)

    for j, (ax, im) in enumerate(
        zip(grid, images.cpu().view(-1, image_size, image_size))
    ):
        ax.imshow(im, cmap="gray")
        if titles is not None:
            ax.set_title(titles[j], fontsize=20)
        ax.axis("off")

    # --- ensure output directory exists ---
    save_dir = os.path.join(current_dir, "Generated_Images_training")
    os.makedirs(save_dir, exist_ok=True)  # crée le dossier si absent

    # --- build filename ---
    figname = os.path.join(save_dir, str(n_epoch))
    if suffix is not None:
        figname += f"_{suffix}"
    figname += ".png"

    fig.savefig(figname, bbox_inches="tight")
    plt.close(fig)


def save_model(address, model, ema_model, optimizer):
    """
    Save a checkpoint containing:
      - model
      - ema_model
      - optimizer
    """
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


def get_conditions_from_labels(labels, class_table, cond_keys, device):
    cond_tensors = class_maker(
        batch_size=labels.size(0),
        labels=labels,
        class_table=class_table,
    )

    cond_dict = {}
    for key, tensor in zip(cond_keys, cond_tensors):
        cond_dict[key] = tensor.long().to(device)

    return cond_dict


def decode_physical_values(cond_dict, decode_maps):
    """
    cond_dict: dict {param: Tensor[B]}
    returns: list of dicts [{param: physical_value}, ...]
    """
    B = next(iter(cond_dict.values())).size(0)
    decoded = []

    for i in range(B):
        entry = {}
        for k, tensor in cond_dict.items():
            idx = tensor[i].item()
            entry[k] = list(decode_maps[k].keys())[idx]
        decoded.append(entry)
    return decoded


def format_physical_label(phys_dict):
    return " | ".join(f"{k}={v}" for k, v in phys_dict.items())


# Define the center crop transform
class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        # Get the dimensions of the image
        _, height, width = img.shape
        # Calculate the starting coordinates for the crop
        start_h = (height - self.size) // 2
        start_w = (width - self.size) // 2
        # Perform the crop
        img = img[:, start_h : start_h + self.size, start_w : start_w + self.size]
        return img


class MultiHorizontalCenterCrop:
    def __init__(self, crop_size=128, n_crops=4):
        self.crop_size = crop_size
        self.n_crops = n_crops

    def __call__(self, img):
        # img is PIL.Image
        w, h = img.size
        cs = self.crop_size
        top = (h - cs) // 2
        max_left = w - cs

        if self.n_crops == 1:
            xs = [max_left // 2]
        else:
            xs = torch.linspace(0, max_left, self.n_crops)
        return [TF.crop(img, top, int(x), cs, cs) for x in xs]


class MultiCropImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, label = self.samples[index]
        img = self.loader(path)
        crops = self.transform(img)
        return crops, label


class FlattenedMultiCropDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset
        self.n_crops = len(base_dataset[0][0])

    def __len__(self):
        return len(self.base) * self.n_crops

    def __getitem__(self, idx):
        img_idx = idx // self.n_crops
        crop_idx = idx % self.n_crops
        crops, label = self.base[img_idx]
        return crops[crop_idx], label
