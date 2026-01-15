import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from numpy.random import randn
import torchvision.utils
from torch.distributions import uniform
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import copy
from utils import *
from modules import *
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 2
n_sampled_images = 2
n_epoch = 400
log_interval = 10  # print loss every 10 batches
n_ax = max(1, int(n_epoch / 40))
total_loss_min = np.inf
image_size = 128
image_shape = (1, image_size, image_size)
image_dim = int(np.prod(image_shape))
learning_rate = 3e-4

# Define paths
current_dir = os.getcwd()
whole_dir = os.path.join(current_dir, "Training", "cropped_images")
training_data_dir = os.path.join(current_dir, "Training", "training_data")
cropped_images_dir = os.path.join(current_dir, "Training", "cropped_images")

# Create the cropped_images directory if it doesn't exist
os.makedirs(cropped_images_dir, exist_ok=True)
df = pd.read_csv("cast_information.csv", sep=";")

# -------------------------------------------------
# Build cast information dictionary
# CT : Cast temperature
# WR : Wheel Roughness
# COMP : Composition
# MFR : Melt flow rate - related to thickness
# -------------------------------------------------

cast_info = {}
for index, row in df.iterrows():
    cast_info[row["Cast_name"]] = {
        "CT": row["CT"],
        "WR": row["WR"],
        "COMP": row["COMP"],
        "MFR": row["MFR"],
    }

# Initialize raw (physical) class table
raw_class_table = [[] for _ in range(5)]  # CT, WR, COMP, MFR, MAG

for subdir, _, files in os.walk(training_data_dir):
    for filename in files:
        if filename in (".gitkeep", "Thumbs.db"):
            continue

        cast_name = filename.split("_")[0]
        magnification = int(filename.split("_")[1])
        parameters = cast_info[cast_name]

        raw_class_table[0].append(parameters["CT"])
        raw_class_table[1].append(parameters["WR"])
        raw_class_table[2].append(parameters["COMP"])
        raw_class_table[3].append(parameters["MFR"])
        raw_class_table[4].append(magnification)

# Convert to tensor for convenience
raw_class_table = torch.tensor(raw_class_table)

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


CT_enc, CT_map = encode_levels(raw_class_table[0])
WR_enc, WR_map = encode_levels(raw_class_table[1])
COMP_enc, COMP_map = encode_levels(raw_class_table[2])
MFR_enc, MFR_map = encode_levels(raw_class_table[3])
MAG_enc, MAG_map = encode_levels(raw_class_table[4])

# Final encoded class table (THIS is used by the model)
class_table = torch.stack([CT_enc, WR_enc, COMP_enc, MFR_enc, MAG_enc])

# Number of discrete levels (for embeddings)
n_CT = len(CT_map)
n_WR = len(WR_map)
n_COMP = len(COMP_map)
n_MFR = len(MFR_map)
n_MAG = len(MAG_map)

embedding_dim = 100  # model choice, not data-dependent

num_levels = {
    "CT": n_CT,
    "WR": n_WR,
    "COMP": n_COMP,
    "MFR": n_MFR,
    "MAG": n_MAG,
}

num_classes = class_table.shape[1]  # number of valid image/parameter combinations

conditioning_config = {
    "num_levels": num_levels,
    "embedding_dim": embedding_dim,
}


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


# Define the whole transform with center crop
# The filter work, however the images are not homogeneous after transform
# automoatic segmentaition might fail for instance.
# some adaptive thresholding could be performed here instead of grayscale conversion
# pay attention to the naming as well
# and check what happens in some images that seem not to be transformed (image 55-4, 56-4 and 7-0)

whole_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Grayscale(),
        CenterCrop(image_size),
        transforms.Lambda(lambda t: (t * 2) - 1),
    ]
)

# Load the dataset and apply the transform
train_dataset = datasets.ImageFolder(training_data_dir, transform=whole_transform)

# Save the cropped images to the cropped_images subfolder
for i, (images, labels) in enumerate(train_dataset):
    image_path = os.path.join(cropped_images_dir, f"image_{i}_{labels}.png")
    torchvision.utils.save_image(images, image_path)

aug_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
    ]
)

reverse_transforms = transforms.Compose(
    [
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t * 255.0),
    ]
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
)


class Diffusion:
    def __init__(
        self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=image_size
    ):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - self.alpha_hat[t])[
            :, None, None, None
        ]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(
        self,
        model,
        n,
        CT,
        WR,
        COMP,
        MFR,
        MAG,
        cfg_scale=0,
        verbose=False,
        print_every=100,
    ):
        """
        Sample images from the diffusion model.

        Args:
            model: UNet_conditional (ou similaire)
            n (int): nombre d'images à générer
            CT, WR, COMP, MFR, MAG: tenseurs de conditions [B]
            cfg_scale (float): classifier-free guidance scale
            verbose (bool): si True, prints timestep progress
            print_every (int): fréquence des prints
        """
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 1, self.img_size, self.img_size)).to(device)

            for i in reversed(range(1, self.noise_steps)):
                t = torch.full((n,), i, device=device, dtype=torch.long)

                predicted_noise = model(x, t, CT, WR, COMP, MFR, MAG)

                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None, None, None, None, None)
                    predicted_noise = torch.lerp(
                        uncond_predicted_noise, predicted_noise, cfg_scale
                    )

                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)

                x = (
                    1
                    / torch.sqrt(alpha)
                    * (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise)
                    + torch.sqrt(beta) * noise
                )

                if verbose and i % print_every == 0:
                    print(f"  Sampling timestep {i}/{self.noise_steps}")

        model.train()
        return x


model = UNet_conditional_small(
    c_in=1, c_out=1, time_dim=512, cond_dims=conditioning_config
).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
mse = nn.MSELoss()
diffusion = Diffusion(img_size=image_size)
l = len(train_loader)
ema = EMA(0.995)
ema_model = copy.deepcopy(model).eval().requires_grad_(False)

# here a window pop upto browse for the model could be implemented
"""load_dir = str(current_dir) + "/All_CDDM_HR_Cat_V_5.pth.tar"
load_model(load_dir)"""


def get_conditions_from_labels(labels, class_table, device):
    """
    labels: Tensor [B]
    class_table: Tensor [n_params, n_classes]
    returns: dict {param_name: Tensor[B]}
    """
    cond_tensors = class_maker(
        batch_size=labels.size(0),
        labels=labels,
        class_table=class_table,
    )

    cond_dict = {}
    for key, tensor in zip(COND_KEYS, cond_tensors):
        cond_dict[key] = tensor.long().to(device)

    return cond_dict


decode_maps = {
    "CT": CT_map,
    "WR": WR_map,
    "COMP": COMP_map,
    "MFR": MFR_map,
    "MAG": MAG_map,
}


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


# ==========================
# CONFIG LOGGING / DEBUG
# ==========================
COND_KEYS = list(num_levels.keys())

PRINT_BATCH_EVERY = 0  # 0 pour désactiver
PRINT_SAMPLING_INFO = True
SAMPLING_VERBOSE = True  # prints pendant le sampling long

# ==========================
# FIXED-CONDITION SAMPLING
# ==========================
FIXED_LABELS = torch.arange(min(4, num_classes), device=device)
FIXED_COND = get_conditions_from_labels(FIXED_LABELS, class_table, device)

for e in range(1, n_epoch + 1):
    loss_epoch = 0.0
    t_values_epoch = []

    # ==========================
    # TRAINING LOOP
    # ==========================
    for step, (images, labels) in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        images = aug_transform(images).to(device)
        labels = labels.long().to(device)

        # --- conditions ---
        cond = get_conditions_from_labels(labels, class_table, device)

        # --- diffusion ---
        t = diffusion.sample_timesteps(images.size(0)).to(device)
        x_t, noise = diffusion.noise_images(images, t)

        predicted_noise = model(x_t, t, **cond)

        loss = mse(noise, predicted_noise)
        loss.backward()

        optimizer.step()
        ema.step_ema(ema_model, model)

        loss_epoch += loss.item()
        t_values_epoch.append(t.detach().cpu())

        # -------- batch-level logging --------
        if PRINT_BATCH_EVERY > 0 and step % PRINT_BATCH_EVERY == 0:
            print(
                f"[Epoch {e}/{n_epoch}] "
                f"Step {step}/{len(train_loader)} | "
                f"Loss: {loss.item():.4e} | "
                f"t: min={t.min().item()}, max={t.max().item()}"
            )

    # ==========================
    # EPOCH SUMMARY
    # ==========================
    avg_loss = loss_epoch / len(train_loader)
    t_cat = torch.cat(t_values_epoch)

    hist_t = torch.histc(
        t_cat.float(),
        bins=10,
        min=0,
        max=diffusion.noise_steps,
    )

    print(
        f"\n=== Epoch {e}/{n_epoch} finished ===\n"
        f"Average loss: {avg_loss:.6f}\n"
        f"t stats: min={t_cat.min().item()} | "
        f"max={t_cat.max().item()} | "
        f"mean={t_cat.float().mean().item():.1f}\n"
        f"t histogram (10 bins): {hist_t.tolist()}\n"
    )

    # ==========================
    # SAMPLING / VISUALIZATION
    # ==========================
    if e % n_ax == 0:
        print(f"[Epoch {e}] Sampling...")

        with torch.no_grad():
            # ---------- RANDOM CONDITIONS ----------
            test_labels = torch.randint(
                0, num_classes, (n_sampled_images,), device=device
            )
            cond_random = get_conditions_from_labels(test_labels, class_table, device)

            ema_random_images = diffusion.sample(
                ema_model,
                n_sampled_images,
                **cond_random,
                cfg_scale=0,
                verbose=SAMPLING_VERBOSE,
            )
            ema_random_images = reverse_transforms(ema_random_images)
            phys_random = decode_physical_values(cond_random, decode_maps)
            titles_random = [format_physical_label(p) for p in phys_random]

            if PRINT_SAMPLING_INFO:
                print("Sampled random conditions:")
                for i, p in enumerate(phys_random):
                    print(f"  [{i}] {format_physical_label(p)}")

            show_grids(
                ema_random_images,
                n_epoch=e,
                current_dir=current_dir,
                titles=titles_random,
                suffix="EMA_RANDOM_CONDITIONS",
                image_size=image_size,
            )

            # ---------- FIXED-CONDITION SAMPLING ----------
            print(f"[Epoch {e}] Fixed-condition EMA sampling...")
            ema_fixed_images = diffusion.sample(
                ema_model,
                len(FIXED_LABELS),
                **FIXED_COND,
                cfg_scale=0,
                verbose=SAMPLING_VERBOSE,
            )
            ema_fixed_images = reverse_transforms(ema_fixed_images)
            fixed_phys = decode_physical_values(FIXED_COND, decode_maps)
            fixed_titles = [format_physical_label(p) for p in fixed_phys]

            show_grids(
                ema_fixed_images,
                n_epoch=e,
                current_dir=current_dir,
                titles=fixed_titles,
                suffix="EMA_FIXED_CONDITIONS",
                image_size=image_size,
            )

            # ---------- EMA vs RAW COMPARISON ----------
            print(f"[Epoch {e}] EMA vs RAW comparison...")
            raw_fixed_images = diffusion.sample(
                model,
                len(FIXED_LABELS),
                **FIXED_COND,
                cfg_scale=0,
                verbose=False,
            )
            raw_fixed_images = reverse_transforms(raw_fixed_images)

            # Show EMA vs RAW side by side
            # You could combine both in a single grid or save separately
            show_grids(
                ema_fixed_images,
                n_epoch=e,
                current_dir=current_dir,
                titles=fixed_titles,
                suffix="EMA_FIXED_COMPARISON",
                image_size=image_size,
            )

            show_grids(
                raw_fixed_images,
                n_epoch=e,
                current_dir=current_dir,
                titles=fixed_titles,
                suffix="RAW_FIXED_COMPARISON",
                image_size=image_size,
            )

            # ---------- CHECKPOINT ----------
            save_dir = os.path.join(
                current_dir, "Generated-Images", "All_CDDM_HR_Cat_V_6.pth.tar"
            )
            save_model(save_dir, model, ema_model, optimizer)

            print(f"[Epoch {e}] Sampling & checkpoint saved.\n")

        # ---------- GPU memory summary ----------
        if device.type == "cuda":
            print(torch.cuda.memory_summary(device=device))


if device.type == "cuda":
    print(torch.cuda.memory_summary(device=device))
else:
    print("Device is CPU, skipping CUDA memory summary.")
