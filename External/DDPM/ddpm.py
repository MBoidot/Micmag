import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from numpy.random import randn
import torchvision.utils
import os
import copy
from tqdm import tqdm
import matplotlib.cm as cm
from utils import (
    encode_levels,
    get_conditions_from_labels,
    show_grids,
    save_model,
    decode_physical_values,
    format_physical_label,
    MultiHorizontalCenterCrop,
    MultiCropImageFolder,
    FlattenedMultiCropDataset,
)

from modules import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 2
N_CROPS = 4
n_epoch = 300
n_ax = max(1, int(n_epoch / 60))
image_size = 256
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

# read process parameters from CSV
# to be modified to pick up parameters in relevant files
df = pd.read_csv("cast_information.csv", sep=";")

# -------------------------------------------------
# Build cast information dictionary
# CT : Cast temperature
# WR : Wheel Roughness
# COMP : Composition or alloy type
# MFR : Melt flow rate - related to thickness unit ??
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

embedding_dim = 32  # model choice, not data-dependent

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


# Define the whole transform with center crop
# The filter work, however the images are not homogeneous after transform
# automoatic segmentaition might fail for instance.
# some adaptive thresholding could be performed here instead of grayscale conversion
# pay attention to the naming as well
# and check what happens in some images that seem not to be transformed (image 55-4, 56-4 and 7-0)

whole_transform = transforms.Compose(
    [
        transforms.Grayscale(),
        MultiHorizontalCenterCrop(crop_size=image_size, n_crops=N_CROPS),
        transforms.Lambda(
            lambda crops: [
                torch.clamp(transforms.ToTensor()(c) * 2 - 1, -1, 1) for c in crops
            ]
        ),
    ]
)


# Load the dataset and apply the transform
base_dataset = MultiCropImageFolder(training_data_dir, transform=whole_transform)
train_dataset = FlattenedMultiCropDataset(base_dataset)

# Save the cropped images to the cropped_images subfolder
for i in range(len(base_dataset)):
    crops, label = base_dataset[i]
    for j, crop in enumerate(crops):
        path = os.path.join(cropped_images_dir, f"img_{i}_crop_{j}_label_{label}.png")
        torchvision.utils.save_image((crop + 1) / 2, path)


def add_noise(x, sigma=0.01):
    return x + sigma * torch.randn_like(x)


aug_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.05, contrast=0.05),
        transforms.RandomApply(
            [transforms.Lambda(lambda x: x ** torch.empty(1).uniform_(0.9, 1.1))], p=0.3
        ),
        transforms.RandomApply([transforms.Lambda(add_noise)], p=0.3),
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
        self, noise_steps=500, beta_start=1e-4, beta_end=0.02, img_size=image_size
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

    def sample_timesteps(self, n, t_min=1):
        """
        Sample timesteps uniformly but never below t_min
        """
        return torch.randint(low=t_min, high=self.noise_steps, size=(n,), device=device)

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


model = UNet_conditional(
    c_in=1,
    c_out=1,
    image_size=image_size,
    time_dim=128,
    attention_from=16,  # activate when using unet_conditional
    attention_to=8,  # activate when using unet_conditional
    attention_on="up",  # up, down or both, depending on where to put attention layers
    cond_dims=conditioning_config,
).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
mse = nn.MSELoss()
diffusion = Diffusion(img_size=image_size)
l = len(train_loader)
ema = EMA(0.9995)
ema_model = copy.deepcopy(model).eval().requires_grad_(False)

# here a window pop upto browse for the model could be implemented
"""load_dir = str(current_dir) + "/All_CDDM_HR_Cat_V_5.pth.tar"
load_model(load_dir)"""


decode_maps = {
    "CT": CT_map,
    "WR": WR_map,
    "COMP": COMP_map,
    "MFR": MFR_map,
    "MAG": MAG_map,
}

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
FIXED_LABELS = torch.arange(min(2, num_classes), device=device)
FIXED_COND = get_conditions_from_labels(FIXED_LABELS, class_table, COND_KEYS, device)

images, labels = next(iter(train_loader))
"""print("DEBUG: images NaN check:", torch.isnan(images).any())  # should be False
print(
    "DEBUG: images min/max:", images.min().item(), images.max().item()
)  # should be ~[-1,1]
print("DEBUG: labels min/max:", labels.min().item(), labels.max().item())"""

ema_start = 1000  # steps
global_step = 0

for e in range(1, n_epoch + 1):
    global_step += 1
    loss_epoch = 0.0
    t_values_epoch = []

    # ==========================
    # TRAINING LOOP
    # ==========================
    for step, (images, labels) in enumerate(
        tqdm(train_loader, desc=f"Epoch {e}/{n_epoch}", leave=False), start=1
    ):

        optimizer.zero_grad()
        images = aug_transform(images).to(device)
        labels = labels.long().to(device)

        # --- conditions ---
        cond = get_conditions_from_labels(labels, class_table, COND_KEYS, device)

        # --- diffusion ---
        t = diffusion.sample_timesteps(images.size(0)).to(device)
        x_t, noise = diffusion.noise_images(images, t)

        # --- model prediction and loss ---
        predicted_noise = model(x_t, t, **cond, debug=False)
        predicted_noise = 10.0 * torch.tanh(predicted_noise / 10.0)

        loss = mse(noise, predicted_noise)
        loss.backward()

        optimizer.step()

        if global_step > ema_start:
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
    if e % n_ax == 1:
        print(f"[Epoch {e}] Sampling...")

        with torch.no_grad():

            # ---------- FIXED-CONDITION EMA SAMPLING ----------
            print(f"[Epoch {e}] Fixed-condition EMA sampling...")

            # Progress bar over diffusion timesteps
            ema_fixed_images = diffusion.sample(
                ema_model,
                len(FIXED_LABELS),
                **FIXED_COND,
                cfg_scale=0,
                verbose=True,  # IMPORTANT: diffusion.sample must update tqdm internally
            )

            # Reverse to [0,255] for visualization
            ema_fixed_images_vis = reverse_transforms(ema_fixed_images)  # [B,1,H,W]

            # Decode physical labels for titles
            fixed_phys = decode_physical_values(FIXED_COND, decode_maps)
            fixed_titles = [format_physical_label(p) for p in fixed_phys]

            # ---------- ATTENTION EXTRACTION ----------
            attn_maps = ema_model.get_last_attention_maps()

            # Always define base RGB images
            imgs_rgb = ema_fixed_images_vis.repeat(1, 3, 1, 1)

            # Defaults (safe fallback)
            overlay_keys = imgs_rgb.clone()
            overlay_queries = imgs_rgb.clone()
            overlay_diag = imgs_rgb.clone()
            heatmap_keys = heatmap_queries = heatmap_diag = None

            if len(attn_maps) > 0:
                attn_module = list(attn_maps.values())[0]

                attn = attn_module.last_attn  # [B, HW, HW]
                H = attn_module.last_H
                W = attn_module.last_W
                B = attn.shape[0]

                # ---------- MULTI-VIEW ATTENTION MAPS ----------
                heatmap_keys = attn.mean(dim=1).view(B, 1, H, W)
                heatmap_queries = attn.mean(dim=2).view(B, 1, H, W)
                heatmap_diag = attn.diagonal(dim1=1, dim2=2).view(B, 1, H, W)

                def resize_and_norm(hm, q=0.99):
                    hm = F.interpolate(
                        hm,
                        size=(image_size, image_size),
                        mode="bilinear",
                        align_corners=False,
                    )

                    flat = hm.flatten(start_dim=1)
                    scale = torch.quantile(flat, q, dim=1).view(-1, 1, 1, 1)

                    hm = hm / (scale + 1e-8)
                    return hm.clamp(0, 1)

                heatmap_keys = resize_and_norm(heatmap_keys)
                heatmap_queries = resize_and_norm(heatmap_queries)
                heatmap_diag = resize_and_norm(heatmap_diag)

                # ---------- OVERLAY ----------
                ATTN_THRESHOLD = 0.5
                ALPHA = 0.6

                COLORMAPS = {
                    "keys": cm.inferno,
                    "queries": cm.viridis,
                    "diag": cm.cividis,
                }

                def apply_overlay(heatmap, cmap):
                    overlay = imgs_rgb.clone()
                    mask = heatmap > ATTN_THRESHOLD

                    for i in range(B):
                        hm = heatmap[i, 0].cpu().numpy()
                        colored_np = cmap(hm)[..., :3]

                        colored = (
                            torch.from_numpy(colored_np)
                            .permute(2, 0, 1)
                            .to(imgs_rgb.device)
                            .type_as(imgs_rgb)
                            * 255.0
                        )

                        for c in range(3):
                            overlay[i, c][mask[i, 0]] = (1 - ALPHA) * imgs_rgb[i, c][
                                mask[i, 0]
                            ] + ALPHA * colored[c][mask[i, 0]]

                    return overlay.clamp(0, 255)

                overlay_keys = apply_overlay(heatmap_keys, COLORMAPS["keys"])
                overlay_queries = apply_overlay(heatmap_queries, COLORMAPS["queries"])
                overlay_diag = apply_overlay(heatmap_diag, COLORMAPS["diag"])

            # ---------- GRID ASSEMBLY ----------
            grid_imgs = torch.cat(
                [imgs_rgb, overlay_keys, overlay_queries, overlay_diag],
                dim=0,
            )

            titles = (
                [f"Raw\n{t}" for t in fixed_titles]
                + ["Attn mean(dim=1)\nKeys"] * len(fixed_titles)
                + ["Attn mean(dim=2)\nQueries"] * len(fixed_titles)
                + ["Attn diagonal\nSelf"] * len(fixed_titles)
            )

            heatmaps_for_plot = (
                [None] * len(fixed_titles)
                + (
                    [heatmap_keys[i, 0].cpu().numpy() for i in range(len(fixed_titles))]
                    if heatmap_keys is not None
                    else [None] * len(fixed_titles)
                )
                + (
                    [
                        heatmap_queries[i, 0].cpu().numpy()
                        for i in range(len(fixed_titles))
                    ]
                    if heatmap_queries is not None
                    else [None] * len(fixed_titles)
                )
                + (
                    [heatmap_diag[i, 0].cpu().numpy() for i in range(len(fixed_titles))]
                    if heatmap_diag is not None
                    else [None] * len(fixed_titles)
                )
            )

            show_grids(
                grid_imgs,
                n_epoch=e,
                current_dir=current_dir,
                titles=titles,
                suffix="EMA_ATTENTION_COMPARISON",
                image_size=image_size,
                show_colorbar=False,
                heatmaps=heatmaps_for_plot,
                attn_threshold=ATTN_THRESHOLD,
            )

            # ---------- CHECKPOINT ----------
            save_dir = os.path.join(
                current_dir,
                "Generated_Images_training",
                "UNET_conditional_Up_att_256_300ep.tar",
            )
            save_model(save_dir, model, ema_model, optimizer)

            print(f"[Epoch {e}] Sampling & checkpoint saved.\n")

        # ---------- GPU memory summary ----------
        if device.type == "cuda":
            print(torch.cuda.memory_summary(device=device))
