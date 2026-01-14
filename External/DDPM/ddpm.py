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

batch_size = 2
n_sampled_images = 4
n_epoch = 400
n_ax = int(n_epoch / 20)
total_loss_min = np.inf
image_shape = (1, 512, 512)
image_size = 512
image_dim = int(np.prod(image_shape))
learning_rate = 3e-4

shapes = 2
locations = 3
cooling_rates = 3
soaking_times = 3
forging_temps = 3
heat_treatments = 2
magnifications = 6
embedding_dim = 100
num_classes = 114

# Define paths
current_dir = os.getcwd()
whole_dir = os.path.join(current_dir, "Training", "cropped_images")
training_data_dir = os.path.join(current_dir, "Training", "training_data")
cropped_images_dir = os.path.join(current_dir, "Training", "cropped_images")

# Create the cropped_images directory if it doesn't exist
os.makedirs(cropped_images_dir, exist_ok=True)

# try to make the class_table automatically, mapping name/magnification with cast_parameters.csv information
# Read the CSV file
df = pd.read_csv("cast_information.csv", sep=";")

# Create a dictionary to map the cast names to their parameters
cast_info = {}
for index, row in df.iterrows():
    cast_info[row["Cast_name"]] = {
        "CT": row["CT"],
        "WR": row["WR"],
        "COMP": row["COMP"],
        "MFR": row["MFR"],
    }

# Initialize one list per parameter (rows)
class_table = [
    [] for _ in range(len(df.columns))
]  # should actually be len(df.columns) -1 (cast_name) +1 (magnification)

for subdir, _, files in os.walk(training_data_dir):
    for filename in files:
        if filename in (".gitkeep", "Thumbs.db"):
            continue
        cast_name = filename.split("_")[0]
        magnification = int(filename.split("_")[1])
        parameters = cast_info[cast_name]
        # Append each value to its parameter row
        class_table[0].append(parameters["CT"])
        class_table[1].append(parameters["WR"])
        class_table[2].append(parameters["COMP"])
        class_table[3].append(parameters["MFR"])
        class_table[4].append(magnification)

# Convert directly to a tensor (already transposed)
class_table = torch.tensor(class_table)

# fmt: off
label_dict = {
    0: 'CM01-0500', 1: 'CM01-1000',2: 'CM01-1500',3: 'CM01-2000',4: 'CM04-0100',5: 'CM04-0500',
    6: 'CM04-1000',7: 'CM04-1500',8: 'CM04-2000',9: 'CM04-3000',10: 'CM10-0500',
    11: 'CM10-1000',12: 'CM10-1500',13: 'CM10-2000',14: 'CM13-0500',15: 'CM13-1000',
    16: 'CM13-2000',17: 'CM13-3000',18: 'CM16-0500',19: 'CM16-1000',20: 'CM16-1500',
    21: 'CM16-2000',22: 'CM16-3000',23: 'CS01-0100',24: 'CS01-0500',25: 'CS01-1000',
    26: 'CS01-1500',27: 'CS01-2000',28: 'CS04-0500',29: 'CS04-1500',30: 'CS04-3000',
    31: 'CS10-0100',32: 'CS10-0500',33: 'CS10-1000',34: 'CS10-1500',35: 'CS10-3000',
    36: 'CS13-0100',37: 'CS13-0500',38: 'CS13-1000',39: 'CS13-2000',40: 'CS13-3000',
    41: 'CS16-0500',42: 'CS16-1000',43: 'CS16-2000',44: 'PL03-0100',45: 'PL03-0500',
    46: 'PL03-1000',47: 'PL03-1500',48: 'PL03-2000',49: 'PL03-3000',50: 'PL11-0100',
    51: 'PL11-0500',52: 'PL11-1000',53: 'PL11-1500',54: 'PL11-2000',55: 'PL11-3000',
    56: 'PL13-0100',57: 'PL13-0500',58: 'PL13-1000',59: 'PL13-2000',60: 'PL13-3000',
    61: 'PL16-0100',62: 'PL16-0500',63: 'PL16-1000',64: 'PL16-1500',65: 'PL16-2000',
    66: 'PL16-3000',67: 'PL18-0100',68: 'PL18-0500',69: 'PL18-1000',70: 'PL18-1500',
    71: 'PL18-2000',72: 'PM03-0100',73: 'PM03-0500',74: 'PM03-1000',75: 'PM03-2000',
    76: 'PM03-3000',77: 'PM11-0100',78: 'PM11-0500',79: 'PM11-1000',80: 'PM11-2000',
    81: 'PM13-0100',82: 'PM13-0500',83: 'PM13-1000',84: 'PM13-2000',85: 'PM13-3000',
    86: 'PM16-0100',87: 'PM16-0500',88: 'PM16-1000',89: 'PM16-2000',90: 'PM16-3000',
    91: 'PM18-0500',92: 'PM18-1000',93: 'PM18-1500',94: 'PM18-2000',95: 'PM18-3000',
    96: 'PS03-0500',97: 'PS03-1000',98: 'PS03-1500',99: 'PS03-2000',100: 'PS03-3000',
    101: 'PS11-0100',102: 'PS11-0500',103: 'PS11-1000',104: 'PS11-2000',105: 'PS16-0100',
    106: 'PS16-0500',107: 'PS16-1000',108: 'PS16-2000',109: 'PS16-3000',110: 'PS18-0100',
    111: 'PS18-0500',112: 'PS18-1000',113: 'PS18-2000'
}

# fmt: on


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
        CenterCrop(512),
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
    dataset=train_dataset, batch_size=batch_size, shuffle=True
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

    def sample(self, model, n, CT, WR, COMP, MFR, MAG, cfg_scale=0):
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

        model.train()
        return x


model = UNet_conditional(num_classes=num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
mse = nn.MSELoss()
diffusion = Diffusion(img_size=image_size)
l = len(train_loader)
ema = EMA(0.995)
ema_model = copy.deepcopy(model).eval().requires_grad_(False)

# here a window pop upto browse for the model could be implemented
load_dir = str(current_dir) + "/All_CDDM_HR_Cat_V_5.pth.tar"
load_model(load_dir)

for e in range(1, n_epoch + 1):
    loss_epoch = 0

    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        images = aug_transform(images).to(device)
        labels = labels.long().to(device)

        CT, WR, COMP, MFR, MAG = class_maker(
            batch_size=labels.size(0), labels=labels, class_table=class_table
        )

        CT = CT.long().to(device)
        WR = WR.long().to(device)
        COMP = COMP.long().to(device)
        MFR = MFR.long().to(device)
        MAG = MAG.long().to(device)

        t = diffusion.sample_timesteps(images.shape[0]).to(device)
        x_t, noise = diffusion.noise_images(images, t)

        predicted_noise = model(x_t, t, CT, WR, COMP, MFR, MAG)

        loss = mse(noise, predicted_noise)
        loss.backward()

        optimizer.step()
        ema.step_ema(ema_model, model)

        loss_epoch += loss.item()

    print(f"Epoch [{e}/{n_epoch}] - Loss: {loss_epoch:.3f}")

    if e % n_ax == 0:
        # Randomly sample class labels
        test_labels = torch.randint(0, num_classes, (n_sampled_images,))
        test_labels = test_labels.long().to(device)

        # Build conditioning parameters from class_table
        CT, WR, COMP, MFR, MAG = class_maker(
            batch_size=test_labels.size(0), labels=test_labels, class_table=class_table
        )

        CT = CT.long().to(device)
        WR = WR.long().to(device)
        COMP = COMP.long().to(device)
        MFR = MFR.long().to(device)
        MAG = MAG.long().to(device)

        # Sample images using EMA model
        ema_sampled_images = diffusion.sample(
            ema_model,
            n_sampled_images,
            CT,
            WR,
            COMP,
            MFR,
            MAG,
            cfg_scale=0,
        )

        # Post-process and visualize
        ema_sampled_images = reverse_transforms(ema_sampled_images)
        show_grids(ema_sampled_images, test_labels, e, label_dict)

        # Save model checkpoint
        save_dir = str(current_dir) + "/Generated-Images/All_CDDM_HR_Cat_V_6.pth.tar"
        save_model(save_dir)

print(torch.cuda.memory_summary(device=device))
