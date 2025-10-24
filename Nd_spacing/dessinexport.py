# -*- coding: utf-8 -*-
"""
dessinexport.py
---------------
Python translation of MATLAB's dessinexport.m

Generates visual outputs and exports data to XLS for each analyzed image,
including a circular legend for orientation angles.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_circular_legend(cmap="twilight", radius=100, bg_color=(1, 1, 1)):
    """
    Returns an RGB array for a circular orientation legend (-90° to +90°)
    with a white background.
    """
    y, x = np.ogrid[-radius:radius, -radius:radius]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    theta_deg = np.clip(np.degrees(theta), -90, 90)
    norm = (theta_deg + 90) / 180.0
    mask_circle = r <= radius

    img = np.ones((2 * radius, 2 * radius, 3), dtype=np.float32)  # white background
    cmap_func = plt.get_cmap(cmap)
    img[mask_circle] = cmap_func(norm[mask_circle])[:, :3]
    return img


def dessinexport(N, DM, Classes, distri, chePH, cheRES, photo):
    """
    Save visualizations and export data results for one analyzed image.

    Parameters
    ----------
    N : ndarray
        Binary image (0/1).
    DM : ndarray
        Processed image or result matrix (e.g. distance or angle map).
    Classes : ndarray
        X-axis for distribution (depth or size bins).
    distri : ndarray
        Distribution data corresponding to Classes.
    chePH : str
        Path prefix for image files (without extension).
    cheRES : str
        Full path for the XLS result file.
    photo : str
        Photo identifier (for title and filenames).
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(chePH), exist_ok=True)

    # --- Figure 1: binarized image ---
    plt.figure(figsize=(6, 5), facecolor="w")
    plt.imshow(N, cmap="gray", interpolation="nearest")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{chePH}-binaire_clahe+BGcorr+adaptive thresholding+OC.TIF", dpi=120)
    plt.close()

    # --- Figure 2: processed (DM) image + synthetic legend ---
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), facecolor="w")

    # Left: DM image
    im = ax[0].imshow(DM, cmap="twilight", vmin=-90, vmax=90)
    ax[0].axis("off")
    ax[0].set_title("Local orientation (DM)")
    fig.colorbar(im, ax=ax[0], label="Angle (°)")

    # Right: synthetic circular legend using dummy angle field
    res = 200  # resolution of the legend
    y, x = np.meshgrid(np.linspace(-1, 1, res), np.linspace(-1, 1, res))
    r = np.sqrt(x**2 + y**2)
    mask = r <= 1.0
    theta = np.degrees(np.arctan2(y, x))  # -180..180
    theta = np.clip(theta, -90, 90)  # match DM range
    legend = np.full_like(theta, np.nan)
    legend[mask] = theta[mask]

    ax[1].imshow(legend, cmap="twilight", vmin=-90, vmax=90, origin="lower")
    ax[1].axis("off")
    ax[1].set_title("Angle legend (°)")
    ax[1].set_aspect("equal")

    plt.tight_layout()
    plt.savefig(f"{chePH}-traite_clahe+BGcorr+adaptive thresholding+OC.TIF", dpi=120)
    plt.close()

    # --- Figure 3: distribution curve ---
    plt.figure(figsize=(6, 5), facecolor="w")
    plt.plot(Classes, distri, linewidth=1.5)
    plt.title(photo)
    plt.xlabel("Classes")
    plt.ylabel("Distribution")
    plt.tight_layout()
    plt.savefig(f"{chePH}-data_clahe+BGcorr+adaptive thresholding+OC.TIF", dpi=120)
    plt.close()

    # --- Excel export ---
    df = pd.DataFrame({"Classes": Classes.flatten(), "distri": distri.flatten()})
    df.to_excel(cheRES, index=False)

    print(f"✅ Exported: {photo}")
    print(f"   ├─ {chePH}-binaire.TIF")
    print(f"   ├─ {chePH}-traite.TIF")
    print(f"   ├─ {chePH}-data.TIF")
    print(f"   └─ {cheRES}")
