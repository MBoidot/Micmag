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
from tkinter import Tk, filedialog, messagebox


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
    # --- Figure 2: processed (DM) image with hemispheric legend below ---
    fig, ax = plt.subplots(2, 1, figsize=(6, 7), facecolor="w", height_ratios=[4, 1])
    # --- Use masked DM directly ---
    cmap = plt.cm.viridis
    cmap.set_bad(color="black")  # masked regions → black
    # --- Top: orientation map ---
    im = ax[0].imshow(DM, cmap=cmap, vmin=-90, vmax=90)
    ax[0].set_title("Local orientation (DM)")
    ax[0].axis("off")
    # --- Bottom: hemispheric legend ---
    res = 300
    radius = 1.0
    x, y = np.meshgrid(np.linspace(-1, 1, 2 * res), np.linspace(0, 1, res))
    r = np.sqrt(x**2 + y**2)
    theta = np.degrees(np.arctan2(x, y))
    theta = np.clip(theta, -90, 90)
    legend = np.full_like(theta, np.nan)
    mask = r <= radius
    legend[mask] = theta[mask]
    ax[1].imshow(
        legend, cmap="viridis", vmin=-90, vmax=90, origin="lower", aspect="equal"
    )
    ax[1].axis("off")
    ax[1].set_title("Orientation legend (°)", pad=8)
    plt.tight_layout()
    plt.savefig(f"{chePH}-traite_clahe+BGcorr+adaptive thresholding+OC.TIF", dpi=150)
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


def select_and_plot_files(results_dir):
    """
    Open a file selection dialog to choose XLSX files and plot their data.

    Args:
        results_dir (str): Path to the directory containing the XLSX files.
    """
    # Initialize Tkinter
    root = Tk()
    root.withdraw()  # Hide the main window

    # Ask the user to select files
    file_paths = filedialog.askopenfilenames(
        title="Select XLSX files to plot",
        initialdir=results_dir,
        filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
    )

    if not file_paths:
        return

    # Determine the type of data based on the first selected file
    first_file = file_paths[0]
    df = pd.read_excel(first_file)

    # Check if the file contains distance data or angle data
    if "Classes" in df.columns and "distri" in df.columns:
        x_col = "Classes"
        y_col = "distri"
        title = "Superimposed Distribution"
    else:
        messagebox.showinfo(
            "Info", "Could not determine the type of data in the selected files."
        )
        return

    # Create the plot
    plt.figure(figsize=(10, 6))

    for file in file_paths:
        df = pd.read_excel(file)
        plt.plot(df[x_col], df[y_col], label=os.path.basename(file))

    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_filename = os.path.join(results_dir, "superimposed_plot.png")
    plt.savefig(plot_filename)
    messagebox.showinfo("Info", f"Superimposed plot saved as: {plot_filename}")

    plt.show()
