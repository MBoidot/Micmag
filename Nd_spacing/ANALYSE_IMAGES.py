# -*- coding: utf-8 -*-
"""
analyse_images.py
-----------------
Main script to analyze images and export one .xlsx file per image.

Python equivalent of MATLAB's ANALYSE_IMAGES.m
"""
import os
import numpy as np
import imageio
import time
from PARAMETRES import *
from binarisation import binarisation
from distance import distance
from distribution import distribution
from calculangle import calculangle
from dessinexport import dessinexport
from prolong_nd import (
    simple_closing,
    multi_angle_closing,
    watershed_and_edge_prolong,
    watershed_grains_pro,
)
from matplotlib import pyplot as plt
from skimage import morphology, segmentation, feature
from scipy import ndimage as ndi

# --------------------------------------------------------------------------
# --- Main analysis loop ---
# --------------------------------------------------------------------------
step = 1  # adjust as needed

global_start = time.perf_counter()
print("🟢 Starting image analysis...")

for ech in range(len(ECHTS)):  # For each sample
    echt = ECHTS[ech]
    print(f"\n📦 Processing sample {ech+1}/{len(ECHTS)}: {echt}")

    nbtypes = len(TYPES) // 3
    for typ in range(nbtypes):  # For each type/depth
        type_str = TYPES[typ * 3 : typ * 3 + 3]

        # Last type can be 'TR' (2 letters only)
        if typ == nbtypes - 1:
            type_str = type_str[:2]

        nbrephotos = NBPHOTOS[5 * ech + typ]
        print(f"  🔹 Type {typ+1}/{nbtypes}: {type_str}, {nbrephotos} photo(s)")

        for fich in range(1, nbrephotos + 1):  # For each image
            photo = f"{nomanip}{echt}-{type_str}{fich}"
            chePH = os.path.join(cheM, photo)
            cheRES = os.path.join(cheM, "Résultats", f"{photo}.xlsx")

            # Ensure results directory exists
            os.makedirs(os.path.dirname(cheRES), exist_ok=True)

            img_path = chePH + ".TIF"
            if not os.path.exists(img_path):
                print(f"    ⚠️  Image not found: {img_path}")
                continue

            print(f"    🖼️  Processing photo {fich}/{nbrephotos}: {img_path}")

            # Read and normalize image
            M = imageio.v2.imread(img_path).astype(np.float64) / 255.0
            hauteur, largeur = M.shape[:2]

            # For quick tests increase step, for precision keep at 1

            plagex = np.arange(1, hauteur, step)
            plagey = np.arange(1, largeur, step)

            # Choose processing path
            if "T" not in type_str:  # face views → spacing calculation
                print("      ⚙️  Running binarisation...")

                M_bin = binarisation(
                    M,
                    clip_limit=2.0,
                    tile_grid_size=(8, 8),
                    blur_kernel=101,
                    block_size=51,
                    C=3,
                    kernel_size=2,
                    n_close=1,
                    n_open=1,
                )

                # 1️⃣ Binarized
                imageio.imwrite(
                    f"{chePH}-binarized.TIF", (M_bin * 255).astype(np.uint8)
                )

                # 2️⃣ Morphological closing
                M_bin_closed = simple_closing(M_bin, disk_radius=3)
                imageio.imwrite(
                    f"{chePH}-closed.TIF", (M_bin_closed * 255).astype(np.uint8)
                )

                # 3️⃣ Directional closing
                M_bin_dir = multi_angle_closing(
                    M_bin, line_length=3, angles=np.arange(0, 180, 1)
                )
                imageio.imwrite(
                    f"{chePH}-directional_closing.TIF",
                    (M_bin_dir * 255).astype(np.uint8),
                )

                # 4️⃣ Watershed + edge prolongation
                M_bin_prolonged, labels = watershed_and_edge_prolong(
                    M_bin, min_distance=1, min_size=50, edge_width=2
                )
                imageio.imwrite(
                    f"{chePH}-prolonged.TIF", (M_bin_prolonged * 255).astype(np.uint8)
                )
                # --- Prepare binarized variants
                bin_variants = {
                    "Binarized": M_bin,
                    "Closed": M_bin_closed,
                    "Directional Closing": M_bin_dir,
                    "Prolonged": M_bin_prolonged,
                }

                # Dictionaries to store watershed outputs
                dil_eroded_masks = {}
                grain_masks = {}
                grain_labels_dict = {}

                # --- Run watershed grains on each variant
                for name, img in bin_variants.items():
                    mask, labels, dil_eroded = watershed_grains_pro(
                        img,  # use each variant
                        pre_dilate_steps=2,
                        pre_erode_steps=1,
                        selem_radius=1,
                        marker_distance=20,
                        min_grain_size=40,
                        edge_smooth=0,
                        return_preprocessed=True,
                    )
                    dil_eroded_masks[name] = dil_eroded
                    grain_masks[name] = mask
                    grain_labels_dict[name] = labels

                # --- Summary figure: 4 rows (original mask, dilated/eroded, grain mask, labels)
                n_variants = len(bin_variants)
                fig, axes = plt.subplots(4, n_variants, figsize=(5 * n_variants, 16))

                # Flatten axes for indexing
                axes = axes if axes.ndim == 2 else axes.reshape(4, n_variants)

                # 1️⃣ Top row: original Nd-rich masks
                for i, name in enumerate(bin_variants.keys()):
                    axes[0, i].imshow(bin_variants[name], cmap="gray")
                    axes[0, i].set_title(f"{name} Mask")
                    axes[0, i].axis("off")

                # 2️⃣ Second row: masks after dilation → erosion
                for i, name in enumerate(bin_variants.keys()):
                    axes[1, i].imshow(dil_eroded_masks[name], cmap="gray")
                    axes[1, i].set_title(f"{name} Dilated/Eroded")
                    axes[1, i].axis("off")

                # 3️⃣ Third row: grain masks
                for i, name in enumerate(bin_variants.keys()):
                    axes[2, i].imshow(grain_masks[name], cmap="gray")
                    axes[2, i].set_title(f"{name} Grain Mask")
                    axes[2, i].axis("off")

                # 4️⃣ Bottom row: labeled grains
                for i, name in enumerate(bin_variants.keys()):
                    axes[3, i].imshow(grain_labels_dict[name], cmap="tab20")
                    axes[3, i].set_title(f"{name} Labels")
                    axes[3, i].axis("off")

                plt.tight_layout()
                plt.savefig(f"{chePH}-mask_dilated_grains_labels.png", dpi=200)
                plt.close(fig)

                # Select the image used for measurements
                M_for_measure = M_bin_prolonged  # choose which binary mask to use

                print("      ⚙️  Running distance calculation...")
                D, DM = distance(M_for_measure, plagex, plagey)

                print("      ⚙️  Running distribution calculation...")
                Classes, distri = distribution(DM)

                print(f"      💾 Exporting results to {cheRES}")
                dessinexport(M_for_measure, DM, Classes, distri, chePH, cheRES, photo)

                # --- Watershed grains
                M_grain_mask, grain_labels = watershed_grains_pro(
                    M_bin,
                    pre_dilate_steps=2,
                    pre_erode_steps=1,
                    selem_radius=1,
                    marker_distance=5,
                    min_grain_size=50,
                    edge_smooth=1,
                )

                # Save outputs for inspection
                import imageio

                imageio.imwrite(
                    f"{chePH}-grains_mask.TIF", (M_grain_mask * 255).astype(np.uint8)
                )
                plt.imsave(f"{chePH}-grains_labels.TIFF", grain_labels, cmap="tab20")

            else:  # cross-sections → angle calculation

                print("      ⚙️  Running binarisation...")
                M_bin = binarisation(
                    M,
                    clip_limit=2.0,
                    tile_grid_size=(8, 8),
                    blur_kernel=101,
                    block_size=51,
                    C=3,
                    kernel_size=2,
                    n_close=1,
                    n_open=1,
                )

                print("      ⚙️  Running angle calculation...")

                start = time.perf_counter()
                M_angle, Classes, distri = calculangle(
                    M_bin, smooth_sigma=0.3, tensor_blur=15, eps=1e-12
                )
                elapsed = time.perf_counter() - start
                print(f"⏱️ Orientation calculation: {elapsed:.2f} s")
                print(f"      💾 Exporting results to {cheRES}")
                dessinexport(
                    M_bin, M_angle, Classes, distri, chePH, cheRES, photo
                )  # distribution may not apply

global_elapsed = time.perf_counter() - global_start
print(f"\n⏱️ Total elapsed time: {global_elapsed:.2f} s")
print("\n✅ Image analysis complete. Results saved in the 'Résultats' folder.")
