# -*- coding: utf-8 -*-
"""
analyse_images.py
-----------------
Main script to analyze images and export one .xlsx file per image.

Python equivalent of MATLAB's ANALYSE_IMAGES.m
"""
import os
import numpy as np
import imageio.v2 as imageio
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
    orientation_based_prolong,
)
from matplotlib import pyplot as plt

# --------------------------------------------------------------------------
# --- Main analysis loop ---
# --------------------------------------------------------------------------
step = 1  # adjust as needed
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
            M = imageio.imread(img_path).astype(np.float64) / 255.0
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
                plt.imsave(f"{chePH}-watershed_labels.TIFF", labels, cmap="tab20")

                # --- Summary figure with all steps ---
                fig, axes = plt.subplots(1, 5, figsize=(20, 4))
                axes = axes.ravel()

                axes[0].imshow(M, cmap="gray")
                axes[0].set_title("Original")

                axes[1].imshow(M_bin, cmap="gray")
                axes[1].set_title("Binarized")

                axes[2].imshow(M_bin_closed, cmap="gray")
                axes[2].set_title("Closed")

                axes[3].imshow(M_bin_dir, cmap="gray")
                axes[3].set_title("Directional Closing")

                axes[4].imshow(M_bin_prolonged, cmap="gray")
                axes[4].set_title("Prolonged (Watershed)")

                for ax in axes:
                    ax.axis("off")

                plt.tight_layout()
                plt.savefig(f"{chePH}-summary.png", dpi=200)
                plt.close(fig)

                # Select the image used for measurements
                M_for_measure = M_bin_prolonged  # choose which binary mask to use

                print("      ⚙️  Running distance calculation...")
                D, DM = distance(M_for_measure, plagex, plagey)

                print("      ⚙️  Running distribution calculation...")
                Classes, distri = distribution(DM)

                print(f"      💾 Exporting results to {cheRES}")
                dessinexport(M_for_measure, DM, Classes, distri, chePH, cheRES, photo)

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

                print("DEBUG M_angle:", type(M_angle), getattr(M_angle, "shape", None))
                print(f"      💾 Exporting results to {cheRES}")
                dessinexport(
                    M_bin, M_angle, Classes, distri, chePH, cheRES, photo
                )  # distribution may not apply

print("\n✅ Image analysis complete. Results saved in the 'Résultats' folder.")
