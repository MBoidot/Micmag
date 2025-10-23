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

# Import configuration and processing modules
from PARAMETRES import *
from binarisation import binarisation
from distance import distance
from distribution import distribution
from calculangle import calculangle
from dessinexport import dessinexport

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
                M_bin = binarisation(M)  # binary image

                print("      ⚙️  Running distance calculation...")
                D, DM = distance(
                    M_bin, plagex, plagey
                )  # D: raw spacing, DM: smoothed spacing

                print("      ⚙️  Running distribution calculation...")
                Classes, distri = distribution(DM)  # compute histogram/distribution

                print(f"      💾 Exporting results to {cheRES}")
                dessinexport(
                    M_bin, DM, Classes, distri, chePH, cheRES, photo
                )  # export images and data
            else:  # cross-sections → angle calculation
                print("      ⚙️  Running binarisation...")
                M_bin = binarisation(M, plagex, plagey)

                print("      ⚙️  Running angle calculation...")
                M_angle = calculangle(M_bin, plagex, plagey)

                print(f"      💾 Exporting results to {cheRES}")
                dessinexport(
                    M_bin, M_angle, None, None, chePH, cheRES, photo
                )  # distribution may not apply

print("\n✅ Image analysis complete. Results saved in the 'Résultats' folder.")
