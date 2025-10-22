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

for ech in range(len(ECHTS)):  # For each sample
    echt = ECHTS[ech]
    nbtypes = len(TYPES) // 3

    for typ in range(nbtypes):  # For each type/depth
        type_str = TYPES[typ * 3 : typ * 3 + 3]

        # Last type can be 'TR' (2 letters only)
        if typ == nbtypes - 1:
            type_str = type_str[:2]

        nbrephotos = NBPHOTOS[5 * ech + typ]

        for fich in range(1, nbrephotos + 1):  # For each image
            photo = f"{nomanip}{echt}-{type_str}{fich}"
            chePH = os.path.join(cheM, photo)
            cheRES = os.path.join(cheM, "Résultats", f"{photo}.xlsx")

            # Ensure results directory exists
            os.makedirs(os.path.dirname(cheRES), exist_ok=True)

            img_path = chePH + ".TIF"
            if not os.path.exists(img_path):
                print(f"⚠️  Image not found: {img_path}")
                continue

            # Read and normalize image
            M = imageio.imread(img_path).astype(np.float64) / 255.0
            hauteur, largeur = M.shape[:2]

            # For quick tests (start from pixel 10)
            step = 5  # adjust as needed
            plagex = np.arange(1, hauteur + 1, step)
            plagey = np.arange(1, largeur + 1, step)

            # Choose processing path
            if "T" not in type_str:  # face views → spacing calculation
                M = binarisation(M)
                M = distance(M)
                M = distribution(M)
            else:  # cross-sections → angle calculation
                M = binarisation(M)
                M = calculangle(M)

            # Export results
            dessinexport(M, cheRES)

print("\n✅ Image analysis complete. Results saved in the 'Résultats' folder.")
