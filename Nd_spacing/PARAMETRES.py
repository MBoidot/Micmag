# -*- coding: utf-8 -*-
"""
PARAMETRES.py
-------------
Equivalent of the MATLAB setup script 'PARAMETRES.m'.
This defines all configuration variables used in the image analysis project.

Usage:
    from PARAMETRES import *

Then use the variables directly (e.g., manip, cheM, NBPHOTOS, etc.)
"""

import os
import numpy as np

# --------------------------------------------------------------------------
# COULEE À ANALYSER (CAST TO ANALYZE)
# --------------------------------------------------------------------------

nomanip = "18"  # For the SC ribbons, put the casting number (e.g. '18' for SC18)
manip = f"SC{nomanip}"  # 'SC' or 'VAC' or another prefix
ECHTS = "A"  # Sample identifiers (e.g., 'ABC' for three ribbons A, B, C)

TYPES = "RRRRRARAAAAATR_"  # Each type is 3 letters, concatenated one after another
NB_PHOTOS = [0, 0, 0, 0, 11]  # Per type, 5 values per sample
nb_identiques = 1  # If all samples have same number of photos → 1, else 0
epaisseurs = [155, 185]  # One value per sample (µm)
rmax = 15  # Exploration radius in µm (standard: 5)
dmaxfigures = 25  # Standard: 2*dmax, can be smaller for RASSEMBLE
bpf = np.array(
    [[10, 20], [30, 40], [60, 70], [80, 90]]  # Boundaries for depths (% from the wheel)
)

# --------------------------------------------------------------------------
# PATHS
# --------------------------------------------------------------------------

che = r"C:/Users/MB232649/Documents/M.Boidot Local/Informatique/Projets-info/2025-Nd_Rich_Spacing/Nd_spacing/"
cheM = os.path.join(
    che, manip, ""
)  # Folder for the current manipulation (must contain 'RESULTATS' subfolder)

# --------------------------------------------------------------------------
# IMAGE PROCESSING PARAMETERS
# --------------------------------------------------------------------------

dmax = rmax
echelle = 5.2  # µm per pixel for 1000x magnification
echelleTR = 2.6  # µm per pixel for 500x magnification
lex = round(dmax * echelle)  # Exploration length from the current point (pixels)
fl = 2  # Multiplication factor for finer angular exploration
nbangles = 40  # Number of angles between 0 and 180° to find shortest segment
nbamoy = 4  # Number of angles averaged for distance calculation
dec = 30  # Number of histogram bins for quantile estimation
q2 = 0.88  # Upper quantile for Nd-rich local classification
largeur_moyennage = 3  # Width of the smoothing zone in final rendering
pas = 3  # Angular step during slice angle analysis (degrees)
envergure = 15  # Pixels to each side for local angle calculation
pasdistri = 2 / echelle  # Step for granulometric distribution (µm)
pasdistrifinal = 0.2  # Step for final synthetic distribution
bords = 0.6  # Border elimination threshold (fraction of intersections)

# --------------------------------------------------------------------------
# DERIVED PARAMETERS
# --------------------------------------------------------------------------

nbtypes = len(TYPES) // 3

# Manage number of photos per sample
NBPHOTOS = []

if nb_identiques:
    for _ in ECHTS:
        NBPHOTOS.extend(NB_PHOTOS)
else:
    NBPHOTOS = NB_PHOTOS

# --------------------------------------------------------------------------
# Debug print (optional)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("PARAMETRES configuration loaded.")
    print(f"Manip: {manip}, Samples: {ECHTS}, Types: {nbtypes}")
    print(f"Images per sample: {NBPHOTOS}")
    print(f"Working directory: {cheM}")
