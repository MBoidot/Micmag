# -*- coding: utf-8 -*-
"""
calculangle_matlab.py
--------------
Python translation of MATLAB's calculangle.m

Determines the dominant local angle/orientation of bright regions in a binary image N.
Used for "tranche" (cross-section) images to later correct spacing.
"""

import numpy as np
from PARAMETRES import pas, nbangles, fl, lex, envergure


def calculangle_matlab(N):
    """
    Compute dominant local angle map (A) and angle distribution along depth.

    Parameters
    ----------
    N : ndarray
        Binary image (1 for bright pixels, 0 for dark).

    Returns
    -------
    DM : ndarray
        Angle map (degrees), same shape as N.
    Classes : ndarray
        Vertical coordinate array (depth).
    distri : ndarray
        Average deviation from vertical per image row.
    """
    hauteur, largeur = N.shape

    # Initialize angle matrix
    A = -120 * np.ones((hauteur, largeur))

    # Define angle ranges
    angleverti = np.concatenate(
        (np.arange(-90, -45 + pas, pas), np.arange(45, 90 + pas, pas))
    )
    nombredanglesverti = len(angleverti)
    vverti = np.zeros(nombredanglesverti)

    anglehoriz = np.arange(-45, 45 + pas, pas)
    nombredangleshoriz = len(anglehoriz)
    vhoriz = np.zeros(nombredangleshoriz)

    # Precalculations
    an = np.arange(1, nbangles + 1)
    Angles = -90 + an * 180 / nbangles
    llex = 2 * fl * lex + 1

    # --- Determine local dominant angle for each bright pixel ---
    for i in range(hauteur):
        for j in range(largeur):
            if N[i, j] == 1:
                # --- Vertical-type angles ---
                for k, angle in enumerate(angleverti):
                    total = 0
                    allumes = 0
                    for m in range(i - envergure, i + envergure + 1):
                        if 0 <= m < hauteur:
                            jj = int(
                                round(
                                    j
                                    + np.sign(angle)
                                    * (m - i)
                                    * np.cos(np.deg2rad(angle))
                                )
                            )
                            if 1 <= jj < largeur - 1:
                                total += 3
                                allumes += np.sum(N[m, jj - 1 : jj + 2])
                    vverti[k] = allumes / total if total > 0 else 0

                # --- Horizontal-type angles ---
                for k, angle in enumerate(anglehoriz):
                    total = 0
                    allumes = 0
                    for m in range(j - envergure, j + envergure + 1):
                        if 0 <= m < largeur:
                            ii = int(round(i + (m - j) * np.sin(np.deg2rad(angle))))
                            if 1 <= ii < hauteur - 1:
                                total += 3
                                allumes += np.sum(N[ii - 1 : ii + 2, m])
                    vhoriz[k] = allumes / total if total > 0 else 0

                # Choose between vertical and horizontal orientation
                valeurverti = np.max(vverti)
                indicemaxverti = np.argmax(vverti)
                valeurhoriz = np.max(vhoriz)
                indicemaxhoriz = np.argmax(vhoriz)

                if valeurverti > valeurhoriz:
                    A[i, j] = angleverti[indicemaxverti]
                else:
                    A[i, j] = anglehoriz[indicemaxhoriz]

    # --- Post-processing ---
    DM = A.copy()
    Classes = np.arange(1, hauteur + 1)
    distri = np.zeros(hauteur)

    # Compute average deviation from vertical for each image row
    for i in range(hauteur):
        releve = A[i, N[i, :] == 1]
        if len(releve) > 0:
            distri[i] = np.mean(90 - np.abs(releve))

    return DM, Classes, distri
