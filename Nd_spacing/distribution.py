# -*- coding: utf-8 -*-
"""
distribution.py
---------------
Python translation of MATLAB's distribution.m

Computes the histogram/distribution of the local mean distances (DM).
"""

import numpy as np
from PARAMETRES import dmax, pasdistri


def distribution(DM):
    """
    Compute the normalized distribution of DM values.

    Parameters
    ----------
    DM : ndarray
        Local mean spacing map (e.g., output from distance.py).

    Returns
    -------
    classes : ndarray
        Class boundaries for the histogram (same as MATLAB 'classes').
    distri : ndarray
        Normalized frequency of each class.
    """

    # Define histogram class boundaries (same as MATLAB)
    classes = np.arange(0, 2 * dmax + (pasdistri * 0.9), pasdistri)
    L = len(classes)

    districum = np.zeros(L)
    hauteur, largeur = DM.shape[:2]

    # --- Cumulative distribution counting ---
    for i in range(hauteur):
        for j in range(largeur):
            val = DM[i, j]
            districum += (classes > val).astype(float)

    # --- Convert cumulative into differential distribution ---
    distri = np.zeros(L)
    distri[:-1] = districum[1:] - districum[:-1]

    # Normalize
    total = np.sum(distri)
    if total > 0:
        distri = distri / total

    return classes, distri
