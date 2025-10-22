# -*- coding: utf-8 -*-
"""
binarisation.py
----------------
Python translation of MATLAB's binarisation.m

Performs local adaptive binarization on grayscale images
based on local intensity quantiles (Nd-rich detection).
"""

import numpy as np
from PARAMETRES import dmax, nbangles, fl, lex, dec, q2


def binarisation(M, plagex=None, plagey=None):
    """
    Performs local adaptive binarization on image M.

    Parameters
    ----------
    M : ndarray
        2D grayscale image (values between 0 and 1).
    plagex, plagey : optional
        Ranges of pixels to process (for speed).

    Returns
    -------
    N : ndarray
        Binary image (1 = matrix, 0 = Nd-rich phase).
    """

    hauteur, largeur = M.shape[:2]

    # Initialize matrices
    MM = -(dmax / 2) * np.ones((hauteur, largeur))
    D = -(dmax / 2) * np.ones((hauteur, largeur))
    N = np.ones((hauteur, largeur))
    DM = -(dmax / 2) * np.ones((hauteur, largeur))

    # Pre-calculations
    an = np.arange(1, nbangles + 1)
    Angles = -90 + an * 180 / nbangles
    llex = int(2 * fl * lex + 1)

    if plagex is None:
        plagex = np.arange(hauteur)
    if plagey is None:
        plagey = np.arange(largeur)

    # --- Main loop ---
    for x in plagex:
        for y in plagey:
            # Local random rotation (unused here, but preserved for consistency)
            _ = (Angles + 90) * np.pi / 180 + np.pi * np.random.rand()

            # Local region
            x_min = max(0, x - lex)
            x_max = min(hauteur, x + lex)
            y_min = max(0, y - lex)
            y_max = min(largeur, y + lex)

            Mloc = M[x_min:x_max, y_min:y_max]
            a, b = Mloc.shape
            T = a * b

            mi = np.min(Mloc)
            ma = np.max(Mloc)

            w = np.zeros(dec)
            W = np.zeros(dec + 1)
            es = (ma - mi) / dec

            i2 = 0
            for i in range(dec):
                mask = (Mloc < mi + (i + 1) * es) & (Mloc >= mi + i * es)
                w[i] = np.sum(mask)
                W[i + 1] = np.sum(w[: i + 1]) / T
                if W[i] < q2:
                    i2 = i

            # Linear interpolation to find threshold m2
            denom = W[i2 + 1] - W[i2] if (W[i2 + 1] - W[i2]) != 0 else 1e-9
            m2 = (
                (q2 - W[i2]) * (mi + (i2 + 1) * es)
                + (W[i2 + 1] - q2) * (mi + (i2 + 1) * es - es)
            ) / denom

            # Binarization condition (magnetic phase or isolated pixel)
            if M[x, y] < m2 or (
                2 < x < hauteur - 2
                and 2 < y < largeur - 2
                and np.sum(M[x - 2 : x + 3, y - 2 : y + 3] > m2) < 5
            ):
                N[x, y] = 0  # Nd-rich phase

    return N
