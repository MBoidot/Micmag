# -*- coding: utf-8 -*-
"""
binarisation_quantile.py
------------------------
Local adaptive binarization using intensity quantiles.
"""

import numpy as np
from scipy.ndimage import uniform_filter


def binarisation_quantile(M, window_size=15, quantile=0.05):
    """
    Local quantile-based binarization.

    Parameters
    ----------
    M : ndarray
        2D grayscale image (values 0-1)
    window_size : int
        Size of the square window for local quantile calculation
    quantile : float
        Quantile threshold (e.g., 0.05 for 5th percentile)

    Returns
    -------
    N : ndarray
        Binary image (1=matrix, 0=Nd-rich)
    """
    hauteur, largeur = M.shape
    N = np.ones_like(M, dtype=np.uint8)

    half = window_size // 2

    # Pad image to handle borders
    Mp = np.pad(M, pad_width=half, mode="reflect")

    for x in range(hauteur):
        for y in range(largeur):
            x_min = x
            x_max = x + window_size
            y_min = y
            y_max = y + window_size

            window = Mp[x_min:x_max, y_min:y_max]
            threshold = np.quantile(window, quantile)

            if M[x, y] < threshold:
                N[x, y] = 0  # Nd-rich

    return N
