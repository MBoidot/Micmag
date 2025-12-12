# -*- coding: utf-8 -*-
"""
binarisation.py
----------------
Python version using Sauvola adaptive thresholding for Nd-rich detection.
"""

import numpy as np
from skimage.filters import threshold_sauvola
from skimage.morphology import opening, closing, square


def binarisation(M, plagex=None, plagey=None, window_size=20, k=0.2, morph_size=3):
    """
    Perform local adaptive binarization using Sauvola method.

    Parameters
    ----------
    M : ndarray
        2D grayscale image (values between 0 and 1).
    plagex, plagey : optional
        Pixel ranges to process (for speed).
    window_size : int
        Neighborhood size for Sauvola threshold.
    k : float
        Sauvola parameter (typically 0.2 to 0.5).
    morph_size : int
        Size of morphological opening/closing to remove small noise.

    Returns
    -------
    N : ndarray
        Binary image (1 = matrix, 0 = Nd-rich phase).
    """

    hauteur, largeur = M.shape[:2]

    # If no subgrid is provided, process the whole image
    if plagex is None:
        plagex = np.arange(hauteur)
    if plagey is None:
        plagey = np.arange(largeur)

    # Compute Sauvola threshold map
    thresh_map = threshold_sauvola(M, window_size=window_size, k=k)

    # Binarize: 1 = matrix, 0 = Nd-rich
    N = np.ones_like(M)
    N[M <= thresh_map] = 0

    # Optional: morphological cleaning on the full image
    selem = square(morph_size)
    N = opening(N, selem)
    N = closing(N, selem)

    # If only subgrid was intended, return the full N anyway (for compatibility)
    return N
