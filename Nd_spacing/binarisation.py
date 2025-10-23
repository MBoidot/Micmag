# -*- coding: utf-8 -*-
"""
binarisation.py
----------------
CLAHE + background correction + Otsu thresholding.
Compatible with legacy call: binarisation(M, plagex=None, plagey=None).

Returns
-------
N : ndarray (uint8)
    Binary mask: 1 = matrix (bright), 0 = Nd-rich (dark).
"""

import numpy as np
import cv2
from PARAMETRES import dmax, nbangles, fl, lex, dec, q2  # kept for compatibility


def binarisation(
    M,
    plagex=None,
    plagey=None,
    *,
    clip_limit=2.0,
    tile_grid_size=(8, 8),
    blur_kernel=51
):
    """
    Parameters
    ----------
    M : ndarray
        Input grayscale image (float 0..1 or uint8 0..255).
    plagex, plagey : ignored (kept for compatibility)
    clip_limit : float
        CLAHE clip limit.
    tile_grid_size : tuple
        CLAHE tile grid size.
    blur_kernel : int
        Kernel size for Gaussian blur used to estimate background (must be odd).
        Set to None or <=1 to skip background correction.
    return_enhanced : bool
        If True, return (N, M_eq) where M_eq is the contrast-enhanced image (uint8).
    debug : bool
        If True, print basic stats for tuning.

    Returns
    -------
    N or (N, M_eq)
    """
    # --- Normalize input to 8-bit grayscale ---
    if M is None:
        raise ValueError("Input image M is None")
    M_arr = np.asarray(M)
    if M_arr.ndim == 3 and M_arr.shape[2] > 1:
        # convert color to grayscale by luminosity if needed
        M_gray = cv2.cvtColor(
            (
                (M_arr * 255).astype(np.uint8)
                if M_arr.max() <= 1.0
                else M_arr.astype(np.uint8)
            ),
            cv2.COLOR_BGR2GRAY,
        )
    else:
        if M_arr.max() <= 1.0:
            M_gray = (M_arr * 255).astype(np.uint8)
        else:
            M_gray = M_arr.astype(np.uint8)

    # --- Background correction (large-scale illumination) ---
    if blur_kernel is not None and int(blur_kernel) > 1:
        # ensure odd kernel size
        bk = int(blur_kernel) if int(blur_kernel) % 2 == 1 else int(blur_kernel) + 1
        background = cv2.GaussianBlur(M_gray, (bk, bk), 0)
        # Subtract background and rescale
        M_corr = cv2.subtract(M_gray, background)
        M_corr = cv2.normalize(M_corr, None, 0, 255, cv2.NORM_MINMAX)
    else:
        M_corr = M_gray

    # --- CLAHE (local contrast enhancement) ---
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid_size)
    M_eq = clahe.apply(M_corr)

    # --- Otsu thresholding (on enhanced image) ---
    _, thresh_img = cv2.threshold(M_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # By convention in this project: Nd-rich (dark) = 0, matrix (bright) = 1
    N = (thresh_img > 0).astype(np.uint8)  # 1 for matrix
    # ensure values are 0/1
    N = N.copy()

    return N
