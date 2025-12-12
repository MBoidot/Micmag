# -*- coding: utf-8 -*-
"""
binarisation.py
----------------
CLAHE + background correction + adaptive thresholding +Open close .

"""

import numpy as np
import cv2
from PARAMETRES import dmax, nbangles, fl, lex, dec, q2  # kept for compatibility


def binarisation(
    M,
    clip_limit=2.0,
    tile_grid_size=(8, 8),
    blur_kernel=51,
    block_size=51,
    C=2,
    kernel_size=5,
    n_close=2,
    n_open=1,
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
        Gaussian blur kernel for background correction (odd).
    block_size : int
        Block size for adaptive threshold (odd).
    C : float
        Constant subtracted in adaptive threshold (smaller = more white retained).
    kernel_size : int
        Morphological kernel size (odd).
    n_close : int
        Number of closing iterations (fill small holes in white regions).
    n_open : int
        Number of opening iterations (remove small white specks).

    Returns
    -------
    N : ndarray
        Binary mask (uint8): 1 = Nd-rich, 0 = matrix.
    """

    if M is None:
        raise ValueError("Input image M is None")

    # --- Normalize input to 8-bit grayscale ---
    M_arr = np.asarray(M)
    if M_arr.ndim == 3 and M_arr.shape[2] > 1:
        if M_arr.max() <= 1.0:
            M_gray = cv2.cvtColor((M_arr * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:
            M_gray = cv2.cvtColor(M_arr.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    else:
        M_gray = (
            (M_arr * 255).astype(np.uint8)
            if M_arr.max() <= 1.0
            else M_arr.astype(np.uint8)
        )

    # --- Background correction ---
    if blur_kernel is not None and int(blur_kernel) > 1:
        bk = int(blur_kernel) if int(blur_kernel) % 2 == 1 else int(blur_kernel) + 1
        background = cv2.GaussianBlur(M_gray, (bk, bk), 0)
        M_corr = cv2.subtract(M_gray, background)
        M_corr = cv2.normalize(M_corr, None, 0, 255, cv2.NORM_MINMAX)
    else:
        M_corr = M_gray

    # --- CLAHE enhancement ---
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid_size)
    M_eq = clahe.apply(M_corr)

    # --- Adaptive Gaussian thresholding ---
    block_size = block_size if block_size % 2 == 1 else block_size + 1
    N = cv2.adaptiveThreshold(
        M_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C
    )

    # Convert to binary 0/1
    N = (N > 0).astype(np.uint8)

    # --- Morphological cleanup (Nd-rich = white) ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Step 1: close dark holes inside white Nd-rich regions
    if n_close > 0:
        N = cv2.morphologyEx(N, cv2.MORPH_CLOSE, kernel, iterations=n_close)

    # Step 2: remove small isolated white specks
    if n_open > 0:
        N = cv2.morphologyEx(N, cv2.MORPH_OPEN, kernel, iterations=n_open)

    return N
