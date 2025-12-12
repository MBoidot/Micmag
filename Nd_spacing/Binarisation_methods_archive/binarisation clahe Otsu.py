import numpy as np
import cv2
from PARAMETRES import dmax, nbangles, fl, lex, dec, q2  # kept for compatibility


def binarisation(M, plagex=None, plagey=None, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Performs local binarization with CLAHE + Otsu thresholding.

    Parameters
    ----------
    M : ndarray
        2D grayscale image (values between 0 and 1 or 0–255).
    plagex, plagey : ignored
        Kept for compatibility with analyse_image.
    clip_limit : float, optional
        CLAHE contrast limit (default=2.0).
    tile_grid_size : tuple(int, int), optional
        Size of local CLAHE tiles (default=(8, 8)).

    Returns
    -------
    N : ndarray
        Binary image (1 = matrix, 0 = Nd-rich phase).
    """

    # --- Normalize image ---
    if M.max() <= 1.0:
        M = (M * 255).astype(np.uint8)
    else:
        M = M.astype(np.uint8)

    # --- Apply CLAHE (contrast enhancement) ---
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    M_eq = clahe.apply(M)

    # --- Otsu thresholding ---
    _, N = cv2.threshold(M_eq, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert binary map → Nd-rich (dark) = 0, matrix (bright) = 1
    # N = 1 - N

    return N
