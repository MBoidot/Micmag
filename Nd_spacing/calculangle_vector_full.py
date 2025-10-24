# -*- coding: utf-8 -*-
"""
calculangle.py (vectorized, structure-tensor)
--------------------------------------------
Fast, fully vectorized orientation estimator using the structure tensor.

Returns:
    DM      : angle map (degrees), shape = N.shape, -120 for undefined
    Classes : array([1..hauteur])
    distri  : per-row mean of (90 - |angle|) computed on bright pixels

    _______________________________

No nested loops over pixels/angles — everything is array operations and a few image filters.

Uses cv2.Sobel for gradients and cv2.GaussianBlur to smooth the tensor components (both were already present in your environment).

Returns the same outputs as your MATLAB/Python function: (DM, Classes, distri).

DM contains angles in degrees in range (-90, 90] for pixels where an orientation is defined; background pixels keep the -120 sentinel like your original implementation.

distri is computed exactly as before: per-row mean of 90 - abs(angle) over bright pixels.


Tuning tips

smooth_sigma controls pre-smoothing of the input; increase if very noisy.

tensor_blur (aggregation window) controls the scale of orientation detection. Make it ~2*envergure+1 like above; larger → smoother orientation fields.

If you want exactly the same behavior as the original
 (which scores along discrete angles and uses a 3-pixel thickness), you can post-process
   the DM to quantize to the nearest angle in your angleverti/anglehoriz sets —
     but usually the continuous angle is better.
"""


import numpy as np
import cv2
from PARAMETRES import pas, nbangles, fl, lex, envergure


def calculangle(N, smooth_sigma=None, tensor_blur=None, eps=1e-12):
    """
    Vectorized orientation estimation using the structure tensor.

    Parameters
    ----------
    N : ndarray (2D)
        Binary image (1 = bright / feature, 0 = background/dark).
    smooth_sigma : float or None
        Optional Gaussian smoothing sigma applied to the input before gradient.
        If None, defaults to max(1.0, envergure/3).
    tensor_blur : int or None
        Kernel size (odd) used to smooth the tensor components Jxx, Jxy, Jyy.
        If None, defaults to 2*envergure+1 (odd).
    eps : float
        Small epsilon to avoid division by zero.

    Returns
    -------
    DM, Classes, distri
    """
    # ensure 2D float image
    N_arr = np.asarray(N, dtype=np.float32)
    if N_arr.ndim != 2:
        raise ValueError("calculangle expects a 2D array")

    hauteur, largeur = N_arr.shape

    # default smoothing params
    if smooth_sigma is None:
        smooth_sigma = max(1.0, envergure / 3.0)
    if tensor_blur is None:
        tb = int(2 * envergure + 1)
        tensor_blur = tb if tb % 2 == 1 else tb + 1

    # 1) Optional small smoothing to reduce noise (input smoothing)
    # convert to 0..255 uint8 for cv2 GaussianBlur then back to float
    tmp = (np.clip(N_arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    ksize = int(max(3, int(round(smooth_sigma * 4 + 1)) // 2 * 2 + 1))  # odd
    tmp_blur = cv2.GaussianBlur(
        tmp, (ksize, ksize), sigmaX=smooth_sigma, sigmaY=smooth_sigma
    )
    I = tmp_blur.astype(np.float32) / 255.0  # smoothed float image

    # 2) compute gradients (Ix, Iy)
    # Sobel returns derivatives scaled; use cv2.CV_32F to keep float
    Ix = cv2.Sobel(I, cv2.CV_32F, 1, 0, ksize=3, scale=1.0, delta=0.0)
    Iy = cv2.Sobel(I, cv2.CV_32F, 0, 1, ksize=3, scale=1.0, delta=0.0)

    # 3) structure tensor components
    Jxx = Ix * Ix
    Jyy = Iy * Iy
    Jxy = Ix * Iy

    # 4) smooth the tensor components with a larger Gaussian (aggregation window)
    # convert kernel size to odd integer
    kb = int(tensor_blur) if int(tensor_blur) % 2 == 1 else int(tensor_blur) + 1
    Jxx_s = cv2.GaussianBlur(Jxx, (kb, kb), 0)
    Jyy_s = cv2.GaussianBlur(Jyy, (kb, kb), 0)
    Jxy_s = cv2.GaussianBlur(Jxy, (kb, kb), 0)

    # 5) compute orientation angle from the tensor:
    #    theta = 0.5 * arctan2(2 Jxy, Jxx - Jyy)
    # this yields angle in radians, where angle corresponds to dominant orientation
    # convert to degrees in range (-90, 90]
    denom = Jxx_s - Jyy_s
    ang_rad = 0.5 * np.arctan2(2.0 * Jxy_s, denom + eps)
    ang_deg = np.degrees(ang_rad)  # -90..90

    # 6) Mask out places with too little signal (optional): keep only where N==1 (bright)
    DM = -120.0 * np.ones_like(ang_deg, dtype=np.float32)
    mask = N_arr > 0.5  # bright pixels mask (1)
    DM[mask] = ang_deg[mask]

    # 7) Prepare Classes and distri (same meaning as original)
    Classes = np.arange(1, hauteur + 1, dtype=np.int32)
    distri = np.zeros(hauteur, dtype=np.float32)

    # compute per-row mean of (90 - abs(angle)) on bright pixels
    # avoid empty-slice warnings by checking counts per row
    abs_ang = np.abs(DM)  # -120 will have abs 120 but is masked out by mask below
    for i in range(hauteur):
        row_mask = mask[i, :]
        if np.count_nonzero(row_mask) > 0:
            # angles only where mask true
            row_angles = DM[i, row_mask]
            # exclude sentinel -120 just in case
            if row_angles.size > 0:
                distri[i] = np.mean(90.0 - np.abs(row_angles))

    return DM, Classes, distri
