# -*- coding: utf-8 -*-
"""
calculangle.py (Radon transform version)
---------------------------------------
Compute orientation distribution and dominant angle using the Radon transform.

Returns:
    DM      : 2D angle map (uniform or local)
    Classes : tested angles (degrees)
    distri  : Radon projection intensity per angle
"""

import numpy as np
from skimage.transform import radon
import cv2


def calculangle(M_bin, theta_res=1, smooth_sigma=2, local=False, patch_size=64):
    """
    Estimate dominant orientation using the Radon transform.

    Parameters
    ----------
    M_bin : ndarray
        Binary image (1 = Nd-rich, 0 = background).
    theta_res : float
        Angular resolution (°) for Radon projections.
    smooth_sigma : float
        Optional Gaussian blur to smooth before Radon.
    local : bool
        If True, compute local Radon on small patches (slower).
        If False, compute global dominant orientation.
    patch_size : int
        Size of square patch if local=True.

    Returns
    -------
    DM : ndarray
        Orientation map (°), same shape as input if local=True,
        else uniform image filled with dominant angle.
    Classes : ndarray
        Tested angles (°).
    distri : ndarray
        Normalized Radon projection per angle.
    """

    # --- Preprocess image ---
    I = cv2.GaussianBlur(M_bin.astype(np.float32), (0, 0), smooth_sigma)
    I = I / np.max(I) if np.max(I) > 0 else I

    # --- Define angle range ---
    Classes = np.arange(-90, 90, theta_res)

    if not local:
        # --- Global Radon transform ---
        sinogram = radon(I, theta=Classes, circle=False)
        # Sum intensity along projection axis
        projection_intensity = np.sum(sinogram, axis=0)
        distri = projection_intensity / np.max(projection_intensity)

        # Dominant orientation
        dominant_angle = Classes[np.argmax(distri)]

        # Fill entire DM with dominant angle
        DM = np.ones_like(I) * dominant_angle

    else:
        # --- Local (patch-based) Radon ---
        h, w = I.shape
        DM = np.zeros_like(I, dtype=np.float32)
        distri = np.zeros_like(Classes, dtype=np.float32)

        for i in range(0, h, patch_size):
            for j in range(0, w, patch_size):
                patch = I[i : i + patch_size, j : j + patch_size]
                if np.count_nonzero(patch) < 10:
                    continue
                sinogram = radon(patch, theta=Classes, circle=False)
                projection_intensity = np.sum(sinogram, axis=0)
                projection_intensity /= np.max(projection_intensity)
                angle_local = Classes[np.argmax(projection_intensity)]
                DM[i : i + patch_size, j : j + patch_size] = angle_local
                distri += projection_intensity

        distri /= np.max(distri)

    return DM, Classes, distri
