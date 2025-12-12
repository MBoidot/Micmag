# -*- coding: utf-8 -*-
"""
binarisation_kmeans.py
----------------------
Local adaptive binarization using K-means clustering (K=2).
"""

import numpy as np
from sklearn.cluster import KMeans
from scipy.ndimage import binary_opening, binary_closing


def binarisation_kmeans(M, window_size=15, step=5, morph_clean=True):
    """
    Binarize image using local K-means clustering.

    Parameters
    ----------
    M : ndarray
        2D grayscale image (0-1)
    window_size : int
        Size of local square window for clustering
    step : int
        Subsampling step for speed
    morph_clean : bool
        Apply morphological opening/closing

    Returns
    -------
    N : ndarray
        Binary image (1=matrix, 0=Nd-rich)
    """
    hauteur, largeur = M.shape
    N = np.ones((hauteur, largeur), dtype=np.uint8)

    half = window_size // 2

    for x in range(0, hauteur, step):
        for y in range(0, largeur, step):
            x_min = max(0, x - half)
            x_max = min(hauteur, x + half + 1)
            y_min = max(0, y - half)
            y_max = min(largeur, y + half + 1)

            window = M[x_min:x_max, y_min:y_max].reshape(-1, 1)

            if len(window) < 2:
                continue

            # K-means clustering
            kmeans = KMeans(n_clusters=2, n_init=5, random_state=42)
            labels = kmeans.fit_predict(window)
            centers = kmeans.cluster_centers_.flatten()

            # Determine which label is Nd-rich (darker intensity)
            nd_label = np.argmin(centers)

            # Assign binary values to the window
            labels_2d = labels.reshape(x_max - x_min, y_max - y_min)
            N[x_min:x_max, y_min:y_max] = np.where(labels_2d == nd_label, 0, 1)

"""     if morph_clean:
        N = binary_opening(N, structure=np.ones((3, 3))).astype(np.uint8)
        N = binary_closing(N, structure=np.ones((3, 3))).astype(np.uint8) """

    return N
