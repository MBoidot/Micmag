# -*- coding: utf-8 -*-
"""
distance.py
------------
Python translation of MATLAB's distance.m

Computes local maximum disk diameters centered on Nd-rich pixels,
based on angular exploration in a binarized image.
"""

import numpy as np
from PARAMETRES import (
    dmax,
    nbangles,
    fl,
    lex,
    echelle,
    bords,
    nbamoy,
    largeur_moyennage,
)


def distance(N, plagex=None, plagey=None):
    """
    Compute the local spacing (maximal inscribed disk diameter) for Nd-rich regions.

    Parameters
    ----------
    N : ndarray
        Binary image (1 = matrix phase, 0 = Nd-rich phase).
    plagex, plagey : optional
        Pixel ranges to process (for test speed).

    Returns
    -------
    D : ndarray
        Final spacing matrix (µm).
    DM : ndarray
        Smoothed spacing matrix (averaged locally).
    """

    hauteur, largeur = N.shape[:2]

    # Initialize matrices
    MM = -(dmax / 2) * np.ones((hauteur, largeur))
    D = -(dmax / 2) * np.ones((hauteur, largeur))
    DM = -(dmax / 2) * np.ones((hauteur, largeur))

    # Precompute angles
    an = np.arange(1, nbangles + 1)
    Angles = -90 + an * 180 / nbangles
    llex = int(2 * fl * lex + 1)

    if plagex is None:
        plagex = np.arange(hauteur)
    if plagey is None:
        plagey = np.arange(largeur)

    # ----------------------------------------------------------------------
    # First pass: compute the largest disk centered at each Nd-rich pixel
    # ----------------------------------------------------------------------
    for x in plagex:
        for y in plagey:
            if N[x, y] == 0:
                stop = 0
                d = np.full(nbangles, dmax, dtype=float)

                angles = (Angles + 90) * np.pi / 180 + np.pi * np.random.rand()
                for ang_idx, angle in enumerate(angles):
                    v = -np.ones(llex)
                    for i in range(llex):
                        ec = i - fl * lex - 1
                        ecx = int(round(np.cos(angle) * ec / fl))
                        ecy = int(round(np.sin(angle) * ec / fl))
                        xx = x + ecx
                        yy = y + ecy
                        if 0 <= xx < hauteur and 0 <= yy < largeur:
                            v[i] = N[xx, yy]

                    vtest = v < 1  # 1 inside Nd-rich or outside
                    vtest2 = 2 * (v >= 0) - 1  # -1 outside image, +1 inside

                    # find right boundary
                    i1, rd = 0, 1
                    while i1 < fl * lex and rd == 1:
                        rd = min(rd, min(vtest[fl * lex + i1], vtest2[fl * lex + i1]))
                        i1 += 1

                    # find left boundary
                    i2, rg = 0, 1
                    while i2 < fl * lex and rg == 1:
                        rg = min(rg, min(vtest[fl * lex - i2], vtest2[fl * lex - i2]))
                        i2 += 1

                    if rd >= 0 and rg >= 0:
                        d[ang_idx] = min(i1 - 1, i2 - 2) / echelle / fl
                    else:
                        stop += 1  # exclude edge cases

                if stop < bords * nbangles:
                    ds = np.sort(d)
                    MM[x, y] = np.mean(ds[:nbamoy])

    # ----------------------------------------------------------------------
    # Second pass: refine using MM to get D (final spacing map)
    # ----------------------------------------------------------------------
    for x in plagex:
        for y in plagey:
            if N[x, y] == 0:
                d_val = -dmax / 2
                angles = (Angles + 90) * np.pi / 180 + np.pi * np.random.rand()
                for angle in angles:
                    pasbloqued = 1
                    pasbloqueg = 1
                    for ec in range(1, lex + 1):
                        ecx = int(round(np.cos(angle) * ec))
                        ecy = int(round(np.sin(angle) * ec))

                        # Forward direction
                        xx = x + ecx
                        yy = y + ecy
                        if 0 <= xx < hauteur and 0 <= yy < largeur:
                            if pasbloqued:
                                if N[xx, yy] == 1:
                                    pasbloqued = 0
                                elif ec / echelle < MM[xx, yy]:
                                    d_val = max(d_val, 2 * MM[xx, yy])

                        # Backward direction
                        xx = x - ecx
                        yy = y - ecy
                        if 0 <= xx < hauteur and 0 <= yy < largeur:
                            if pasbloqueg:
                                if N[xx, yy] == 1:
                                    pasbloqueg = 0
                                elif MM[xx, yy] > 0 and ec / echelle < MM[xx, yy]:
                                    d_val = max(d_val, 2 * MM[xx, yy])

                D[x, y] = d_val

    # ----------------------------------------------------------------------
    # Third pass: local averaging for smoothing (DM)
    # ----------------------------------------------------------------------
    lrg = largeur_moyennage
    for x in plagex:
        for y in plagey:
            if D[x, y] > 0:
                x_min = max(0, x - lrg)
                x_max = min(hauteur, x + lrg)
                y_min = max(0, y - lrg)
                y_max = min(largeur, y + lrg)

                Dloc = D[x_min:x_max, y_min:y_max]
                vals = Dloc[Dloc > 0]
                if vals.size > 0:
                    DM[x, y] = np.mean(vals)

    return D, DM
