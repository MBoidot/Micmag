# -*- coding: utf-8 -*-
"""
distance.py
------------
Python translation of MATLAB's distance.m, optimized with Numba.

Computes local maximum disk diameters centered on Nd-rich pixels,
based on angular exploration in a binarized image.
"""

import numpy as np
from numba import njit
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


# ----------------------------- Numba functions -----------------------------
@njit
def first_pass_numba(
    N, MM, Angles, llex, fl, lex, echelle, dmax, bords, nbamoy, nbangles
):
    xs, ys = np.where(N == 0)
    n_pixels = xs.size

    for idx in range(n_pixels):
        x = xs[idx]
        y = ys[idx]

        stop = 0
        d = np.full(nbangles, dmax, dtype=np.float64)

        angles = (Angles + 90) * np.pi / 180 + np.pi * np.random.rand()
        for ang_idx in range(nbangles):
            angle = angles[ang_idx]
            v = -np.ones(llex)
            for i in range(llex):
                ec = i - fl * lex - 1
                ecx = int(round(np.cos(angle) * ec / fl))
                ecy = int(round(np.sin(angle) * ec / fl))
                xx = x + ecx
                yy = y + ecy
                if 0 <= xx < N.shape[0] and 0 <= yy < N.shape[1]:
                    v[i] = N[xx, yy]

            vtest = v < 1
            vtest2 = 2 * (v >= 0) - 1

            # Right boundary
            i1, rd = 0, 1
            while i1 < fl * lex and rd == 1:
                rd = min(rd, min(vtest[fl * lex + i1], vtest2[fl * lex + i1]))
                i1 += 1

            # Left boundary
            i2, rg = 0, 1
            while i2 < fl * lex and rg == 1:
                rg = min(rg, min(vtest[fl * lex - i2], vtest2[fl * lex - i2]))
                i2 += 1

            if rd >= 0 and rg >= 0:
                d[ang_idx] = min(i1 - 1, i2 - 2) / echelle / fl
            else:
                stop += 1

        if stop < bords * nbangles:
            ds = np.sort(d)
            MM[x, y] = np.mean(ds[:nbamoy])


@njit
def second_pass_numba(N, MM, D, Angles, lex, echelle, dmax):
    xs, ys = np.where(N == 0)
    n_pixels = xs.size

    for idx in range(n_pixels):
        x = xs[idx]
        y = ys[idx]
        d_val = -dmax / 2

        angles = (Angles + 90) * np.pi / 180 + np.pi * np.random.rand()
        for angle in angles:
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            pasbloqued = 1
            pasbloqueg = 1
            for ec in range(1, lex + 1):
                ecx = int(round(cos_angle * ec))
                ecy = int(round(sin_angle * ec))

                # Forward
                xx, yy = x + ecx, y + ecy
                if 0 <= xx < N.shape[0] and 0 <= yy < N.shape[1]:
                    if pasbloqued:
                        if N[xx, yy] == 1:
                            pasbloqued = 0
                        elif ec / echelle < MM[xx, yy]:
                            d_val = max(d_val, 2 * MM[xx, yy])

                # Backward
                xx, yy = x - ecx, y - ecy
                if 0 <= xx < N.shape[0] and 0 <= yy < N.shape[1]:
                    if pasbloqueg:
                        if N[xx, yy] == 1:
                            pasbloqueg = 0
                        elif MM[xx, yy] > 0 and ec / echelle < MM[xx, yy]:
                            d_val = max(d_val, 2 * MM[xx, yy])

        D[x, y] = d_val


@njit
def third_pass_numba(D, DM, largeur_moyennage):
    lrg = largeur_moyennage
    hauteur, largeur = D.shape
    for x in range(hauteur):
        for y in range(largeur):
            if D[x, y] > 0:
                x_min = max(0, x - lrg)
                x_max = min(hauteur, x + lrg + 1)
                y_min = max(0, y - lrg)
                y_max = min(largeur, y + lrg + 1)
                Dloc = D[x_min:x_max, y_min:y_max]
                s = 0.0
                count = 0
                for i in range(Dloc.shape[0]):
                    for j in range(Dloc.shape[1]):
                        if Dloc[i, j] > 0:
                            s += Dloc[i, j]
                            count += 1
                if count > 0:
                    DM[x, y] = s / count


# ----------------------------- Main distance function -----------------------------
def distance(N, plagex=None, plagey=None):
    hauteur, largeur = N.shape[:2]
    MM = -(dmax / 2) * np.ones((hauteur, largeur))
    D = -(dmax / 2) * np.ones((hauteur, largeur))
    DM = -(dmax / 2) * np.ones((hauteur, largeur))

    Angles = -90 + np.arange(1, nbangles + 1) * 180 / nbangles
    llex = int(2 * fl * lex + 1)

    if plagex is None:
        plagex = np.arange(hauteur)
    if plagey is None:
        plagey = np.arange(largeur)

    print("⚡ First pass: computing maximal disks MM")
    first_pass_numba(
        N, MM, Angles, llex, fl, lex, echelle, dmax, bords, nbamoy, nbangles
    )
    print("✅ First pass complete.")

    print("⚡ Second pass: computing final spacing map D")
    second_pass_numba(N, MM, D, Angles, lex, echelle, dmax)
    print("✅ Second pass complete.")

    print("⚡ Third pass: smoothing to get DM")
    third_pass_numba(D, DM, largeur_moyennage)
    print("✅ Third pass complete.")

    return D, DM
