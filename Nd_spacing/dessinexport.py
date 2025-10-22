# -*- coding: utf-8 -*-
"""
dessinexport.py
---------------
Python translation of MATLAB's dessinexport.m

Generates visual outputs and exports data to XLS for each analyzed image.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def dessinexport(N, DM, Classes, distri, chePH, cheRES, photo):
    """
    Save visualizations and export data results for one analyzed image.

    Parameters
    ----------
    N : ndarray
        Binary image (0/1).
    DM : ndarray
        Processed image or result matrix (e.g. distance or angle map).
    Classes : ndarray
        X-axis for distribution (depth or size bins).
    distri : ndarray
        Distribution data corresponding to Classes.
    chePH : str
        Path prefix for image files (without extension).
    cheRES : str
        Full path for the XLS result file.
    photo : str
        Photo identifier (for title and filenames).
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(chePH), exist_ok=True)

    # --- Figure 1: binarized image ---
    plt.figure(figsize=(6, 5), facecolor="w")
    plt.imshow(N, cmap="gray", interpolation="nearest")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{chePH}-binaire.TIF", dpi=120)
    plt.close()

    # --- Figure 2: processed (DM) image ---
    plt.figure(figsize=(6, 5), facecolor="w")
    plt.imshow(DM, cmap="viridis", interpolation="nearest")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{chePH}-traite.TIF", dpi=120)
    plt.close()

    # --- Figure 3: distribution curve ---
    plt.figure(figsize=(6, 5), facecolor="w")
    plt.plot(Classes, distri, linewidth=1.5)
    plt.title(photo)
    plt.xlabel("Classes")
    plt.ylabel("Distribution")
    plt.tight_layout()
    plt.savefig(f"{chePH}-data.TIF", dpi=120)
    plt.close()

    # --- Excel export ---
    df = pd.DataFrame({"Classes": Classes.flatten(), "distri": distri.flatten()})
    df.to_excel(cheRES, index=False)

    print(f"✅ Exported: {photo}")
    print(f"   ├─ {chePH}-binaire.TIF")
    print(f"   ├─ {chePH}-traite.TIF")
    print(f"   ├─ {chePH}-data.TIF")
    print(f"   └─ {cheRES}")
