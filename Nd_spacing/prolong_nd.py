# prolong_nd.py
import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage import morphology, segmentation, feature, measure


def simple_closing(N, disk_radius=1):
    """
    Fast isotropic closing to join near domains.
    N: binary (0/1) numpy array (Nd-rich = 1 or adapt to your convention)
    returns: closed binary image
    """
    selem = morphology.disk(disk_radius)
    closed = morphology.binary_closing(N.astype(bool), selem)
    return closed.astype(np.uint8)


def multi_angle_closing(N, line_length=15, angles=None):
    """
    Close gaps by performing morphological closing with line SE at multiple angles
    and taking the union of results.
    Works on all skimage versions (no morphology.line dependency).

    Parameters
    ----------
    N : ndarray
        Binary (0/1) image.
    line_length : int
        Length of the line kernel in pixels.
    angles : list or ndarray
        List of angles in degrees (0 = horizontal). Default every 15°.

    Returns
    -------
    closed_union : ndarray
        Binary image (0/1) after union of directional closings.
    """
    if angles is None:
        angles = np.arange(0, 180, 15)

    N_bool = N.astype(bool)
    result = np.zeros_like(N_bool)

    # Create base horizontal line kernel
    base_kernel = np.zeros((line_length, line_length), dtype=np.uint8)
    base_kernel[line_length // 2, :] = 1

    for ang in angles:
        # Rotate the kernel
        M = cv2.getRotationMatrix2D((line_length / 2, line_length / 2), ang, 1.0)
        rotated = cv2.warpAffine(
            base_kernel.astype(np.uint8) * 255,
            M,
            (line_length, line_length),
            flags=cv2.INTER_NEAREST,
            borderValue=0,
        )
        kernel = (rotated > 128).astype(np.uint8)

        # Morphological closing with the rotated line
        closed = cv2.morphologyEx(N_bool.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        result |= closed.astype(bool)

    return result.astype(np.uint8)


def watershed_and_edge_prolong(M_bin, min_distance=10, min_size=50, edge_width=2):
    """
    Segment the white magnetic phase and add white contours to close gaps.
    """
    # Ensure binary: 1 = white (magnetic phase)
    fg = M_bin > 0

    # --- 1️⃣ Distance transform inside white regions
    dist = ndi.distance_transform_edt(fg)

    # --- 2️⃣ Local maxima (potential dendrite centers)
    coords = feature.peak_local_max(dist, min_distance=min_distance, labels=fg)
    markers = np.zeros_like(dist, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i

    # --- 3️⃣ Watershed segmentation (labels each dendrite)
    labels = segmentation.watershed(-dist, markers, mask=fg)

    # --- 4️⃣ Optional: remove very small regions
    labels = morphology.remove_small_objects(labels, min_size=min_size)

    # --- 5️⃣ Create edges/contours between regions
    boundaries = segmentation.find_boundaries(labels, mode="outer")

    # --- 6️⃣ Dilate edges to thicken them
    if edge_width > 0:
        se = morphology.disk(edge_width)
        boundaries = morphology.binary_dilation(boundaries, footprint=se)

    # --- 7️⃣ Combine: add white edges to original magnetic regions
    prolonged = np.logical_or(fg, boundaries)

    return prolonged.astype(np.uint8), labels


def orientation_based_prolong(M_bin, M_angle, line_length=10):
    """
    Prolong white pixels along their local orientation (from angular map).

    Parameters
    ----------
    M_bin : 2D ndarray (0/1)
        Binary image (Nd-rich phase = 1).
    M_angle : 2D ndarray (float)
        Orientation map in degrees (0–180).
    line_length : int
        Length of each prolongation line (pixels).

    Returns
    -------
    prolonged : 2D ndarray (0/1)
        Binary image after directional prolongation.
    """
    from skimage.draw import line

    M_bin = M_bin.astype(bool)
    prolonged = M_bin.copy()
    rows, cols = np.nonzero(M_bin)

    nrows, ncols = M_bin.shape
    L = line_length // 2

    for y, x in zip(rows, cols):
        theta = np.deg2rad(M_angle[y, x])
        dy = int(np.round(np.sin(theta) * L))
        dx = int(np.round(np.cos(theta) * L))

        # Start and end points, clipped safely inside bounds
        y0 = int(np.clip(y - dy, 0, nrows - 1))
        x0 = int(np.clip(x - dx, 0, ncols - 1))
        y1 = int(np.clip(y + dy, 0, nrows - 1))
        x1 = int(np.clip(x + dx, 0, ncols - 1))

        rr, cc = line(y0, x0, y1, x1)

        # Clip again in case of rounding errors from line()
        rr = rr[(rr >= 0) & (rr < nrows)]
        cc = cc[(cc >= 0) & (cc < ncols)]

        prolonged[rr, cc] = True

    return prolonged.astype(np.uint8)
