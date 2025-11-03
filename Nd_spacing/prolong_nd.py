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


"""
def orientation_based_prolong(M_bin, M_angle, line_length=10):

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
"""


def watershed_grains_pro(
    M_bin,
    hole_area=50,
    pre_dilate_steps=2,
    pre_erode_steps=1,
    selem_radius=1,
    marker_distance=5,
    min_grain_size=50,
    edge_smooth=1,
    return_preprocessed=False,
):
    """
    Watershed-based grain reconstruction for Nd-rich alloys.

    Parameters
    ----------
    M_bin : ndarray
        Binary image of Nd-rich phase (1=Nd-rich, 0=matrix)
    hole_area : int
        Maximum area of holes to fill in the Nd-rich phase
    pre_dilate_steps : int
        Number of dilation steps applied to the Nd-rich mask before inversion
    pre_erode_steps : int
        Number of erosion steps applied to the Nd-rich mask after dilation
    selem_radius : int
        Radius of the disk structuring element for dilation/erosion
    marker_distance : int
        Minimum distance between watershed markers
    min_grain_size : int
        Minimum size of grains to keep
    edge_smooth : int
        Radius of disk to smooth edges of grains
    return_preprocessed : bool
        If True, returns the dilated/eroded mask before inversion

    Returns
    -------
    grains_mask : ndarray
        Binary mask of grains (1=grain, 0=matrix)
    labels : ndarray
        Labeled grains
    dend_mask_preprocessed : ndarray (optional)
        Dilated/eroded Nd-rich mask (if return_preprocessed=True)
    """

    # --- 1️⃣ Work on Nd-rich mask (dendrite mask)
    dend_mask = M_bin > 0
    selem = morphology.disk(selem_radius)

    # Dilation
    for _ in range(pre_dilate_steps):
        dend_mask = morphology.binary_dilation(dend_mask, footprint=selem)

    # Erosion
    for _ in range(pre_erode_steps):
        dend_mask = morphology.binary_erosion(dend_mask, footprint=selem)

    dend_mask_preprocessed = dend_mask.copy()  # save for plotting if needed

    # Fill small holes in dendrite mask
    dend_mask = morphology.remove_small_holes(dend_mask, area_threshold=hole_area)

    # --- 2️⃣ Invert to get grain mask
    grain_mask_initial = np.logical_not(dend_mask)

    # --- 3️⃣ Distance transform on grain mask
    dist = ndi.distance_transform_edt(grain_mask_initial)

    # --- 4️⃣ Local maxima for watershed markers
    coords = feature.peak_local_max(
        dist, min_distance=marker_distance, labels=grain_mask_initial
    )
    markers = np.zeros_like(dist, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i

    # --- 5️⃣ Watershed
    labels = segmentation.watershed(-dist, markers, mask=grain_mask_initial)

    # --- 6️⃣ Remove small grains
    labels = morphology.remove_small_objects(labels, min_size=min_grain_size)

    # --- 7️⃣ Optional: smooth edges
    if edge_smooth > 0:
        boundaries = segmentation.find_boundaries(labels, mode="outer")
        se = morphology.disk(edge_smooth)
        boundaries = morphology.binary_dilation(boundaries, footprint=se)
        grains_mask = np.logical_or(boundaries, grain_mask_initial)
    else:
        grains_mask = grain_mask_initial

    if return_preprocessed:
        return grains_mask.astype(np.uint8), labels, dend_mask_preprocessed
    else:
        return grains_mask.astype(np.uint8), labels
