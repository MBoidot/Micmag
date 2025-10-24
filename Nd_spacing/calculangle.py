import numpy as np
import cv2


def calculangle(
    M_bin, smooth_sigma=2, tensor_blur=3, coherence_threshold=0.3, eps=1e-12
):
    """
    Compute local orientation (DM) from binary Nd-rich mask using structure tensor.

    Parameters
    ----------
    M_bin : ndarray
        Binary image (1 = Nd-rich (white), 0 = magnetic phase (black)).
    smooth_sigma : float
        Gaussian blur before gradient computation.
    tensor_blur : float
        Gaussian blur for tensor components.
    coherence_threshold : float
        Minimum coherence for valid orientation (masking criterion).
    eps : float
        Small constant to prevent division by zero.

    Returns
    -------
    DM_masked : ndarray
        Orientation map (-90° to +90°) masked by coherence.
    Classes : ndarray
        Angle class centers (for histogram bins).
    distri : ndarray
        Normalized distribution of orientation angles.
    """

    # --- Convert to float32 for OpenCV ---
    gray = M_bin.astype(np.float32)

    # --- Pre-smoothing ---
    gray_blur = cv2.GaussianBlur(gray, (0, 0), smooth_sigma)

    # --- Compute gradients ---
    Ix = cv2.Sobel(gray_blur, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray_blur, cv2.CV_32F, 0, 1, ksize=3)

    # --- Structure tensor components ---
    Jxx = cv2.GaussianBlur(Ix * Ix, (0, 0), tensor_blur)
    Jyy = cv2.GaussianBlur(Iy * Iy, (0, 0), tensor_blur)
    Jxy = cv2.GaussianBlur(Ix * Iy, (0, 0), tensor_blur)

    # --- Orientation ---
    DM = 0.5 * np.degrees(np.arctan2(2 * Jxy, Jxx - Jyy + eps))

    # --- Coherence ---
    sqrt_term = np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy**2)
    coherence = sqrt_term / (Jxx + Jyy + eps)
    coherence = np.clip(coherence, 0, 1)

    # --- Mask low-coherence regions ---
    DM_masked = np.ma.masked_where(coherence < coherence_threshold, DM)

    # --- Compute histogram of valid orientations ---
    valid_angles = DM_masked.compressed()
    Classes = np.linspace(-90, 90, 181)  # 1° bins
    distri, _ = np.histogram(valid_angles, bins=Classes, density=True)

    # Center Classes for plotting consistency
    Classes = (Classes[:-1] + Classes[1:]) / 2

    return DM_masked, Classes, distri
