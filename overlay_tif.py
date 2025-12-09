# ...existing code...
import os
from pathlib import Path
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog
from IPython.display import display
from datetime import datetime


def overlay_blend_arr(base, overlay):
    """Darken blend: per-channel minimum of base and overlay.
    Both inputs are float arrays in [0,1]. Supports shape (...,3) or (...)."""
    base = np.clip(base, 0.0, 1.0)
    overlay = np.clip(overlay, 0.0, 1.0)
    return np.minimum(base, overlay)


def open_file_dialog():
    root = tk.Tk()
    root.withdraw()
    # keep dialog on top
    try:
        root.wm_attributes("-topmost", 1)
    except Exception:
        pass
    paths = filedialog.askopenfilenames(
        title="Select TIFF files to overlay (order matters)",
        filetypes=[("TIFF files", ("*.tif", "*.tiff")), ("All files", "*.*")],
    )
    root.destroy()
    return list(paths)


def load_image_as_rgb(path, size=None):
    img = Image.open(path).convert("RGB")
    if size is not None and img.size != size:
        img = img.resize(size, Image.BILINEAR)
    return img


def overlay_files(paths, save_folder=None):
    if not paths:
        print("No files selected.")
        return None

    base_img = load_image_as_rgb(paths[0])
    base_arr = np.asarray(base_img, dtype=np.float32) / 255.0

    for p in paths[1:]:
        overlay_img = load_image_as_rgb(p, size=base_img.size)
        overlay_arr = np.asarray(overlay_img, dtype=np.float32) / 255.0
        base_arr = overlay_blend_arr(base_arr, overlay_arr)

    out_arr = (np.clip(base_arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    out_img = Image.fromarray(out_arr, mode="RGB")

    if save_folder is None:
        save_folder = Path(paths[0]).parent
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = save_folder / f"overlay_result_{ts}.tif"
    out_img.save(out_name, format="TIFF")
    return out_img, out_name


def main():
    paths = open_file_dialog()
    if not paths:
        return
    print("Files chosen (in order):")
    for p in paths:
        print(" -", p)
    result = overlay_files(paths)
    if result is None:
        return
    out_img, out_path = result
    print("Saved result to:", out_path)
    # display in interactive window
    display(out_img)


if __name__ == "__main__":
    main()
