import argparse
import numpy as np
from PIL import Image
import math
import os


def create_striped_tif(angle=45, pixel_size=5, output_file="stripes.tif"):
    """
    Create a TIFF image with black and white stripes.

    Args:
        angle: Stripe angle in degrees (default: 45)
        pixel_size: Stripe width in pixels (default: 5)
        output_file: Output TIFF filename (default: stripes.tif)
    """
    # Image dimensions
    width, height = 644, 484

    # Create coordinate grids
    x = np.arange(width)
    y = np.arange(height)
    xx, yy = np.meshgrid(x, y)

    # Convert angle to radians
    angle_rad = math.radians(angle)

    # Rotate coordinates
    rotated = xx * math.cos(angle_rad) + yy * math.sin(angle_rad)

    # Create stripes based on pixel_size
    stripes = np.floor(rotated / pixel_size) % 2

    # Convert to 8-bit grayscale (0-255)
    img_array = (stripes * 255).astype(np.uint8)

    # Create and save image in current directory
    img = Image.fromarray(img_array, mode="L")
    img.save(output_file, format="TIFF")
    print(f"Created {output_file} with angle={angle}°, stripe width={pixel_size}px")
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create striped TIFF image")
    parser.add_argument(
        "--angle", type=float, default=60, help="Stripe angle in degrees (default: 45)"
    )
    parser.add_argument(
        "--pixel-size", type=int, default=2, help="Stripe width in pixels (default: 5)"
    )
    parser.add_argument(
        "--output",
        default="stripes60-2.tif",
        help="Output TIFF filename (default: stripes.tif)",
    )

    args = parser.parse_args()
    create_striped_tif(args.angle, args.pixel_size, args.output)

# Interactive Jupyter/IPython execution
if __name__ != "__main__":
    create_striped_tif()
