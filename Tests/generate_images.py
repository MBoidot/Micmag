import numpy as np
from scipy.spatial import Voronoi
from matplotlib.path import Path
from PIL import Image
import os


def generate_colonies(width, height, n_colonies, angle_deg):
    """
    Generate parallel colony bands.
    """
    theta = np.deg2rad(angle_deg)
    direction = np.array([np.cos(theta), np.sin(theta)])

    x, y = np.meshgrid(np.arange(width), np.arange(height))
    coords = np.stack((x, y), axis=-1)

    projection = coords @ direction
    bins = np.linspace(projection.min(), projection.max(), n_colonies + 1)

    colony_map = np.digitize(projection, bins) - 1
    colony_map = np.clip(colony_map, 0, n_colonies - 1)

    return colony_map


def anisotropic_transform(points, angle_deg, aspect_ratio):
    """
    Stretch space along growth direction.
    """
    theta = np.deg2rad(angle_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    S = np.array([[aspect_ratio, 0], [0, 1]])

    return points @ (R @ S @ R.T)


def generate_directional_grains(
    width,
    height,
    colony_map,
    seeds_per_colony=40,
    base_angle=90,
    angle_jitter=5,
    aspect_ratio=3.0,
):
    labels = np.zeros((height, width), dtype=np.int32)
    grain_id = 1

    for colony in np.unique(colony_map):
        mask = colony_map == colony
        ys, xs = np.where(mask)

        if len(xs) < seeds_per_colony:
            continue

        idx = np.random.choice(len(xs), seeds_per_colony, replace=False)
        points = np.column_stack((xs[idx], ys[idx]))

        angle = base_angle + np.random.uniform(-angle_jitter, angle_jitter)

        warped_points = anisotropic_transform(points, angle, aspect_ratio)
        vor = Voronoi(warped_points)

        coords = np.column_stack(np.where(mask))
        coords_xy = np.column_stack((coords[:, 1], coords[:, 0]))
        warped_coords = anisotropic_transform(coords_xy, angle, aspect_ratio)

        for i, region_idx in enumerate(vor.point_region):
            region = vor.regions[region_idx]
            if -1 in region or len(region) == 0:
                continue

            polygon = vor.vertices[region]
            path = Path(polygon)
            inside = path.contains_points(warped_coords)

            pix = coords[inside]
            labels[pix[:, 0], pix[:, 1]] = grain_id
            grain_id += 1

    return labels


def detect_grain_boundaries(labels):
    gb = np.zeros_like(labels, dtype=bool)
    gb[:-1, :] |= labels[:-1, :] != labels[1:, :]
    gb[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    return gb


def render_microstructure(labels, gb_mask, grain_grey=120, nd_white=255):
    img = np.full(labels.shape, grain_grey, dtype=np.uint8)
    img[gb_mask] = nd_white
    return img


def generate_dendritic_colonies(
    width=512,
    height=512,
    n_colonies=6,
    base_angle=90,
    seeds_per_colony=40,
    angle_jitter=5,
    aspect_ratio=3.0,
):
    colony_map = generate_colonies(width, height, n_colonies, base_angle)

    labels = generate_directional_grains(
        width,
        height,
        colony_map,
        seeds_per_colony=seeds_per_colony,
        base_angle=base_angle,
        angle_jitter=angle_jitter,
        aspect_ratio=aspect_ratio,
    )

    gb = detect_grain_boundaries(labels)
    img = render_microstructure(labels, gb)

    return img, labels, gb, colony_map


if __name__ == "__main__":

    img, labels, gb, colonies = generate_dendritic_colonies(
        width=512,
        height=512,
        n_colonies=6,
        base_angle=0,
        seeds_per_colony=45,
        angle_jitter=3,
        aspect_ratio=4.0,
    )

    Image.fromarray(img).save("debug_dendritic_colonies.png")
    print("Baseline dendritic colonies generated.")
Image.fromarray((colonies / colonies.max() * 255).astype(np.uint8)).save("colonies.png")
Image.fromarray((labels > 0).astype(np.uint8) * 255).save("grains.png")
Image.fromarray(gb.astype(np.uint8) * 255).save("gb.png")
