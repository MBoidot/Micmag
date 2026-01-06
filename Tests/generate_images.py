import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


def generate_microstructure(width, height, grain_size, noise_level):
    """
    Génère une microstructure de matériau en cours de solidification.

    :param width: Largeur de l'image
    :param height: Hauteur de l'image
    :param grain_size: Taille moyenne des grains
    :param noise_level: Niveau de bruit ajouté à l'image
    :return: Image en niveau de gris de la microstructure
    """
    # Création d'une grille de points de départ pour les grains
    x = np.arange(0, width, grain_size)
    y = np.arange(0, height, grain_size)
    xx, yy = np.meshgrid(x, y)

    # Initialisation de l'image avec des valeurs aléatoires
    image = np.random.rand(height, width) * 255

    # Création des grains
    for i in range(len(xx)):
        for j in range(len(yy)):
            # Centre du grain
            cx, cy = xx[i, j], yy[i, j]

            # Taille aléatoire du grain
            size = grain_size * (0.5 + np.random.rand())

            # Création d'un cercle autour du centre du grain
            for k in range(height):
                for l in range(width):
                    distance = np.sqrt((k - cy) ** 2 + (l - cx) ** 2)
                    if distance < size:
                        # Valeur du pixel en fonction de la distance au centre
                        value = 255 - distance * (255 / size)
                        image[k, l] = min(image[k, l], value)

    # Ajout de bruit
    image = image + noise_level * np.random.randn(height, width)
    image = np.clip(image, 0, 255)

    return image.astype(np.uint8)


def save_images(
    num_images, output_dir, width=512, height=512, grain_size=50, noise_level=10
):
    """
    Génère et sauvegarde un certain nombre d'images de microstructures.

    :param num_images: Nombre d'images à générer
    :param output_dir: Répertoire de sortie pour les images
    :param width: Largeur de l'image
    :param height: Hauteur de l'image
    :param grain_size: Taille moyenne des grains
    :param noise_level: Niveau de bruit ajouté à l'image
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(num_images):
        microstructure = generate_microstructure(width, height, grain_size, noise_level)
        image = Image.fromarray(microstructure)
        image.save(os.path.join(output_dir, f"microstructure_{i}.png"))


# Exemple d'utilisation
num_images = 10
output_dir = "microstructures"
width, height = 512, 512
grain_size = 50
noise_level = 10

save_images(num_images, output_dir, width, height, grain_size, noise_level)
