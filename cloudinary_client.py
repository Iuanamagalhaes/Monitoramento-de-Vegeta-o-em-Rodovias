import os
import requests
import numpy as np
import cv2

import cloudinary
import cloudinary.api


cloudinary.config()


def list_images(folder="rodovias", max_results=500):

    resources = cloudinary.api.resources(
        type="upload",
        prefix=folder,
        max_results=max_results
    )

    return resources["resources"]


def download_image_rgb(url: str):

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    image = np.frombuffer(response.content, np.uint8)

    img = cv2.imdecode(image, cv2.IMREAD_COLOR)

    if img is None:
        raise Exception(f"Erro lendo imagem {url}")

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)