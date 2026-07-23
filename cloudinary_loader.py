import requests
import numpy as np
import cv2


def load_image_from_url(url: str):
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    image = np.frombuffer(response.content, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    if image is None:
        raise Exception("Não foi possível decodificar a imagem")

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)