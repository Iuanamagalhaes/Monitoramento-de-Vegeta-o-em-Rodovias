"""
segmentation.py
----------------
Modulo responsavel por gerar a mascara pixel-a-pixel de vegetacao em uma imagem.

Duas estrategias sao oferecidas:

1. `segment_classical`: metodo baseado em indices de vegetacao (Excess Green -
   Excess Red) + limiarizacao de Otsu + limpeza morfologica. NAO precisa de
   treino nem de dataset rotulado. Funciona bem como baseline imediato e como
   gerador automatico de "pseudo-rotulos" para acelerar a anotacao futura.

2. `segment_dl`: hook para um modelo de deep learning (ex: YOLOv8-seg,
   DeepLabV3+) treinado com fotos reais das rodovias. Quando voce tiver dados
   anotados, basta treinar o modelo e plugar aqui — o resto do pipeline
   (temporal_tracker.py, pipeline.py) nao muda.

A funcao `segment()` unifica as duas, escolhendo o metodo via parametro.
"""

from __future__ import annotations
import numpy as np
import cv2


def _excess_green_minus_red(image_rgb: np.ndarray) -> np.ndarray:
    """Calcula o indice ExG - ExR, classico em agricultura de precisao para
    realcar vegetacao verde independente de iluminacao.

    image_rgb: array HxWx3, uint8, canais em ordem RGB.
    Retorna: array HxW float, quanto maior, mais "vegetacao" o pixel parece.
    """
    img = image_rgb.astype(np.float32)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    total = r + g + b + 1e-6
    r_n, g_n, b_n = r / total, g / total, b / total

    exg = 2 * g_n - r_n - b_n
    exr = 1.4 * r_n - g_n
    return exg - exr


def segment_classical(image_rgb: np.ndarray, min_blob_area: int = 150) -> np.ndarray:
    """Segmentacao classica de vegetacao sem necessidade de treino.

    Passos:
      1. Calcula indice ExG-ExR (realca verde)
      2. Limiarizacao automatica (Otsu) sobre o indice
      3. Remove ruido (abertura morfologica) e buracos pequenos (fechamento)
      4. Remove blobs muito pequenos (provavel ruido, nao vegetacao real)

    Retorna mascara booleana HxW (True = vegetacao).
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb deve ser um array HxWx3 (RGB)")

    idx = _excess_green_minus_red(image_rgb)

    # Normaliza para 0-255 para usar Otsu do OpenCV
    idx_norm = cv2.normalize(idx, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(idx_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Limpeza morfologica
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Remove blobs pequenos
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean_mask = np.zeros_like(mask)
    for label_id in range(1, num_labels):
        if stats[label_id, cv2.CC_STAT_AREA] >= min_blob_area:
            clean_mask[labels == label_id] = 255

    return clean_mask.astype(bool)


def segment_dl(image_rgb: np.ndarray, model) -> np.ndarray:
    """Hook para segmentacao via deep learning (ex: YOLOv8-seg da Ultralytics).

    `model` deve ser um objeto ja carregado (ex: `YOLO('best.pt')`).
    Esta funcao assume uma unica classe "vegetacao" e retorna a uniao de
    todas as mascaras preditas.

    Exemplo de uso quando o modelo estiver pronto:
        from ultralytics import YOLO
        model = YOLO('runs/segment/train/weights/best.pt')
        mask = segment_dl(image_rgb, model)
    """
    results = model.predict(image_rgb, verbose=False)
    h, w = image_rgb.shape[:2]
    mask = np.zeros((h, w), dtype=bool)

    for r in results:
        if r.masks is None:
            continue
        for m in r.masks.data.cpu().numpy():
            m_resized = cv2.resize(m, (w, h))
            mask |= (m_resized > 0.5)

    return mask


def segment(image_rgb: np.ndarray, method: str = "classical", model=None,
            min_blob_area: int = 150) -> np.ndarray:
    """Interface unica de segmentacao.

    method: "classical" (padrao, sem treino) ou "dl" (requer `model` treinado).
    """
    if method == "classical":
        return segment_classical(image_rgb, min_blob_area=min_blob_area)
    elif method == "dl":
        if model is None:
            raise ValueError("Metodo 'dl' requer um modelo carregado (parametro model=)")
        return segment_dl(image_rgb, model)
    else:
        raise ValueError(f"Metodo desconhecido: {method}")


def coverage_percent(mask: np.ndarray) -> float:
    """Percentual de pixels classificados como vegetacao (0-100)."""
    return 100.0 * mask.sum() / mask.size