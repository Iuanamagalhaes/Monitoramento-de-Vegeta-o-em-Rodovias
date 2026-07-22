"""
pipeline.py
------------
Orquestra o fluxo completo para uma imagem recebida de uma camera:

  imagem -> segmentacao -> metrica de cobertura -> grava no historico
         -> compara com medicao anterior -> gera relatorio (com alerta se
            o crescimento ultrapassar o limiar)

Este modulo e agnostico de hardware/transporte: voce chama `process_image()"
tanto a partir de um script que le uma pasta local (edge, embarcado) quanto a
partir de uma API que recebe upload de imagens (servidor central). O que
muda entre esses cenarios fica fora deste arquivo.
"""

from __future__ import annotations
import os
import datetime as dt
import numpy as np
import cv2

import segmentation
import temporal_tracker as tt
import visualize


def load_image_rgb(path: str) -> np.ndarray:
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise FileNotFoundError(f"Nao foi possivel ler a imagem: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def process_image(image_path: str, camera_id: str, db_path: str,
                   output_dir: str = "output", method: str = "classical",
                   model=None, timestamp: dt.datetime = None,
                   alert_rate_pct_per_day: float = 0.5,
                   alert_absolute_pct: float = 30.0) -> dict:
    """Processa uma unica imagem e retorna um relatorio (dict) com:
       coverage_pct, area_px, overlay_path, growth (GrowthReport), alert (bool)
    """
    os.makedirs(output_dir, exist_ok=True)
    tt.init_db(db_path)

    image_rgb = load_image_rgb(image_path)
    mask = segmentation.segment(image_rgb, method=method, model=model)
    coverage_pct = segmentation.coverage_percent(mask)
    area_px = int(mask.sum())

    timestamp = timestamp or dt.datetime.utcnow()
    ts_tag = timestamp.strftime("%Y%m%dT%H%M%S")
    overlay_path = os.path.join(output_dir, f"{camera_id}_{ts_tag}_overlay.png")
    visualize.save_overlay(image_rgb, mask, overlay_path)

    tt.record_measurement(
        db_path=db_path,
        camera_id=camera_id,
        coverage_pct=coverage_pct,
        area_px=area_px,
        mask_path=overlay_path,
        image_path=image_path,
        timestamp=timestamp,
    )

    growth = tt.compute_growth(
        db_path, camera_id,
        alert_rate_pct_per_day=alert_rate_pct_per_day,
        alert_absolute_pct=alert_absolute_pct,
    )

    return {
        "camera_id": camera_id,
        "timestamp": timestamp.isoformat(),
        "coverage_pct": coverage_pct,
        "area_px": area_px,
        "overlay_path": overlay_path,
        "growth": growth,
        "alert": growth.alert,
    }