"""
visualize.py
-------------
Funcoes de apoio para inspecao visual: overlay da mascara sobre a imagem
original, e grafico de evolucao da cobertura de vegetacao ao longo do tempo.
"""

from __future__ import annotations
import numpy as np
import cv2
import datetime as dt


def overlay_mask(image_rgb: np.ndarray, mask: np.ndarray, color=(255, 0, 0),
                  alpha: float = 0.45) -> np.ndarray:
    """Desenha a mascara de vegetacao sobre a imagem original (para conferencia
    visual / relatorios). Retorna imagem RGB uint8."""
    overlay = image_rgb.copy()
    color_layer = np.zeros_like(image_rgb)
    color_layer[..., 0], color_layer[..., 1], color_layer[..., 2] = color
    overlay[mask] = (
        (1 - alpha) * image_rgb[mask] + alpha * color_layer[mask]
    ).astype(np.uint8)
    return overlay


def save_overlay(image_rgb: np.ndarray, mask: np.ndarray, out_path: str) -> None:
    overlay = overlay_mask(image_rgb, mask)
    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def plot_growth(history, out_path: str, camera_id: str = "") -> None:
    """history: lista de tuplas (timestamp_iso, coverage_pct, area_px),
    como retornado por temporal_tracker.get_history()."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    timestamps = [dt.datetime.fromisoformat(h[0]) for h in history]
    coverage = [h[1] for h in history]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(timestamps, coverage, marker="o", color="#2e7d32")
    ax.set_title(f"Evolucao da cobertura de vegetacao — {camera_id}")
    ax.set_xlabel("Data")
    ax.set_ylabel("Cobertura (%)")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)