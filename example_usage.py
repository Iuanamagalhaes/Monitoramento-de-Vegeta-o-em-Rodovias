"""
example_usage.py
------------------
Demonstra o pipeline completo usando imagens SINTETICAS (geradas por
codigo), simulando fotos da mesma camera fixa em 4 dias diferentes, com uma
mancha verde (vegetacao) crescendo progressivamente sobre o acostamento.

Isso serve para validar que a arquitetura funciona de ponta a ponta mesmo
antes de voce ter fotos reais das rodovias. Quando tiver fotos reais, basta
trocar `gerar_imagem_sintetica()` por `pipeline.load_image_rgb(caminho_real)`.
"""

import os
import datetime as dt
import numpy as np
import cv2

import pipeline
import visualize
import temporal_tracker as tt


def gerar_imagem_sintetica(dia: int, out_path: str, tamanho=(480, 640)) -> str:
    """Gera uma imagem simulando uma cena de rodovia (ceu, asfalto cinza) com
    uma mancha verde de vegetacao no acostamento, crescendo a cada 'dia'."""
    h, w = tamanho
    img = np.zeros((h, w, 3), dtype=np.uint8)

    img[:] = (150, 170, 190)  # tom acinzentado de fundo (ceu/asfalto)
    img[int(h*0.55):, :] = (90, 90, 95)  # asfalto

    # Mancha de vegetacao crescendo com o tempo (simula avanco sobre a pista)
    base_radius = 25 + dia * 12
    center = (int(w * 0.75), int(h * 0.62))
    rng = np.random.default_rng(dia)
    overlay = img.copy()
    for _ in range(30):
        offset_x = int(rng.normal(0, base_radius * 0.5))
        offset_y = int(rng.normal(0, base_radius * 0.3))
        r = int(rng.uniform(base_radius * 0.4, base_radius))
        green = (int(rng.uniform(60, 100)), int(rng.uniform(120, 170)), int(rng.uniform(40, 70)))
        cv2.circle(overlay, (center[0] + offset_x, center[1] + offset_y), r, green, -1)

    img = cv2.addWeighted(overlay, 0.9, img, 0.1, 0)

    cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return out_path


def main():
    work_dir = "demo_run"
    os.makedirs(work_dir, exist_ok=True)
    db_path = os.path.join(work_dir, "vegetacao.db")
    output_dir = os.path.join(work_dir, "output")
    camera_id = "km_42_sentido_norte"

    base_date = dt.datetime(2026, 6, 1)

    print("=== Simulando 4 capturas ao longo do tempo (a cada 10 dias) ===\n")
    for i, dia in enumerate([0, 1, 2, 3]):
        img_path = os.path.join(work_dir, f"foto_dia_{dia}.png")
        gerar_imagem_sintetica(dia, img_path)
        timestamp = base_date + dt.timedelta(days=10 * i)

        report = pipeline.process_image(
            image_path=img_path,
            camera_id=camera_id,
            db_path=db_path,
            output_dir=output_dir,
            method="classical",
            timestamp=timestamp,
            alert_rate_pct_per_day=0.05,
            alert_absolute_pct=8.0,
        )

        g = report["growth"]
        print(f"[{timestamp.date()}] cobertura = {report['coverage_pct']:.2f}%  "
              f"| overlay salvo em: {report['overlay_path']}")
        if g.previous_coverage_pct is not None:
            print(f"    delta = {g.delta_pct:+.2f} pp em {g.days_between:.1f} dias "
                  f"-> taxa = {g.rate_pct_per_day:.3f} pp/dia")
        if report["alert"]:
            print(f"    !! ALERTA: {g.alert_reason}")
        print()

    history = tt.get_history(db_path, camera_id)
    chart_path = os.path.join(output_dir, "evolucao_cobertura.png")
    visualize.plot_growth(history, chart_path, camera_id=camera_id)
    print(f"Grafico de evolucao salvo em: {chart_path}")


if __name__ == "__main__":
    main()