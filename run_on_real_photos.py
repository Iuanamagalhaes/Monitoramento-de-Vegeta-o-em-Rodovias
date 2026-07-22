"""
run_on_real_photos.py
-----------------------
Roda o pipeline (metodo classico, sem treino) em fotos REAIS, organizadas
em pastas por camera. Estrutura de pastas esperada:

    fotos/
      km_42_sentido_norte/
        2026-06-01.jpg
        2026-06-11.jpg
        2026-06-21.jpg
      km_58_sentido_sul/
        2026-06-01.jpg
        ...

O nome da subpasta vira o `camera_id`. A data e extraida do nome do arquivo
se possivel (formato YYYY-MM-DD); se nao conseguir, usa a data de
modificacao do arquivo.

Uso:
    python run_on_real_photos.py --input fotos --db vegetacao.db --output output
"""

import argparse
import os
import re
import datetime as dt
from pathlib import Path

import pipeline
import temporal_tracker as tt
import visualize

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DATE_PATTERN = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def extract_timestamp(file_path: Path) -> dt.datetime:
    match = DATE_PATTERN.search(file_path.stem)
    if match:
        y, m, d = map(int, match.groups())
        return dt.datetime(y, m, d)
    # Fallback: data de modificacao do arquivo
    return dt.datetime.fromtimestamp(file_path.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(description="Roda o pipeline de vegetacao em fotos reais.")
    parser.add_argument("--input", required=True, help="Pasta raiz com subpastas por camera")
    parser.add_argument("--db", default="vegetacao.db", help="Caminho do banco SQLite")
    parser.add_argument("--output", default="output", help="Pasta para overlays e graficos")
    parser.add_argument("--alert-rate", type=float, default=0.3,
                         help="Limiar de alerta: pp de cobertura por dia")
    parser.add_argument("--alert-abs", type=float, default=15.0,
                         help="Limiar de alerta: cobertura absoluta (%%)")
    args = parser.parse_args()

    input_root = Path(args.input)
    if not input_root.is_dir():
        raise SystemExit(f"Pasta de entrada nao encontrada: {input_root}")

    os.makedirs(args.output, exist_ok=True)

    camera_dirs = [d for d in sorted(input_root.iterdir()) if d.is_dir()]
    if not camera_dirs:
        raise SystemExit(
            f"Nenhuma subpasta encontrada em {input_root}. "
            "Organize as fotos em subpastas por camera (ver docstring do script)."
        )

    for camera_dir in camera_dirs:
        camera_id = camera_dir.name
        photos = sorted(
            [p for p in camera_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS],
            key=extract_timestamp,
        )
        if not photos:
            print(f"[{camera_id}] nenhuma foto encontrada, pulando.")
            continue

        print(f"\n=== Camera: {camera_id} ({len(photos)} fotos) ===")
        for photo_path in photos:
            timestamp = extract_timestamp(photo_path)
            report = pipeline.process_image(
                image_path=str(photo_path),
                camera_id=camera_id,
                db_path=args.db,
                output_dir=args.output,
                method="classical",
                timestamp=timestamp,
                alert_rate_pct_per_day=args.alert_rate,
                alert_absolute_pct=args.alert_abs,
            )
            g = report["growth"]
            line = f"  [{timestamp.date()}] {photo_path.name}: cobertura = {report['coverage_pct']:.2f}%"
            if g.rate_pct_per_day is not None:
                line += f" (taxa = {g.rate_pct_per_day:.3f} pp/dia)"
            print(line)
            if report["alert"]:
                print(f"    !! ALERTA: {g.alert_reason}")

        # Grafico de evolucao para essa camera
        history = tt.get_history(args.db, camera_id)
        if len(history) >= 2:
            chart_path = os.path.join(args.output, f"{camera_id}_evolucao.png")
            visualize.plot_growth(history, chart_path, camera_id=camera_id)
            print(f"  Grafico salvo em: {chart_path}")

    print(f"\nConcluido. Overlays e graficos em: {args.output}")
    print(f"Historico completo em: {args.db}")


if __name__ == "__main__":
    main()