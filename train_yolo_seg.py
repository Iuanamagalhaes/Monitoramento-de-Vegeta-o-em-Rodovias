"""
train_yolo_seg.py
-------------------
Treina um modelo YOLOv8-seg para deteccao pixel-a-pixel de vegetacao, a
partir de um dataset ja organizado no formato YOLO (ver prepare_yolo_dataset.py
ou export direto do Roboflow no formato "YOLOv8").

Uso:
    python train_yolo_seg.py --data dataset/data.yaml --epochs 100 --model n

Depois do treino, o modelo treinado fica em:
    runs/segment/train/weights/best.pt

Para usar no pipeline principal:
    from ultralytics import YOLO
    import segmentation

    model = YOLO("runs/segment/train/weights/best.pt")
    mask = segmentation.segment(image_rgb, method="dl", model=model)
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Treina YOLOv8-seg para vegetacao.")
    parser.add_argument("--data", required=True, help="Caminho para data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--model", default="n", choices=["n", "s", "m", "l", "x"],
                         help="Tamanho do YOLOv8-seg: n(ano) e o mais leve, "
                              "indicado para embarcado (Jetson/RPi); m/l para servidor")
    parser.add_argument("--device", default=None,
                         help='"cpu", "0" (primeira GPU), etc. Default: auto-detecta')
    args = parser.parse_args()

    weights = f"yolov8{args.model}-seg.pt"
    model = YOLO(weights)  # baixa pesos pre-treinados no COCO automaticamente

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name="train",
    )

    metrics = model.val()
    print("\n=== Metricas de validacao ===")
    print(metrics)
    print("\nModelo treinado salvo em: runs/segment/train/weights/best.pt")


if __name__ == "__main__":
    main()