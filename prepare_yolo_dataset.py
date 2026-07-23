from __future__ import annotations
import os
import shutil
import cv2
import numpy as np
from pathlib import Path


def mask_to_yolo_polygons(mask: np.ndarray, class_id: int = 0,
                           min_contour_area: int = 50) -> list[str]:
    h, w = mask.shape[:2]
    mask_u8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_contour_area:
            continue
        cnt = cnt.reshape(-1, 2).astype(np.float32)
        cnt[:, 0] /= w
        cnt[:, 1] /= h
        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in cnt)
        lines.append(f"{class_id} {coords}")

    return lines


def build_dataset(image_mask_pairs: list[tuple[str, np.ndarray]],
                   output_dir: str, val_fraction: float = 0.2,
                   class_names: list[str] = None) -> str:
    class_names = class_names or ["vegetacao"]
    output_dir = Path(output_dir)

    n_val = max(1, int(len(image_mask_pairs) * val_fraction))
    val_set = set(range(len(image_mask_pairs) - n_val, len(image_mask_pairs)))

    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for i, (img_path, mask) in enumerate(image_mask_pairs):
        split = "val" if i in val_set else "train"
        img_name = Path(img_path).name
        stem = Path(img_path).stem

        shutil.copy(img_path, output_dir / "images" / split / img_name)

        lines = mask_to_yolo_polygons(mask)
        label_path = output_dir / "labels" / split / f"{stem}.txt"
        label_path.write_text("\n".join(lines))

    data_yaml_path = output_dir / "data.yaml"
    data_yaml_path.write_text(
        f"path: {output_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n" + "\n".join(f"  {i}: {name}" for i, name in enumerate(class_names)) + "\n"
    )

    return str(data_yaml_path)