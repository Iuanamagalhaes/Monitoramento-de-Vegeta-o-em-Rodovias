import os
import re

import cloudinary
import cloudinary.api

import pipeline

cloudinary.config()


cloudinary.config(
    cloud_name="tde2cyks",
    api_key="579279342685698",
    api_secret="pzM1Va6u-VkW_K-3eU_dDPTTUjI"
)

DB_PATH = "vegetacao.db"
OUTPUT_DIR = "output"
CAMERA_ID = "rodovia"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def image_number(resource):
    m = re.match(r"(\d+)", resource["public_id"])

    if m:
        return int(m.group(1))

    return 999999


def main():

    resources = cloudinary.api.resources(
        type="upload",
        max_results=500
    )["resources"]

    # mantém apenas imagens que começam com número
    resources = [
        r for r in resources
        if re.match(r"^\d+", r["public_id"])
    ]

    # remove duplicatas, mantendo a mais recente
    latest = {}

    for r in resources:
        numero = image_number(r)
        latest[numero] = r

    resources = list(latest.values())
    resources.sort(key=image_number)

    print(f"{len(resources)} imagens encontradas.\n")

    for r in resources:

        print("=" * 60)
        print(r["public_id"])

        report = pipeline.process_image(
            image_path=r["secure_url"],
            camera_id=CAMERA_ID,
            db_path=DB_PATH,
            output_dir=OUTPUT_DIR,
            method="classical"
        )

        print(f"Cobertura: {report['coverage_pct']:.2f}%")

    print("\nFinalizado.")


if __name__ == "__main__":
    main()