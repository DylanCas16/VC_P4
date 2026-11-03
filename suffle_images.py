import random
import shutil
from pathlib import Path

BASE = Path(__file__).parent
IMAGES_DIR = BASE / "images"
LABELS_DIR = BASE / "labels"
OUT_DIR = BASE / "dataset"

TRAIN, VAL, TEST = 0.80, 0.10, 0.10
SEED = 42
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def main():
    labels = {p.stem: p for p in LABELS_DIR.glob("*.json")}
    images = sorted(p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in IMG_EXTS)

    pairs = [(img, labels[img.stem]) for img in images if img.stem in labels]
    for img in images:
        if img.stem not in labels:
            print(f"Sin label, se omite: {img.name}")

    random.Random(SEED).shuffle(pairs)

    n = len(pairs)
    n_train = round(n * TRAIN)
    n_val = round(n * VAL)
    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:],
    }

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    for name, items in splits.items():
        img_out = OUT_DIR / name / "images"
        lbl_out = OUT_DIR / name / "labels"
        img_out.mkdir(parents=True)
        lbl_out.mkdir(parents=True)
        for img, lbl in items:
            shutil.copy2(img, img_out / img.name)
            shutil.copy2(lbl, lbl_out / lbl.name)
        print(f"{name}: {len(items)} imágenes")


if __name__ == "__main__":
    main()
