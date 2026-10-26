import argparse
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def mask_to_boxes(mask):

    boxes = []
    ids = np.unique(mask)
    ids = ids[ids != 0]
    for obj_id in ids:
        ys, xs = np.where(mask == obj_id)
        boxes.append((xs.min(), ys.min(), xs.max(), ys.max()))

    return boxes


def box_to_yolo(box, img_w, img_h):

    x_min, y_min, x_max, y_max = box
    x_center = (x_min + x_max) / 2 / img_w
    y_center = (y_min + y_max) / 2 / img_h
    width = (x_max - x_min) / img_w
    height = (y_max - y_min) / img_h
    return x_center, y_center, width, height


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="PennFudanPed", help="unzipped Penn-Fudan folder")
    parser.add_argument("--out", default="data", help="output folder in YOLO format")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src, out = Path(args.src), Path(args.out)
    images = sorted((src / "PNGImages").glob("*.png"))
    print(f"Found {len(images)} images")


    random.seed(args.seed)
    random.shuffle(images)
    n = len(images)
    n_train, n_val = int(0.70 * n), int(0.15 * n)
    splits = {
        "train": images[:n_train], #118
        "val": images[n_train:n_train + n_val], #25
        "test": images[n_train + n_val:], #27
    }

    total_people = 0
    for split, files in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

        for img_path in files:
            mask_path = src / "PedMasks" / f"{img_path.stem}_mask.png"
            img_w, img_h = Image.open(img_path).size
            mask = np.array(Image.open(mask_path))

            boxes = mask_to_boxes(mask)
            total_people += len(boxes)

            lines = []
            for box in boxes:
                xc, yc, w, h = box_to_yolo(box, img_w, img_h)
                lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

            shutil.copy(img_path, out / "images" / split / img_path.name)
            (out / "labels" / split / f"{img_path.stem}.txt").write_text("\n".join(lines) + "\n")

        print(f"{split}: {len(files)} images")


    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: person\n"
    )
    print(f"Total people (boxes): {total_people}")
    print(f"Wrote {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
