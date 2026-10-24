import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data") #option called --data which folder the YOLO dataset is in
    parser.add_argument("--split", default="train")#option which split to check train, val or test
    parser.add_argument("--n", type=int, default=4)#how many images to draw, default is 4
    parser.add_argument("--out", default="label_check")#which folder to save the checked images info
    args = parser.parse_args()

    data, out = Path(args.data), Path(args.out) #folder paths input and output folders
    out.mkdir(exist_ok=True)#mkdir creates folder
    images = sorted((data / "images" / args.split).glob("*.png"))#folder where images are ready to be trained
    random.seed(0)#we get same pics again, not differents if we want to fix something

    for img_path in random.sample(images, min(args.n, len(images))):
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        draw = ImageDraw.Draw(img)
        label_path = data / "labels" / args.split / f"{img_path.stem}.txt"

        for line in label_path.read_text().strip().splitlines():
            _, xc, yc, bw, bh = map(float, line.split())

            x_min, y_min = (xc - bw / 2) * w, (yc - bh / 2) * h
            x_max, y_max = (xc + bw / 2) * w, (yc + bh / 2) * h
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)

        img.save(out / img_path.name)
        print(f"Saved {out / img_path.name}")


if __name__ == "__main__":
    main()
