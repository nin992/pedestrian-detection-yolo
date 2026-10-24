
import argparse
import csv
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/data.yaml", help="YOLO dataset config")
    parser.add_argument("--model", default="yolov8n.pt",
                        help="yolov8n.pt = pretrained, yolov8n.yaml = from scratch")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640, help="training image size")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", default="pretrained_n", help="name of this experiment")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ---- 1. Train ----
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        project="runs",
        name=args.name,
        exist_ok=True,
    )


    save_dir = Path(getattr(model.trainer, "save_dir", Path("runs") / args.name))
    best_path = save_dir / "weights" / "best.pt"
    best = YOLO(str(best_path))
    metrics = best.val(data=args.data, split="val", imgsz=args.imgsz)

    row = {
        "experiment": args.name,
        "model": args.model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "mAP50": round(metrics.box.map50, 4),
        "mAP50-95": round(metrics.box.map, 4),
        "precision": round(metrics.box.mp, 4),
        "recall": round(metrics.box.mr, 4),
        "weights": str(best_path),
    }
    print("\nValidation results:", row)

    # ---- 3. Save the results in one table for all experiments ----
    csv_path = Path("experiments.csv")
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    print(f"Added results to {csv_path}")


if __name__ == "__main__":
    main()
