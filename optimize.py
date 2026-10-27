import argparse
import csv
from pathlib import Path

from ultralytics import YOLO


def size_mb(path):

    p = Path(path)
    if p.is_dir():
        total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    else:
        total = p.stat().st_size
    return round(total / 1e6, 2)


def evaluate(model_path, data, imgsz, device):

    model = YOLO(str(model_path), task="detect")
    m = model.val(data=data, split="val", imgsz=imgsz, device=device, batch=1, verbose=False)
    return {
        "mAP50": round(float(m.box.map50), 4),
        "mAP50-95": round(float(m.box.map), 4),
        "inference_ms": round(float(m.speed["inference"]), 2),   # per image
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="runs/detect/runs/pretrained_n/weights/best.pt")
    parser.add_argument("--data", default="data/data.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu", help="cpu = realistic deployment speed")
    args = parser.parse_args()

    weights = Path(args.weights)
    base = YOLO(str(weights))


    onnx_path = base.export(format="onnx", imgsz=args.imgsz)
    ov_fp32_path = base.export(format="openvino", imgsz=args.imgsz)

    ov_int8_path = base.export(format="openvino", imgsz=args.imgsz, int8=True, data=args.data)

    versions = [
        ("PyTorch FP32 (baseline)", weights),
        ("ONNX FP32", onnx_path),
        ("OpenVINO FP32", ov_fp32_path),
        ("OpenVINO INT8", ov_int8_path),
    ]


    rows = []
    for name, path in versions:
        result = evaluate(path, args.data, args.imgsz, args.device)
        row = {"version": name, "size_MB": size_mb(path), **result, "device": args.device,
               "path": str(path)}
        rows.append(row)
        print(row)


    with open("optimization.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    base_row = rows[0]
    print(f"\n{'version':26} {'size MB':>8} {'mAP50':>7} {'mAP50-95':>9} {'ms/img':>8} {'speedup':>8}")
    for r in rows:
        speedup = base_row["inference_ms"] / r["inference_ms"] if r["inference_ms"] else 0
        print(f"{r['version']:26} {r['size_MB']:>8} {r['mAP50']:>7} {r['mAP50-95']:>9} "
              f"{r['inference_ms']:>8} {speedup:>7.2f}x")
    print("\nSaved optimization.csv")


if __name__ == "__main__":
    main()
