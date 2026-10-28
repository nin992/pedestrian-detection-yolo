"""
evaluate.py
1. Scores the final model ONCE on the held-out test split (mAP, precision, recall).
2. Error analysis: for every test image, matches predicted boxes to the real boxes
   and counts correct detections (TP), false alarms (FP) and misses (FN).
   Images with mistakes are saved with colored boxes:
       green  = real person that was found      (TP, ground truth box)
       red    = real person that was MISSED     (FN)
       orange = box with no matching real person (FP)
3. Checks whether the missed people are the "small / heavily occluded" pedestrians
   listed in PennFudanPed/added-object-list.txt.

Usage:
    python evaluate.py --weights runs/detect/runs/pretrained_n/weights/best.pt
"""


import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw
from ultralytics import YOLO


def load_ground_truth(label_path, img_w, img_h):

    boxes = []
    for line in label_path.read_text().strip().splitlines():
        _, xc, yc, bw, bh = map(float, line.split())
        boxes.append(((xc - bw / 2) * img_w, (yc - bh / 2) * img_h,
                      (xc + bw / 2) * img_w, (yc + bh / 2) * img_h))
    return boxes


def iou(a, b):

    inter_w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = inter_w * inter_h
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match(preds, gts, iou_threshold):

    order = sorted(range(len(preds)), key=lambda i: preds[i][1], reverse=True)
    matched_gt, false_pos = set(), []
    for i in order:
        box, _ = preds[i]
        best_j, best_iou = -1, 0.0
        for j, gt in enumerate(gts):
            if j in matched_gt:
                continue
            v = iou(box, gt)
            if v > best_iou:
                best_j, best_iou = j, v
        if best_iou >= iou_threshold:
            matched_gt.add(best_j)
        else:
            false_pos.append(i)
    missed = [j for j in range(len(gts)) if j not in matched_gt]
    return matched_gt, false_pos, missed


def load_added_list(path):

    added = set()
    if not path.exists():
        return added
    for line in path.read_text().strip().splitlines()[1:]:      # skip header line
        parts = line.split()
        if len(parts) == 2:
            added.add((parts[0], int(parts[1])))
    return added


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="runs/detect/runs/pretrained_n/weights/best.pt")
    parser.add_argument("--data", default="data/data.yaml")
    parser.add_argument("--images", default="data/images/test")
    parser.add_argument("--labels", default="data/labels/test")
    parser.add_argument("--added", default="PennFudanPed/added-object-list.txt")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold for error analysis")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU needed to count a box as correct")
    parser.add_argument("--out", default="error_analysis")
    args = parser.parse_args()

    model = YOLO(args.weights)
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    # ---- 1. Official test-set metrics (computed by Ultralytics) ----
    metrics = model.val(data=args.data, split="test")
    test_scores = {
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50-95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
    }
    print("\nTEST SET RESULTS:", test_scores)


    added = load_added_list(Path(args.added))
    tp = fp = fn = 0
    missed_people = []
    gt_heights = []

    for img_path in sorted(Path(args.images).glob("*.png")):
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        gts = load_ground_truth(Path(args.labels) / f"{img_path.stem}.txt", w, h)
        gt_heights += [g[3] - g[1] for g in gts]

        r = model.predict(str(img_path), conf=args.conf, verbose=False)[0]
        preds = list(zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()))

        matched, false_pos, missed = match(preds, gts, args.iou)
        tp += len(matched)
        fp += len(false_pos)
        fn += len(missed)

        for j in missed:

            person_number = j + 1
            missed_people.append({
                "image": img_path.name,
                "person": person_number,
                "height_px": round(gts[j][3] - gts[j][1]),
                "in_added_list": (img_path.name, person_number) in added,
            })

        if false_pos or missed:
            draw = ImageDraw.Draw(img)
            for j, gt in enumerate(gts):
                color = "red" if j in missed else "lime"
                draw.rectangle(gt, outline=color, width=4)
            for i in false_pos:
                box, conf = preds[i]
                draw.rectangle(box, outline="orange", width=4)
                draw.text((box[0] + 4, box[1] + 4), f"{conf:.2f}", fill="orange")
            img.save(out / img_path.name)


    summary = {
        "test_scores": test_scores,
        "conf_threshold": args.conf,
        "iou_threshold": args.iou,
        "true_positives": tp,
        "false_positives": fp,
        "missed_people": fn,
        "total_real_people": tp + fn,
        "missed_details": missed_people,
        "missed_in_added_list": sum(m["in_added_list"] for m in missed_people),
        "median_person_height_px": sorted(gt_heights)[len(gt_heights) // 2] if gt_heights else None,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nAt conf >= {args.conf}, IoU >= {args.iou}:")
    print(f"  Real people: {tp + fn} | found: {tp} | missed: {fn} | false alarms: {fp}")
    print(f"  Missed people that are in added-object-list (small/occluded): "
          f"{summary['missed_in_added_list']} of {fn}")
    for m in missed_people:
        print(f"    missed: {m['image']} person {m['person']}, height {m['height_px']} px, "
              f"in added list: {m['in_added_list']}")
    print(f"  Median real person height: {summary['median_person_height_px']:.0f} px")
    print(f"Images with mistakes saved in {out}/ ; summary in {out / 'summary.json'}")


if __name__ == "__main__":
    main()
