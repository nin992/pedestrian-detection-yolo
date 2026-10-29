# Pedestrian Detection with YOLOv8 

An end-to-end computer vision project: a model that **finds pedestrians in images** (bounding boxes + people count), built by fine-tuning YOLOv8 on the Penn-Fudan Pedestrian dataset. It covers data preparation, training experiments, evaluation with error analysis, model optimization (ONNX / OpenVINO / INT8), and a FastAPI demo.

## Results at a glance

| | Value |
|---|---|
| Final model | YOLOv8n, fine-tuned from COCO-pretrained weights |
| **Test set** (27 unseen images, 67 people) | **mAP50 0.949**, mAP50-95 0.782, precision 0.883, recall 0.896 |
| Error analysis | All 9 misses are very small (19–67 px tall) or heavily occluded people; all 6 "false positives" are real people missing from the labels |
| Optimization (CPU) | OpenVINO INT8: **1.46× faster** and **3.4× smaller** than FP32, with almost no accuracy loss |
| API speed (M1 MacBook Air, CPU) | ~190 ms per image after warm-up |

### Experiments (validation set, 25 images)

| Experiment | Epochs | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|---|
| YOLOv8n pretrained | 50 | 0.982 | 0.842 | 0.971 | 0.967 |
| YOLOv8n from scratch | 50 | 0.696 | 0.356 | 0.580 | 0.672 |
| YOLOv8n from scratch | 150 | 0.924 | 0.604 | 0.897 | 0.870 |
| YOLOv8s pretrained | 50 | 0.988 | 0.842 | 0.972 | 0.984 |

- **Pretraining matters:** with only 118 training images, the pretrained model clearly beats training from scratch, even when scratch gets 3× more epochs.
- **YOLOv8n was chosen over YOLOv8s:** same accuracy, but YOLOv8s is 3.5× larger (22.5 MB vs 6.5 MB) and needs 3.3× more computation.

### Optimization (validation set, Colab CPU, batch 1)

| Version | Size | mAP50 | mAP50-95 | ms / image | Speedup |
|---|---|---|---|---|---|
| PyTorch `.pt` (stored as FP16) | 6.51 MB | 0.9801 | 0.8397 | 123 | 1.00× |
| ONNX FP32 | 12.77 MB | 0.9797 | 0.8319 | 124 | 0.99× |
| OpenVINO FP32 | 12.86 MB | 0.9797 | 0.8319 | 108 | 1.14× |
| **OpenVINO INT8** | **3.73 MB** | **0.9794** | **0.8294** | **84** | **1.46×** |

Full details, reasoning and example images are in [REPORT.md](REPORT.md).

## Project structure

```
pedestrian-detection-yolo/
├── prepare_data.py     # Penn-Fudan masks -> YOLO labels, 70/15/15 split (seed 42), data.yaml
├── check_labels.py     # draws the saved labels on images to verify the conversion
├── train.py            # trains YOLOv8, validates best.pt, appends a row to experiments.csv
├── evaluate.py         # final test-set score + error analysis (misses / false positives)
├── optimize.py         # exports ONNX, OpenVINO FP32/INT8 and compares size, speed, accuracy
├── app.py              # FastAPI demo: /detect (JSON) and /detect/image (annotated image)
├── experiments.csv     # results of all training experiments
├── optimization.csv    # results of the optimization comparison
├── requirements.txt
├── REPORT.md           # technical report
└── AI_USAGE.md         # how AI tools were used and how their output was verified
```

## Setup

Python 3.10+ recommended.

```bash
git clone https://github.com/nin992/pedestrian-detection-yolo.git
cd pedestrian-detection-yolo
pip install -r requirements.txt
```

Training was done on a free Google Colab T4 GPU (about 2 minutes per 50-epoch run). Data preparation, evaluation and the API also run on a normal laptop CPU.

## Reproduce the results

Run the scripts in this order.

**1. Get the dataset.** The Penn-Fudan dataset is not included in this repo (its license does not allow reposting). Download it from the official site:

```bash
curl -L -O https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip
unzip -q PennFudanPed.zip
```

**2. Convert to YOLO format and split** (170 images → 118 train / 25 val / 27 test, 423 pedestrians):

```bash
python3 prepare_data.py
```

**3. (Optional) Check the labels visually:**

```bash
python3 check_labels.py --split train --n 4      # images saved to label_check/
```

**4. Train** (each run adds a row to `experiments.csv`):

```bash
python3 train.py --model yolov8n.pt   --name pretrained_n                 # final model
python3 train.py --model yolov8n.yaml --name scratch_n                    # from scratch
python3 train.py --model yolov8n.yaml --epochs 150 --name scratch_n_150
python3 train.py --model yolov8s.pt   --name pretrained_s                 # bigger model
```

Weights are saved to `runs/detect/runs/<name>/weights/best.pt`.

**5. Evaluate on the test set + error analysis:**

```bash
python3 evaluate.py --weights runs/detect/runs/pretrained_n/weights/best.pt
```

Images with mistakes are saved to `error_analysis/` (green = found, red = missed, orange = false positive).

**6. Optimize and compare formats:**

```bash
python3 optimize.py --weights runs/detect/runs/pretrained_n/weights/best.pt
```

## Trained model

## Trained model

Download the trained model (`best.pt`, 6.5 MB) from the [Releases page](https://github.com/nin992/pedestrian-detection-yolo/releases), or from the terminal:

```bash
mkdir -p models
curl -L -o models/best.pt https://github.com/nin992/pedestrian-detection-yolo/releases/download/v1.0/best.pt
```

With the downloaded model you can skip training (step 4) and run steps 5 and 6 with `--weights models/best.pt`:

```bash
python3 evaluate.py --weights models/best.pt
```

Or retrain it with step 4 above.

```
models/best.pt
```

Or retrain it with step 4 above.

## Run the API

```bash
uvicorn app:app --reload
```

Open http://127.0.0.1:8000/docs to try it in the browser.

| Endpoint | Input | Output |
|---|---|---|
| `GET /` | — | status and model path |
| `POST /detect` | image file, optional `conf` (default 0.25) | JSON: `people_count`, boxes, confidences, inference time |
| `POST /detect/image` | image file, optional `conf` | the image with boxes drawn (people count in the `X-People-Count` header) |

Example with curl:

```bash
curl -X POST "http://127.0.0.1:8000/detect?conf=0.25" -F "file=@street.jpg"
```

To serve the INT8 OpenVINO model instead (fastest on Intel CPUs), first create it with step 6; `optimize.py` saves it next to the weights file (e.g. `models/best_int8_openvino_model/` when run with `--weights models/best.pt`):

```bash
MODEL_PATH=models/best_int8_openvino_model uvicorn app:app
```

Invalid uploads (non-image files) return a clear `400` error.

## Limitations

- Trained only on upright, full-body street pedestrians: it struggles with **seated, close-up or partly visible people** (domain shift).
- **Very small or heavily occluded** pedestrians are the main source of misses.
- The validation and test sets are small (25 and 27 images), so scores can vary by a few percent.

## Acknowledgements

- Dataset: [Penn-Fudan Pedestrian Database](https://www.cis.upenn.edu/~jshi/ped_html/), L. Wang, J. Shi, G. Song, I. Shen, ACCV 2007.
- Model: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics).
- AI tools were used during this project; see [AI_USAGE.md](AI_USAGE.md).
