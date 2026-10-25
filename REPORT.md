# Technical Report: Pedestrian Detection with YOLOv8

## 1. What problem does the project solve?
- Detect pedestrians in images (boxes + count)
- Use case ideas: people counting, safety zones, ...

## 2. Why this data, model and approach?
- Penn-Fudan: 170 images, 423 pedestrians, masks → converted to YOLO boxes
- YOLOv8n: small, fast, trains in ~2 min on free Colab T4
- Fine-tuning pretrained weights because only 118 training images

## 3. Experiments and results
- Exp 1: pretrained vs scratch → 0.982 vs 0.696 mAP50 (50 epochs)
- Scratch 150 epochs → 0.924: was undertrained, but still worse
- (Exp 2: n vs s, coming)

## 4. Problems and how I solved them
- Readme said 345 people, masks gave 423 → checked Annotation files: 423
- Split 118/25/27 not 119/25/26 → floating point (0.7 × 170 = 118.999...)
- Private repo couldn't clone in Colab → made it public

## 5. Strengths and weaknesses
- Strong on upright street pedestrians (mAP50 0.98)
- Fails on seated / close-up people (domain shift); COCO model did better

## 6. What I'd improve with more time
- More varied training data (indoor, seated, crowds)
- ...
