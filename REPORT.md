# Technical Report: Pedestrian Detection with YOLOv8

## 1. What problem does the project solve?
- Detect pedestrians in images (boxes + count)
- Use case ideas: people counting, safety zones, ...

## 2. Why this data, model and approach?
- Penn-Fudan: 170 images, 423 pedestrians, masks → converted to YOLO boxes
- YOLOv8n: small, fast, trains in ~2 min on free Colab T4
- Fine-tuning pretrained weights because only 118 training images
- Final model YOLOv8n: same accuracy as YOLOv8s but 3.5× smaller (see Exp 2)

## 3. Experiments and results
- Exp 1: pretrained vs scratch → 0.982 vs 0.696 mAP50 (50 epochs)
- Scratch 150 epochs → 0.924: was undertrained, but still worse
- - Exp 1: pretrained vs scratch → 0.982 vs 0.696 mAP50 (50 epochs)
- Scratch 150 epochs → 0.924: was undertrained, but still worse
- Exp 2: YOLOv8n vs YOLOv8s → mAP50 0.982 vs 0.988, mAP50-95 0.842 vs 0.842
  - s is 3.5× bigger (22.5 MB vs 6.5 MB), 3.3× more compute (28.4 vs 8.6 GFLOPs)
  - Conclusion: YOLOv8s gave essentially the same accuracy, while being 3.5× larger and needing
    3.3× more computation. For this single-class task with limited data, the extra capacity
    brings no meaningful benefit, so I chose YOLOv8n as the final model.
- Final test set (27 unseen images): mAP50 0.949, mAP50-95 0.782, precision 0.883, recall 0.896
  (lower than val 0.982, as expected: val was used to choose the model; test is the honest score)

## 4. Problems and how I solved them
- Readme said 345 people, masks gave 423 → checked Annotation files: 423
- Split 118/25/27 not 119/25/26 → floating point (0.7 × 170 = 118.999...)
- Private repo couldn't clone in Colab → made it public
- The original paper reports 345 pedestrians, but my 
  conversion found 423. The dataset readme explains that this release added labels 
  for very small and heavily occluded pedestrians. added-object-list.txt lists 78 of them, 
  and 345 + 78 = 423, which also matches the Annotation files. So the conversion is correct, 
  and the dataset includes harder, partly hidden people.
- Evaluation counted 6 "false positives" → inspected each by eye: all were unlabeled real people
  → reported both the measured precision and this finding (label quality limits evaluation)

## 5. Strengths and weaknesses
- Strong on upright street pedestrians (mAP50 0.98)
- Fails on seated / close-up people (domain shift); COCO model did better
- Street photo, not from the dataset: detected the pedestrian at 0.88 confidence with a 
  tight full-body box and no false alarms on vehicles.
- Test set error analysis: found 58 of 67 people; all 9 misses have a clear cause:
  - 6 very small, distant pedestrians (19–67 px tall vs median 282 px)
  - 3 heavily occluded, incl. one seated person with only a leg visible (hard even for a human)
  - 8 of the 9 are in the dataset's added-object-list (small/occluded people added after the original paper)
  - No clearly visible, upright pedestrian was missed
- No real false alarms on the test set: all 6 "false positives" were real people missing from the
  labels (e.g. a woman riding a bicycle, which Penn-Fudan doesn't label as a pedestrian).
  Checked each one by eye: every box at conf ≥ 0.25 was on a real person.
- So the measured precision (0.883) underestimates the model's real precision on these images.

## 6. What I'd improve with more time
- More varied training data (indoor, seated, crowds)
- Better small-object detection: train at a higher input resolution (e.g. imgsz 960), more small examples
- Add occluded and seated people to the training data
- Re-check / complete the labels (add cyclists and other unlabeled people) for a fairer evaluation