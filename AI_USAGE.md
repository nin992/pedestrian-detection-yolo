# AI Tools Usage

Tool used: Claude (Anthropic)

## How I used AI

| Date        | What I asked / what AI helped with | How I verified it |
|-------------|---|---|
| 2026-10-23  | Choosing dataset (Penn-Fudan) and model (YOLOv8n) based on the project brief | Checked dataset size and labels myself; confirmed free T4 GPU works on Colab |
| 2026-10-24  | AI predicted the split would be 119/25/26 and the dataset would have 345 people. Both were wrong. | My run gave 118/25/27: found it’s a floating-point rounding effect (0.7 × 170 = 118.999...). Counted people in the Annotation files myself: 423, not 345.|
| 2026-10-25  | AI wrote train.py (train YOLOv8, validate best.pt, log results to experiments.csv) and suggested Experiment 1: pretrained vs from scratch. | Ran both on Colab. Read the per-epoch output myself, noticed the scratch model's best score was at the last epoch, guessed it was undertrained, and tested it with a 150-epoch run (mAP50 rose 0.70 → 0.92, still below pretrained 0.98). |
| 2026-10-27  | Writing evaluate.py (error analysis) and optimize.py | Ran both; checked every miss and "false positive" image by eye; confirmed the optimization numbers in optimization.csv |
| 2026-10-27  | Building the FastAPI app | Tested every endpoint in /docs, including an invalid file (got the expected 400 error) |
| 2026-10-28  | README and report writing | Checked every number against my own CSV files and outputs; tested the README from a fresh clone (reproduced test mAP50 0.949) |
| 2026-10-28  | Uploading the model as a GitHub Release | The file was 21 MB instead of the expected ~6.5 MB. Checked it with model.info() (11.1M parameters = YOLOv8s), replaced it with the correct YOLOv8n file (3.1M) and re-checked |
