# AI Tools Usage

Tool used: Claude (Anthropic)

## How I used AI
| Date | What I asked / what AI helped with | How I verified it |
|------|------------------------------------|-------------------|
| 2026-10-23 | Choosing dataset (Penn-Fudan) and model (YOLOv8n) based on the project brief | Checked dataset size and labels myself; confirmed free T4 GPU works on Colab |
|2026-10-24| AI predicted the split would be 119/25/26 and the dataset would have 345 people. Both were wrong. | My run gave 118/25/27: found it’s a floating-point rounding effect (0.7 × 170 = 118.999...). Counted people in the Annotation files myself: 423, not 345.|

## What I did myself
-

## How I made sure I understand the code
-