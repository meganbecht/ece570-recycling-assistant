# Personal Recycling Assistant (ECE 570)

## Overview
This project implements a recycling sort assistant that classifies an item photo into 6 classes:
cardboard, glass, metal, paper, plastic, trash. The project includes a Gradio UI that displays
prediction, confidence, top-3 probabilities, and disposal guidance.

## Environment
- Python 3.11 recommended
- Install dependencies:
  pip install -r requirements.txt
  pip install gradio

## Files
- src/app_final.py: final UI (upload image -> prediction + confidence + top-3 + guidance)
- src/eval_final.py: evaluation script (accuracy + classification report + confusion matrix)
- src/utils_model_final.py: checkpoint loader + top-k prediction helper
- outputs/model.pt: pretrained checkpoint from checkpoint 2 (ResNet-18 fine-tuned)

## How to Run
### 1) Run the UI
python src/app_final.py

### 2) Run Evaluation
python src/eval_final.py
When prompted, paste the path to the extracted TrashNet dataset-original folder.

## Notes on Authorship / LLM
- Code written by me: all files in src/*_final.py, and overall system integration.
- Model checkpoint: outputs/model.pt produced by my training script (checkpoint 2).
- If any code was adapted from external sources, list them here with links and what was changed.