# Personal Recycling Assistant (ECE 570 Final Project)

This repository contains my complete, runnable recycling sort assistant:
- a trained ResNet-18 image classifier checkpoint included in the repository
- a styled Gradio web UI for interactive image uploads and predictions
- an evaluation script that produces accuracy + a full classification report + a confusion matrix plot

My GitHub repo: https://github.com/meganbecht/ece570-recycling-assistant

--------------------------------------------------------------------

## What my project does

Given an image of an item, my system predicts one of 6 waste categories:
- cardboard
- glass
- metal
- paper
- plastic
- trash

My UI displays:
- predicted category (or "UNSURE" if confidence is low)
- confidence score (top-1 probability)
- top-3 alternatives (so the user can see similar classes)
- short disposal guidance text for the predicted class

--------------------------------------------------------------------

## Repository structure (what you should see)

After downloading or cloning, your repo should include these main files and look roughly like this:

- requirements.txt
- src/
  - app_final_styled.py
  - eval_final.py
  - utils_model_final.py
  - utils_data_final.py
  - outputs/
    - model.pt

The trained model checkpoint is included here:
- src/outputs/model.pt

--------------------------------------------------------------------

## Download

if you just want to demo the project, you only need to:
1) install dependencies
2) run the app

You can stop after running the app. I include instructions on how I ran the evaluation, but running evaluation is optional bonus.

--------------------------------------------------------------------

## Step-by-step setup

### 1) Install python
Use python 3.10 or 3.11.

check your version:
```bash
python --version
````

### 2) Download the repository

Option a: download zip

1. go to the repo page:
   [https://github.com/meganbecht/ece570-recycling-assistant](https://github.com/meganbecht/ece570-recycling-assistant)
2. click the green "code" button
3. click "download zip"
4. extract it somewhere you can find easily

My recommended locations:

* windows: C:\Users<you>\Documents\ece570-recycling-assistant\
* mac/linux: ~/Documents/ece570-recycling-assistant/

Option b: clone with git

```bash
git clone https://github.com/meganbecht/ece570-recycling-assistant.git
cd ece570-recycling-assistant
```

### 3) Open a terminal in the correct folder

You must run commands from the repo root (the folder that contains src/ and requirements.txt).

quick checks:

windows powershell:

```powershell
dir
dir src
```

mac/linux:

```bash
ls
ls src
```

You should see requirements.txt and a src folder.

### 4) Create and activate a virtual environment

I recommend this to avoids package conflicts.

windows powershell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

if activation is blocked, run once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

mac/linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

you’ll know it worked when your terminal shows something like (.venv) at the start of the prompt.

### 5) Install dependencies

From the repo root:

```bash
pip install -r requirements.txt
```

if pip is outdated and install fails:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 6) Confirm the model checkpoint exists

This repo includes the trained checkpoint at:

* src/outputs/model.pt

check it exists:

windows powershell:

```powershell
dir src\outputs\model.pt
```

mac/linux:

```bash
ls -lh src/outputs/model.pt
```

if model.pt is missing, the UI will not run. (in this repo it should already be included.)

---

## Run the project

### Run the styled UI (this is the required demo)

From the repo root:

```bash
python src/app_final_styled.py
```

expected behavior:

* the terminal prints a local url (usually [http://127.0.0.1:7860](http://127.0.0.1:7860))
* open that url in your browser
* upload an image (jpg/png/webp)
* click the classify button
* you’ll see prediction + confidence + top-3 + guidance
* if confidence is low, it will show "UNSURE" and suggest retaking the photo

once you have the app running successfully, you can stop here. evaluation is optional bonus.

---

## Optional bonus: run evaluation (metrics + confusion matrix)

Evaluation is not required to run the project, but it is useful if you want to reproduce the metrics and regenerate plots.

From the repo root:

```bash
python src/eval_final.py
```

What evaluation does:

* loads the trained checkpoint from src/outputs/model.pt
* downloads/locates the dataset (depending on the eval script version)
* runs inference in batches and prints progress
* saves outputs into src/outputs/

expected output files:

* src/outputs/final_metrics.txt
* src/outputs/final_confusion_matrix.png

typical runtime:

* cpu evaluation can take several minutes depending on your machine

dataset download behavior (hugging face):

* the dataset is not stored in this repository (too large)
* the eval code can download it automatically via hugging face and reuse the local cache later

hugging face cache locations:

* windows: C:\Users<you>.cache\huggingface\hub\
* mac/linux: ~/.cache/huggingface/hub/

if hugging face download is slow or rate-limited, you can optionally login:

```bash
huggingface-cli login
```

---

## Troubleshooting

module not found / missing packages:

* make sure your virtual environment is activated (you should see (.venv))
* rerun: pip install -r requirements.txt

model not found:

* confirm src/outputs/model.pt exists
* confirm you are running from the repo root

gradio not installed:

```bash
pip install gradio
```

slow downloads:

* first run can be slow because hugging face is downloading the dataset
* later runs use cache
* optional: huggingface-cli login

---

## My file guide (what each file does)

* src/app_final_styled.py
  styled gradio ui (upload image → prediction + confidence/top-3 + guidance + UNSURE threshold)

* src/eval_final.py
  evaluation script (accuracy + classification report + confusion matrix)

* src/utils_model_final.py
  loads checkpoint + top-k prediction helper

* src/utils_data_final.py
  dataset download + dataset root locator (used for auto-download eval)

* src/outputs/model.pt
  trained model checkpoint included in the repo
