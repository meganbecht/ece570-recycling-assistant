import os
import time
import random
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

from utils_model_final import load_checkpoint


#i use a fixed seed so my results are repeatable
SEED = 570
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

#i set up paths relative to this file so i can run it from anywhere
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

#i load the same checkpoint my UI uses
MODEL_PATH = os.path.join(PROJECT_ROOT, "src", "outputs", "model.pt")

#i write all evaluation outputs into my outputs folder
OUT_DIR = os.path.join(PROJECT_ROOT, "src", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

#i keep these settings here so i can make eval faster/slower depending on my machine
BATCH_SIZE = 32
NUM_WORKERS = 0
PRINT_EVERY = 10


#this function just makes a confusion matrix image and saves it for my report
def plot_confusion(cm, labels, out_path, title="Confusion Matrix (Final Eval)"):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    #i load my trained model once at the start
    model, class_names = load_checkpoint(MODEL_PATH)

    #this matches the preprocessing i use during inference (same style as the UI)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    #i ask for the dataset folder here so i can run eval on whatever local location i have
    dataset_folder = input(
        "Paste the path to your extracted TrashNet 'dataset-original' folder:\n"
        "(example: ...\\_extracted\\dataset-original)\n> "
    ).strip().strip('"')

    #i do a quick check so i can fail fast if the path is wrong
    if not os.path.isdir(dataset_folder):
        raise FileNotFoundError(f"Folder not found: {dataset_folder}")

    #i load the dataset using ImageFolder so class folders become labels automatically
    ds = ImageFolder(root=dataset_folder, transform=preprocess)

    #i use a dataloader so i can run evaluation in batches instead of one image at a time
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    total_batches = len(loader)
    total_images = len(ds)

    #i print a quick summary so i know what i'm evaluating on
    print(f"\nLoaded dataset from: {dataset_folder}")
    print(f"Classes: {ds.classes}")
    print(f"Total images: {total_images} | Batch size: {BATCH_SIZE} | Total batches: {total_batches}\n")

    #i store all predictions and ground truth labels so i can compute metrics at the end
    all_preds, all_true = [], []
    start_time = time.time()

    #i turn off gradients because i'm only doing inference here
    model.eval()
    with torch.no_grad():
        for i, (xb, yb) in enumerate(loader):
            #i run the model forward pass on this batch
            logits = model(xb)

            #i convert logits to predicted class indices
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            #i save predictions and ground truth so sklearn can compute metrics later
            all_preds.extend(preds.tolist())
            all_true.extend(yb.numpy().tolist())

            #i print progress every few batches so it doesn't look stuck
            if i % PRINT_EVERY == 0 or i == total_batches - 1:
                elapsed = time.time() - start_time
                done = min((i + 1) * BATCH_SIZE, total_images)
                print(f"Processed batch {i+1}/{total_batches}  ({done}/{total_images} images)  elapsed={elapsed:.1f}s")

    #i compute the main metrics i want to report
    acc = accuracy_score(all_true, all_preds)
    cm = confusion_matrix(all_true, all_preds)
    report = classification_report(all_true, all_preds, target_names=ds.classes, digits=4)

    #i save the metrics and the confusion matrix image so i can include them in my paper
    metrics_path = os.path.join(OUT_DIR, "final_metrics.txt")
    cm_path = os.path.join(OUT_DIR, "final_confusion_matrix.png")

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.6f}\n\n")
        f.write(report)

    plot_confusion(cm, ds.classes, cm_path)

    #i print a final summary so i can copy/paste results quickly
    total_time = time.time() - start_time
    print("\nDONE.")
    print(f"Final eval accuracy: {acc:.6f}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved confusion matrix to: {cm_path}")
    print(f"Total eval time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
