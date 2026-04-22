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
from utils_data_final import download_trashnet, is_valid_file


#i use a fixed seed so anything random stays repeatable
SEED = 570
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

#i set up paths relative to this file so this works no matter where i run it from
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "src", "outputs", "model.pt")
OUT_DIR = os.path.join(PROJECT_ROOT, "src", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

#i keep these settings here so i can tune speed if needed
BATCH_SIZE = 32
NUM_WORKERS = 0
PRINT_EVERY = 10


#this saves a confusion matrix image so i can include it in my report
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
    #i load the trained checkpoint that i committed to github
    model, class_names = load_checkpoint(MODEL_PATH)

    #this matches the preprocessing i use for inference in my UI
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    #this is the big improvement: i auto-download/find the dataset so the user doesn't paste paths
    print("downloading/finding trashnet from huggingface...")
    dataset_folder = download_trashnet()
    print("using dataset folder:", dataset_folder)

    #i load the dataset with ImageFolder so folder names become labels automatically
    ds = ImageFolder(root=dataset_folder, transform=preprocess, is_valid_file=is_valid_file)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    total_batches = len(loader)
    total_images = len(ds)

    print(f"classes: {ds.classes}")
    print(f"total images: {total_images} | batch size: {BATCH_SIZE} | total batches: {total_batches}\n")

    #i store predictions and true labels so i can compute metrics at the end
    all_preds, all_true = [], []
    start_time = time.time()

    model.eval()
    with torch.no_grad():
        for i, (xb, yb) in enumerate(loader):
            logits = model(xb)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds.tolist())
            all_true.extend(yb.numpy().tolist())

            #i print progress so it doesn't look stuck
            if i % PRINT_EVERY == 0 or i == total_batches - 1:
                elapsed = time.time() - start_time
                done = min((i + 1) * BATCH_SIZE, total_images)
                print(f"processed batch {i+1}/{total_batches}  ({done}/{total_images} images)  elapsed={elapsed:.1f}s")

    #i compute the metrics i want to report
    acc = accuracy_score(all_true, all_preds)
    cm = confusion_matrix(all_true, all_preds)
    report = classification_report(all_true, all_preds, target_names=ds.classes, digits=4)

    metrics_path = os.path.join(OUT_DIR, "final_metrics.txt")
    cm_path = os.path.join(OUT_DIR, "final_confusion_matrix.png")

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.6f}\n\n")
        f.write(report)

    plot_confusion(cm, ds.classes, cm_path)

    total_time = time.time() - start_time
    print("\ndone.")
    print(f"final eval accuracy: {acc:.6f}")
    print(f"saved metrics to: {metrics_path}")
    print(f"saved confusion matrix to: {cm_path}")
    print(f"total eval time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
