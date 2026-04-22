import os
import time
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

from utils_data_final import download_trashnet, is_valid_file
from utils_model_final import build_resnet18, save_checkpoint

# -----------------------------
# Config
# -----------------------------
SEED = 570
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(OUTPUTS_DIR, "model.pt")

BATCH_SIZE = 32
EPOCHS = 5
LR = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
PRINT_EVERY = 10

TRAIN_FRAC = 0.8  # 80/20 split


def plot_confusion(cm, labels, out_path, title):
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


def eval_metrics(model, loader, device):
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_true.extend(yb.numpy().tolist())

    acc = accuracy_score(all_true, all_preds)
    cm = confusion_matrix(all_true, all_preds)
    report = classification_report(all_true, all_preds, digits=4)
    return acc, cm, report


def main():
    print("Downloading + preparing dataset...")
    data_root = download_trashnet()
    print("Using ImageFolder root:", data_root)

    # CPU-friendly transforms (fast + consistent with your CP2 success)
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    full_ds = ImageFolder(root=data_root, transform=train_tf, is_valid_file=is_valid_file)
    class_names = full_ds.classes
    num_classes = len(class_names)

    print("Classes:", class_names)
    print("Total images:", len(full_ds))

    # Reproducible split indices
    indices = list(range(len(full_ds)))
    random.Random(SEED).shuffle(indices)
    n_train = int(TRAIN_FRAC * len(indices))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    # Build test dataset with test_tf
    full_ds_test = ImageFolder(root=data_root, transform=test_tf, is_valid_file=is_valid_file)

    train_ds = Subset(full_ds, train_idx)
    test_ds = Subset(full_ds_test, test_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print(f"Train batches/epoch={len(train_loader)} | Test batches={len(test_loader)}")

    model = build_resnet18(num_classes=num_classes, pretrained=True).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.3)

    best_acc = -1.0
    best_report = ""
    best_cm = None

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0

        print(f"\n=== Epoch {epoch+1}/{EPOCHS} (lr={optimizer.param_groups[0]['lr']:.2e}) ===")
        for batch_i, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_i % PRINT_EVERY == 0:
                elapsed = time.time() - epoch_start
                print(f"  batch {batch_i:>3}/{len(train_loader)} | loss={loss.item():.4f} | elapsed={elapsed:.1f}s")

        scheduler.step()

        avg_loss = running_loss / len(train_loader)
        acc, cm, report = eval_metrics(model, test_loader, device)
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{EPOCHS} DONE | avg_loss={avg_loss:.4f} | test_acc={acc:.4f} | epoch_time={epoch_time:.1f}s")

        if acc > best_acc:
            best_acc = acc
            best_cm = cm
            best_report = report

            # Save checkpoint + split indices for reproducibility
            save_checkpoint(
                model_path=MODEL_PATH,
                model=model,
                class_names=class_names,
                split_indices={"train": train_idx, "test": test_idx},
                extra={"best_acc": best_acc, "seed": SEED}
            )

            plot_confusion(best_cm, class_names,
                           os.path.join(OUTPUTS_DIR, "confusion_matrix.png"),
                           title="Confusion Matrix (Best Model)")
            with open(os.path.join(OUTPUTS_DIR, "metrics.txt"), "w", encoding="utf-8") as f:
                f.write(f"Best test accuracy: {best_acc:.6f}\n\n")
                f.write("Classification report:\n")
                f.write(best_report)

            print(f"  New best! Saved outputs/model.pt, outputs/confusion_matrix.png, outputs/metrics.txt (best_acc={best_acc:.4f})")

    print("\nTraining complete.")
    print("Best test accuracy:", best_acc)
    print("Saved best model to:", MODEL_PATH)


if __name__ == "__main__":
    main()