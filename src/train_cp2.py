import os
import random
import zipfile
import tarfile
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.models import resnet18
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from huggingface_hub import snapshot_download
from PIL import ImageFile

# Robustness for odd image files
ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# -----------------------------
# Config (CPU-friendly)
# -----------------------------
SEED = 570
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

HF_REPO_ID = "garythung/trashnet"

BATCH_SIZE = 32
EPOCHS = 5                 # faster than 10; still meaningful
LR = 3e-4                  # good fine-tuning LR
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0            # keep 0 on Windows

PRINT_EVERY = 10           # print progress more often


def plot_confusion(cm, labels, out_path):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (CP2 Best)")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def is_valid_file(path: str) -> bool:
    base = os.path.basename(path)
    if base.startswith("."):
        return False
    if "__MACOSX" in path:
        return False
    return True


def has_class_folders(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    ignore_dirs = {"__MACOSX"}
    subdirs = []
    for d in os.listdir(path):
        full = os.path.join(path, d)
        if os.path.isdir(full) and d not in ignore_dirs:
            subdirs.append(full)
    if len(subdirs) < 2:
        return False

    img_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    for sd in subdirs:
        for root, _, files in os.walk(sd):
            real_imgs = [
                f for f in files
                if f.lower().endswith(img_exts) and not os.path.basename(f).startswith(".")
            ]
            if len(real_imgs) > 0:
                return True
    return False


def find_archive(repo_dir: str) -> str | None:
    for root, _, files in os.walk(repo_dir):
        for f in files:
            lf = f.lower()
            if lf.endswith(".zip") or lf.endswith(".tar") or lf.endswith(".tar.gz") or lf.endswith(".tgz"):
                return os.path.join(root, f)
    return None


def extract_archive(archive_path: str, extract_dir: str) -> None:
    os.makedirs(extract_dir, exist_ok=True)
    if any(os.scandir(extract_dir)):
        return

    lp = archive_path.lower()
    if lp.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_dir)
    elif lp.endswith(".tar") or lp.endswith(".tar.gz") or lp.endswith(".tgz"):
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(extract_dir)
    else:
        raise ValueError(f"unsupported archive type {archive_path}")


def find_imagefolder_root(repo_dir: str) -> str:
    candidates = [
        os.path.join(repo_dir, "_extracted", "dataset-original"),
        os.path.join(repo_dir, "_extracted", "dataset-resized"),
        os.path.join(repo_dir, "dataset-resized"),
        os.path.join(repo_dir, "dataset"),
        os.path.join(repo_dir, "data"),
        os.path.join(repo_dir, "_extracted"),
        repo_dir,
    ]
    for c in candidates:
        if has_class_folders(c):
            return c

    archive = find_archive(repo_dir)
    if archive is None:
        raise FileNotFoundError(f"couldn't find class folders or archive in {repo_dir}")

    extract_dir = os.path.join(repo_dir, "_extracted")
    extract_archive(archive, extract_dir)

    for root, dirs, _ in os.walk(extract_dir):
        for d in dirs:
            cand = os.path.join(root, d)
            if has_class_folders(cand):
                return cand

    raise FileNotFoundError(f"couldn't find class folders in extracted dir {extract_dir}")


def eval_accuracy(model, loader, device):
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_true.extend(yb.numpy().tolist())
    return accuracy_score(all_true, all_preds), confusion_matrix(all_true, all_preds)


def main():
    os.makedirs("outputs", exist_ok=True)

    print("Downloading dataset repo from HF:", HF_REPO_ID)
    repo_dir = snapshot_download(repo_id=HF_REPO_ID, repo_type="dataset")
    data_root = find_imagefolder_root(repo_dir)
    print("Using data root:", data_root)

    # -----------------------------
    # Faster CPU-friendly transforms
    # -----------------------------
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

    # Full dataset (train transform)
    full_ds = ImageFolder(root=data_root, transform=train_tf, is_valid_file=is_valid_file)
    class_names = full_ds.classes
    num_classes = len(class_names)
    print("Classes:", class_names)
    print("Total images:", len(full_ds))

    # 80/20 split (more data than CP1 subset)
    n_total = len(full_ds)
    n_train = int(0.8 * n_total)
    n_test = n_total - n_train
    generator = torch.Generator().manual_seed(SEED)

    train_ds, test_indices_ds = random_split(full_ds, [n_train, n_test], generator=generator)
    full_ds_test = ImageFolder(root=data_root, transform=test_tf, is_valid_file=is_valid_file)
    test_ds = torch.utils.data.Subset(full_ds_test, test_indices_ds.indices)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print(f"Train batches per epoch: {len(train_loader)} | Test batches: {len(test_loader)}")

    # Model
    model = resnet18(weights="DEFAULT")
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.3)

    best_acc = -1.0
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
        acc, cm = eval_accuracy(model, test_loader, device)
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{EPOCHS} DONE | avg_loss={avg_loss:.4f} | test_acc={acc:.4f} | epoch_time={epoch_time:.1f}s")

        if acc > best_acc:
            best_acc = acc
            torch.save(
                {"model_state": model.state_dict(), "class_names": class_names},
                os.path.join("outputs", "model.pt"),
            )
            plot_confusion(cm, class_names, os.path.join("outputs", "confusion_matrix_cp2.png"))
            print(f"  New best! Saved outputs/model.pt and outputs/confusion_matrix_cp2.png (best_acc={best_acc:.4f})")

    print("\nBest test accuracy:", best_acc)
    print("Saved best model to outputs/model.pt")


if __name__ == "__main__":
    main()