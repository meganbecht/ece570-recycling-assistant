import os
import random
import zipfile
import tarfile
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.models import resnet18
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from huggingface_hub import snapshot_download
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

SEED = 570
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

HF_REPO_ID = "garythung/trashnet"

MAX_TRAIN = 800
MAX_TEST = 200

BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3
NUM_WORKERS = 0

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def plot_confusion(cm, labels, out_path):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_loss_curve(losses, out_path):
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(losses) + 1), losses, marker="o")
    plt.title("Training Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Average Training Loss")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


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
    count_with_imgs = 0
    for sd in subdirs:
        for root, _, files in os.walk(sd):
            real_imgs = [
                f for f in files
                if f.lower().endswith(img_exts)
                and not os.path.basename(f).startswith(".")
            ]
            if len(real_imgs) > 0:
                count_with_imgs += 1
                break
    return count_with_imgs >= 2


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
        for root, dirs, _ in os.walk(repo_dir):
            for d in dirs:
                cand = os.path.join(root, d)
                if has_class_folders(cand):
                    return cand
        raise FileNotFoundError(f"couldn't find class folders in {repo_dir}")

    extract_dir = os.path.join(repo_dir, "_extracted")
    print("archive:", archive)
    print("extracting to:", extract_dir)
    extract_archive(archive, extract_dir)

    for root, dirs, _ in os.walk(extract_dir):
        for d in dirs:
            cand = os.path.join(root, d)
            if has_class_folders(cand):
                return cand

    raise FileNotFoundError(f"couldn't find class folders in {extract_dir}")


def is_valid_file(path: str) -> bool:
    base = os.path.basename(path)
    if base.startswith("."):
        return False
    if "__MACOSX" in path:
        return False
    return True


def evaluate(model, loader, device):
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
    return acc, cm


def main():
    repo_dir = snapshot_download(repo_id=HF_REPO_ID, repo_type="dataset")
    data_root = find_imagefolder_root(repo_dir)

    train_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    test_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    full_ds = ImageFolder(root=data_root, transform=train_tf, is_valid_file=is_valid_file)
    class_names = full_ds.classes
    num_classes = len(class_names)

    indices = list(range(len(full_ds)))
    random.shuffle(indices)

    n_train = min(MAX_TRAIN, len(indices))
    n_test = min(MAX_TEST, max(0, len(indices) - n_train))

    train_idx = indices[:n_train]
    test_idx = indices[n_train:n_train + n_test]

    full_ds_test = ImageFolder(root=data_root, transform=test_tf, is_valid_file=is_valid_file)

    train_ds = Subset(full_ds, train_idx)
    test_ds = Subset(full_ds_test, test_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model = resnet18(weights="DEFAULT")
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    os.makedirs("outputs", exist_ok=True)

    epoch_losses = []
    epoch_accs = []

    # ---- Train for multiple epochs ----
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        epoch_losses.append(avg_loss)

        acc, _ = evaluate(model, test_loader, device)
        epoch_accs.append(acc)

        print(f"Epoch {epoch+1}/{EPOCHS} loss={avg_loss:.4f} test_acc={acc:.4f}")

    # ---- Final evaluation artifacts ----
    final_acc, final_cm = evaluate(model, test_loader, device)

    cm_path = os.path.join("outputs", "confusion_matrix.png")
    plot_confusion(final_cm, class_names, cm_path)

    loss_path = os.path.join("outputs", "loss_curve.png")
    plot_loss_curve(epoch_losses, loss_path)

    print(f"Final test accuracy: {final_acc:.4f}")
    print("Saved confusion matrix to:", cm_path)
    print("Saved loss curve to:", loss_path)


if __name__ == "__main__":
    main()