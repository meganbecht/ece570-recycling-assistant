import os
import zipfile
import tarfile
from huggingface_hub import snapshot_download
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def is_valid_file(path: str) -> bool:
    base = os.path.basename(path)
    if base.startswith("."):
        return False
    if "__MACOSX" in path:
        return False
    return True


def has_class_folders(path: str) -> bool:
    """True if `path` contains >=2 class subfolders each containing at least one image file."""
    if not os.path.isdir(path):
        return False

    ignore_dirs = {"__MACOSX"}
    subdirs = [
        os.path.join(path, d) for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d)) and d not in ignore_dirs
    ]
    if len(subdirs) < 2:
        return False

    img_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    for sd in subdirs:
        for root, _, files in os.walk(sd):
            if any(f.lower().endswith(img_exts) and not f.startswith(".") for f in files):
                return True
    return False


def find_archive(repo_dir: str):
    for root, _, files in os.walk(repo_dir):
        for f in files:
            lf = f.lower()
            if lf.endswith(".zip") or lf.endswith(".tar") or lf.endswith(".tar.gz") or lf.endswith(".tgz"):
                return os.path.join(root, f)
    return None


def extract_archive(archive_path: str, extract_dir: str):
    os.makedirs(extract_dir, exist_ok=True)
    # Skip extraction if directory already has content
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
        raise ValueError(f"Unsupported archive type: {archive_path}")


def find_imagefolder_root(repo_dir: str) -> str:
    """Finds the directory that ImageFolder can load (class subfolders)."""
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
        raise FileNotFoundError(f"Could not find class folders or archive inside: {repo_dir}")

    extract_dir = os.path.join(repo_dir, "_extracted")
    extract_archive(archive, extract_dir)

    for root, dirs, _ in os.walk(extract_dir):
        for d in dirs:
            cand = os.path.join(root, d)
            if has_class_folders(cand):
                return cand

    raise FileNotFoundError(f"Extracted archive, but still couldn't find class folders under: {extract_dir}")


def download_trashnet(repo_id: str = "garythung/trashnet") -> str:
    """Downloads the dataset repo and returns the ImageFolder root directory."""
    repo_dir = snapshot_download(repo_id=repo_id, repo_type="dataset")
    return find_imagefolder_root(repo_dir)