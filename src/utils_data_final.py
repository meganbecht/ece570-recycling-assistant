import os
import zipfile
import tarfile
from huggingface_hub import snapshot_download
from PIL import ImageFile

#i turn this on so PIL doesn't crash if it hits a weird/truncated image file
ImageFile.LOAD_TRUNCATED_IMAGES = True

#i set this so huggingface doesn't spam the symlink warning on windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


#i use this to filter out hidden/mac junk files that can break ImageFolder/PIL
def is_valid_file(path: str) -> bool:
    base = os.path.basename(path)
    if base.startswith("."):
        return False
    if "__MACOSX" in path:
        return False
    return True


#i use this to check if a folder looks like an ImageFolder dataset root
def has_class_folders(path: str) -> bool:
    #i bail early if the path isn't even a directory
    if not os.path.isdir(path):
        return False

    #i ignore this mac folder because it isn't a real class folder
    ignore_dirs = {"__MACOSX"}

    #i collect the immediate subfolders (these would be class names for ImageFolder)
    subdirs = [
        os.path.join(path, d) for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d)) and d not in ignore_dirs
    ]

    #i need at least 2 class folders or it's probably not the dataset root
    if len(subdirs) < 2:
        return False

    #i look for at least one real image file inside the class folders
    img_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    for sd in subdirs:
        for root, _, files in os.walk(sd):
            if any(f.lower().endswith(img_exts) and not f.startswith(".") for f in files):
                return True

    return False


#i scan the downloaded huggingface snapshot to see if there's a zip/tar archive to extract
def find_archive(repo_dir: str):
    for root, _, files in os.walk(repo_dir):
        for f in files:
            lf = f.lower()
            if lf.endswith(".zip") or lf.endswith(".tar") or lf.endswith(".tar.gz") or lf.endswith(".tgz"):
                return os.path.join(root, f)
    return None


#i extract the archive into an _extracted folder (and skip if it already extracted before)
def extract_archive(archive_path: str, extract_dir: str):
    os.makedirs(extract_dir, exist_ok=True)

    #if there's already stuff in here, i assume i extracted already and skip
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


#i try a few common locations and then fall back to extracting/scanning until i find the real class-folder root
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

    #i check the obvious candidate folders first
    for c in candidates:
        if has_class_folders(c):
            return c

    #if i didn't find it, i look for an archive inside the repo snapshot
    archive = find_archive(repo_dir)
    if archive is None:
        raise FileNotFoundError(f"Could not find class folders or archive inside: {repo_dir}")

    #i extract into _extracted and then search for the first folder that looks like ImageFolder root
    extract_dir = os.path.join(repo_dir, "_extracted")
    extract_archive(archive, extract_dir)

    for root, dirs, _ in os.walk(extract_dir):
        for d in dirs:
            cand = os.path.join(root, d)
            if has_class_folders(cand):
                return cand

    raise FileNotFoundError(f"Extracted archive, but still couldn't find class folders under: {extract_dir}")


#this is my one-liner helper that downloads the hf dataset repo and returns the ImageFolder root path
def download_trashnet(repo_id: str = "garythung/trashnet") -> str:
    repo_dir = snapshot_download(repo_id=repo_id, repo_type="dataset")
    return find_imagefolder_root(repo_dir)
