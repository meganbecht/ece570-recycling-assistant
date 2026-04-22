import os
import torch
from torchvision.models import resnet18


def load_checkpoint(model_path: str):
    """
    Loads your existing outputs/model.pt from CP2.
    Expected checkpoint format:
      {"model_state": ..., "class_names": ...}
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at: {model_path}\n"
            "Make sure outputs/model.pt exists (your CP2 training should have created it)."
        )

    ckpt = torch.load(model_path, map_location="cpu")
    class_names = ckpt["class_names"]

    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, class_names


def predict_topk(model, class_names, preprocess, img, k=3):
    """
    Returns top-k labels and probabilities for a PIL image.
    """
    x = preprocess(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    k = min(k, len(class_names))
    topk = torch.topk(probs, k=k)
    labels = [class_names[i] for i in topk.indices.tolist()]
    scores = topk.values.tolist()
    return labels, scores