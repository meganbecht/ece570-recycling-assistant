import os
import torch
from torchvision.models import resnet18


#this loads my saved checkpoint and rebuilds the resnet18 model so i can run inference
def load_checkpoint(model_path: str):
    #i fail fast here if the checkpoint path is wrong so i don't get confusing errors later
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at: {model_path}\n"
            "Make sure outputs/model.pt exists (my training script should have created it)."
        )

    #i load the checkpoint on cpu so it works on any machine
    ckpt = torch.load(model_path, map_location="cpu")

    #i pull the class names out so i can map indices -> labels later
    class_names = ckpt["class_names"]

    #i rebuild resnet18 with the correct final layer size for my number of classes
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))

    #i load the trained weights into the model and switch to eval mode
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, class_names


#this runs inference on one image and returns the top-k labels + probabilities
def predict_topk(model, class_names, preprocess, img, k=3):
    #i preprocess the image the same way i did during evaluation and then add a batch dimension
    x = preprocess(img.convert("RGB")).unsqueeze(0)

    #i turn off gradients since i'm not training here
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    #i make sure k isn't bigger than the number of classes
    k = min(k, len(class_names))

    #i grab the top-k probabilities and convert indices back to readable class names
    topk = torch.topk(probs, k=k)
    labels = [class_names[i] for i in topk.indices.tolist()]
    scores = topk.values.tolist()

    return labels, scores
