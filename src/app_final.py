import os
import gradio as gr
from PIL import Image
from torchvision import transforms

from utils_model_final import load_checkpoint, predict_topk

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "src", "outputs", "model.pt")
EXAMPLES_DIR = os.path.join(PROJECT_ROOT, "src", "examples")

TOPK = 3
UNSURE_THRESHOLD = 0.55  # if confidence below this, show "UNSURE"

GUIDANCE = {
    "cardboard": "Recycle if clean and dry. Flatten if possible.",
    "glass": "Recycle if empty/rinsed. No ceramics or mirrors.",
    "metal": "Recycle if empty/rinsed. Remove food residue.",
    "paper": "Recycle if clean and dry. Avoid food-soiled paper.",
    "plastic": "Recycle if empty/rinsed. Check local rules for bags/film.",
    "trash": "Landfill/trash if not recyclable or heavily contaminated.",
}

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

model, class_names = load_checkpoint(MODEL_PATH)

def format_output(labels, probs):
    pred = labels[0]
    conf = probs[0]
    top_lines = "\n".join([f"{l}: {p*100:.1f}%" for l, p in zip(labels, probs)])

    if conf < UNSURE_THRESHOLD:
        header = f"Prediction: UNSURE (top guess: {pred})\nConfidence: {conf*100:.1f}%"
        note = "Tip: try a clearer photo (better lighting, closer crop, simpler background)."
    else:
        header = f"Prediction: {pred}\nConfidence: {conf*100:.1f}%"
        note = ""

    out = f"{header}\n\nTop-{len(labels)}:\n{top_lines}"
    if note:
        out += f"\n\n{note}"
    return out, GUIDANCE.get(pred, "")

def predict_ui(img: Image.Image):
    if img is None:
        return "No image provided.", ""
    labels, probs = predict_topk(model, class_names, preprocess, img, k=TOPK)
    return format_output(labels, probs)

# Build examples list (optional)
examples = []
if os.path.isdir(EXAMPLES_DIR):
    for fn in sorted(os.listdir(EXAMPLES_DIR)):
        if fn.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            examples.append([os.path.join(EXAMPLES_DIR, fn)])

demo = gr.Interface(
    fn=predict_ui,
    inputs=gr.Image(type="pil", label="Upload an item photo"),
    outputs=[gr.Textbox(label="Model Output", lines=10),
             gr.Textbox(label="Guidance", lines=4)],
    examples=examples if examples else None,
    title="Personal Recycling Assistant",
    description=(
        "Upload a photo to classify it into one of 6 categories. "
        "The UI shows confidence + top-3 predictions and disposal guidance. "
        "If confidence is low, the system returns 'UNSURE' and shows alternatives."
    )
)

if __name__ == "__main__":
    demo.launch()