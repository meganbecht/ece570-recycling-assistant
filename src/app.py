# import os
# import torch
# import gradio as gr
# from PIL import Image
# from torchvision import transforms
# from torchvision.models import resnet18
#
# MODEL_PATH = os.path.join("outputs", "model.pt")
#
# # Same preprocessing style as test_tf (resize->centercrop->normalize)
# preprocess = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225]),
# ])
#
# GUIDANCE = {
#     "cardboard": "Recycle if clean and dry. Flatten if possible.",
#     "glass": "Recycle if empty/rinsed. No ceramics or mirrors.",
#     "metal": "Recycle if empty/rinsed. Remove food residue.",
#     "paper": "Recycle if clean and dry. Avoid food-soiled paper.",
#     "plastic": "Recycle if empty/rinsed. Check local rules for bags/film.",
#     "trash": "Landfill/trash if not recyclable or heavily contaminated.",
# }
#
# def load_model():
#     ckpt = torch.load(MODEL_PATH, map_location="cpu")
#     class_names = ckpt["class_names"]
#     model = resnet18(weights=None)
#     model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
#     model.load_state_dict(ckpt["model_state"])
#     model.eval()
#     return model, class_names
#
# model, class_names = load_model()
#
# def predict(img: Image.Image):
#     if img is None:
#         return "No image provided.", ""
#
#     x = preprocess(img.convert("RGB")).unsqueeze(0)
#     with torch.no_grad():
#         logits = model(x)
#         probs = torch.softmax(logits, dim=1).squeeze(0)
#
#     topk = torch.topk(probs, k=min(3, len(class_names)))
#     top_indices = topk.indices.tolist()
#     top_probs = topk.values.tolist()
#
#     pred_label = class_names[top_indices[0]]
#     confidence = top_probs[0]
#
#     top3_lines = []
#     for i, p in zip(top_indices, top_probs):
#         top3_lines.append(f"{class_names[i]}: {p*100:.1f}%")
#     top3_text = "\n".join(top3_lines)
#
#     guidance = GUIDANCE.get(pred_label, "")
#     result = f"Prediction: {pred_label}\nConfidence: {confidence*100:.1f}%\n\nTop-3:\n{top3_text}"
#     return result, guidance
#
# # demo = gr.Interface(
# #     fn=predict,
# #     inputs=gr.Image(type="pil", label="Upload an item photo"),
# #     outputs=[gr.Textbox(label="Model Output"), gr.Textbox(label="Guidance")],
# #     examples=[["examples/example.jpg"]] if os.path.exists("examples/example.jpg") else None,
# #     title="Recycling Sort Assistant (CP2 Demo)",
# #     description="Upload an image and get a predicted waste category + confidence + guidance."
# # )
# import os
# import torch
# import gradio as gr
# from PIL import Image
# from torchvision import transforms
# from torchvision.models import resnet18
#
# MODEL_PATH = os.path.join("outputs", "model.pt")
#
# # Same preprocessing style as test_tf (resize->centercrop->normalize)
# preprocess = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225]),
# ])
#
# GUIDANCE = {
#     "cardboard": "Recycle if clean and dry. Flatten if possible.",
#     "glass": "Recycle if empty/rinsed. No ceramics or mirrors.",
#     "metal": "Recycle if empty/rinsed. Remove food residue.",
#     "paper": "Recycle if clean and dry. Avoid food-soiled paper.",
#     "plastic": "Recycle if empty/rinsed. Check local rules for bags/film.",
#     "trash": "Landfill/trash if not recyclable or heavily contaminated.",
# }
#
# def load_model():
#     ckpt = torch.load(MODEL_PATH, map_location="cpu")
#     class_names = ckpt["class_names"]
#     model = resnet18(weights=None)
#     model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
#     model.load_state_dict(ckpt["model_state"])
#     model.eval()
#     return model, class_names
#
# model, class_names = load_model()
#
# def predict(img: Image.Image):
#     if img is None:
#         return "No image provided.", ""
#
#     x = preprocess(img.convert("RGB")).unsqueeze(0)
#     with torch.no_grad():
#         logits = model(x)
#         probs = torch.softmax(logits, dim=1).squeeze(0)
#
#     topk = torch.topk(probs, k=min(3, len(class_names)))
#     top_indices = topk.indices.tolist()
#     top_probs = topk.values.tolist()
#
#     pred_label = class_names[top_indices[0]]
#     confidence = top_probs[0]
#
#     top3_lines = []
#     for i, p in zip(top_indices, top_probs):
#         top3_lines.append(f"{class_names[i]}: {p*100:.1f}%")
#     top3_text = "\n".join(top3_lines)
#
#     guidance = GUIDANCE.get(pred_label, "")
#     result = f"Prediction: {pred_label}\nConfidence: {confidence*100:.1f}%\n\nTop-3:\n{top3_text}"
#     return result, guidance
#
# demo = gr.Interface(
#     fn=predict,
#     inputs=gr.Image(type="pil", label="Upload an item photo"),
#     outputs=[gr.Textbox(label="Model Output"), gr.Textbox(label="Guidance")],
#     examples=[["examples/example.jpg"]] if os.path.exists("examples/example.jpg") else None,
#     title="Recycling Sort Assistant (CP2 Demo)",
#     description="Upload an image and get a predicted waste category + confidence + guidance."
# )
#
# if __name__ == "__main__":
#     demo.launch()

import os
import torch
import gradio as gr
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18

# MODEL_PATH = os.path.join("outputs", "model.pt")
# EXAMPLE_PATH = os.path.join("examples", "example4.jpg")
# Put this near the top of app.py
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "model.pt")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(PROJECT_ROOT, "src", "outputs", "model.pt")

EXAMPLE_PATH = os.path.join(PROJECT_ROOT, "examples", "example2.jpg")

# Same preprocessing style as test_tf (resize->centercrop->normalize)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

GUIDANCE = {
    "cardboard": "Recycle if clean and dry. Flatten if possible.",
    "glass": "Recycle if empty/rinsed. No ceramics or mirrors.",
    "metal": "Recycle if empty/rinsed. Remove food residue.",
    "paper": "Recycle if clean and dry. Avoid food-soiled paper.",
    "plastic": "Recycle if empty/rinsed. Check local rules for bags/film.",
    "trash": "Landfill/trash if not recyclable or heavily contaminated.",
}

def load_model():
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    class_names = ckpt["class_names"]
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, class_names

model, class_names = load_model()

# --- KEEP THIS FUNCTION UNCHANGED FOR YOUR CODE SNIPPET ---
def predict(img: Image.Image):
    if img is None:
        return "No image provided.", ""

    x = preprocess(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    topk = torch.topk(probs, k=min(3, len(class_names)))
    top_indices = topk.indices.tolist()
    top_probs = topk.values.tolist()

    pred_label = class_names[top_indices[0]]
    confidence = top_probs[0]

    top3_lines = []
    for i, p in zip(top_indices, top_probs):
        top3_lines.append(f"{class_names[i]}: {p*100:.1f}%")
    top3_text = "\n".join(top3_lines)

    guidance = GUIDANCE.get(pred_label, "")
    result = f"Prediction: {pred_label}\nConfidence: {confidence*100:.1f}%\n\nTop-3:\n{top3_text}"
    return result, guidance
# --- END UNCHANGED FUNCTION ---


def load_fixed_example():
    """Loads examples/example4.jpg and returns (image, model_output, guidance)."""
    if not os.path.exists(EXAMPLE_PATH):
        return None, f"Missing file: {EXAMPLE_PATH}", ""
    img = Image.open(EXAMPLE_PATH).convert("RGB")
    result, guidance = predict(img)
    return img, result, guidance


# Precompute so the page shows results immediately
default_img, default_result, default_guidance = load_fixed_example()

with gr.Blocks() as demo:
    gr.Markdown("## Recycling Sort Assistant (CP2 Demo — Example Results)")

    with gr.Row():
        img_box = gr.Image(
            value=default_img,
            type="pil",
            label="Example image (placeholder — uploads coming next checkpoint)",
            interactive=False
        )
        out_box = gr.Textbox(value=default_result, label="Model Output", lines=8)
        guide_box = gr.Textbox(value=default_guidance, label="Guidance", lines=4)

if __name__ == "__main__":
    demo.launch()