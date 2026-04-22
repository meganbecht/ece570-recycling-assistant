import os
import gradio as gr
from PIL import Image
from torchvision import transforms

from utils_model_final import load_checkpoint, predict_topk

#i use this to anchor all my file paths so the app works no matter where i run it from
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

#i point to the model checkpoint i trained earlier (this is what the UI loads)
MODEL_PATH = os.path.join(PROJECT_ROOT, "src", "outputs", "model.pt")

#i point to my local examples folder so i can show clickable demo images in the UI
EXAMPLES_DIR = os.path.join(PROJECT_ROOT, "src", "examples")

#i show the top 3 classes so users can see alternatives when categories look similar
TOPK = 3

#i flag low-confidence predictions as "UNSURE" so the UI is a little more honest
UNSURE_THRESHOLD = 0.55

#i keep short guidance text here so i can display a helpful message with each prediction
GUIDANCE = {
    "cardboard": "Recycle if clean and dry. Flatten if possible.",
    "glass": "Recycle if empty/rinsed. No ceramics or mirrors.",
    "metal": "Recycle if empty/rinsed. Remove food residue.",
    "paper": "Recycle if clean and dry. Avoid food-soiled paper.",
    "plastic": "Recycle if empty/rinsed. Check local rules for bags/film.",
    "trash": "Landfill/trash if not recyclable or heavily contaminated.",
}

#this matches the preprocessing i used for inference (resize/crop + ImageNet normalization)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

#i load the saved model checkpoint once so every request is fast
model, class_names = load_checkpoint(MODEL_PATH)

#this is my custom CSS to make the UI look earth/recycling themed
CUSTOM_CSS = """
body, .gradio-container {
    background: linear-gradient(180deg, #f4f1e8 0%, #eef4ea 100%) !important;
    font-family: Arial, sans-serif;
}

.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
}

#hero {
    background: linear-gradient(135deg, #2f6b3a 0%, #4f8a54 100%);
    color: white;
    border-radius: 20px;
    padding: 24px 28px;
    margin-bottom: 18px;
    box-shadow: 0 10px 24px rgba(0,0,0,0.12);
}

#hero h1 {
    margin: 0 0 8px 0;
    font-size: 2.1rem;
}

#hero p {
    margin: 0;
    font-size: 1rem;
    opacity: 0.96;
}

.feature-card {
    background: white;
    border-radius: 18px;
    padding: 16px;
    border: 1px solid #d9e4d3;
    box-shadow: 0 6px 16px rgba(0,0,0,0.06);
    text-align: center;
    min-height: 88px;
}

.feature-card h3 {
    margin: 0 0 8px 0;
    color: #2f6b3a;
    font-size: 1.05rem;
}

.feature-card p {
    margin: 0;
    font-size: 0.92rem;
    color: #4d5a4f;
}

.panel-card {
    background: rgba(255,255,255,0.90);
    border-radius: 20px;
    border: 1px solid #d9e4d3;
    box-shadow: 0 8px 20px rgba(0,0,0,0.06);
    padding: 12px;
}

.result-card {
    background: #ffffff;
    border: 1px solid #d8e6d3;
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 8px 18px rgba(0,0,0,0.06);
}

.result-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
    margin-bottom: 12px;
}

.metric-box {
    background: #f5fbf2;
    border: 1px solid #d8e6d3;
    border-radius: 14px;
    padding: 12px;
}

.metric-label {
    font-size: 0.82rem;
    color: #5c6c5e;
    margin-bottom: 4px;
}

.metric-value {
    font-size: 1.25rem;
    font-weight: bold;
    color: #244d2b;
}

.status-pill {
    display: inline-block;
    padding: 7px 12px;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: bold;
    margin-bottom: 12px;
}

.status-good {
    background: #dff2df;
    color: #1f6a2b;
    border: 1px solid #b9e0bb;
}

.status-unsure {
    background: #fff0cc;
    color: #8a6500;
    border: 1px solid #f0d887;
}

.topk-title {
    font-weight: bold;
    color: #2f6b3a;
    margin: 12px 0 8px 0;
    font-size: 1rem;
}

.topk-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.95rem;
}

.topk-table th, .topk-table td {
    padding: 10px 8px;
    border-bottom: 1px solid #e7eee4;
    text-align: left;
}

.topk-table th {
    color: #355d3c;
    background: #f7fbf5;
}

.guidance-card {
    background: #ffffff;
    border: 1px solid #d8e6d3;
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 8px 18px rgba(0,0,0,0.06);
    min-height: 180px;
}

.guidance-card h3 {
    margin-top: 0;
    color: #2f6b3a;
}

.guidance-card p {
    margin-bottom: 0;
    color: #4d5a4f;
    line-height: 1.5;
}

.footer-note {
    text-align: center;
    color: #5f6f61;
    font-size: 0.9rem;
    margin-top: 8px;
}

button.primary {
    background: #2f6b3a !important;
    border: none !important;
}

button.primary:hover {
    background: #285c31 !important;
}

button.secondary {
    border-color: #2f6b3a !important;
    color: #2f6b3a !important;
}

/*these selectors are my attempt to hide the default gradio footer (it can change between versions) */
footer { display: none !important; }
.gradio-container footer { display: none !important; }
div[class*="footer"] { display: none !important; }
div[class*="footer-container"] { display: none !important; }
"""

#i define the theme once and pass it into launch() (gradio 6+ prefers it this way)
THEME = gr.themes.Soft(
    primary_hue="green",
    secondary_hue="emerald",
    neutral_hue="stone"
)

#this function builds the HTML card for the prediction + confidence + top-k table
def build_result_html(labels, probs):
    pred = labels[0]
    conf = probs[0]
    status_class = "status-good"
    status_text = "High-confidence prediction"

    #if the model is unsure, i change the headline and the pill styling
    if conf < UNSURE_THRESHOLD:
        headline = f"UNSURE (top guess: {pred})"
        status_class = "status-unsure"
        status_text = "Low-confidence prediction"
    else:
        headline = pred.capitalize()

    #i build the rows for the top-k table
    rows = ""
    for rank, (label, prob) in enumerate(zip(labels, probs), start=1):
        rows += f"""
        <tr>
            <td>{rank}</td>
            <td>{label.capitalize()}</td>
            <td>{prob*100:.1f}%</td>
        </tr>
        """

    html = f"""
    <div class="result-card">
        <div class="status-pill {status_class}">{status_text}</div>
        <div class="result-grid">
            <div class="metric-box">
                <div class="metric-label">Predicted Category</div>
                <div class="metric-value">{headline}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Confidence</div>
                <div class="metric-value">{conf*100:.1f}%</div>
            </div>
        </div>

        <div class="topk-title">Top {len(labels)} Predictions</div>
        <table class="topk-table">
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Class</th>
                    <th>Probability</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </div>
    """
    return html

#this function builds the guidance card (and adds an extra note if confidence is low)
def build_guidance_html(labels, probs):
    pred = labels[0]
    conf = probs[0]
    guidance = GUIDANCE.get(pred, "No guidance available.")

    extra_note = ""
    if conf < UNSURE_THRESHOLD:
        extra_note = (
            "<p><strong>Suggestion:</strong> Try retaking the photo with better lighting, "
            "a cleaner background, or a closer crop of the object.</p>"
        )

    return f"""
    <div class="guidance-card">
        <h3>♻️ Disposal Guidance</h3>
        <p><strong>Recommended class:</strong> {pred.capitalize()}</p>
        <p>{guidance}</p>
        {extra_note}
    </div>
    """

#this is the function gradio calls when the user clicks "Classify Item"
def predict_ui(img: Image.Image):
    #if they haven't provided an image yet, i show a friendly placeholder state
    if img is None:
        empty_result = """
        <div class="result-card">
            <div class="status-pill status-unsure">Waiting for image</div>
            <div class="metric-box">
                <div class="metric-label">Status</div>
                <div class="metric-value">Upload an item photo to begin</div>
            </div>
        </div>
        """
        empty_guidance = """
        <div class="guidance-card">
            <h3>♻️ Disposal Guidance</h3>
            <p>Upload an image to receive a classification, confidence score, and recycling guidance.</p>
        </div>
        """
        return empty_result, empty_guidance

    #i run my model inference here and then convert it into the two html panels
    labels, probs = predict_topk(model, class_names, preprocess, img, k=TOPK)
    return build_result_html(labels, probs), build_guidance_html(labels, probs)

#i scan the examples directory and build a list gradio can display as clickable examples
examples = []
if os.path.isdir(EXAMPLES_DIR):
    for fn in sorted(os.listdir(EXAMPLES_DIR)):
        if fn.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            examples.append([os.path.join(EXAMPLES_DIR, fn)])

#this is the main UI layout using gradio blocks
with gr.Blocks(title="Personal Recycling Assistant") as demo:

    #this is the hero banner at the top
    gr.HTML("""
    <div id="hero">
        <h1>🌎 Personal Recycling Assistant ♻️</h1>
        <p>
            Upload a photo of an item to classify it into one of 6 waste categories:
            <strong>cardboard, glass, metal, paper, plastic, trash</strong>.
            This assistant displays confidence, top-3 predictions, and disposal guidance.
        </p>
    </div>
    """)

    #these are the three feature cards i show under the hero banner
    with gr.Row():
        gr.HTML("""
        <div class="feature-card">
            <h3>📷 Image Classification</h3>
            <p>Uses a fine-tuned ResNet-18 model to identify waste categories from photos.</p>
        </div>
        """)
        gr.HTML("""
        <div class="feature-card">
            <h3>📊 Confidence + Top-3</h3>
            <p>Displays the most likely classes so the user can judge uncertainty.</p>
        </div>
        """)
        gr.HTML("""
        <div class="feature-card">
            <h3>♻️ Practical Guidance</h3>
            <p>Returns category-specific recycling guidance to support better disposal choices.</p>
        </div>
        """)

    gr.Markdown("### Upload or choose an example image")

    #left side is the image input + buttons + examples
    #right side is the result panels
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group(elem_classes=["panel-card"]):
                img_input = gr.Image(type="pil", label="Item Photo")
                with gr.Row():
                    submit_btn = gr.Button("Classify Item", variant="primary")
                    clear_btn = gr.ClearButton([img_input], value="Clear", variant="secondary")

                #i only show the examples widget if i actually found images in the folder
                if examples:
                    gr.Examples(
                        examples=examples,
                        inputs=img_input,
                        label="Example Images"
                    )

        with gr.Column(scale=1):
            result_html = gr.HTML()
            guidance_html = gr.HTML()

    #this wires the button click to my prediction function
    submit_btn.click(fn=predict_ui, inputs=img_input, outputs=[result_html, guidance_html])

    #this is a small footer note that matches the overall project vibe
    gr.HTML('<div class="footer-note">ECE 570 Final Project • AI-Powered Recycling Classification Prototype</div>')

#this actually launches the web app and applies my css/theme cleanly
if __name__ == "__main__":
    demo.launch(css=CUSTOM_CSS, theme=THEME)
