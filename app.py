# app.py
import os, io, json
from pathlib import Path
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, abort
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import models

# ---------- CONFIG ----------
PROJECT_DIR = Path(__file__).parent
SAVED_MODELS_DIR = PROJECT_DIR / 'saved_models'
WEIGHTS_PATH = SAVED_MODELS_DIR / 'cnn_v1.pth'
CLASSES_PATH = SAVED_MODELS_DIR / 'classes.json'

# Adjust this to your dataset root used by preprocess.load_data
DATA_DIR = Path(r"C:\Endsemlab\DL\DATASET\masterCategory\SportsClassification")  # <<-- change if needed

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # matches training
])
def load_model(weights_path: Path, classes_path: Path):
    if not weights_path.exists() or not classes_path.exists():
        raise FileNotFoundError("Weights or classes.json not found. Run training (main.py) first.")
    with open(classes_path, 'r') as f:
        classes = json.load(f)
    num_classes = len(classes)
    # prefer explicit CNN
    model = models.CNNClassifier_regularization(num_classes=num_classes)
    state = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model, classes

def predict_image_bytes(model, image_bytes, topk=3):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    inp = INFERENCE_TRANSFORM(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(inp)
        if outputs.shape[1] == 1:
            prob = torch.sigmoid(outputs).cpu().item()
            label_idx = 1 if prob > 0.5 else 0
            return [(label_idx, float(prob))]
        else:
            probs = F.softmax(outputs, dim=1)
            top_probs, top_idx = probs.topk(topk, dim=1)
            top_probs = top_probs.cpu().numpy()[0].tolist()
            top_idx = top_idx.cpu().numpy()[0].tolist()
            return list(zip(top_idx, top_probs))

def get_images_for_category(category_name, limit=60):
    matches = []
    for root, dirs, files in os.walk(DATA_DIR):
        if os.path.basename(root).lower() == category_name.lower():
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    rel = os.path.relpath(os.path.join(root, f), DATA_DIR)
                    matches.append(rel.replace('\\','/'))
    if not matches:
        for root, dirs, files in os.walk(DATA_DIR):
            if category_name.lower() in root.lower():
                for f in files:
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        rel = os.path.relpath(os.path.join(root, f), DATA_DIR)
                        matches.append(rel.replace('\\','/'))
    return matches[:limit]

# ---------- FLASK APP ----------
app = Flask(__name__, template_folder='templates')

MODEL, CLASSES = load_model(WEIGHTS_PATH, CLASSES_PATH)

@app.route('/')
def index():
    return render_template('index.html', classes=CLASSES)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(url_for('index'))
    file = request.files['image']
    img_bytes = file.read()

    results = predict_image_bytes(MODEL, img_bytes, topk=5)
    preds = []
    for idx, prob in results:
        label = CLASSES[idx]
        preds.append({'label': label, 'confidence': round(prob, 4)})

    # Get related images based on top predicted label
    top_label = preds[0]['label']
    related_images = get_images_for_category(top_label, limit=12)

    return render_template(
        'result.html',
        predictions=preds,
        top_label=top_label,
        related_images=related_images
    )


@app.route('/browse')
def browse():
    category = request.args.get('category', '')
    if not category:
        return render_template('index.html', classes=CLASSES, message='Type a category to browse images')
    images = get_images_for_category(category, limit=60)
    return render_template('browse.html', category=category, images=images)

@app.route('/data/<path:filename>')
def serve_data_image(filename):
    safe_path = (DATA_DIR / filename).resolve()
    if not str(safe_path).startswith(str(DATA_DIR.resolve())):
        abort(403)
    return send_from_directory(str(DATA_DIR), filename)



# ...rest of function
if __name__ == '__main__':
    print(f"Starting app on {DEVICE}. Serving dataset from: {DATA_DIR}")
    app.run(host='0.0.0.0', port=5000, debug=True)
