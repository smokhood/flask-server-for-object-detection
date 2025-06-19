from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import os
import gdown

app = Flask(__name__)

# Download model if not present
model_path = "best.pt"
gdrive_file_id = "YOUR_FILE_ID_HERE"  # <-- Replace with your real file ID

if not os.path.exists(model_path):
    print("Downloading YOLOv8 model...")
    gdown.download(f"https://drive.google.com/uc?id={gdrive_file_id}", model_path, quiet=False)

# Load model
model = YOLO(model_path)

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = Image.open(request.files['image'].stream)
    results = model(image)

    output = []
    for r in results:
        for box in r.boxes:
            cls = r.names[int(box.cls[0])]
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            output.append({
                "class": cls,
                "confidence": conf,
                "bbox": xyxy
            })

    return jsonify(output)
