import os
import requests
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
import json

app = Flask(__name__)

# ============================================
# MODEL DOWNLOAD
# ============================================

MODEL_PATH = "plant_model.h5"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1u67A5wWdcu9b5PPdDvs6OufIecfY4diG"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("⬇️ Downloading model...")
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        print("✅ Model downloaded")

download_model()

# ============================================
# LOAD MODEL
# ============================================

model = tf.keras.models.load_model(MODEL_PATH)

print("🔥 Warming up model...")
dummy = np.zeros((1, 224, 224, 3))
model.predict(dummy)
print("✅ Model ready")

# ============================================
# LOAD CLASSES
# ============================================

with open("classes.json", "r") as f:
    class_indices = json.load(f)

classes = {v: k for k, v in class_indices.items()}

# ============================================
# NORMALIZE
# ============================================

def normalize(text):
    if not text:
        return ""
    return text.lower().replace("_", " ").strip()

# ============================================
# PREDICTION
# ============================================

def predict(img, crop_type=None):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]
    top_indices = predictions.argsort()[-5:][::-1]

    crop_type = normalize(crop_type)
    filtered = []

    for i in top_indices:
        class_name = classes[i]
        confidence = float(predictions[i])
        class_norm = normalize(class_name)

        if crop_type and crop_type in class_norm:
            filtered.append((class_name, confidence))

    if filtered:
        best_class, best_conf = max(filtered, key=lambda x: x[1])
    else:
        best_class = classes[top_indices[0]]
        best_conf = float(predictions[top_indices[0]])

        if best_conf < 0.3:
            return "Unknown", best_conf

    disease_name = best_class.replace("_", " ").title()

    return disease_name, best_conf

# ============================================
# API
# ============================================

@app.route("/predict", methods=["POST"])
def predict_api():
    try:
        file = request.files.get("image")
        crop_type = request.form.get("cropType")

        if not file:
            return jsonify({"error": "No image provided"}), 400

        img = Image.open(file.stream).convert("RGB")

        disease, confidence = predict(img, crop_type)

        return jsonify({
            "disease": disease,
            "confidence": confidence
        })

    except Exception as e:
        print("❌ Prediction Error:", str(e))
        return jsonify({"error": "Prediction failed"}), 500

# ============================================
# HEALTH CHECK (IMPORTANT)
# ============================================

@app.route("/")
def home():
    return "ML Service Running ✅"

# ============================================
# RUN (RENDER FIX)
# ============================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)