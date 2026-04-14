import os
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
import json
import traceback

app = Flask(__name__)

# ✅ USE YOUR NEW TRAINED MODEL
MODEL_PATH = "plant_model.keras"

model = None  # GLOBAL MODEL

# ============================================
# LOAD MODEL (LAZY + SAFE)
# ============================================

def get_model():
    global model

    if model is None:
        try:
            print("📦 Loading model...")

            model = tf.keras.models.load_model(
                MODEL_PATH,
                compile=False
            )

            print("🔥 Warming up model...")
            dummy = np.zeros((1, 224, 224, 3))
            model.predict(dummy, verbose=0)

            print("✅ Model ready")

        except Exception as e:
            print("❌ MODEL LOAD ERROR:")
            traceback.print_exc()
            model = None

    return model


# ============================================
# LOAD CLASSES
# ============================================

try:
    with open("classes.json", "r") as f:
        class_indices = json.load(f)
    classes = {v: k for k, v in class_indices.items()}
    print("✅ Classes loaded")
except Exception as e:
    print("❌ Failed to load classes.json")
    traceback.print_exc()
    classes = {}


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
    model = get_model()

    if model is None:
        return "Model not loaded", 0

    try:
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array, verbose=0)[0]
        top_indices = predictions.argsort()[-5:][::-1]

        crop_type = normalize(crop_type)
        filtered = []

        for i in top_indices:
            class_name = classes.get(i, "Unknown")
            confidence = float(predictions[i])
            class_norm = normalize(class_name)

            if crop_type and crop_type in class_norm:
                filtered.append((class_name, confidence))

        if filtered:
            best_class, best_conf = max(filtered, key=lambda x: x[1])
        else:
            best_class = classes.get(top_indices[0], "Unknown")
            best_conf = float(predictions[top_indices[0]])

            if best_conf < 0.3:
                return "Unknown", best_conf

        disease_name = best_class.replace("_", " ").title()

        return disease_name, best_conf

    except Exception as e:
        print("❌ PREDICTION ERROR:")
        traceback.print_exc()
        return "Prediction error", 0


# ============================================
# ROUTES
# ============================================

@app.route("/")
def home():
    return "ML Service Running ✅"


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
        print("❌ API ERROR:")
        traceback.print_exc()

        return jsonify({
            "error": str(e)
        }), 500


# ============================================
# RUN
# ============================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)