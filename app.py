from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
import json

app = Flask(__name__)

# ===============================
# LOAD MODEL
# ===============================
model = tf.keras.models.load_model("plant_model.keras")

# Warmup (VERY IMPORTANT - avoids slow first request)
print("🔥 Warming up model...")
dummy = np.zeros((1, 224, 224, 3))
model.predict(dummy)
print("✅ Model ready")

# ===============================
# LOAD CLASSES
# ===============================
with open("classes.json", "r") as f:
    class_indices = json.load(f)

classes = {v: k for k, v in class_indices.items()}


# ===============================
# HELPER: NORMALIZE TEXT
# ===============================
def normalize(text):
    if not text:
        return ""
    return text.lower().replace("_", " ").strip()


# ===============================
# PREDICTION FUNCTION
# ===============================
def predict(img, crop_type=None):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]

    # Top 5 predictions
    top_indices = predictions.argsort()[-5:][::-1]

    crop_type = normalize(crop_type)

    filtered = []

    print("\n🔍 Incoming cropType:", crop_type)
    print("🔝 Top predictions:")

    for i in top_indices:
        class_name = classes[i]
        confidence = float(predictions[i])
        class_norm = normalize(class_name)

        print(f"{class_name} → {confidence:.4f}")

        # Flexible match
        if crop_type and crop_type in class_norm:
            filtered.append((class_name, confidence))

    # ===============================
    # DECISION LOGIC
    # ===============================
    if filtered:
        best_class, best_conf = max(filtered, key=lambda x: x[1])
        print("✅ Filtered match used:", best_class)

    else:
        # fallback → use top prediction
        best_class = classes[top_indices[0]]
        best_conf = float(predictions[top_indices[0]])

        print("⚠️ No crop match → fallback:", best_class)

        # If confidence too low → Unknown
        if best_conf < 0.3:
            print("❌ Low confidence → returning Unknown")
            return "Unknown", best_conf

    disease_name = best_class.replace("_", " ").title()

    return disease_name, best_conf


# ===============================
# API ROUTE
# ===============================
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
        return jsonify({
            "error": "Prediction failed"
        }), 500


# ===============================
# RUN SERVER
# ===============================
if __name__ == "__main__":
    app.run(port=8000)