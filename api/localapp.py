from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import os
import sys
import logging

# ✅ Fix import path for local helpers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.helperslocal import predict_image

# 🔹 Load model & class names
model = load_model(r"C:\Users\Dell\Desktop\anjori\veinsecure-palm-vein-authentication\results\models\final_model.h5")
class_names = [f"{i:03d}" for i in range(1, 42)]  # '001' to '041'

# 🔧 App Config
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
TEMPLATES_FOLDER = "templates"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# 🪵 Logging
logging.basicConfig(filename="logs/api.log", level=logging.INFO)

# 🔹 Route: Home Page (upload form)
@app.route('/')
def index():
    return render_template("index.html")

# 🔹 Route: Prediction
@app.route("/predict", methods=["POST"])
def predict():
    prediction = None
    confidence = None
    filename = None
    error = None

    if "file" not in request.files:
        error = "No file part"
    else:
        file = request.files["file"]
        if file.filename == "":
            error = "No selected file"
        else:
            filename = file.filename
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            try:
                class_id, class_name, confidence = predict_image(file_path, model, class_names)
                prediction = class_name
                logging.info(f"Predicted: {class_name} (Confidence: {confidence:.2f})")
                confidence = round(confidence * 100, 2)  # % format
            except Exception as e:
                error = str(e)
                logging.error(f"Prediction error: {error}")

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence,
                           filename=filename,
                           error=error)

# 🔧 Run locally
if __name__ == '__main__':
    app.run(debug=True)

