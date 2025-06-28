from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import os
import sys
import logging

# ✅ Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.helperslocal import predict_image  # Local-only version

# 🔹 Load model
model = load_model(r"C:\Users\Dell\Desktop\anjori\veinsecure-palm-vein-authentication\results\models\final_model.h5")
class_names = [f"{i:03d}" for i in range(1, 42)]  # '001' to '041'

# 🔧 Flask App Config
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 🪵 Logging Setup
os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename="logs/api.log", level=logging.INFO)

# 🔹 Route: Home Page with Upload Form
@app.route('/')
def index():
    return render_template("index.html")  # Ensure templates/index.html exists

# 🔹 Route: Prediction Logic
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    try:
        class_id, class_name, confidence = predict_image(file_path, model, class_names)
        logging.info(f"Predicted: {class_name} (Confidence: {confidence:.2f})")

        return jsonify({
            "predicted_class": class_name,
            "confidence": round(confidence, 2),
            "class_id": int(class_id)
        })

    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        return jsonify({"error": str(e)}), 500

# 🔧 Run Server
if __name__ == '__main__':
    app.run(debug=True)


