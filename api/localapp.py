from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
import os
import sys
from datetime import datetime

# Get current script's base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Set paths
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
LOGS_FOLDER = os.path.join(BASE_DIR, "logs")
MODEL_PATH = os.path.join(BASE_DIR, "..", "results", "models", "final_model.h5")
HELPER_PATH = os.path.abspath(os.path.join(BASE_DIR, '..'))

# Add helper path
sys.path.append(HELPER_PATH)
from utils.helperslocal import predict_image  # Local version of predict_image

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LOGS_FOLDER, exist_ok=True)

# Initialize Flask app
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB

# Load model and class names
try:
    model = load_model(MODEL_PATH)
    class_names = [f"{i:03d}" for i in range(1, 42)]
except Exception as e:
    print(f"Failed to load model: {e}")
    model = None
    class_names = [f"{i:03d}" for i in range(1, 42)]

# 🔸 Simple log function (write directly to .log file)
def log_auth_attempt(claimed_id, predicted_id, status):
    log_file_path = os.path.join(LOGS_FOLDER, "login_attempts.log")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file_path, "a") as f:
        f.write(f"[{timestamp}] Claimed: {claimed_id}, Predicted: {predicted_id}, Access: {status}\n")

# Utility
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route
@app.route('/')
def index():
    return render_template("index.html")

# Authenticate route
@app.route('/authenticate', methods=["POST"])
def authenticate():
    response = {
        "prediction": None,
        "confidence": None,
        "filename": None,
        "error": None,
        "access_granted": False
    }

    if model is None:
        response["error"] = "Model not loaded."
        return jsonify(response), 500

    claimed_identity = request.form.get("claimed_identity")
    if not claimed_identity:
        response["error"] = "No identity selected."
        return jsonify(response), 400
    response["claimed_identity"] = claimed_identity

    if "file" not in request.files:
        response["error"] = "No file uploaded."
        return jsonify(response), 400
    
    file = request.files["file"]
    if file.filename == "":
        response["error"] = "No file selected."
        return jsonify(response), 400

    if not allowed_file(file.filename):
        response["error"] = "Unsupported file type."
        return jsonify(response), 415

    filename = file.filename
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)
    response["filename"] = filename

    try:
        class_id, class_name, confidence = predict_image(file_path, model, class_names)
        response["prediction"] = class_name
        response["confidence"] = round(confidence * 100, 2)

        if claimed_identity == class_name:
            response["access_granted"] = True
            log_auth_attempt(claimed_identity, class_name, "GRANTED")
        else:
            response["access_granted"] = False
            log_auth_attempt(claimed_identity, class_name, "DENIED")

    except Exception as e:
        response["error"] = f"Prediction failed: {str(e)}"
        log_auth_attempt(claimed_identity, "ERROR", "FAILED")
        return jsonify(response), 500

    return jsonify(response), 200

# Run app
if __name__ == "__main__":
    app.run(debug=True)
