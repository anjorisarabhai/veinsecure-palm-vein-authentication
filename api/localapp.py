from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
from collections import defaultdict
import os
import sys

# ----- Paths -----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
LOGS_FOLDER = os.path.join(BASE_DIR, "logs")
MODEL_PATH = os.path.join(BASE_DIR, "..", "results", "models", "final_model.h5")
HELPER_PATH = os.path.abspath(os.path.join(BASE_DIR, '..'))
sys.path.append(HELPER_PATH)

from utils.helperslocal import predict_image

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LOGS_FOLDER, exist_ok=True)

# ----- Flask App -----
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB

# ----- Load Model -----
try:
    model = load_model(MODEL_PATH)
    class_names = [f"{i:03d}" for i in range(1, 42)]
except Exception as e:
    print(f"Model load failed: {e}")
    model = None
    class_names = [f"{i:03d}" for i in range(1, 42)]

# ----- Logging -----
def log_auth_attempt(claimed_id, predicted_id, status, note=None):
    log_file = os.path.join(LOGS_FOLDER, "login_attempts.log")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] Claimed: {claimed_id}, Predicted: {predicted_id}, Access: {status}"
    if note:
        entry += f" | Note: {note}"
    with open(log_file, "a") as f:
        f.write(entry + "\n")

# ----- Rate Limiting (Lockout) -----
failed_attempts = defaultdict(lambda: {"count": 0, "last_failed_time": None})
LOCKOUT_THRESHOLD = 5
LOCKOUT_TIME = timedelta(seconds=60)

# ----- Helpers -----
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ----- Routes -----
@app.route('/')
def index():
    return render_template("index.html")

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

    # 🔒 Rate-limiting check
    user_state = failed_attempts[claimed_identity]
    if user_state["count"] >= LOCKOUT_THRESHOLD:
        time_diff = datetime.now() - user_state["last_failed_time"]
        if time_diff < LOCKOUT_TIME:
            wait_time = int((LOCKOUT_TIME - time_diff).total_seconds())
            response["error"] = f"Too many failed attempts. Try again in {wait_time} seconds."
            log_auth_attempt(claimed_identity, "N/A", "LOCKED", f"Anomaly detected, retry allowed in {wait_time} sec")
            return jsonify(response), 429  # Too Many Requests

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
            failed_attempts[claimed_identity] = {"count": 0, "last_failed_time": None}
            log_auth_attempt(claimed_identity, class_name, "GRANTED")
        else:
            response["access_granted"] = False
            failed_attempts[claimed_identity]["count"] += 1
            failed_attempts[claimed_identity]["last_failed_time"] = datetime.now()
            log_auth_attempt(claimed_identity, class_name, "DENIED")

    except Exception as e:
        response["error"] = f"Prediction failed: {str(e)}"
        log_auth_attempt(claimed_identity, "ERROR", "FAILED")
        return jsonify(response), 500

    return jsonify(response), 200

# ----- Run App -----
if __name__ == "__main__":
    app.run(debug=True)