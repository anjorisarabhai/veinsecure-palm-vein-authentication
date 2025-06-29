from flask import Flask, request, render_template, jsonify # Added jsonify
from tensorflow.keras.models import load_model
import os
import sys
import logging
from datetime import datetime # Added for logging timestamps

# ✅ Fix import path for local helpers
# Ensure this path correctly points to your 'utils' directory if it's not directly in 'api'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.helperslocal import predict_image

# 🔹 Load model & class names
try:
    # Ensure this path is correct for your local setup
    model = load_model(r"C:\Users\Dell\Desktop\anjori\veinsecure-palm-vein-authentication\results\models\final_model.h5")
    class_names = [f"{i:03d}" for i in range(1, 42)]  # '001' to '041'
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    model = None
    class_names = [f"{i:03d}" for i in range(1, 42)] # Fallback class names
    # You might want to exit or handle this more gracefully in production

# 🔧 App Config
app = Flask(__name__)
# For simplicity, remove secret_key if not using Flask sessions/flash messages that require it
# app.secret_key = os.environ.get("SESSION_SECRET", "fallback-secret-key-for-development")

UPLOAD_FOLDER = "static/uploads"
TEMPLATES_FOLDER = "templates" # Not strictly needed if not rendering Flask templates from backend for API calls
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'} # Define allowed extensions

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# 🪵 Logging (Enhanced to match frontend's error logging)
logging.basicConfig(
    level=logging.DEBUG, # Set to DEBUG to capture more info
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api.log'),
        logging.StreamHandler() # Also log to console
    ]
)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def log_authentication_attempt(claimed_id, predicted_id, match_status, filename, error_msg=None):
    """Log authentication attempt with all relevant details"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp} | Claimed: {claimed_id} | Predicted: {predicted_id} | Status: {match_status} | File: {filename}"
    if error_msg:
        log_entry += f" | Error: {error_msg}"
    logging.info(f"AUTH_ATTEMPT: {log_entry}")


# 🔹 Route: Home Page (initial render)
@app.route('/')
def index():
    # In a real setup, this would serve your index.html from the templates folder
    # For now, it's less critical as index.html is self-contained.
    return render_template("index.html")

# 🔹 Route: Authentication (Changed from /predict to /authenticate)
@app.route("/authenticate", methods=["POST"])
def authenticate():
    response_data = {
        'prediction': None,
        'filename': None,
        'error': None,
        'claimed_identity': None,
        'access_granted': False,
        'match_status': None
    }

    # Check if model is loaded
    if model is None:
        response_data['error'] = "Authentication system not available. Please contact administrator."
        log_authentication_attempt("N/A", "N/A", "ERROR", "N/A", response_data['error'])
        return jsonify(response_data), 500 # Internal Server Error

    # Get claimed identity
    claimed_identity = request.form.get('claimed_identity')
    if not claimed_identity:
        response_data['error'] = "Please select your identity before uploading palm vein scan."
        log_authentication_attempt("N/A", "N/A", "ERROR", "N/A", response_data['error'])
        return jsonify(response_data), 400 # Bad Request

    response_data['claimed_identity'] = claimed_identity

    # Check file upload
    if "file" not in request.files:
        response_data['error'] = "No file uploaded. Please select a palm vein image."
        log_authentication_attempt(claimed_identity, "N/A", "ERROR", "N/A", response_data['error'])
        return jsonify(response_data), 400 # Bad Request

    file = request.files["file"]
    if file.filename == "":
        response_data['error'] = "No file selected. Please choose a palm vein image."
        log_authentication_attempt(claimed_identity, "N/A", "ERROR", "N/A", response_data['error'])
        return jsonify(response_data), 400 # Bad Request

    if not allowed_file(file.filename):
        response_data['error'] = f"Invalid file type. Please upload one of: {', '.join(ALLOWED_EXTENSIONS)}"
        log_authentication_attempt(claimed_identity, "N/A", "ERROR", file.filename, response_data['error'])
        return jsonify(response_data), 415 # Unsupported Media Type

    # Save uploaded file
    filename = file.filename # Using original filename for simplicity
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    try:
        file.save(file_path)
        response_data['filename'] = filename
    except Exception as e:
        response_data['error'] = f"Failed to save uploaded file: {str(e)}"
        logging.error(f"File save error: {response_data['error']}")
        log_authentication_attempt(claimed_identity, "N/A", "ERROR", filename, response_data['error'])
        return jsonify(response_data), 500 # Internal Server Error

    # Perform prediction
    try:
        class_id, class_name, confidence = predict_image(file_path, model, class_names)
        response_data['prediction'] = class_name
        # Removed confidence from response_data based on your previous request

        # Verify identity (claimed vs predicted)
        if claimed_identity == class_name:
            response_data['access_granted'] = True
            response_data['match_status'] = "MATCH"
        else:
            response_data['access_granted'] = False
            response_data['match_status'] = "MISMATCH"

        # Log authentication attempt
        log_authentication_attempt(
            claimed_identity,
            class_name,
            response_data['match_status'],
            filename
        )

    except Exception as e:
        response_data['error'] = f"Authentication failed during prediction: {str(e)}"
        logging.error(f"Prediction error: {response_data['error']}")
        log_authentication_attempt(
            claimed_identity,
            "ERROR",
            "ERROR",
            filename,
            response_data['error']
        )

    # Return JSON response
    return jsonify(response_data), 200 # OK

# 🔧 Run locally
if __name__ == '__main__':
    app.run(debug=True) # Ensure host is 0.0.0.0 for broader access