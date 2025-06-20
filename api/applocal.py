from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import os
import sys

# ✅ Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.helperslocal import predict_image

# Load model
model = load_model("results/models/final_model.h5")

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return "✅ Flask App Running! (UI coming soon)"

if __name__ == '__main__':
    app.run(debug=True)
