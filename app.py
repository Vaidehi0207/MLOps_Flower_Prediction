from flask import Flask, request, jsonify
import joblib
import os
from pathlib import Path

app = Flask(__name__)
MODEL_PATH = Path("artifacts/model.pkl")

# Load the model globally
if not MODEL_PATH.exists():
    # convenience: train if model missing
    import train as _train
    _train.main()

model = joblib.load(MODEL_PATH)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "features" not in data:
        return jsonify({"error": "send JSON with key 'features'"}), 400
    
    features = data["features"]
    try:
        # LogisticRegression expects a 2D array: [ [f1, f2, f3, f4] ]
        pred = model.predict([features])
        return jsonify({"prediction": int(pred[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Ensure port is an integer and remove the trailing dot from your snippet
    app.run(host="0.0.0.0", port=5001)