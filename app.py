import joblib
from flask import Flask, request, jsonify

# Load the trained model
model = joblib.load("SVC.pkl")

app = Flask(__name__)

# Root URL for a simple welcome message (GET request)
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to the Credit Defaults Prediction API. Use POST requests to /predict to get predictions."})

# Prediction endpoint (POST request)
@app.route("/predict", methods=["GET", "POST"])
def predict():
    # Check if the request content type is JSON
    if request.content_type != 'application/json':
        return jsonify({"error": "Unsupported Media Type. Content-Type must be application/json"}), 415

    # Try to get JSON data from the request
    data = request.get_json(silent=True)
    if data is None or "features" not in data:
        return jsonify({"error": "Invalid input. JSON payload with 'features' key is required."}), 400

    # Get the features from the JSON data
    features = data["features"]

    try:
        # Perform the prediction
        prediction = model.predict([features])
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        # Handle errors during prediction
        return jsonify({"error": str(e)}), 500

# Start the Flask application
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)