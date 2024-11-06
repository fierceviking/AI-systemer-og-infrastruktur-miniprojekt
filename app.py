# app.py
import joblib
from flask import Flask, request, jsonify

# Load the model
model = joblib.load("SVC.pkl")

app = Flask(__name__)

@app.route("/predict", methods=["POST", "GET"])
def predict():
    data = request.get_json()
    prediction = model.predict([data["features"]])
    return jsonify({"prediction": prediction.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)