from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
import sklearn
import xgboost
import onnxruntime as ort
import logging
import uvicorn

app = FastAPI()

# Loading the ONNX model
model_path = "deploy/xgb_model.onnx"  # Update based on your chosen model's file name
try:
    model = ort.InferenceSession(model_path)
except Exception as e:
    logging.error(f"Error loading ONNX model: {e}")
    raise e

# Loading the scaler
model_path = "deploy/scaler_model.onnx"  # Update based on your chosen model's file name
try:
    scaler = ort.InferenceSession(model_path)
except Exception as e:
    logging.error(f"Error loading ONNX model: {e}")
    raise e

class PizzaSalesFeatures(BaseModel):
    hour: float
    day: float
    month: float

@app.get("/")
def index():
    return {
        "Hello": "Welcome to the pizza sales prediction service! Access the API docs at /docs."
    }

@app.post("/predict")
def predict_pizza_sales(features: PizzaSalesFeatures):
    # Prepare the input data
    input_data = np.array([[features.hour, features.day, features.month]], dtype=np.float32)
    print(f"Input data for prediction: {input_data}")

    try:
        # Perform the prediction
        predicted_sales = pipeline.predict(input_data)
        
        # Convert numpy.float32 to standard float for JSON serialization
        predicted_sales_value = float(predicted_sales[0])

        # Return the response with the converted float value
        return {
            "input": {
                "hour": features.hour,
                "day": features.day,
                "month": features.month
            },
            "predicted_sales": predicted_sales_value
        }
    except Exception as e:
        print(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)