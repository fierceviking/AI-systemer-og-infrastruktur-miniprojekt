from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

model_path = "SVR_pipeline.joblib"
if not os.path.exists(model_path):
    raise RuntimeError(f"Model file '{model_path}' not found. Please ensure 'system.py' has generated it correctly.")

pipeline = joblib.load(model_path)

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
    input_data = np.array([[features.hour, features.day, features.month]], dtype=np.float32)
    print(f"Input data for prediction: {input_data}")

    try:
        predicted_sales = pipeline.predict(input_data)
        
        return {
            "input": {
                "hour": features.hour,
                "day": features.day,
                "month": features.month
            },
            "predicted_sales": predicted_sales[0]
        }
    except Exception as e:
        print(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)