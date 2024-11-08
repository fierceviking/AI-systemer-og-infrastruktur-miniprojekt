from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
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
    # Prepare the input data as a numpy array
    input_data = np.array([[features.hour, features.day, features.month]], dtype=np.float32)
    print(f"Original input data for prediction: {input_data}")

    try:
        # Perform the prediction using the ONNX model
        model_input_name = model.get_inputs()[0].name
        model_output_name = model.get_outputs()[0].name
        predicted_sales = model.run([model_output_name], {model_input_name: input_data})[0]

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