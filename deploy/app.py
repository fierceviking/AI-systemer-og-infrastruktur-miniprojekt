from fastapi import FastAPI, HTTPException
import numpy as np
import onnxruntime as ort
import pandas as pd
import logging
import uvicorn

app = FastAPI()

# Load the ONNX model
model_path = "ridge_model.onnx"  # Update based on your chosen model's file name
try:
    session = ort.InferenceSession(model_path)
except Exception as e:
    logging.error(f"Error loading ONNX model: {e}")
    raise e

@app.get("/")
async def root():
    return {"message": "Hello, World!"}

@app.post("/predict/")
async def predict(data: dict):
    try:
        # Extract and prepare the input data
        logging.info(f"Received data: {data}")
        features = np.array([data[key] for key in ['day', 'month', 'hour']]).reshape(1, -1)
        input_name = session.get_inputs()[0].name
        pred = session.run(None, {input_name: features.astype(np.float32)})

        logging.info(f"Model prediction: {pred[0][0]}")
        # Ensure the response is serializable
        return {"prediction": pred[0][0].item()} 
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)