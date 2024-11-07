from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import uvicorn
from typing import List

# Load the model and scaler
model = joblib.load("SVC.pkl")
scaler = joblib.load("scaler.pkl")

app = FastAPI()

# Set up templates for serving HTML files
templates = Jinja2Templates(directory="templates")

# Route to serve the HTML page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # By default, show the page without a prediction
    return templates.TemplateResponse("index.html", {"request": request, "pred": None})

# Prediction endpoint to handle form submission
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, features: List[float] = Form(...)):
    try:
        # Scale the features if needed and perform prediction
        scaled_features = scaler.transform([features])
        prediction = model.predict(scaled_features)
        
        # Pass the prediction to the template
        return templates.TemplateResponse("index.html", {"request": request, "pred": prediction[0]})
    except Exception as e:
        # Handle errors during prediction
        raise HTTPException(status_code=500, detail=str(e))

# Start the FastAPI application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)