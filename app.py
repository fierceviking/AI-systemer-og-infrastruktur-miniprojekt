from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import onnx
import uvicorn
from typing import List

from variables import Variables

# Load the model and scaler
model = joblib.load("SVC.pkl")
scaler = joblib.load("scaler.pkl")

app = FastAPI()
# Load model scalar
pickle_in = open("artifacts/model-scaler.pkl", "rb")
scaler = pickle.load(pickle_in)
# Load the model
sess = rt.InferenceSession("artifacts/svc.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name