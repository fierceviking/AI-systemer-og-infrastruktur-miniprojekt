import onnxruntime as ort
import numpy as np

model_path = "deploy/ridge_model.onnx"
session = ort.InferenceSession(model_path)

# Test data input
features = np.array([[15, 6, 14, 100, 50, 30, 20]]).astype(np.float32)
input_name = session.get_inputs()[0].name
pred = session.run(None, {input_name: features})

# Extract the prediction safely
print("Prediction:", pred[0][0].item())
