# imports
import pandas as pd
import numpy
from skl2onnx import convert_sklearn
from onnxmltools import convert_xgboost
from skl2onnx.common.data_types import FloatTensorType
import os
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


def save_model(pipeline, x_train, model_type):
    # Define initial type for the conversion
    initial_type = [('float_input', FloatTensorType([None, x_train.shape[1]]))]

    if model_type == "XGB":
        # Convert the pipeline to ONNX format
        model = convert_xgboost(pipeline, initial_types=initial_type)
    else:
        model = convert_sklearn(pipeline, initial_types=initial_type)

    # Ensure the output directory exists
    output_dir = "deploy"
    os.makedirs(output_dir, exist_ok=True)

    # Save the model within the deploy folder
    model_path = os.path.join(output_dir, f"{model_type.lower()}_model.onnx")
    with open(model_path, "wb") as f:
        f.write(model.SerializeToString())
    print(f"Model saved to {model_path}")

# main function
def main():
    # loading the data
    df = pd.read_csv('data.csv', parse_dates=['order_timestamp_hour'], index_col='order_timestamp_hour')

    cols = ["hour", "day", "month"]
    target = 'total_sales'
    x = df[cols]
    y = df[target]

    split = int(len(x)*0.8)

    x_train, x_test = x.values[:split], x.values[split:]
    y_train, y_test = y.values[:split], y.values[split:]

    model = xgb.XGBRegressor(
        objective = 'reg:squarederror',
        max_depth=5,
        learning_rate=0.05,
        n_estimators=1000,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        alpha=0.1,
        random_state = 42)

    scaler = StandardScaler()

    save_model(scaler.fit(x_train), x_train, "scaler")

    save_model(model.fit(x_train, y_train), x_train, "XGB")
    
if __name__ == "__main__":
    main()