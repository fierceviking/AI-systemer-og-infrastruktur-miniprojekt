import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline

def load_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Feature Engineering
    if 'order_timestamp_hour' in data.columns:
        data['order_timestamp_hour'] = pd.to_datetime(data['order_timestamp_hour'])
    else:
        print("Column 'order_timestamp_hour' not found. Using index as time axis for plotting.")

    # Define the feature set and the target variable
    features = ['day', 'month', 'hour'] # , 'lag_total_sales_1', 'lag_total_sales_3', 'lag_total_sales_5'
    x = data[features]
    y = data['total_quantity']
    return data, x, y

def cross_val_evaluate(model, x, y, tscv):
    fold_predictions = []
    mse_scores = []
    rmse_scores = []
    mae_scores = []
    
    for train_index, test_index in tscv.split(x):
        # Split data into training and testing sets
        x_train, x_test = x.values[train_index], x.values[test_index]
        y_train, y_test = y.values[train_index], y.values[test_index]

        # Create a pipeline with the scaler and the model
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', model)
        ])
        
        # Fit the pipeline on the training data
        pipeline.fit(x_train, y_train)
        
        # Predict and store
        y_pred = pipeline.predict(x_test)
        fold_predictions.append((test_index, y_pred))
        mse = mean_squared_error(y_test, y_pred)
        mse_scores.append(mse)
        rmse_scores.append(np.sqrt(mse))
        mae_scores.append(mean_absolute_error(y_test, y_pred))
    
    # Average metrics over all folds
    avg_mse = np.mean(mse_scores)
    avg_rmse = np.mean(rmse_scores)
    avg_mae = np.mean(mae_scores)
    
    return fold_predictions, avg_mse, avg_mae, avg_rmse, pipeline, x_train


def plot(data, y, preds, model_name):
    # Plot actual vs. predicted for the specified model
    plt.figure(figsize=(10, 6))

    # Initialize a placeholder array for averaged predictions
    y_pred_combined = np.full(y.shape, np.nan)
    
    # Fill in predictions for each fold in the placeholder array
    for test_index, y_pred in preds:
        y_pred_combined[test_index] = y_pred
    
    # Plot the entire series
    if 'order_timestamp_hour' in data.columns:
        plt.plot(data['order_timestamp_hour'], y, label='Actual', alpha=0.7)
        plt.plot(data['order_timestamp_hour'], y_pred_combined, label=f'Predicted - {model_name}', alpha=0.7)
    else:
        # Use the index if 'order_timestamp_hour' is missing
        plt.plot(y.index, y, label='Actual', alpha=0.7)
        plt.plot(y.index, y_pred_combined, label=f'Predicted - {model_name}', alpha=0.7)

    plt.xlabel("Time" if 'order_timestamp_hour' in data.columns else "Index")
    plt.ylabel("Total Quantity")
    plt.title(f"{model_name} Regression: Actual vs. Predicted (Full Series)")
    plt.legend()
    plt.show()

def save_model(pipeline, x_train, model_type):
    # Define initial type for the conversion
    initial_type = [('float_input', FloatTensorType([None, x_train.shape[1]]))]

    # Convert the pipeline to ONNX format
    onnx_model = convert_sklearn(pipeline, initial_types=initial_type)

    # Ensure the output directory exists
    output_dir = "deploy"
    os.makedirs(output_dir, exist_ok=True)

    # Save the model within the deploy folder
    model_path = os.path.join(output_dir, f"{model_type.lower()}_model.onnx")
    with open(model_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"Model saved to {model_path}")


def main():
    # Variable
    file_path = 'deploy/new_pizza_sales.csv'
    model_type = 'Ridge'  # 'Ridge', 'Lasso', or 'ElasticNet'

    data, x, y = load_data(file_path)

    # Initialize TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)

    # Choose model based on model_type
    if model_type == 'Ridge':
        model = Ridge(alpha=1.0)
    elif model_type == 'Lasso':
        model = Lasso(alpha=0.1, max_iter=10000)
    elif model_type == 'ElasticNet':
        model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)
    else:
        print("Invalid model type selected.")
        return

    # Evaluate the chosen model
    preds, avg_mse, avg_mae, avg_rmse, pipeline, x_train = cross_val_evaluate(model, x, y, tscv)
    
    # Print out average metrics
    print(f"{model_type} - Average MSE: {avg_mse:.2f}, Average MAE: {avg_mae:.2f}, Average RMSE: {avg_rmse:.2f}")

    # Save model
   # save_model(pipeline, x_train, model_type)

    # Plot results
    plot(data, y, preds, model_type)

if __name__ == "__main__":
    main()
