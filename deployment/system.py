# imports
import os
import numpy as np
import pandas as pd
from onnxmltools import convert_xgboost
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt

def plot_y_test_vs_y_pred(y_test, y_pred, title_size=14, label_size=12, legend_size=10, tick_size=10):
    plt.figure(figsize=(10, 6))
    
    # Ensure y_test and y_pred are pandas Series to retain their indices (which should be datetime)
    if isinstance(y_test, np.ndarray):
        y_test = pd.Series(y_test)
    if isinstance(y_pred, np.ndarray):
        y_pred = pd.Series(y_pred)
    
    # Sort values for a smoother plot, useful if y_test is not sorted by time or index
    y_test_sorted = y_test.sort_index()
    y_pred_sorted = y_pred.sort_index()

    # Plot both y_test and y_pred, using the index (which is datetime)
    plt.plot(y_test_sorted.index, y_test_sorted, label='Actual Values (y_test)', color='blue', alpha=0.7)
    plt.plot(y_pred_sorted.index, y_pred_sorted, label='Predicted Values (y_pred)', color='red', alpha=0.7, linestyle='dashed')
    
    # Set axis labels and title with font sizes
    plt.xlabel("Date", fontsize=label_size)
    plt.ylabel("Target Value", fontsize=label_size)
    plt.title("Comparison of Actual and Predicted Values", fontsize=title_size)
    
    # Set legend font size
    plt.legend(fontsize=legend_size)
    
    # Set tick label font sizes
    plt.tick_params(axis='both', which='major', labelsize=tick_size)
    
    # Rotate date labels on the x-axis for better readability
    plt.xticks(rotation=20, ha="right")
    
    plt.show()

def cross_val_evaluate(model, x, y, filter_outliers: bool, tscv):
    fold_predictions = []
    mse_scores = []
    rmse_scores = []
    mae_scores = []
    
    # Track last fold data for plotting
    last_y_test = None
    last_y_pred = None
    
    for train_index, test_index in tscv.split(x):
        # Split data into training and testing sets
        x_train, x_test = x.values[train_index], x.values[test_index]
        y_train, y_test = y.values[train_index], y.values[test_index]

        if filter_outliers:
            iqr = np.quantile(y_train, 0.75) - np.quantile(y_train, 0.25)
            upper_bound = 1.5 * iqr + np.quantile(y_train, 0.75)
            print(upper_bound)

        # Create a pipeline with the scaler and the model        
        # Fit the pipeline on the training data
        model.fit(x_train, y_train)
        
        # Predict and store
        y_pred = model.predict(x_test)
        fold_predictions.append((test_index, y_pred))
        mse = mean_squared_error(y_test, y_pred)
        mse_scores.append(mse)
        rmse_scores.append(np.sqrt(mse))
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        
        # Save the last fold's test data and predictions for plotting
        last_y_test, last_y_pred = pd.Series(y_test, index=y.iloc[test_index].index), pd.Series(y_pred, index=y.iloc[test_index].index)
    
    # Plot the last fold's results
    if last_y_test is not None and last_y_pred is not None:
        plot_y_test_vs_y_pred(last_y_test, last_y_pred)
    
    return fold_predictions, mse_scores, mae_scores, rmse_scores, model, x_train

def save_model(pipeline, x_train, model_type):
    # Define initial type for the conversion
    initial_type = [('float_input', FloatTensorType([None, x_train.shape[1]]))]

    if model_type == "XGB":
        # Convert the pipeline to ONNX format
        model = convert_xgboost(pipeline, initial_types=initial_type)
    else:
        model = convert_sklearn(pipeline, initial_types=initial_type)

    # Ensure the output directory exists
    output_dir = "deployment/deploy"
    os.makedirs(output_dir, exist_ok=True)

    # Save the model within the deploy folder
    model_path = os.path.join(output_dir, f"{model_type.lower()}_model.onnx")
    with open(model_path, "wb") as f:
        f.write(model.SerializeToString())
    print(f"Model saved to {model_path}")

# main function
def main():
    # loading the data
    df = pd.read_csv('new_pizza_sales.csv', parse_dates=['order_timestamp_hour'], index_col='order_timestamp_hour')
    df = df.reset_index()  # Reset index to make 'order_timestamp_hour' available as a column

    cols = ["hour", "month", "day_of_week"]
    target = 'total_quantity'
    x = df[cols]
    y = df[target]

    tscv = TimeSeriesSplit()

    model = xgb.XGBRegressor(
        objective = 'reg:squarederror',
        max_depth=3,
        learning_rate=0.05,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        alpha=0.1,
        random_state = 42)
    
    preds, mse, mae, rmse, best_model, x_train = cross_val_evaluate(model, x, y, True, tscv)
    for i in range(len(mse)):
        print(f"Scores for model @ iteration {i}:\n MSE: {mse[i]}\n MAE: {mae[i]}\n RMSE: {rmse[i]}")

    save_model(best_model, x_train, "XGB")

# Run the main function
if __name__ == "__main__":
    main()