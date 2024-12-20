import os
import numpy as np
import pandas as pd
from onnxmltools import convert_xgboost
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet

def plot_y_test_vs_y_pred(y_test, y_pred, title_size=16, label_size=14, legend_size=10, tick_size=10):
    plt.figure(figsize=(10, 6))
    
    if isinstance(y_test, np.ndarray):
        y_test = pd.Series(y_test)
    if isinstance(y_pred, np.ndarray):
        y_pred = pd.Series(y_pred)
    
    y_test_sorted = y_test.sort_index()
    y_pred_sorted = y_pred.sort_index()

    plt.plot(y_test_sorted.index, y_test_sorted, label='Actual Values (y_test)', color='blue', alpha=0.7)
    plt.plot(y_pred_sorted.index, y_pred_sorted, label='Predicted Values (y_pred)', color='red', alpha=0.7, linestyle='dashed')
    plt.xlabel("Hour", fontsize=label_size)
    plt.ylabel("Target Value", fontsize=label_size)
    plt.title("Comparison of Actual and Predicted Values", fontsize=title_size)
    plt.legend(fontsize=legend_size)
    plt.xticks(rotation=20, ha="right")
    plt.tick_params(axis='both', which='major', labelsize=tick_size)
    plt.show()
    
    # Calculate residuals
    residuals = y_test - y_pred

    # Plot residuals
    plt.figure(figsize=(10, 6))
    plt.plot(residuals.index, residuals, label='Residuals (y_test - y_pred)', color='purple', alpha=0.7)
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.xlabel("Hour", fontsize=label_size)
    plt.ylabel("Residual Value", fontsize=label_size)
    plt.title("Residuals Plot", fontsize=title_size)
    plt.legend(fontsize=legend_size)
    plt.xticks(rotation=20, ha="right")
    plt.tick_params(axis='both', which='major', labelsize=tick_size)
    plt.show()

def cross_val_evaluate(model, x, y, tscv, filter_outliers: bool = False):
    fold_predictions = []
    mse_scores = []
    rmse_scores = []
    mae_scores = []
    
    # Track last fold data for plotting
    last_y_test = None
    last_y_pred = None
    
    for train_index, test_index in tscv.split(x):
        x_train, x_test = x.values[train_index], x.values[test_index]
        y_train, y_test = y.values[train_index], y.values[test_index]

        if filter_outliers:
            # Calculate IQR and both upper and lower bounds for y_train (target variable)
            Q1 = np.quantile(y_train, 0.25)
            Q3 = np.quantile(y_train, 0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Identify outliers in y_train and filter x_train and y_train accordingly
            inliers = (y_train >= lower_bound) & (y_train <= upper_bound)
            x_train, y_train = x_train[inliers], y_train[inliers]

            print(f"Outliers removed: {np.sum(~inliers)}")

        # Fit the model on the filtered training data
        model.fit(x_train, y_train)
        
        # Predict and store
        y_pred = model.predict(x_test)
        fold_predictions.append((test_index, y_pred))
        mse = mean_squared_error(y_test, y_pred)
        mse_scores.append(mse)
        rmse_scores.append(np.sqrt(mse))
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        
        # Save the last fold's test data and predictions for plotting
        last_y_test = pd.Series(y_test, index=y.iloc[test_index].index)
        last_y_pred = pd.Series(y_pred, index=y.iloc[test_index].index)
    
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

def plot_data_with_outlier_threshold(data, iqr_factor = 1.5):
    data = pd.Series(data)

    training_data = data[:int(0.8 * len(data))]

    Q1 = training_data.quantile(0.25)
    Q3 = training_data.quantile(0.75)

    IQR = Q3 - Q1

    lower_threshold = Q1 - iqr_factor * IQR
    upper_threshold = Q3 + iqr_factor * IQR

    outliers = training_data[(training_data < lower_threshold) | (training_data > upper_threshold)]
    non_outliers = training_data[(training_data >= lower_threshold) & (training_data <= upper_threshold)]
    
    plt.figure(figsize=(10, 6))
    
    # Plot the non-outliers as a plain line
    plt.plot(non_outliers.index, non_outliers.values, label='Training Data (Non-Outliers)', color='blue', marker='', linestyle='-', linewidth=2)
    
    # Plot the outliers as dots
    plt.scatter(outliers.index, outliers.values, color='red', label='Outliers', zorder=5, s=50)
    
    # Plot the outlier threshold lines (lower and upper)
    plt.axhline(y=upper_threshold, color='red', linestyle='--', label=f'Upper Threshold ({upper_threshold:.2f})')

    plt.title("Training Data Outliers", fontsize=14)
    plt.xlabel("Hour", fontsize=16)
    plt.ylabel("Total quantity", fontsize=16)
    plt.legend()
    plt.show()

def main():
    # 'evaluate' multiple models with different lag features. 'final' model is saved
    evaluate_final = 'final' # 'evaluate' or 'final'
    remove_outliers = False # True or False

    df = pd.read_csv('new_pizza_sales.csv', parse_dates=['order_timestamp_hour'], index_col='order_timestamp_hour')
    df = df.reset_index()  # Reset index to make 'order_timestamp_hour' available as a column

    models = {}
    cols_of_features = {}

    cols = ["hour", "month", "day_of_week"]
    target = 'total_quantity'
    x_no_lag = df[cols]
    y = df[target]

   # plot_data_with_outlier_threshold(y)

    tscv = TimeSeriesSplit()

    model_xgb = xgb.XGBRegressor(
        objective = 'reg:squarederror',
        max_depth=3,
        learning_rate=0.05,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        alpha=0.1,
        random_state = 42
    )
    
    models['XGB'] = model_xgb
    cols_of_features['No lag'] = x_no_lag

    if evaluate_final == 'evaluate':
        model_svm = Pipeline([
            ('scaler', StandardScaler()), 
            ('svr', SVR(kernel='rbf', C=1.0, epsilon=0.1))])
        model_elastic = ElasticNet(
            alpha=0.1, 
            l1_ratio=0.5, 
            max_iter=1000,    
            random_state=42)
        models['SVM'] = model_svm
        models['Elastic Net'] = model_elastic
        
        cols.extend(['lag_total_quantity_1','lag_total_quantity_3','lag_total_quantity_5','lag_total_sales_1','lag_total_sales_3','lag_total_sales_5'])
        x_quant_sales = df[cols]
        cols_of_features['Quantity and sales lag'] = x_quant_sales
        
        cols.extend(['lag_S_count_1','lag_S_count_3','lag_S_count_5','lag_M_count_1','lag_M_count_3','lag_M_count_5','lag_L_count_1','lag_L_count_3',
                     'lag_L_count_5','lag_XL_count_1','lag_XL_count_3','lag_XL_count_5','lag_XXL_count_1','lag_XXL_count_3','lag_XXL_count_5',
                     'lag_Classic_1','lag_Classic_3','lag_Classic_5','lag_Chicken_1','lag_Chicken_3','lag_Chicken_5','lag_Supreme_1','lag_Supreme_3',
                     'lag_Supreme_5','lag_Veggie_1','lag_Veggie_3','lag_Veggie_5'])
        x_all_lag = df[cols]
        cols_of_features['All lag features'] = x_all_lag

    for model_name, model in models.items():
        print('---------------------------------------------------------------')
        print(model_name)
        for features_name, x in cols_of_features.items():
            print(f'\n{features_name}')
            _, mse, mae, rmse, best_model, x_train = cross_val_evaluate(model, x, y, tscv, remove_outliers)
            for i in range(len(mse)):
                print(f"Scores for model @ split {i+1}:\n MSE: {mse[i]}\n MAE: {mae[i]}\n RMSE: {rmse[i]}")

    if evaluate_final == 'final':
        save_model(best_model, x_train, "XGB")

if __name__ == "__main__":
    main()