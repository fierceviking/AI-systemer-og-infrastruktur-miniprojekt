import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'new_pizza_sales.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Feature Engineering
#data['order_timestamp_hour'] = pd.to_datetime(data['order_timestamp_hour'])
#data['day_of_week'] = data['order_timestamp_hour'].dt.dayofweek
#data['is_weekend'] = data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

  #  'lag_total_sales_1', 'lag_total_sales_3', 'lag_total_sales_5', 
   # 'day_of_week', 'is_weekend'
# Define the feature set and the target variable
features = [
    'day', 'month', 'hour',
    'Classic', 'Chicken', 'Supreme', 'Veggie', 'total_quantity']
X = data[features]
y = data['total_sales']

# Initialize TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

# Initialize models
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.1, max_iter=10000)
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)

# Function to cross-validate and collect fold predictions
def cross_val_evaluate(model, X, y, tscv):
    fold_predictions = []
    mse_scores = []
    mae_scores = []
    
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Fit the model
        model.fit(X_train, y_train)
        
        # Predict and store
        y_pred = model.predict(X_test)
        fold_predictions.append((test_index, y_pred))  # Store predictions with test indices for later plotting
        mse_scores.append(mean_squared_error(y_test, y_pred))
        mae_scores.append(mean_absolute_error(y_test, y_pred))
    
    # Average metrics over all folds
    avg_mse = np.mean(mse_scores)
    avg_mae = np.mean(mae_scores)
    return fold_predictions, avg_mse, avg_mae

# Collect predictions for each model
ridge_preds, ridge_mse, ridge_mae = cross_val_evaluate(ridge, X, y, tscv)
lasso_preds, lasso_mse, lasso_mae = cross_val_evaluate(lasso, X, y, tscv)
elastic_net_preds, elastic_net_mse, elastic_net_mae = cross_val_evaluate(elastic_net, X, y, tscv)

# Print out average metrics
print(f"Ridge - Average MSE: {ridge_mse:.2f}, Average MAE: {ridge_mae:.2f}")
print(f"Lasso - Average MSE: {lasso_mse:.2f}, Average MAE: {lasso_mae:.2f}")
print(f"ElasticNet - Average MSE: {elastic_net_mse:.2f}, Average MAE: {elastic_net_mae:.2f}")

# Plotting actual vs. predicted for each model
plt.figure(figsize=(18, 6))
model_predictions = {
    "Ridge": ridge_preds,
    "Lasso": lasso_preds,
    "ElasticNet": elastic_net_preds
}

for i, (model_name, preds) in enumerate(model_predictions.items(), 1):
    # Initialize a placeholder array for averaged predictions
    y_pred_combined = np.full(y.shape, np.nan)
    
    # Fill in predictions for each fold in the placeholder array
    for test_index, y_pred in preds:
        y_pred_combined[test_index] = y_pred
    
    # Plot actual vs. combined predictions
    plt.subplot(1, 3, i)
    plt.plot(data['order_timestamp_hour'], y, label='Actual', alpha=0.7)
    plt.plot(data['order_timestamp_hour'], y_pred_combined, label=f'Predicted - {model_name}', alpha=0.7)
    plt.xlabel("Time")
    plt.ylabel("Total Sales")
    plt.title(f"{model_name} Regression: Actual vs. Predicted")
    plt.legend()

plt.tight_layout()
plt.show()


for i, (model_name, preds) in enumerate(model_predictions.items(), 1):
    # Initialize a placeholder array for averaged predictions
    y_pred_combined = np.full(y.shape, np.nan)
    
    # Fill in predictions for each fold in the placeholder array
    for test_index, y_pred in preds:
        y_pred_combined[test_index] = y_pred
    
    # Plot actual vs. combined predictions for the last 25 points
    plt.subplot(1, 3, i)
    plt.plot(data['order_timestamp_hour'][-50:], y[-50:], label='Actual', alpha=0.7)
    plt.plot(data['order_timestamp_hour'][-50:], y_pred_combined[-50:], label=f'Predicted - {model_name}', alpha=0.7)
    plt.xlabel("Time")
    plt.ylabel("Total Sales")
    plt.title(f"{model_name} Regression: Actual vs. Predicted (Last 50 Points)")
    plt.legend()

plt.tight_layout()
plt.show()