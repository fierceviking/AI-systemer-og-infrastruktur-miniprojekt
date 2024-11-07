import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'new_pizza_sales.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Feature Engineering
data['order_timestamp_hour'] = pd.to_datetime(data['order_timestamp_hour'])
data['day_of_week'] = data['order_timestamp_hour'].dt.dayofweek
data['is_weekend'] = data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# Define the feature set and the target variable
features = [
    'day', 'month', 'hour', 'S_count', 'M_count', 'L_count', 'XL_count', 'XXL_count',
    'Classic', 'Chicken', 'Supreme', 'Veggie', 'total_quantity', 
    'lag_total_sales_1', 'lag_total_sales_3', 'lag_total_sales_5', 
    'day_of_week', 'is_weekend'
]
X = data[features]
y = data['total_sales']

# Split the data into training, validation, and testing sets (60% train, 20% validation, 20% test)
#X_train, X_test, y_train, y_test = TimeSeriesSplit(n_splits=5)
#X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, shuffle=False)

train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# Initialize and train models
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.1, max_iter=10000)
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)

ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)
elastic_net.fit(X_train, y_train)

# Make predictions on validation set
#ridge_val_pred = ridge.predict(X_val)
#lasso_val_pred = lasso.predict(X_val)
#elastic_net_val_pred = elastic_net.predict(X_val)

# Evaluate the models on the validation set
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{model_name} - Validation MSE: {mse:.2f}, Validation MAE: {mae:.2f}")

# Make predictions on the test set
ridge_test_pred = ridge.predict(X_test)
lasso_test_pred = lasso.predict(X_test)
elastic_net_test_pred = elastic_net.predict(X_test)

evaluate_model(y_test, ridge_test_pred, "Ridge")
evaluate_model(y_test, lasso_test_pred, "Lasso")
evaluate_model(y_test, elastic_net_test_pred, "ElasticNet")


# Plotting with the time frame
test_dates = data['order_timestamp_hour'][train_size:]

plt.figure(figsize=(18, 6))
models = {"Ridge": ridge_test_pred, "Lasso": lasso_test_pred, "ElasticNet": elastic_net_test_pred}
for i, (model_name, predictions) in enumerate(models.items(), 1):
    plt.subplot(1, 3, i)
    plt.plot(test_dates, y_test.values, label='Actual', alpha=0.7)
    plt.plot(test_dates, predictions, label=f'Predicted - {model_name}', alpha=0.7)
    plt.xlabel("Time")
    plt.ylabel("Total Sales")
    plt.title(f"{model_name} Regression: Actual vs. Predicted")
    plt.legend()

plt.tight_layout()
plt.show()
