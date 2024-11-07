import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Load and preprocess the dataset
series = pd.read_csv(
    'new_pizza_sales.csv',
    parse_dates=['order_timestamp_hour'],
    index_col='order_timestamp_hour'
)

# Ensure the index is sorted
series = series.sort_index()

# Use the 'total_sales' column for forecasting
series = series[['total_sales', 'lag_total_sales_1', 'lag_total_sales_3', 'lag_total_sales_5']]

# Drop any remaining NaN values (if present)
series = series.dropna()

# Split into train and test sets
X = series['total_sales'].values
size = int(len(X) * 0.8)
train, test = X[:size], X[size:]

# Split lag features into train and test sets
exog_train = series[['lag_total_sales_1', 'lag_total_sales_3', 'lag_total_sales_5']].iloc[:size]
exog_test = series[['lag_total_sales_1', 'lag_total_sales_3', 'lag_total_sales_5']].iloc[size:]

# Use auto_arima to find the best (p, d, q) order with exogenous variables
model = auto_arima(train, exogenous=exog_train, seasonal=False, trace=True, stepwise=True)

# Print the selected order
print(f"Selected ARIMA order: {model.order}")

# Convert the model to a statsmodels ARIMA object for walk-forward validation
history = list(train)
history_exog = exog_train.values.tolist()
predictions = []

# Walk-forward validation using the best ARIMA order from pmdarima
for t in range(len(test)):
    try:
        model_fit = model.fit(history, exogenous=history_exog)
        yhat = model_fit.predict(n_periods=1, exogenous=[exog_test.iloc[t]])[0]
        predictions.append(yhat)
        history.append(test[t])
        history_exog.append(exog_test.iloc[t].values)
        print(f'predicted={yhat:.3f}, expected={test[t]:.3f}')
    except Exception as e:
        print(f"An error occurred: {e}")
        predictions.append(history[-1])  # Use the last known value as a fallback

# Evaluate the forecasts
rmse = sqrt(mean_squared_error(test, predictions))
print(f'Test RMSE: {rmse:.3f}')

# Use the test indices from your original DataFrame
test_dates = series.index[size:]

# Plot the forecasts against actual outcomes
plt.figure(figsize=(10, 6))
plt.plot(test_dates, test, label='Actual')
plt.plot(test_dates, predictions, color='red', label='Predicted')
plt.title('Pizza Sales Prediction using ARIMA with Pre-Shifted Lag Features')
plt.xlabel('Time')
plt.ylabel('Total Sales')
plt.legend()
plt.show()
