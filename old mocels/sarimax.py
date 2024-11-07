import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
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
series = series[['total_sales']]

# Add lag features
series['lag_1'] = series['total_sales'].shift(1)
series['lag_2'] = series['total_sales'].shift(2)
series['lag_3'] = series['total_sales'].shift(3)

# Add additional features for exogenous variables
series['day_of_week'] = series.index.dayofweek
series['hour_of_day'] = series.index.hour

# Drop rows with NaN values created by lagging
series = series.dropna()

# Split into train and test sets
size = int(len(series) * 0.8)

# Apply log transformation
series['log_sales'] = np.log1p(series['total_sales'])

# Use the log-transformed sales for modeling
train, test = series.iloc[:size], series.iloc[size:]
exog_train = train[['day_of_week', 'hour_of_day', 'lag_1', 'lag_2', 'lag_3']]
exog_test = test[['day_of_week', 'hour_of_day', 'lag_1', 'lag_2', 'lag_3']]

# Fit the SARIMAX model
model = SARIMAX(train['log_sales'], exog=exog_train, order=(2, 1, 2))
model_fit = model.fit()

# Forecast and transform predictions back to the original scale
log_predictions = model_fit.get_forecast(steps=len(test), exog=exog_test).predicted_mean
predictions = np.expm1(log_predictions)  # Reverse the log transformation

# Align the predictions with the test index and clip negative values
predictions.index = test.index
predictions = predictions.clip(lower=0)

# Evaluate the forecasts
rmse = sqrt(mean_squared_error(test['total_sales'], predictions))
print(f'Test RMSE: {rmse:.3f}')

# Plot the actual and predicted values
plt.figure(figsize=(10, 6))
plt.plot(test.index, test['total_sales'], label='Actual')
plt.plot(predictions.index, predictions, color='red', label='Predicted')
plt.title('Pizza Sales Prediction using SARIMAX with Log Transformation and Lag Features')
plt.xlabel('Time')
plt.ylabel('Total Sales')
plt.legend()
plt.show()
