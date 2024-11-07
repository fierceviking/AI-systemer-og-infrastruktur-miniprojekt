import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

# Load dataset and parse the date
def parser(x):
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

# Read the CSV file, specifying the date column and parsing it
series = pd.read_csv(
    'new_pizza_sales.csv',
    header=0,
    index_col='order_timestamp_hour',  # Ensure this is the correct datetime column
    parse_dates=True,
    date_parser=parser
)

# Use the 'total_sales' column for forecasting
series = series[['total_sales']]  # Extract the relevant feature for ARIMA

# Optionally, convert the index to a period index if needed
# Uncomment this line if your time series is on a monthly frequency:
# series.index = series.index.to_period('M')

# Split into train and test sets
X = series['total_sales'].values
size = int(len(X) * 0.8)
train, test = X[:size], X[size:]
history = list(train)
predictions = []

# Walk-forward validation
for t in range(len(test)):
    model = ARIMA(history, order=(5, 1, 0))  # Adjust the ARIMA order if necessary
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print(f'predicted={yhat:.3f}, expected={obs:.3f}')

# Evaluate forecasts
rmse = sqrt(mean_squared_error(test, predictions))
print(f'Test RMSE: {rmse:.3f}')


# Use the test indices from your original DataFrame
test_dates = series.index[size:]

# Update the plot to use the actual timestamps
plt.figure(figsize=(10, 6))
plt.plot(test_dates, test, label='Actual')
plt.plot(test_dates, predictions, color='red', label='Predicted')
plt.xlabel('Time')
plt.ylabel('Total Sales')
plt.title('Pizza Sales Prediction using ARIMA')
plt.legend()
plt.show()

