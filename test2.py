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
    index_col='order_timestamp_hour',  # Use the correct date column
    parse_dates=True,
    date_parser=parser
).squeeze()

# Convert the index to a period index if your time series is monthly
series.index = series.index.to_period('M')

# Split into train and test sets
X = series['total_sales'].values  # Use the 'total_sales' column for forecasting
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()

# Walk-forward validation
for t in range(len(test)):
    model = ARIMA(history, order=(5, 1, 0))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))

# Evaluate forecasts
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)

# Plot forecasts against actual outcomes
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()

