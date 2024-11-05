import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def plot_model_performance(y_test, y_pred):
    # Residual plot
    residuals = y_test - y_pred
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.hlines(0, min(y_pred), max(y_pred), colors='red', linestyles='dashed')
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    
    # Prediction vs. True plot
    plt.subplot(1, 3, 2)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='dashed')
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Predicted vs. True Values")
    
    # Distribution of Residuals
    plt.subplot(1, 3, 3)
    plt.hist(residuals, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title("Distribution of Residuals")
    
    plt.tight_layout()
    plt.show()

# Define the main function to handle the ML pipeline
def main():
    # Load data
    df = pd.read_csv('data.csv', parse_dates=['order_timestamp_hour'], index_col='order_timestamp_hour')
    
    target = 'total_sales'

    x = df[[col for col in df.columns if col != target]]

    y = df[target]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    
    # Scale features - if numerical scaling is desired
    scaler = StandardScaler()
    scaler = scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)


    # Initialize model with chosen objective
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions on test data
    y_pred = model.predict(X_test)
    
    # Evaluate model performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("Mean Squared Error:", mse)
    print("R^2 Score:", r2)

    plot_model_performance(y_test, y_pred)

if __name__ == "__main__":
    main()