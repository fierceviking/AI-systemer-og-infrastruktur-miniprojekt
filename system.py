import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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

if __name__ == "__main__":
    main()