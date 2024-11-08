import xgboost as xgb
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

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

def plot_y_test_vs_y_pred(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    
    # Sort values for a smoother plot, useful if y_test is not sorted by time or index
    y_test_sorted = y_test.sort_index()
    y_pred_sorted = pd.Series(y_pred, index=y_test.index).sort_index()
    
    # Plot both y_test and y_pred
    plt.plot(y_test, label='Actual Values (y_test)', color='blue', alpha=0.7)
    plt.plot(y_pred_sorted, label='Predicted Values (y_pred)', color='red', alpha=0.7, linestyle='dashed')
    
    plt.xlabel("Index")
    plt.ylabel("Target Value")
    plt.title("Comparison of Actual and Predicted Values")
    plt.legend()
    plt.show()

def plot_correlation_matrix(df):
    """
    Plots the correlation matrix for the features in a DataFrame.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the features to analyze.
    """
    plt.figure(figsize=(12, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
    plt.title("Feature Correlation Matrix")
    plt.show()

def plot_last_25_samples(y_test, y_pred):
    """
    Plots the last 25 samples of the test set for actual vs. predicted values.

    Parameters:
        y_test (pd.Series): The actual test values.
        y_pred (array-like): The predicted values for the test set.
    """
    # Select the last 25 samples
    y_test_last_25 = y_test.iloc[-25:]
    y_pred_last_25 = pd.Series(y_pred, index=y_test.index).iloc[-25:]
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_last_25, label='Actual Values (y_test)', color='blue', alpha=0.7)
    plt.plot(y_pred_last_25, label='Predicted Values (y_pred)', color='red', linestyle='dashed', alpha=0.7)
    plt.xlabel("Index")
    plt.ylabel("Target Value")
    plt.title("Comparison of Actual and Predicted Values (Last 25 Test Samples)")
    plt.legend()
    plt.show()

def main():
    df = pd.read_csv('data.csv', parse_dates=['order_timestamp_hour'], index_col='order_timestamp_hour')

    cols = ["hour", "day", "month"]
    target = 'total_sales'
    X = df[cols]
    y = df[target]

    split_index = int(len(df) * 0.8)
    X_train, y_train = X.iloc[:split_index], y.iloc[:split_index]
    X_test, y_test = X.iloc[split_index:], y.iloc[split_index:]

    scaler = StandardScaler()
    # model = SVR(kernel='sigmoid', gamma='scale', degree=8, coef0=0.01, C=1)
    model = xgb.XGBRegressor(
        max_depth=5,
        learning_rate=0.05,
        n_estimators=1000,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        alpha=0.1,
        random_state = 42)

    pipeline = Pipeline([
        ('scaler', scaler),
        ('model', model)
    ])

    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, "SVR_pipeline.joblib")
    print("Model saved as SVR_pipeline.joblib")

    y_pred = pipeline.predict(X_test)

    plot_y_test_vs_y_pred(y_test, y_pred)

if __name__ == "__main__":
    main()