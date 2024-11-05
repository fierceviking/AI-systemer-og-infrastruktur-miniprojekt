import xgboost
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    df = pd.read_csv("data.csv")

    model = xgboost.XGBRegressor(objective='reg:squarederror')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

if __name__ == "__main__":
    main()