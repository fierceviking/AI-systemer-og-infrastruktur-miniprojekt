import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Preprocessing
def preprocess_data(df, seq_length):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[['total_sales']].values)
    X, y = [], []
    for i in range(len(df_scaled) - seq_length):
        X.append(df_scaled[i:i + seq_length])
        y.append(df_scaled[i + seq_length])
    return np.array(X), np.array(y), scaler

# Custom Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

# Improved Model Definition
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=4):
        super(RNNModel, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.5)  # Increased dropout for regularization

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# Training Function with Loss Monitoring
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, targets in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Print average loss per epoch
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

# Main Code
if __name__ == "__main__":
    # Load the CSV file
    df = pd.read_csv('new_pizza_sales.csv', parse_dates=['order_timestamp_hour'], index_col='order_timestamp_hour')

    # Use the 'total_sales' column for prediction
    seq_length = 24

    # Preprocess data
    X, y, scaler = preprocess_data(df, seq_length)
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Data Loaders
    train_dataset = TimeSeriesDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Model, Loss, Optimizer
    model = RNNModel(input_size=X_train.shape[2], hidden_size=256, output_size=1, num_layers=2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)  # Adjusted learning rate

    # Train the model
    train_model(model, train_loader, criterion, optimizer, num_epochs=50)

    # Evaluation
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_pred = model(X_test_tensor).cpu().numpy()
        y_pred = scaler.inverse_transform(y_pred)
        y_test = scaler.inverse_transform(y_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100

    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")




    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(df.index[train_size + seq_length:], y_test, label='Actual')
    plt.plot(df.index[train_size + seq_length:], y_pred, label='Predicted')
    plt.xlabel('Timestamp')
    plt.ylabel('Total Sales')
    plt.title('Pizza Sales Prediction using Improved RNN (PyTorch)')
    plt.legend()
    plt.show()
