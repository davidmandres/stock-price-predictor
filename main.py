from functions import get_x_ticks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

file_path = "yahoo_data.xlsx"
data = pd.read_excel(file_path)
data = data.iloc[::-1]

plt.figure(figsize=(10, 6))
plt.plot(data["Date"], data["Close*"], label="Closing Price", color="blue")
plt.xticks(get_x_ticks(date_list=data["Date"].to_list(), num_years=5, num_month=10, min_year=2018))
plt.title("Stock Closing Price Over Time")
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.legend()
plt.show()

data["50_MA"] = data["Close*"].rolling(window=50).mean()

data = data.dropna()

X = data[["Open", "High", "Low", "Volume", "50_MA"]]
y = data["Close*"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title("Actual vs. Predicted Stock Prices")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.show()

data.set_index("Date", inplace=True)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data["Close*"].values.reshape(-1, 1))

# Prepare sequences for LSTM
sequence_length = 3
X, y = [], []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i - sequence_length:i, 0])  # Last 3 days
    y.append(scaled_data[i, 0])  # Current day

X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape for LSTM (samples, timesteps, features)

# Split into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))  # Output layer

# Compile the model
model.compile(optimizer="adam", loss="mean_squared_error")

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

# Predict on test data
predicted_prices = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_prices)  # Rescale back to original prices

# Predict future prices
future_days = 5
last_sequence = scaled_data[-sequence_length:]  # Get the last sequence
future_predictions = []

for _ in range(future_days):
    pred = model.predict(last_sequence.reshape(1, sequence_length, 1))[0][0]
    future_predictions.append(pred)
    last_sequence = np.append(last_sequence[1:], pred).reshape(sequence_length, 1)

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(data.index, data["Close*"], label="Historical Prices", marker="o")
test_dates = data.index[-len(y_test):]
plt.plot(test_dates, predicted_prices, label="Predicted Prices", marker="x")

future_dates = pd.date_range(data.index[-1] + "{}".format(pd.Timedelta(days=1)), periods=future_days)
plt.plot(future_dates, future_predictions, label="Future Predictions", marker="s")

plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("Stock Price Prediction with LSTM")
plt.legend()
plt.show()

