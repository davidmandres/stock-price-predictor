from functions import generate_future_dates, get_x_ticks, turn_strs_into_dates
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

data = pd.read_excel("yahoo_data.xlsx")
data = data.iloc[::-1]

data.set_index("Date", inplace=True)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data["Close*"].values.reshape(-1, 1))

sequence_length = 3
X, y = [], []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i - sequence_length:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

model.compile(optimizer="adam", loss="mean_squared_error")

model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

# predicted_prices = model.predict(X_test)
# predicted_prices = scaler.inverse_transform(predicted_prices)  

future_days = 365
last_sequence = scaled_data[-sequence_length:]
future_predictions = []

for _ in range(future_days):
    pred = model.predict(last_sequence.reshape(1, sequence_length, 1))[0][0]
    future_predictions.append(pred)
    last_sequence = np.append(last_sequence[1:], pred).reshape(sequence_length, 1)

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

plt.figure(figsize=(10, 6))
plt.plot(data.index, data["Close*"], label="Historical Prices", color="black")

# test_dates = data.index[-len(y_test):]
# plt.plot(test_dates, predicted_prices, label="Predicted Prices", color="red")

future_dates = generate_future_dates(data.index[-1], future_days)
plt.plot(future_dates, future_predictions, 'b--', label="Future Predictions")
all_dates = turn_strs_into_dates(data.index + future_dates)
print(all_dates)
plt.xticks(get_x_ticks(date_list=all_dates, num_years={all_dates[-1].year - all_dates[0].year}, num_month=10, min_year=all_dates[0].year))

plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("Stock Price Prediction with LSTM")
plt.legend()
plt.show()

