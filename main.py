from functions import generate_future_dates, get_x_ticks, convert_strs_into_dates
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional

# setup to keep predictions consistent across runs
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

tf.config.experimental.enable_op_determinism()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# data used: https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data, cleaned up using clean_csv.py
data = pd.read_csv("btcusd_1-min_data_cleaned.csv")

data.set_index("Timestamp", inplace=True)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1, 1))

sequence_length = 3
X, y = [], []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i - sequence_length:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))

train_size = int(len(X) * 0.8)
X_train = X[:train_size]
y_train = y[:train_size]

model = Sequential()
model.add(Bidirectional(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1))))
model.add(Bidirectional(LSTM(units=50)))
model.add(Dense(units=1))

model.compile(optimizer="adam", loss="mean_squared_error")

model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1, shuffle=False)

future_days = 60
last_sequence = scaled_data[-sequence_length:]
future_predictions = []

for i in range(future_days):
    pred = model.predict(last_sequence.reshape(1, sequence_length, 1))[0][0]
    
    # 🆕 Mix real data for the first few steps
    if i < sequence_length:
        last_sequence = np.append(last_sequence[1:], scaled_data[-sequence_length + i][0])
    else:
        last_sequence = np.append(last_sequence[1:], pred)
    
    last_sequence = last_sequence.reshape(sequence_length, 1)
    future_predictions.append(pred)

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

plt.figure(figsize=(10, 6))
plt.plot(data.index, data["Close"], label="Historical Prices", color="black")

future_dates = generate_future_dates(data.index[-1], future_days)
plt.plot(future_dates, future_predictions, '--', color="black", label="Future Predictions")
all_dates = convert_strs_into_dates(data.index.tolist() + future_dates)
plt.xticks(get_x_ticks(date_list=all_dates, num_years=(all_dates[-1].year - all_dates[0].year), num_month=10, min_year=all_dates[0].year))

plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("Stock Price Prediction with LSTM")
plt.legend()
plt.show()

