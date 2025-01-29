from functions import generate_future_dates, get_x_ticks, convert_strs_into_dates
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

data = pd.read_csv("btcusd_1-min_data_cleaned.csv")

data.set_index("Timestamp", inplace=True)

plt.figure(figsize=(10, 6))
plt.plot(data.index, data["Close*"], label="Historical Prices", color="black")
# plt.xticks(get_x_ticks(date_list=data.index.to_list(), num_years=5, num_month=10, min_year=2018))

future_days = 365
future_predictions = np.full(future_days, 30000)

future_dates = generate_future_dates(data.index[-1], future_days)
plt.plot(future_dates, future_predictions, 'b--', label="Future Predictions")
all_dates = convert_strs_into_dates(data.index.to_list() + future_dates)
print(all_dates)
plt.xticks(get_x_ticks(date_list=all_dates, num_years=(all_dates[-1].year - all_dates[0].year), num_month=10, min_year=all_dates[0].year))

plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("Stock Price Prediction with LSTM")
plt.legend()
plt.show()