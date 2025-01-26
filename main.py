from functions import get_x_ticks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

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
