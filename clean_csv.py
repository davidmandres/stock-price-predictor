import pandas as pd
import numpy as np
from functions import convert_unix_to_datetime

start_in_unix = 1641013200 # 2022-01-01 00:00:00
day_in_secs = 86400

# Load CSV
df = pd.read_csv("btcusd_1-min_data_raw.csv")

# Remove rows where Timestamp is NaN or infinite
df = df.dropna(subset=["Timestamp"])
df = df[(df["Timestamp"] != np.inf) & (df["Timestamp"] >= start_in_unix) & (df["Timestamp"] % day_in_secs == 0)]

# Convert to date format: "Jan 01, 2022"
df["Timestamp"] = df["Timestamp"].apply(convert_unix_to_datetime)

# Save cleaned CSV
df.to_csv("btcusd_1-min_data_cleaned.csv", index=False)
print("Cleaned CSV saved successfully.")