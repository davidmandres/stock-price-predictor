from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pandas import DatetimeIndex

def get_x_ticks(date_list: list, num_years: int, min_year: int, num_month: int) -> list:
  x_ticks = np.zeros(num_years)
  for i in range(num_years):
    year = min_year + i
    x_ticks[i] = date_list.index(find_earliest_date(num_month, year, date_list))
  
  return x_ticks

def find_earliest_date(num_month: int, year: int, dates: list) -> datetime:
  for date in dates:
    if date.month == num_month and date.year == year:
         return date
    
def generate_future_dates(start_date: str, end: int) -> DatetimeIndex:
  future_dates = []
  start_date = pd.to_datetime(start_date)

  if (end < 1):
    return future_dates
  
  for _ in range(end):
    start_date += timedelta(days=1)
    future_dates.append(start_date.strftime("%b %d, %Y"))
    
  return future_dates

def convert_strs_into_dates(dates: list) -> DatetimeIndex:
  return [datetime.strptime(date, "%b %d, %Y") for date in dates]

def convert_unix_to_str(unix: int) -> str:
  return datetime.fromtimestamp(unix).strftime("%b %d, %Y")
    