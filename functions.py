from datetime import datetime
import numpy as np

def get_x_ticks(date_list: list, num_years: int, min_year: int, num_month) -> list:
  x_ticks = np.zeros(num_years)
  for i in range(num_years):
    year = min_year + i
    x_ticks[i] = date_list.index(find_earliest_date(num_month, year, date_list))
  
  return x_ticks

def find_earliest_date(num_month: int, year: int, dates: list) -> str:
  for date in dates:
    parsed_date = datetime.strptime(date, "%b %d, %Y")
    if parsed_date.month == num_month and parsed_date.year == year:
         return parsed_date.strftime("%b %d, %Y")
    