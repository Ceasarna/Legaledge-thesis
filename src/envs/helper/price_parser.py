import pandas as pd
import random
from pathlib import Path
import random
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.interpolate import make_interp_spline

# Fixed paths to data files ...
DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"

PATHS = {
    2024: DATA_DIR / "AuctionPrice_2024_DayAhead_SE3_EUR_None.csv",
    2025: DATA_DIR / "AuctionPrice_2025_DayAhead_SE3_EUR_None.csv",
    2026: DATA_DIR / "AuctionPrice_2026_DayAhead_SE3_EUR_None.csv",
}

_cache: dict = {}

def parse_price_data(year: int) -> pd.DataFrame:
    if year not in _cache:
        df = pd.read_csv(PATHS[year], sep=';', decimal='.')
        df['Delivery Start (CET)'] = pd.to_datetime(df['Delivery Start (CET)'], format='%d.%m.%Y %H:%M:%S')
        df['Delivery End (CET)'] = pd.to_datetime(df['Delivery End (CET)'], format='%d.%m.%Y %H:%M:%S')
        _cache[year] = df
    return _cache[year]

def get_random_price_slice(year: int, duration: int) -> pd.DataFrame:
    """
    Returns a random slice of the dataframe for the given year with the specified duration in hours.
    """
    df = parse_price_data(year)
    
    if duration > len(df):
        raise ValueError(f"Requested duration {duration} exceeds data length {len(df)}")
    start_index = random.randint(0, len(df) - duration)
    return df.iloc[start_index : start_index + duration]

def get_random_price_slice_by_season(year: int, season: int, duration_hours: int) -> pd.DataFrame:
    """
    Returns a random slice for a specific season (quarter) of the year.
    season: 1 (Jan-Mar), 2 (Apr-Jun), 3 (Jul-Sep), 4 (Oct-Dec)
    """
    df = parse_price_data(year)
    
    quarters = {
        1: [1, 2, 3],
        2: [4, 5, 6],
        3: [7, 8, 9],
        4: [10, 11, 12]
    }
    
    if season not in quarters:
        raise ValueError("Season must be 1, 2, 3, or 4")
        
    season_mask = df['Delivery Start (CET)'].dt.month.isin(quarters[season])
    season_df = df.loc[season_mask]

    if duration_hours > len(season_df):
        raise ValueError(f"Requested duration {duration_hours} exceeds data length for season {season}")

    start_index = random.randint(0, len(season_df) - duration_hours)
    return season_df.iloc[start_index : start_index + duration_hours]

def plot_price_data(df: pd.DataFrame, x_col: str = 'Delivery Start (CET)', y_col: str = 'SE3 Price (EUR)', title: str = None):

    plt.figure(figsize=(15, 7))
    
    x = df[x_col]
    y = df[y_col]
    
    x_num = mdates.date2num(x)
    
    x_smooth_num = np.linspace(x_num.min(), x_num.max(), 300)
    spline = make_interp_spline(x_num, y, k=3) 
    y_smooth = spline(x_smooth_num)
    x_smooth = mdates.num2date(x_smooth_num)
    

    plt.plot(x_smooth, y_smooth, linestyle='-', color='royalblue', linewidth=2)
    plt.plot(x, y, marker='o', linestyle='none', color='navy')

    if title is None:
        start_time = x.iloc[0]
        end_time = x.iloc[-1]
        title = f'Electricity Price from {start_time.strftime("%Y-%m-%d %H:%M")} to {end_time.strftime("%Y-%m-%d %H:%M")}'

    plt.title(title, fontsize=14)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()