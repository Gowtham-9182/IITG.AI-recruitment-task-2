import pandas as pd
import numpy as np

def make_features(pgcb_cleaned, weather_cleaned, economics_df):
    """
    Create final features.
    economics_df should be the split version (econ_train or econ_test).
    """
    df = pgcb_cleaned.copy()
    
    df = df.drop(columns=['generation_mw','load_shedding'], 
                 errors='ignore')
    # Calendar
    df['hour']       = df['datetime'].dt.hour
    df['dayofweek']  = df['datetime'].dt.dayofweek
    df['month']      = df['datetime'].dt.month

    # Lags
    df['lag_1h']   = df['demand_mw'].shift(1)
    df['lag_24h']  = df['demand_mw'].shift(24)
    df['lag_168h'] = df['demand_mw'].shift(168)

    # Rolling means
    df['rolling_mean_24h']  = df['demand_mw'].rolling(24,  min_periods=1).mean().shift(1)
    df['rolling_mean_168h'] = df['demand_mw'].rolling(168, min_periods=1).mean().shift(1)

    # Merge weather
    weather = weather_cleaned.rename(columns={'time': 'datetime'})
    df = df.merge(weather, on='datetime', how='left')

    # Drop redundant weather columns
    df = df.drop(columns=['temperature', 'soil_temperature', 'relative_humidity'], 
                 errors='ignore')

    # Wind cyclical (safe)
    df['wind_sin'] = np.sin(2 * np.pi * df['wind_direction'] / 360)
    df['wind_cos'] = np.cos(2 * np.pi * df['wind_direction'] / 360)

    # Merge economics — handle case where 'year' is index
    econ = economics_df.copy()
    if 'year' not in econ.columns:
        econ = econ.reset_index()   # convert index 'year' to column

    df['year'] = df['datetime'].dt.year
    df = df.merge(econ, on='year', how='left')
    df = df.drop(columns=['year'], errors='ignore')

    # Derived economic features
    df['gdp_per_capita'] = df['NY.GDP.MKTP.CD'] / df['SP.POP.TOTL']
    df['econ_growth']    = df['NY.GDP.MKTP.KD.ZG'] * df['SP.POP.GROW']

    return df
