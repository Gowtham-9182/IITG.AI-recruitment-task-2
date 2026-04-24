import pandas as pd
import numpy as np

def clean_pgcb(raw_df):
    df = raw_df.copy()
    
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    df = df.drop(columns=['remarks'], errors='ignore')
    
    df = df.groupby('datetime').mean().reset_index()
    
    # === HALF-HOURLY FIX (this part was causing the error) ===
    num_cols = df.select_dtypes(include='number').columns
    
    df_00 = df[df['datetime'].dt.minute == 0].copy()
    df_30 = df[df['datetime'].dt.minute == 30].copy()
    df_30['datetime'] = df_30['datetime'].dt.floor('h')
    
    common = df_30['datetime'].isin(df_00['datetime'])
    case1_30 = df_30[common].set_index('datetime')
    case1_00 = df_00[df_00['datetime'].isin(case1_30.index)].set_index('datetime')
    merged1 = ((case1_00[num_cols] + case1_30[num_cols]) / 2).reset_index()
    
    case2_30 = df_30[~common].copy()
    case2_30['prev'] = case2_30['datetime'] - pd.Timedelta(hours=1)
    case2_prev = df_00[df_00['datetime'].isin(case2_30['prev'])].set_index('datetime')
    case2_30 = case2_30.set_index('prev')
    
    merged2 = (0.7 * case2_prev[num_cols] + 0.3 * case2_30[num_cols])
    merged2 = merged2.reset_index(drop=True)
    merged2['datetime'] = case2_30.index.values
    
    used = pd.concat([merged1['datetime'], merged2['datetime']])
    remaining = df_00[~df_00['datetime'].isin(used)]
    
    df = pd.concat([remaining, merged1, merged2], ignore_index=True)\
           .sort_values('datetime').reset_index(drop=True)
    
    # fill missing timestamps
    full_range = pd.date_range(df['datetime'].min(), df['datetime'].max(), freq='H')
    df = df.set_index('datetime').reindex(full_range).ffill().reset_index()
    df = df.rename(columns={'index': 'datetime'})
    
    source_cols = ['gas', 'liquid_fuel', 'coal', 'hydro', 'solar', 'wind',
                   'india_bheramara_hvdc', 'india_tripura']
    df[['india_adani', 'nepal']] = df[['india_adani', 'nepal']].fillna(0)
    df = df.drop(columns=['india_adani', 'nepal'], errors='ignore')
    df[source_cols] = df[source_cols].fillna(0)
    
    # outlier flagging
    def rolling_mad_score(x, w):
        med = x.rolling(w, min_periods=1).median().shift(1)
        mad = (x - med).abs().rolling(w, min_periods=1).median().shift(1)
        return (x - med) / (mad + 1e-6)
    
    target = 'demand_mw'
    z_short = rolling_mad_score(df[target], 24)
    z_long = rolling_mad_score(df[target], 24 * 7)
    lag_diff = df[target].diff().abs()
    lag_med = lag_diff.rolling(24, min_periods=1).median().shift(1)
    jump = lag_diff > 6 * (lag_med + 1e-6)
    demand_flags = ((z_short.abs() > 6) & (z_long.abs() > 4)) | jump
    
    z_gen_short = rolling_mad_score(df['generation_mw'], 24)
    z_gen_long = rolling_mad_score(df['generation_mw'], 24 * 7)
    lag_diff_gen = df['generation_mw'].diff().abs()
    lag_med_gen = lag_diff_gen.rolling(24, min_periods=1).median().shift(1)
    jump_gen = lag_diff_gen > 6 * (lag_med_gen + 1e-6)
    gen_flags = ((z_gen_short.abs() > 6) & (z_gen_long.abs() > 4)) | jump_gen
    
    ls_nonzero = df['load_shedding'][df['load_shedding'] > 0]
    ls_q3 = ls_nonzero.quantile(0.75)
    ls_iqr = ls_q3 - ls_nonzero.quantile(0.25)
    ls_flags = df['load_shedding'] > ls_q3 + 3 * ls_iqr
    
    # demand imputation
    ls_clean = df['load_shedding'].copy()
    ls_clean[ls_flags] = 0
    source_sum = df[source_cols].sum(axis=1)
    gen_source_match = (df['generation_mw'] - source_sum).abs() <= 100
    fallback = df[target].rolling(24, min_periods=6).median().shift(1)
    
    case1a = demand_flags & ~gen_flags & ~ls_flags
    case1b = demand_flags & ~gen_flags & ls_flags
    case2a = demand_flags & gen_flags & gen_source_match & ~ls_flags
    case2b = demand_flags & gen_flags & gen_source_match & ls_flags
    case3 = demand_flags & gen_flags & ~gen_source_match
    
    df.loc[case1a, target] = df.loc[case1a, 'generation_mw'] + ls_clean[case1a]
    df.loc[case1b, target] = df.loc[case1b, 'generation_mw']
    df.loc[case2a, target] = df.loc[case2a, 'generation_mw'] + ls_clean[case2a]
    df.loc[case2b, target] = df.loc[case2b, 'generation_mw']
    df.loc[case3, target] = fallback[case3]
    
    # blackout fix
    total_source = df[source_cols].sum(axis=1)
    blackout = (total_source == 0) & (df[target] > 0)
    if blackout.any():
        ref_sources = df[source_cols].copy()
        ref_sources.loc[blackout] = np.nan
        ref_sources = ref_sources.bfill().ffill()
        ref_total = ref_sources.sum(axis=1)
        scale_factor = df.loc[blackout, target] / (ref_total[blackout] + 1e-8)
        for col in source_cols:
            df.loc[blackout, col] = ref_sources.loc[blackout, col] * scale_factor
    
    # source_cols bad rows fix
    ls_clean = df['load_shedding'].copy()
    ls_clean[ls_flags] = 0
    source_sum = df[source_cols].sum(axis=1)
    balance_diff = (source_sum + ls_clean - df[target]).abs()
    diff_nonzero = balance_diff[balance_diff > 0]
    diff_thresh = diff_nonzero.quantile(0.75) + 5 * (diff_nonzero.quantile(0.75) - diff_nonzero.quantile(0.25))
    bad_rows = balance_diff > diff_thresh
    
    rolling_share = pd.DataFrame(index=df.index, columns=source_cols, dtype=float)
    for col in source_cols:
        rolling_share[col] = (
            df[col]
            .expanding(min_periods=1)
            .median()
            .where(df[col].rolling(24, min_periods=8).count() < 8,
                   df[col].rolling(24, min_periods=8).median())
            .shift(1)
        )
    rolling_share = rolling_share.clip(lower=0)
    row_totals = rolling_share.sum(axis=1).replace(0, np.nan)
    proportions = rolling_share.div(row_totals, axis=0).fillna(1 / len(source_cols))
    
    for idx in bad_rows[bad_rows].index:
        required_sum = df.loc[idx, target] - ls_clean.loc[idx]
        if required_sum < 0:
            continue
        for col in source_cols:
            df.loc[idx, col] = proportions.loc[idx, col] * required_sum
    
    return df


def clean_weather(raw_df):
    df = raw_df.copy()
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    
    df = df.groupby('time').mean().reset_index()
    
    full_range = pd.date_range(df['time'].min(), df['time'].max(), freq='H')
    df = df.set_index('time').reindex(full_range).ffill().reset_index()
    df = df.rename(columns={'index': 'time'})
    
    df.columns = ['time', 'temperature', 'relative_humidity',
                  'apparent_temperature', 'precipitation', 'dew_point',
                  'soil_temperature', 'wind_direction', 'cloud_cover', 'sunshine_duration']
    
    clip_bounds = {
        'temperature': (-10, 50),
        'apparent_temperature': (-10, 50),
        'relative_humidity': (0, 100),
        'precipitation': (0, None),
        'dew_point': (-10, 40),
        'soil_temperature': (0, 50),
        'cloud_cover': (0, 100),
    }
    for col, (lo, hi) in clip_bounds.items():
        if hi is None:
            df[col] = df[col].clip(lower=lo)
        else:
            df[col] = df[col].clip(lower=lo, upper=hi)
    
    return df