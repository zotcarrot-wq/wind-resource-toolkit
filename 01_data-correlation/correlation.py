import pandas as pd
from scipy.stats import pearsonr, linregress
import numpy as np


def extract_wind_speed_column(df, wind_speed_col="M(m/s)"):
    """
    Extract wind speed column from measurement data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Measurement data with wind speed column.
    wind_speed_col : str
        Name of the wind speed column (default: "M(m/s)").
    
    Returns
    -------
    pd.Series
        Wind speed data (numeric).
    """
    # Try exact match first
    if wind_speed_col in df.columns:
        return pd.to_numeric(df[wind_speed_col], errors='coerce')
    
    # Try to find a column starting with 'M' (likely wind speed magnitude)
    for col in df.columns:
        if col.strip().startswith('M') and 'm/s' in col.lower():
            print(f"Using column '{col}' for wind speed")
            return pd.to_numeric(df[col], errors='coerce')
    
    # Print available columns for debugging
    print(f"Available columns in measurement data: {df.columns.tolist()}")
    raise ValueError(f"Column '{wind_speed_col}' not found in measurement data. Available columns: {df.columns.tolist()}")


def extract_wind_speed_column_ref(df):
    """
    Extract wind speed column from reference data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Reference data with time index. Expects the first numeric column to be wind speed.
    
    Returns
    -------
    pd.Series
        Wind speed data (numeric).
    """
    # Find the first column that looks like wind speed (usually starts with MeanWindSpeedUID)
    for col in df.columns:
        if 'MeanWindSpeed' in col or 'wind speed' in col.lower():
            return pd.to_numeric(df[col], errors='coerce')
    
    # Fallback: use the first numeric column
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
            return pd.to_numeric(df[col], errors='coerce')
    
    raise ValueError("Could not find wind speed column in reference data")


def calculate_regression_metrics(y_true, y_pred):
    """
    Calculate linear regression metrics using Least Squares method.
    
    Fits: y_pred = intercept + slope * y_true
    
    Parameters
    ----------
    y_true : array-like
        True values (independent variable, reference data)
    y_pred : array-like
        Predicted values (dependent variable, measurement data)
    
    Returns
    -------
    dict
        Dictionary with keys: 'slope', 'intercept', 'r2', 'n_points'
    """
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    n_points = len(y_true_clean)
    
    if n_points < 2:
        return {'slope': np.nan, 'intercept': np.nan, 'r2': np.nan, 'n_points': 0}
    
    # Linear regression: y_pred = intercept + slope * y_true
    result = linregress(y_true_clean, y_pred_clean)
    
    # Calculate R²
    r2 = result.rvalue ** 2
    
    return {
        'slope': result.slope,
        'intercept': result.intercept,
        'r2': r2,
        'n_points': n_points
    }


def correlate_10min(measurement_data, ref_data_dict):
    """
    Correlate at 10-minute time steps without any grouping.
    
    Match 10-min measurement data to the 1-hour reference data.
    Each 10-min measurement point matches to its corresponding hour in reference data.
    
    Parameters
    ----------
    measurement_data : pd.DataFrame
        Measurement data (10-min resolution)
    ref_data_dict : dict
        Dictionary of reference data (1-hour resolution)
    
    Returns
    -------
    dict
        Dictionary with location names as keys and regression metrics as values
    """
    measurement_data = measurement_data.copy()
    
    if not isinstance(measurement_data.index, pd.DatetimeIndex):
        measurement_data.index = pd.to_datetime(measurement_data.index)
    
    meas_wind_speed = extract_wind_speed_column(measurement_data)
    measurement_data['wind_speed'] = meas_wind_speed
    
    # Floor to hour for matching
    measurement_data['hour'] = measurement_data.index.floor('h')
    
    results = {}
    
    for location_name, ref_df in ref_data_dict.items():
        ref_df_copy = ref_df.copy()
        
        if not isinstance(ref_df_copy.index, pd.DatetimeIndex):
            ref_df_copy.index = pd.to_datetime(ref_df_copy.index)
        
        ref_wind_speed = extract_wind_speed_column_ref(ref_df_copy)
        ref_df_copy['wind_speed'] = ref_wind_speed
        
        # Ensure reference has hour as index
        ref_hourly = ref_df_copy.set_index(ref_df_copy.index.floor('h'))['wind_speed']
        
        # Match measurement to reference by hour
        measurement_data_with_ref = measurement_data.copy()
        measurement_data_with_ref['ref_value'] = measurement_data_with_ref['hour'].map(ref_hourly)
        
        # Remove rows with NaN in either column
        valid_data = measurement_data_with_ref[['wind_speed', 'ref_value']].dropna()
        
        if len(valid_data) > 1:
            metrics = calculate_regression_metrics(valid_data['ref_value'].values, valid_data['wind_speed'].values)
            results[location_name] = metrics
        else:
            results[location_name] = {'slope': np.nan, 'intercept': np.nan, 'r2': np.nan, 'n_points': 0}
    
    return results


def correlate_1h(measurement_data, ref_data_dict):
    """
    Correlate at 1-hour time steps.
    
    Average 10-min measurement data within each hour, then correlate with 1-hour reference data.
    
    Parameters
    ----------
    measurement_data : pd.DataFrame
        Measurement data (10-min resolution)
    ref_data_dict : dict
        Dictionary of reference data (1-hour resolution)
    
    Returns
    -------
    dict
        Dictionary with location names as keys and regression metrics as values
    """
    measurement_data = measurement_data.copy()
    
    if not isinstance(measurement_data.index, pd.DatetimeIndex):
        measurement_data.index = pd.to_datetime(measurement_data.index)
    
    meas_wind_speed = extract_wind_speed_column(measurement_data)
    measurement_data['wind_speed'] = meas_wind_speed
    
    # Group by hour and average
    measurement_data['hour'] = measurement_data.index.floor('h')
    hourly_avg = measurement_data.groupby('hour')['wind_speed'].mean()
    
    results = {}
    
    for location_name, ref_df in ref_data_dict.items():
        ref_df_copy = ref_df.copy()
        
        if not isinstance(ref_df_copy.index, pd.DatetimeIndex):
            ref_df_copy.index = pd.to_datetime(ref_df_copy.index)
        
        ref_wind_speed = extract_wind_speed_column_ref(ref_df_copy)
        ref_df_copy['wind_speed'] = ref_wind_speed
        
        # Extract hour
        ref_hourly = ref_df_copy.set_index(ref_df_copy.index.floor('h'))['wind_speed']
        
        # Merge on hour
        merged = pd.DataFrame({
            'meas': hourly_avg,
            'ref': ref_hourly
        }).dropna()
        
        if len(merged) > 1:
            metrics = calculate_regression_metrics(merged['ref'].values, merged['meas'].values)
            results[location_name] = metrics
        else:
            results[location_name] = {'slope': np.nan, 'intercept': np.nan, 'r2': np.nan, 'n_points': 0}
    
    return results


def correlate_3h(measurement_data, ref_data_dict):
    """
    Correlate at 3-hour time steps.
    
    Average 10-min measurement data within each 3-hour window (starting at 0:00, 3:00, 6:00, etc.),
    then correlate with 1-hour reference data by averaging reference within same 3-hour windows.
    
    Only includes periods with sufficient data (>= 10 samples per period, ~55% of full 3h period).
    
    Parameters
    ----------
    measurement_data : pd.DataFrame
        Measurement data (10-min resolution)
    ref_data_dict : dict
        Dictionary of reference data (1-hour resolution)
    
    Returns
    -------
    dict
        Dictionary with location names as keys and regression metrics as values
    """
    measurement_data = measurement_data.copy()
    
    if not isinstance(measurement_data.index, pd.DatetimeIndex):
        measurement_data.index = pd.to_datetime(measurement_data.index)
    
    meas_wind_speed = extract_wind_speed_column(measurement_data)
    measurement_data['wind_speed'] = meas_wind_speed
    
    # Group by 3-hour windows: floor hour to nearest multiple of 3
    measurement_data['time_3h'] = measurement_data.index.floor('h')
    measurement_data['time_3h'] = measurement_data['time_3h'] - pd.to_timedelta(
        measurement_data['time_3h'].dt.hour % 3, unit='h'
    )
    
    # Count samples per 3h period to filter periods with sufficient data
    period_counts = measurement_data.groupby('time_3h').size()
    # Keep periods with >= 10 samples (~55% of 18 expected samples)
    valid_periods = period_counts[period_counts >= 10].index
    
    # Filter to only valid periods
    measurement_data_filtered = measurement_data[measurement_data['time_3h'].isin(valid_periods)]
    meas_3h_avg = measurement_data_filtered.groupby('time_3h')['wind_speed'].mean()
    
    results = {}
    
    for location_name, ref_df in ref_data_dict.items():
        ref_df_copy = ref_df.copy()
        
        if not isinstance(ref_df_copy.index, pd.DatetimeIndex):
            ref_df_copy.index = pd.to_datetime(ref_df_copy.index)
        
        ref_wind_speed = extract_wind_speed_column_ref(ref_df_copy)
        ref_df_copy['wind_speed'] = ref_wind_speed
        
        # Group reference by 3-hour windows
        ref_df_copy['time_3h'] = ref_df_copy.index.floor('h')
        ref_df_copy['time_3h'] = ref_df_copy['time_3h'] - pd.to_timedelta(
            ref_df_copy['time_3h'].dt.hour % 3, unit='h'
        )
        
        # Count samples per 3h period in reference
        ref_period_counts = ref_df_copy.groupby('time_3h').size()
        ref_valid_periods = ref_period_counts[ref_period_counts >= 1].index
        
        ref_df_copy_filtered = ref_df_copy[ref_df_copy['time_3h'].isin(ref_valid_periods)]
        ref_3h_avg = ref_df_copy_filtered.groupby('time_3h')['wind_speed'].mean()
        
        # Merge on 3-hour window (only intersection of valid periods)
        merged = pd.DataFrame({
            'meas': meas_3h_avg,
            'ref': ref_3h_avg
        }).dropna()
        
        if len(merged) > 1:
            metrics = calculate_regression_metrics(merged['ref'].values, merged['meas'].values)
            results[location_name] = metrics
        else:
            results[location_name] = {'slope': np.nan, 'intercept': np.nan, 'r2': np.nan, 'n_points': 0}
    
    return results


def correlate_24h(measurement_data, ref_data_dict):
    """
    Correlate at 24-hour (daily) time steps.
    
    Average 10-min measurement data within each day (0:00 to 23:50),
    then correlate with 1-hour reference data by averaging reference within same day.
    
    Only includes complete 24h periods (>=144 samples per day).
    
    Parameters
    ----------
    measurement_data : pd.DataFrame
        Measurement data (10-min resolution)
    ref_data_dict : dict
        Dictionary of reference data (1-hour resolution)
    
    Returns
    -------
    dict
        Dictionary with location names as keys and regression metrics as values
    """
    measurement_data = measurement_data.copy()
    
    if not isinstance(measurement_data.index, pd.DatetimeIndex):
        measurement_data.index = pd.to_datetime(measurement_data.index)
    
    meas_wind_speed = extract_wind_speed_column(measurement_data)
    measurement_data['wind_speed'] = meas_wind_speed
    
    # Group by day
    measurement_data['day'] = measurement_data.index.floor('D')
    
    # Count samples per day to filter complete periods
    day_counts = measurement_data.groupby('day').size()
    # Complete day should have ~144 samples (24h * 6 samples/hour)
    complete_days = day_counts[day_counts >= 144].index
    
    # Filter to only complete days
    measurement_data_filtered = measurement_data[measurement_data['day'].isin(complete_days)]
    meas_daily_avg = measurement_data_filtered.groupby('day')['wind_speed'].mean()
    
    results = {}
    
    for location_name, ref_df in ref_data_dict.items():
        ref_df_copy = ref_df.copy()
        
        if not isinstance(ref_df_copy.index, pd.DatetimeIndex):
            ref_df_copy.index = pd.to_datetime(ref_df_copy.index)
        
        ref_wind_speed = extract_wind_speed_column_ref(ref_df_copy)
        ref_df_copy['wind_speed'] = ref_wind_speed
        
        # Group reference by day
        ref_df_copy['day'] = ref_df_copy.index.floor('D')
        
        # Count samples per day in reference
        ref_day_counts = ref_df_copy.groupby('day').size()
        ref_complete_days = ref_day_counts[ref_day_counts >= 1].index
        
        ref_df_copy_filtered = ref_df_copy[ref_df_copy['day'].isin(ref_complete_days)]
        ref_daily_avg = ref_df_copy_filtered.groupby('day')['wind_speed'].mean()
        
        # Merge on day (only intersection of complete periods)
        merged = pd.DataFrame({
            'meas': meas_daily_avg,
            'ref': ref_daily_avg
        }).dropna()
        
        if len(merged) > 1:
            metrics = calculate_regression_metrics(merged['ref'].values, merged['meas'].values)
            results[location_name] = metrics
        else:
            results[location_name] = {'slope': np.nan, 'intercept': np.nan, 'r2': np.nan, 'n_points': 0}
    
    return results


def run_all_correlations(measurement_data, ref_data_dict):
    """
    Run all correlation time steps and return results.
    
    Parameters
    ----------
    measurement_data : pd.DataFrame
        Measurement data
    ref_data_dict : dict
        Dictionary of reference data
    
    Returns
    -------
    dict
        Dictionary with time steps as keys and result dictionaries as values
    """
    results = {
        '10min': correlate_10min(measurement_data, ref_data_dict),
        '1h': correlate_1h(measurement_data, ref_data_dict),
        '3h': correlate_3h(measurement_data, ref_data_dict),
        '24h': correlate_24h(measurement_data, ref_data_dict),
    }
    
    return results


def summarize_correlations(correlation_results):
    """
    Print summary of regression metrics for all time steps and locations.
    
    Parameters
    ----------
    correlation_results : dict
        Output from run_all_correlations function
    """
    print("\n" + "="*100)
    print("LINEAR REGRESSION RESULTS - WINDOGRAPHER VALIDATION")
    print("="*100)
    
    # Get all locations from first time step
    first_timestep = list(correlation_results.values())[0]
    locations = list(first_timestep.keys())
    
    # For each location, print detailed results
    for loc in locations:
        print(f"\n{'Location:':<12} {loc}")
        print("-" * 100)
        print(f"{'Time Step':<12} {'Steps':>10} {'Intercept':>15} {'Slope':>15} {'R²':>15}")
        print("-" * 100)
        
        for timestep in ['10min', '1h', '3h', '24h']:
            if timestep in correlation_results:
                metrics = correlation_results[timestep].get(loc, {})
                n_points = metrics.get('n_points', 0)
                intercept = metrics.get('intercept', np.nan)
                slope = metrics.get('slope', np.nan)
                r2 = metrics.get('r2', np.nan)
                
                if n_points > 0:
                    print(f"{timestep:<12} {n_points:>10} {intercept:>15.3f} {slope:>15.3f} {r2:>15.6f}")
                else:
                    print(f"{timestep:<12} {'N/A':>10} {'N/A':>15} {'N/A':>15} {'N/A':>15}")
    
    print("\n" + "="*100)
    print("COMPARISON TABLE (All Locations)")
    print("="*100)
    
    # Print header
    print(f"\n{'Time Step':<12}", end='')
    for loc in locations:
        loc_short = loc.replace(' ', '_')[:20]
        print(f"{loc_short:>25}", end='')
    print()
    print("-" * (12 + 25 * len(locations)))
    
    # Print R² values for each time step
    for timestep in ['10min', '1h', '3h', '24h']:
        print(f"{timestep:<12}", end='')
        for loc in locations:
            if timestep in correlation_results:
                r2_value = correlation_results[timestep].get(loc, {}).get('r2', np.nan)
                if np.isnan(r2_value):
                    print(f"{'N/A':>25}", end='')
                else:
                    print(f"{r2_value:>25.6f}", end='')
        print()
    
    print("\n" + "="*100)
