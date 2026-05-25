import pandas as pd
from scipy.stats import pearsonr
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


def correlate_by_hour(measurement_data, ref_data_dict):
    """
    Correlate hourly measurement data with reference data by matching hours.
    
    For each hour, calculates hourly statistics of measurement data (mean, min, max, std)
    and correlates these with the reference value for that hour across all common hours.
    
    Parameters
    ----------
    measurement_data : pd.DataFrame
        Measurement data with timestamp index and "M(m/s)" column (sub-hourly samples).
    ref_data_dict : dict
        Dictionary where keys are location names and values are reference DataFrames
        with time index and wind speed columns (hourly data).
    
    Returns
    -------
    dict
        Dictionary with location names as keys and correlation results (DataFrame) as values.
        Each result DataFrame contains:
        - 'hour': Hour timestamp
        - 'meas_mean': Mean of measurement values for that hour
        - 'meas_min': Min of measurement values for that hour
        - 'meas_max': Max of measurement values for that hour
        - 'meas_std': Std dev of measurement values for that hour
        - 'meas_range': Max - Min of measurement values for that hour
        - 'ref_value': Reference wind speed value for that hour
        - 'bias': Difference (ref_value - meas_mean)
        - 'n_samples': Number of measurement samples in that hour
        
    And overall correlation metrics:
        - 'corr_mean_ref': Pearson correlation between hourly mean measurement and ref value
        - 'corr_min_ref': Pearson correlation between hourly min measurement and ref value
        - 'corr_max_ref': Pearson correlation between hourly max measurement and ref value
    """
    
    # Extract wind speed from measurement data
    measurement_data = measurement_data.copy()
    
    # Ensure datetime is a DatetimeIndex
    if not isinstance(measurement_data.index, pd.DatetimeIndex):
        measurement_data.index = pd.to_datetime(measurement_data.index)
    
    meas_wind_speed = extract_wind_speed_column(measurement_data)
    measurement_data['wind_speed'] = meas_wind_speed
    
    # Extract hour from measurement data for grouping
    measurement_data['hour'] = measurement_data.index.floor('h')
    
    # Calculate hourly statistics
    hourly_stats = measurement_data.groupby('hour')['wind_speed'].agg([
        'mean', 'min', 'max', 'std', 'count'
    ]).reset_index()
    hourly_stats.columns = ['hour', 'meas_mean', 'meas_min', 'meas_max', 'meas_std', 'n_samples']
    hourly_stats['meas_range'] = hourly_stats['meas_max'] - hourly_stats['meas_min']
    hourly_stats.set_index('hour', inplace=True)
    
    results = {}
    
    for location_name, ref_df in ref_data_dict.items():
        print(f"\nProcessing location: {location_name}")
        
        # Extract wind speed from reference data
        ref_wind_speed = extract_wind_speed_column_ref(ref_df)
        ref_df_copy = ref_df.copy()
        ref_df_copy['wind_speed'] = ref_wind_speed
        
        # Ensure ref data has DatetimeIndex
        if not isinstance(ref_df_copy.index, pd.DatetimeIndex):
            ref_df_copy.index = pd.to_datetime(ref_df_copy.index)
        
        # Extract hour from reference data
        ref_df_copy['hour'] = ref_df_copy.index.floor('h')
        
        # Group reference data by hour (take first value for each hour)
        ref_hourly = ref_df_copy.groupby('hour')['wind_speed'].first()
        
        # Merge measurement and reference data by hour
        merged = hourly_stats.join(ref_hourly.rename('ref_value'), how='inner')
        
        print(f"  Found {len(merged)} common hours")
        
        if len(merged) > 0:
            # Remove rows with NaN in critical columns
            merged_clean = merged.dropna(subset=['meas_mean', 'ref_value'])
            
            print(f"  Valid rows for correlation: {len(merged_clean)}")
            
            # Calculate correlations
            merged_clean['bias'] = merged_clean['ref_value'] - merged_clean['meas_mean']
            
            # Calculate Pearson correlations
            if len(merged_clean) >= 2:
                corr_mean_ref, p_mean_ref = pearsonr(merged_clean['meas_mean'], merged_clean['ref_value'])
                corr_min_ref, p_min_ref = pearsonr(merged_clean['meas_min'], merged_clean['ref_value'])
                corr_max_ref, p_max_ref = pearsonr(merged_clean['meas_max'], merged_clean['ref_value'])
                
                print(f"  Correlation (mean measurement vs ref): {corr_mean_ref:.4f} (p-value: {p_mean_ref:.4e})")
                print(f"  Correlation (min measurement vs ref): {corr_min_ref:.4f} (p-value: {p_min_ref:.4e})")
                print(f"  Correlation (max measurement vs ref): {corr_max_ref:.4f} (p-value: {p_max_ref:.4e})")
                print(f"  Mean bias (ref - meas_mean): {merged_clean['bias'].mean():.4f} m/s")
                print(f"  Std of bias: {merged_clean['bias'].std():.4f} m/s")
            else:
                corr_mean_ref = corr_min_ref = corr_max_ref = np.nan
                p_mean_ref = p_min_ref = p_max_ref = np.nan
            
            # Add correlation metadata to results
            merged_clean['corr_mean_ref'] = corr_mean_ref
            merged_clean['corr_min_ref'] = corr_min_ref
            merged_clean['corr_max_ref'] = corr_max_ref
        else:
            print(f"  No common hours found")
            merged_clean = pd.DataFrame()
        
        results[location_name] = merged_clean
    
    return results


def summarize_correlations(correlation_results):
    """
    Print a summary of correlation results across all locations.
    
    Parameters
    ----------
    correlation_results : dict
        Output from correlate_by_hour function.
    """
    print("\n" + "="*70)
    print("CORRELATION SUMMARY BY LOCATION")
    print("="*70)
    
    for location_name, corr_df in correlation_results.items():
        print(f"\n{location_name}:")
        if len(corr_df) > 0:
            print(f"  Total hours analyzed: {len(corr_df)}")
            print(f"\n  Hourly Statistics:")
            print(f"    Mean measurement: {corr_df['meas_mean'].mean():.4f} m/s ± {corr_df['meas_mean'].std():.4f}")
            print(f"    Mean reference:   {corr_df['ref_value'].mean():.4f} m/s ± {corr_df['ref_value'].std():.4f}")
            print(f"    Mean bias (ref - meas): {corr_df['bias'].mean():.4f} m/s ± {corr_df['bias'].std():.4f}")
            print(f"    Mean hourly range: {corr_df['meas_range'].mean():.4f} m/s")
            
            print(f"\n  Correlation Coefficients (Pearson):")
            if 'corr_mean_ref' in corr_df.columns:
                print(f"    Mean measurement vs reference: {corr_df['corr_mean_ref'].iloc[0]:.4f}")
                print(f"    Min measurement vs reference:  {corr_df['corr_min_ref'].iloc[0]:.4f}")
                print(f"    Max measurement vs reference:  {corr_df['corr_max_ref'].iloc[0]:.4f}")
        else:
            print("  No valid correlations found")
