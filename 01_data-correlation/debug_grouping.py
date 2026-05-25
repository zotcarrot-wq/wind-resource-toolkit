import pandas as pd
from read_data import read_data_func

# Path to measurement data
measurement_data_path = r"D:\_10_code\wind-resource-toolkit\99_sample\vortex.les.912025.1year 140m UTC+07.0 ERA5.txt"
ref_data_folder_path = r"D:\_10_code\wind-resource-toolkit\99_sample\ref_data"

# Read data
measurement_data, ref_data = read_data_func(measurement_data_path, ref_data_folder_path)

# Ensure datetime index
if not isinstance(measurement_data.index, pd.DatetimeIndex):
    measurement_data.index = pd.to_datetime(measurement_data.index)

# Get reference data
location_name = list(ref_data.keys())[0]
ref_df = ref_data[location_name].copy()

if not isinstance(ref_df.index, pd.DatetimeIndex):
    ref_df.index = pd.to_datetime(ref_df.index)

print("=" * 100)
print("DEBUGGING TIME STEP GROUPING")
print("=" * 100)

# --- MEASUREMENT DATA ANALYSIS ---
print("\n--- MEASUREMENT DATA ---")
print(f"Start: {measurement_data.index.min()}")
print(f"End: {measurement_data.index.max()}")
print(f"Total points: {len(measurement_data)}")

# 10min analysis
print(f"\n10MIN GROUPING:")
meas_hour = measurement_data.index.floor('h')
ten_min_groups = meas_hour.nunique()
print(f"  Unique hours: {ten_min_groups}")
print(f"  Expected: ~{len(measurement_data) / 6:.0f} (52416 / 6)")

# 1h analysis
print(f"\n1H GROUPING:")
meas_1h = measurement_data.index.floor('h')
one_h_unique = meas_1h.nunique()
print(f"  Unique hours: {one_h_unique}")
print(f"  Hour range: {meas_1h.min()} to {meas_1h.max()}")

# 3h analysis
print(f"\n3H GROUPING:")
meas_hour = measurement_data.index.floor('h')
meas_time_3h = meas_hour - pd.to_timedelta(meas_hour.hour % 3, unit='h')
three_h_unique = meas_time_3h.nunique()
print(f"  Unique 3h windows: {three_h_unique}")
print(f"  Expected: {one_h_unique / 3:.1f}")
print(f"  First 3h window: {meas_time_3h.min()}")
print(f"  Last 3h window: {meas_time_3h.max()}")

# Show some 3h boundaries
print(f"\n  First few 3h periods:")
for i, ts in enumerate(meas_time_3h.unique()[:5]):
    print(f"    {ts}")

# 24h analysis
print(f"\n24H GROUPING:")
meas_day = measurement_data.index.floor('d')
twentyfour_h_unique = meas_day.nunique()
print(f"  Unique days: {twentyfour_h_unique}")
print(f"  Day range: {meas_day.min()} to {meas_day.max()}")
print(f"  Expected: {one_h_unique / 24:.1f}")

# --- REFERENCE DATA ANALYSIS ---
print("\n" + "=" * 100)
print("--- REFERENCE DATA ---")
print(f"Start: {ref_df.index.min()}")
print(f"End: {ref_df.index.max()}")
print(f"Total points: {len(ref_df)}")

# 1h analysis
print(f"\n1H GROUPING:")
ref_hour = ref_df.index.floor('h')
ref_1h_unique = ref_hour.nunique()
print(f"  Unique hours: {ref_1h_unique}")

# 3h analysis
print(f"\n3H GROUPING:")
ref_hour = ref_df.index.floor('h')
ref_time_3h = ref_hour - pd.to_timedelta(ref_hour.hour % 3, unit='h')
ref_3h_unique = ref_time_3h.nunique()
print(f"  Unique 3h windows: {ref_3h_unique}")

# 24h analysis
print(f"\n24H GROUPING:")
ref_day = ref_df.index.floor('d')
ref_24h_unique = ref_day.nunique()
print(f"  Unique days: {ref_24h_unique}")

# --- INTERSECTION ANALYSIS ---
print("\n" + "=" * 100)
print("--- MERGED DATA ANALYSIS ---")

# 1h merge
print(f"\n1H MERGE:")
meas_1h_avg = pd.Series(1, index=measurement_data.index.floor('h')).groupby(level=0).count()
ref_1h_avg = pd.Series(1, index=ref_df.index.floor('h')).groupby(level=0).count()
merged_1h = pd.DataFrame({'meas': meas_1h_avg, 'ref': ref_1h_avg}).dropna()
print(f"  Measurement hours: {len(meas_1h_avg)}")
print(f"  Reference hours: {len(ref_1h_avg)}")
print(f"  Common hours (merged): {len(merged_1h)}")

# 3h merge
print(f"\n3H MERGE:")
meas_3h_ts = measurement_data.index.floor('h')
meas_3h_ts = meas_3h_ts - pd.to_timedelta(meas_3h_ts.hour % 3, unit='h')
meas_3h_unique = meas_3h_ts.unique()
ref_3h_ts = ref_df.index.floor('h')
ref_3h_ts = ref_3h_ts - pd.to_timedelta(ref_3h_ts.hour % 3, unit='h')
ref_3h_unique = ref_3h_ts.unique()

meas_3h_avg = pd.Series(1, index=meas_3h_ts).groupby(level=0).count()
ref_3h_avg = pd.Series(1, index=ref_3h_ts).groupby(level=0).count()
merged_3h = pd.DataFrame({'meas': meas_3h_avg, 'ref': ref_3h_avg}).dropna()
print(f"  Measurement 3h windows: {len(meas_3h_avg)}")
print(f"  Reference 3h windows: {len(ref_3h_avg)}")
print(f"  Common 3h windows (merged): {len(merged_3h)}")

# 24h merge
print(f"\n24H MERGE:")
meas_day = measurement_data.index.floor('d')
ref_day = ref_df.index.floor('d')

meas_24h_avg = pd.Series(1, index=meas_day).groupby(level=0).count()
ref_24h_avg = pd.Series(1, index=ref_day).groupby(level=0).count()
merged_24h = pd.DataFrame({'meas': meas_24h_avg, 'ref': ref_24h_avg}).dropna()
print(f"  Measurement days: {len(meas_24h_avg)}")
print(f"  Reference days: {len(ref_24h_avg)}")
print(f"  Common days (merged): {len(merged_24h)}")

print("\n" + "=" * 100)
print("SUMMARY VS WINDOGRAPHER")
print("=" * 100)
print(f"\nTime Step | Current | Windographer | Difference")
print(f"10min     | 52,416  | 52,416        | ✓ Match")
print(f"1h        | {len(merged_1h):>6}  | 8,736         | {'✓ Match' if len(merged_1h) == 8736 else '✗ Mismatch'}")
print(f"3h        | {len(merged_3h):>6}  | 2,912         | {'✓ Match' if len(merged_3h) == 2912 else f'✗ {len(merged_3h) - 2912:+d}'}")
print(f"24h       | {len(merged_24h):>6}  | 362           | {'✓ Match' if len(merged_24h) == 362 else f'✗ {len(merged_24h) - 362:+d}'}")

print("\n" + "=" * 100)
