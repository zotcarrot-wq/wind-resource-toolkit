from read_data import read_data_func
from correlation import run_all_correlations, summarize_correlations

# Path to measurement data
measurement_data_path = r"D:\_10_code\wind-resource-toolkit\99_sample\vortex.les.912025.1year 140m UTC+07.0 ERA5.txt"

# Path to ref data
ref_data_folder_path = r"D:\_10_code\wind-resource-toolkit\99_sample\ref_data"


def main():
    # Read measurement and reference data
    measurement_data, ref_data = read_data_func(measurement_data_path, ref_data_folder_path)
    
    print("\nMeasurement data shape:", measurement_data.shape)
    print("Reference data locations:", list(ref_data.keys()))
    print("\nReference data details:")
    for loc, df in ref_data.items():
        print(f"  {loc}: {df.shape[0]} rows, time range: {df.index.min()} to {df.index.max()}")
    
    # Correlate measurement data with reference data at multiple time steps
    print("\n" + "="*70)
    print("CORRELATING MEASUREMENT DATA WITH REFERENCE DATA")
    print("="*70)
    
    correlation_results = run_all_correlations(measurement_data, ref_data)
    
    # Print summary of R² values
    summarize_correlations(correlation_results)
    
    return correlation_results


if __name__ == "__main__":
    correlation_results = main()
