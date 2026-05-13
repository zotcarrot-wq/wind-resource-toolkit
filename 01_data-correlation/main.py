from read_data import read_data_func

# Path to measurement data
measurement_data_path = r"D:\_10_code\wind-resource-toolkit\99_sample\vortex.les.912025.1year 140m UTC+07.0 ERA5.txt"

# Path to ref data
ref_data_folder_path = r"D:\_10_code\wind-resource-toolkit\99_sample\ref_data"


def main():
    measurement_data, ref_data = read_data_func(measurement_data_path, ref_data_folder_path)
    return None


if __name__ == "__main__":
    # Display the first few rows of the measurement data
    main()
