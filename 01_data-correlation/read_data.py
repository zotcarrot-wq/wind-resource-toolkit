import pandas as pd
import os

# Path to measurement data
measurement_data_path = r"D:\_10_code\wind-resource-toolkit\99_sample\vortex.les.912025.1year 140m UTC+07.0 ERA5.txt"

# Path to ref data
ref_data_folder_path = r"D:\_10_code\wind-resource-toolkit\99_sample\ref_data"

def read_vortex_les_txt(file_path):
    """
    Read Vortex LES measurement data TXT file.
    
    Skips metadata lines and reads the actual data table starting from the header line
    containing "YYYYMMDD" or "HHMM".
    
    Parameters
    ----------
    file_path : str
        Path to Vortex LES file.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with datetime index.
    """
    # Find the header line (contains YYYYMMDD or HHMM)
    header_row_index = None
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if 'YYYYMMDD' in line or (i > 0 and 'HHMM' in line):
                header_row_index = i
                break
    
    if header_row_index is None:
        raise ValueError("Could not find header line with 'YYYYMMDD' or 'HHMM'")
    
    # Read the file starting from the header
    df = pd.read_csv(
        file_path,
        sep=r'\s+',  # Use regex to handle variable whitespace
        skiprows=header_row_index,
        engine='python'
    )
    
    # Parse datetime from YYYYMMDD and HHMM columns
    if 'YYYYMMDD' in df.columns and 'HHMM' in df.columns:
        # Combine date and time
        date_time_str = df['YYYYMMDD'].astype(str) + ' ' + df['HHMM'].astype(str).str.zfill(4)
        df['datetime'] = pd.to_datetime(date_time_str, format='%Y%m%d %H%M', errors='coerce')
        df = df.set_index('datetime')
        df = df.drop(['YYYYMMDD', 'HHMM'], axis=1)
    
    print("Successfully read Vortex LES file: ", file_path)
    return df


def detect_and_read_file(file_path):
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    if ext in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    elif ext == '.csv':
        df = pd.read_csv(file_path)
    elif ext == '.txt':
        # Try to detect delimiter by reading the first line
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            if ',' in first_line:
                delimiter = ','
            elif '\t' in first_line:
                delimiter = '\t'
            elif ';' in first_line:
                delimiter = ';'
            else:
                delimiter = None  # Default: whitespace
        df = pd.read_csv(file_path, delimiter=delimiter)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    print("Successfully read file: ", file_path)
    return df


def read_windpro_reanalysis_txt(
    file_path: str,
    header_start_keyword: str = "TimeStamp",
    timezone: str | None = None,
) -> pd.DataFrame:
    """
    Read WindPRO re-analysis (ERA5, MERRA-2, etc.) TXT export into a clean pandas DataFrame.

    Scope:
    - Time-series WindPRO Meteo Data Export (TXT)
    - Re-analysis data (ERA5 Gaussian/Rectangular, MERRA-2)
    - No aggregation exports

    Parameters
    ----------
    file_path : str
        Path to WindPRO TXT file.
    header_start_keyword : str, default "TimeStamp"
        Keyword identifying the start of the data table.
    timezone : str or None
        Optional timezone label (e.g. 'UTC+07').
        NOTE: Label only; no time conversion is performed.

    Returns
    -------
    pd.DataFrame
        Time-indexed DataFrame containing numeric data and status columns.
    """

    # --------------------------------------------------
    # 1. Locate the start of the data table
    # --------------------------------------------------
    header_row_index = None

    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip().startswith(header_start_keyword):
                header_row_index = i
                break

    if header_row_index is None:
        raise ValueError(
            f"Could not find table header starting with '{header_start_keyword}'"
        )

    # --------------------------------------------------
    # 2. Read the table (WindPRO uses tab delimiter)
    # --------------------------------------------------
    df = pd.read_csv(
        file_path,
        sep="\t",
        skiprows=header_row_index,
        engine="python",
    )

    # --------------------------------------------------
    # 3. Basic column cleanup
    # --------------------------------------------------
    # Drop fully empty columns
    df = df.dropna(axis=1, how="all")

    # Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]

    # --------------------------------------------------
    # 4. Minimal improvement:
    #    Remove WindPRO units row if present
    # --------------------------------------------------
    # WindPRO typically inserts a second header row containing units like [m/s]
    if df.iloc[0].astype(str).str.contains(r"\[.*\]").any():
        df = df.iloc[1:].reset_index(drop=True)

    # --------------------------------------------------
    # 5. Parse timestamp
    # --------------------------------------------------
    if "TimeStamp" not in df.columns:
        raise ValueError("Expected 'TimeStamp' column not found")

    df["TimeStamp"] = pd.to_datetime(
        df["TimeStamp"],
        format="%d-%b-%y %I:%M %p",
        errors="coerce",
    )

    # Drop rows with invalid timestamps
    df = df.dropna(subset=["TimeStamp"])

    # Set time index
    df = df.set_index("TimeStamp")

    # --------------------------------------------------
    # 6. Optional timezone labeling (no conversion)
    # --------------------------------------------------
    if timezone is not None:
        df.index = df.index.tz_localize(
            timezone,
            nonexistent="NaT",
            ambiguous="NaT",
        )

    return df

def convert_underscore_filename(file_name):
    # Remove the file extension
    name_without_ext = os.path.splitext(file_name)[0]
    # Replace underscores with spaces
    converted_name = name_without_ext.replace(' ', '_')
    return converted_name

def read_data_func(measurement_data_path, ref_data_folder_path):
    # Use specialized reader for Vortex LES measurement data
    df_measurement = read_vortex_les_txt(measurement_data_path)

    # (Optional) Display the first few rows
    # print(df_measurement.head())

    # List all files in the folder
    files = os.listdir(ref_data_folder_path)

    # # Filter for Excel files
    data_files = [
        f for f in files
        if f.lower().endswith(('.xlsx', '.xls', '.csv', '.txt'))
    ]

    # Read all Excel files into a dictionary of DataFrames
    ref_data = {}
    for file in data_files:
        file_path = os.path.join(ref_data_folder_path, file)

    file_id = convert_underscore_filename(file)
    # print(file_id)
    ref_data[file_id] = read_windpro_reanalysis_txt(file_path)
    print("Done processing file: ", file)
    return df_measurement, ref_data