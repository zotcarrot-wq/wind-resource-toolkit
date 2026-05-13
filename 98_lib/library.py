import pandas as pd
import os

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
    return df