import pandas as pd
import os

def load_dataset(filepath):
    """
    Loads the dataset handling encoding and separator issues.
    """
    print(f"[Data Loader] Loading from {filepath}...")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        # Try default UTF-8 with BOM
        df = pd.read_csv(filepath, sep=';', encoding='utf-8-sig', parse_dates=['DATA_OPERAZIONE'])
    except Exception as e:
        print(f"[Data Loader] warning: utf-8-sig failed ({e}), trying latin1...")
        try:
            df = pd.read_csv(filepath, sep=';', encoding='latin1', parse_dates=['DATA_OPERAZIONE'])
        except Exception as e2:
            raise ValueError(f"Failed to load dataset: {e2}")

    # Basic Cleaning
    # Drop rows with critical missing values
    df.dropna(subset=['CODICE_CLIENTE', 'QUANTITA_MOVIMENTATA'], inplace=True)
    
    # Ensure Numeric
    df['QUANTITA_MOVIMENTATA'] = pd.to_numeric(df['QUANTITA_MOVIMENTATA'], errors='coerce')
    
    # Fill remaining NaNs in quantity with 0
    df['QUANTITA_MOVIMENTATA'] = df['QUANTITA_MOVIMENTATA'].fillna(0)

    print(f"[Data Loader] Dataset loaded. Shape: {df.shape}")
    return df
