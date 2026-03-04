import pandas as pd
import numpy as np

def create_monthly_features(df):
    """
    Aggregates data to monthly level and creates time-series features.
    """
    print("[Features] Engineering monthly features...")
    
    # Ensure Date
    if not pd.api.types.is_datetime64_any_dtype(df['DATA_OPERAZIONE']):
        df['DATA_OPERAZIONE'] = pd.to_datetime(df['DATA_OPERAZIONE'])

    # Create YearMonth for aggregation
    df['YearMonth'] = df['DATA_OPERAZIONE'].dt.to_period('M')
    
    # Aggregation
    monthly_sales = df[df['CAUSALE_DEL_MOVIMENTO'] == 'VEN'].groupby('YearMonth')['QUANTITA_MOVIMENTATA'].sum().rename('Sales_Quantity')
    monthly_returns = df[df['CAUSALE_DEL_MOVIMENTO'] == 'RES'].groupby('YearMonth')['QUANTITA_MOVIMENTATA'].sum().rename('Return_Quantity')
    
    # Merge
    monthly_features = pd.merge(monthly_sales, monthly_returns, on='YearMonth', how='outer').fillna(0)
    
    # Convert 'YearMonth' index to column and timestamp (start of month)
    monthly_features = monthly_features.reset_index()
    monthly_features['Date'] = monthly_features['YearMonth'].dt.to_timestamp()
    monthly_features = monthly_features.sort_values('Date').set_index('Date') # Set Date as index for easy shifting/rolling
    
    # --- Feature Engineering ---
    
    # 1. Lags (Past Returns and Sales)
    for lag in [1, 2, 3]:
        monthly_features[f'Sales_Lag{lag}'] = monthly_features['Sales_Quantity'].shift(lag)
        monthly_features[f'Returns_Lag{lag}'] = monthly_features['Return_Quantity'].shift(lag)
        
    # 2. Rolling Statistics (on Returns)
    monthly_features['Returns_RollMean_3'] = monthly_features['Return_Quantity'].shift(1).rolling(window=3).mean()
    monthly_features['Returns_RollStd_3'] = monthly_features['Return_Quantity'].shift(1).rolling(window=3).std()
    
    # 3. Calendar Features
    monthly_features['Month'] = monthly_features.index.month
    monthly_features['Quarter'] = monthly_features.index.quarter
    
    # 4. Differencing (for Stationarity - optional feature)
    # monthly_features['Returns_Diff1'] = monthly_features['Return_Quantity'].diff()

    # Drop NaNs created by Lags/Rolling
    monthly_features.dropna(inplace=True)
    
    # Reset index to keep Date as column if preferred, or return as is. The previous script used 'Date' column.
    monthly_features.reset_index(inplace=True)

    print(f"[Features] Monthly Features Shape: {monthly_features.shape}")
    return monthly_features

def calculate_estimated_price(df):
    """
    Parses 'DESCRIZIONE_FASCIA_DI_PREZZO' to estimate unit price and total value.
    Format example: 'da 0 a 125', 'da 126 a 250'.
    """
    print("[Features] Calculating Estimated Value...")
    
    def parse_price_range(desc):
        if pd.isna(desc):
            return 0
        try:
            # Remove 'da ' and split by ' a '
            parts = desc.lower().replace('da ', '').split(' a ')
            if len(parts) == 2:
                min_val = float(parts[0])
                max_val = float(parts[1])
                return round((min_val + max_val) / 2)
            return 0
        except:
            return 0

    if 'DESCRIZIONE_FASCIA_DI_PREZZO' in df.columns:
        df['Stima_Prezzo_Unitario'] = df['DESCRIZIONE_FASCIA_DI_PREZZO'].apply(parse_price_range)
        # Total Value = Unit Price * Quantity (Use ABS for returns to have positive value magnitude, or keep sign for financial flow)
        # Usually Returns Value is negative in financial books, but for "Magnitude of Returns" we might want positive.
        # Let's keep the sign of QUANTITA_MOVIMENTATA so Returns are negative value (cost/refund) and Sales positive.
        df['Valore_Totale'] = df['Stima_Prezzo_Unitario'] * df['QUANTITA_MOVIMENTATA']
    else:
        print("[Warning] 'DESCRIZIONE_FASCIA_DI_PREZZO' not found.")
        df['Stima_Prezzo_Unitario'] = 0
        df['Valore_Totale'] = 0
        
    return df
