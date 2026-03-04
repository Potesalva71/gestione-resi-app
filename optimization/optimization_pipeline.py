import sys
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error

# Add project root to path to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_dataset
from src.features import create_monthly_features

def optimize_models(monthly_df, output_dir):
    """
    Performs Hyperparameter Tuning using GridSearchCV.
    """
    print("[Optimization] Starting Hyperparameter Tuning...")
    
    target_col = 'Return_Quantity'
    exclude_cols = ['Date', 'YearMonth', target_col, 'Quarter']
    feature_cols = [c for c in monthly_df.columns if c not in exclude_cols]
    
    X = monthly_df[feature_cols]
    y = monthly_df[target_col]
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Time Series Split
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Define Models and Parameter Grids
    # Focusing on Random Forest and Gradient Boosting as SVR is harder to interpret/tune blindly
    
    models_params = {
        'RandomForest': {
            'model': RandomForestRegressor(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            }
        },
        'GradientBoosting': {
            'model': GradientBoostingRegressor(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5]
            }
        }
    }
    
    results = []
    
    for name, mp in models_params.items():
        print(f"[Optimization] Tuning {name}...")
        grid = GridSearchCV(
            estimator=mp['model'],
            param_grid=mp['params'],
            cv=tscv,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
        
        grid.fit(X_scaled, y)
        
        best_rmse = -grid.best_score_
        best_params = grid.best_params_
        
        print(f"  Best RMSE: {best_rmse:.4f}")
        print(f"  Best Params: {best_params}")
        
        results.append({
            'Model': name,
            'Best_RMSE': best_rmse,
            'Best_Params': best_params
        })
        
    # Save Optimization Report
    report_path = os.path.join(output_dir, 'optimization_results.md')
    with open(report_path, 'w') as f:
        f.write("# Optimization Results (Hyperparameter Tuning)\n\n")
        f.write("Evaluation Metric: RMSE (Lower is Better)\n\n")
        
        for res in results:
            f.write(f"## {res['Model']}\n")
            f.write(f"- **Best RMSE**: {res['Best_RMSE']:.4f}\n")
            f.write(f"- **Best Parameters**: `{res['Best_Params']}`\n\n")
            
    print(f"[Optimization] Results saved to {report_path}")

def main():
    PROJECT_DIR = r'c:\Vibe_Coding\ProjectWork_ML'
    DATA_PATH = os.path.join(PROJECT_DIR, 'Dataset_ML_Resi.csv')
    OUTPUT_DIR = os.path.join(PROJECT_DIR, 'optimization')
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    df = load_dataset(DATA_PATH)
    if df is not None:
        monthly_df = create_monthly_features(df)
        optimize_models(monthly_df, OUTPUT_DIR)

if __name__ == "__main__":
    main()
