import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

def train_evaluate_models(monthly_df, target_col='Return_Quantity', use_optimized=False):
    """
    Trains and evaluates multiple regression models using TimeSeriesSplit.
    """
    iteration_label = "Ottimizzati" if use_optimized else "Base"
    print(f"[Modeling] Training and evaluating models ({iteration_label})...")
    
    # Define Features
    # Exclude non-numeric or target columns
    exclude_cols = ['Date', 'YearMonth', target_col, 'Quarter']
    feature_cols = [c for c in monthly_df.columns if c not in exclude_cols]
    
    print(f"[Modeling] Features used: {feature_cols}")
    
    X = monthly_df[feature_cols]
    y = monthly_df[target_col]
    
    # Models
    if use_optimized:
        # Second Iteration: Models with tuned hyperparameters
        models = {
            'SVR': SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1),
            'RandomForest': RandomForestRegressor(n_estimators=50, max_depth=None, min_samples_split=5, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.01, max_depth=5, random_state=42)
        }
    else:
        # First Iteration: Models with default/baseline hyperparameters
        models = {
            'SVR': SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        }
    
    results = {}
    tscv = TimeSeriesSplit(n_splits=3)
    
    best_last_test_prediction = None
    best_last_test_actual = None
    best_rmse = float('inf')
    best_model_name = ""
    
    for name, model in models.items():
        rmse_scores = []
        mae_scores = []
        mape_scores = []
        
        # Cross Validation Loop
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Scaling (Important for SVR, good for others)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-6))) * 100
            
            rmse_scores.append(rmse)
            mae_scores.append(mae)
            mape_scores.append(mape)
            
            # Keep track of the last Fold predictions for the main plot (simplification)
            last_pred = y_pred
            last_actual = y_test
            
        avg_rmse = np.mean(rmse_scores)
        avg_mae = np.mean(mae_scores)
        
        results[name] = {
            'RMSE': avg_rmse,
            'MAE': avg_mae,
            'Model_Object': model, # Trained on last fold
            'Last_Test_Actual': last_actual,
            'Last_Test_Pred': last_pred
        }
        
        # Track feature importances if the model exposes them
        if hasattr(model, 'feature_importances_'):
            results[name]['Feature_Importance'] = dict(zip(X.columns, model.feature_importances_))
        else:
            try:
                from sklearn.inspection import permutation_importance
                # Use permutation importance for models without native support (like SVR rbf)
                # Evaluate on the last fold's training set
                perm_imp = permutation_importance(model, X_train_scaled, y_train, n_repeats=10, random_state=42)
                importances = perm_imp.importances_mean
                # Normalize to make them comparable (0-1 range sum)
                importances = np.maximum(importances, 0)
                if importances.sum() > 0:
                    importances = importances / importances.sum()
                results[name]['Feature_Importance'] = dict(zip(X.columns, importances))
            except Exception:
                results[name]['Feature_Importance'] = None

        
        # Track "Best" for plotting purposes
        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best_model_name = name

    print("[Modeling] Evaluation Complete.")
    return results, best_model_name
