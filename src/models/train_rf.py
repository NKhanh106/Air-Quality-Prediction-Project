"""
Random Forest Training Module với Optuna optimization
"""

import pandas as pd
import numpy as np
import os
import joblib
import json
import optuna
from optuna.pruners import MedianPruner
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SelectFromModel

from ..features.engineering import create_all_features, prepare_rf_features
from ..utils.paths import get_data_path, get_model_path, ensure_dirs

# Ensure directories exist
ensure_dirs()

def train_random_forest(use_advanced_features=True, lag_steps=12, n_trials=150, max_features=150):
    """
    Train Random Forest model
    Args:
        use_advanced_features: True = dùng features nâng cao, False = chỉ lag features
        lag_steps: số lag steps
        n_trials: số trials cho Optuna
        max_features: số lượng features tối đa sau khi selection (default: 150)
    """
    print("=" * 60)
    if use_advanced_features:
        print("Training Random Forest với Feature Engineering...")
    else:
        print("Training Random Forest với lag features...")
    print("=" * 60)
    
    print("\n📂 Đang load dữ liệu...")
    csv_path = get_data_path("FinalData.csv", "processed")
    df = pd.read_csv(csv_path, encoding='utf8')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values('Date').reset_index(drop=True)
    print(f"✅ Đã load {len(df)} rows")
    
    target_cols = ['co', 'no2', 'o3', 'pm10', 'pm25', 'so2', 
                   'Temp', 'Rain', 'Cloud', 'Pressure', 'Wind', 'Gust']
    
    if use_advanced_features:
        print("\n🔧 Đang tạo features nâng cao...")
        df_features = create_all_features(
            df,
            lag_steps=[1, 2, 3, 7, 14, 30],
            rolling_windows=[3, 7, 14, 30]
        )
        X, y, dates = prepare_rf_features(df_features, target_cols)
    else:
        print(f"\n🔧 Đang tạo lag features (lag_steps={lag_steps})...")
        df_features = df.copy()
        
        lag_features = {}
        for lag in range(1, lag_steps + 1):
            for col in target_cols:
                lag_features[f'{col}_t-{lag}'] = df_features[col].shift(lag)
        
        lag_df = pd.DataFrame(lag_features, index=df_features.index)
        df_features = pd.concat([df_features, lag_df], axis=1)
        df_features = df_features[lag_steps:-1].reset_index(drop=True)
        
        features = list(lag_features.keys())
        X = df_features[features]
        y = df_features[target_cols]
        print(f"✅ Đã tạo {len(features)} lag features")
    
    # Feature selection nếu có quá nhiều features
    if len(X.columns) > max_features:
        print(f"\n🔍 Đang chọn {max_features} features quan trọng nhất từ {len(X.columns)} features...")
        
        # Train baseline RF để lấy feature importance
        baseline_rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        
        # Chia train/test để tính importance
        test_size = int(0.2 * len(X))
        X_train_sel = X[:-test_size]
        y_train_sel = y[:-test_size]
        
        # Tính importance cho từng target (multi-output)
        importances_all = []
        for target_idx in range(y_train_sel.shape[1]):
            baseline_rf.fit(X_train_sel, y_train_sel.iloc[:, target_idx])
            importances_all.append(baseline_rf.feature_importances_)
        
        # Tính mean importance across all targets
        mean_importance = np.mean(importances_all, axis=0)
        
        # Chọn top features
        feature_importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': mean_importance
        }).sort_values('importance', ascending=False)
        
        selected_features = feature_importance_df.head(max_features)['feature'].tolist()
        X = X[selected_features]
        
        print(f"✅ Đã chọn {len(selected_features)} features quan trọng nhất")
        print(f"   Top 10 features: {selected_features[:10]}")
    else:
        print(f"✅ Số features ({len(X.columns)}) đã phù hợp, không cần selection")
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    X_array = X.values
    y_array = y.values
    
    print(f"\n🔍 Đang tối ưu hyperparameters với Optuna (n_trials={n_trials})...")
    
    def objective(trial):  
        n_estimators = trial.suggest_int('n_estimators', 100, 500)
        max_depth = trial.suggest_int('max_depth', 5, 30)
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.7])
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        bootstrap = trial.suggest_categorical('bootstrap', [True, False])
        
        params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'max_features': max_features,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'bootstrap': bootstrap,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # max_samples: tỷ lệ samples dùng cho mỗi tree (chỉ khi bootstrap=True)
        if bootstrap:
            max_samples = trial.suggest_float('max_samples', 0.6, 1.0)
            params['max_samples'] = max_samples
        
        model = RandomForestRegressor(**params)
        
        scores = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_array)):
            X_train_cv = X_array[train_idx]
            y_train_cv = y_array[train_idx]
            X_val_cv = X_array[val_idx]
            y_val_cv = y_array[val_idx]
            
            model.fit(X_train_cv, y_train_cv)
            y_pred_cv = model.predict(X_val_cv)
            
            rmse = np.sqrt(mean_squared_error(y_val_cv, y_pred_cv, multioutput='uniform_average'))
            scores.append(rmse)
            
            trial.report(rmse, fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return np.mean(scores)
    
    # Tắt logging verbose của Optuna nhưng giữ progress bar
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    study = optuna.create_study(
        direction='minimize',
        study_name='random_forest_optimization',
        pruner=pruner
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print("\n" + "=" * 60)
    print("Kết quả Optuna Optimization:")
    print("=" * 60)
    print(f"Best params: {study.best_params}")
    print(f"Best CV RMSE: {study.best_value:.4f}")
    
    best_params = study.best_params.copy()
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1
    
    # max_depth đã được set trực tiếp trong objective, không cần xử lý thêm
    # max_samples chỉ có khi bootstrap=True, nếu không có thì RandomForest sẽ dùng default
    
    best_model = RandomForestRegressor(**best_params)
    
    print("\n📊 Đánh giá trên test set (20% cuối)...")
    test_size = int(0.2 * len(X))
    X_train = X[:-test_size]
    y_train = y[:-test_size]
    X_test = X[-test_size:]
    y_test = y[-test_size:]
    
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    
    print("\n" + "=" * 60)
    print("Metrics trên Test Set:")
    print("=" * 60)
    print(f"{'Feature':<12} {'R² Score':<12} {'MAE':<12} {'RMSE':<12}")
    print("-" * 60)
    
    metrics_summary = {}
    for i, feature in enumerate(target_cols):
        r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
        mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i]))
        
        metrics_summary[feature] = {
            'r2': float(r2),
            'mae': float(mae),
            'rmse': float(rmse)
        }
        
        print(f"{feature:<12} {r2:<12.4f} {mae:<12.4f} {rmse:<12.4f}")
    
    print("\n" + "=" * 60)
    print("Top 20 Features quan trọng nhất:")
    print("=" * 60)
    
    if len(best_model.feature_importances_.shape) > 1:
        importance_mean = best_model.feature_importances_.mean(axis=0)
    else:
        importance_mean = best_model.feature_importances_
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': importance_mean
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    print(feature_importance.head(20).to_string(index=False))
    
    model_path = get_model_path('random_forest_model.pkl', 'rf')
    joblib.dump(best_model, model_path)
    print(f"\n✅ Đã lưu model tại: {model_path}")
    
    feature_names_path = get_model_path('rf_feature_names.pkl', 'rf')
    joblib.dump(list(X.columns), feature_names_path)
    print(f"✅ Đã lưu feature names tại: {feature_names_path}")
    
    metrics_path = get_model_path('rf_metrics.json', 'rf')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"✅ Đã lưu metrics tại: {metrics_path}")
    
    importance_path = get_model_path('rf_feature_importance.csv', 'rf')
    feature_importance.to_csv(importance_path, index=False)
    print(f"✅ Đã lưu feature importance tại: {importance_path}")
    
    print("\n" + "=" * 60)
    print("Hoàn thành training!")
    print("=" * 60)
    print(f"\n📊 Tổng số features: {len(X.columns)}")
    print(f"📊 Số samples train: {len(X_train)}")
    print(f"📊 Số samples test: {len(X_test)}")

