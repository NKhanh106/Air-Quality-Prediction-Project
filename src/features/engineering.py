"""
Feature Engineering Module cho Air Quality Prediction
Tạo các features mạnh mẽ cho Random Forest và các mô hình ML khác
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def create_lag_features(df: pd.DataFrame, 
                       columns: List[str], 
                       lag_steps: List[int] = [1, 2, 3, 7, 14, 30],
                       use_concat: bool = True) -> pd.DataFrame:
    """
    Tạo lag features một cách hiệu quả
    Args:
        df: DataFrame với Date column
        columns: danh sách columns cần tạo lag
        lag_steps: danh sách số ngày lag (mặc định: 1,2,3,7,14,30)
        use_concat: sử dụng concat thay vì loop (nhanh hơn)
    Returns:
        DataFrame với lag features
    """
    df = df.copy()
    df = df.sort_values('Date').reset_index(drop=True)
    
    lag_features = {}
    
    for col in columns:
        if col not in df.columns:
            continue
        for lag in lag_steps:
            lag_features[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    # Sử dụng concat để tránh fragmentation warning
    if lag_features:
        lag_df = pd.DataFrame(lag_features, index=df.index)
        df = pd.concat([df, lag_df], axis=1)
    
    return df


def create_rolling_features(df: pd.DataFrame,
                           columns: List[str],
                           windows: List[int] = [3, 7, 14, 30],
                           stats: List[str] = ['mean', 'std', 'min', 'max']) -> pd.DataFrame:
    """
    Tạo rolling statistics features
    Args:
        df: DataFrame với Date column
        columns: danh sách columns cần tính rolling stats
        windows: danh sách window sizes (số ngày)
        stats: danh sách statistics ['mean', 'std', 'min', 'max', 'median']
    Returns:
        DataFrame với rolling features
    """
    df = df.copy()
    df = df.sort_values('Date').reset_index(drop=True)
    
    rolling_features = {}
    
    for col in columns:
        if col not in df.columns:
            continue
        for window in windows:
            rolling = df[col].rolling(window=window, min_periods=1)
            
            if 'mean' in stats:
                rolling_features[f'{col}_rolling_mean_{window}'] = rolling.mean()
            if 'std' in stats:
                rolling_features[f'{col}_rolling_std_{window}'] = rolling.std()
            if 'min' in stats:
                rolling_features[f'{col}_rolling_min_{window}'] = rolling.min()
            if 'max' in stats:
                rolling_features[f'{col}_rolling_max_{window}'] = rolling.max()
            if 'median' in stats:
                rolling_features[f'{col}_rolling_median_{window}'] = rolling.median()
    
    if rolling_features:
        rolling_df = pd.DataFrame(rolling_features, index=df.index)
        df = pd.concat([df, rolling_df], axis=1)
    
    return df


def create_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo seasonal/time features đơn giản: chỉ ngày trong tuần và tháng trong năm
    Args:
        df: DataFrame với Date column
    Returns:
        DataFrame với seasonal features
    """
    df = df.copy()
    
    if 'Date' not in df.columns:
        raise ValueError("DataFrame phải có cột 'Date'")
    
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Chỉ tạo 2 features cơ bản:
    # 1. Ngày trong tuần (0=Monday, 6=Sunday)
    df['dayofweek'] = df['Date'].dt.dayofweek
    
    # 2. Tháng trong năm (1-12)
    df['month'] = df['Date'].dt.month
    
    return df


def create_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo ratio features giữa các pollutants
    Args:
        df: DataFrame với pollutant columns
    Returns:
        DataFrame với ratio features
    """
    df = df.copy()
    
    # PM2.5/PM10 ratio (quan trọng cho chất lượng không khí)
    if 'pm25' in df.columns and 'pm10' in df.columns:
        df['pm25_pm10_ratio'] = df['pm25'] / (df['pm10'] + 1e-6)  # Tránh chia 0
    
    # O3/NO2 ratio (chỉ số phản ứng quang hóa)
    if 'o3' in df.columns and 'no2' in df.columns:
        df['o3_no2_ratio'] = df['o3'] / (df['no2'] + 1e-6)
    
    # PM/Weather ratios
    if 'pm25' in df.columns and 'Wind' in df.columns:
        df['pm25_wind_ratio'] = df['pm25'] / (df['Wind'] + 1e-6)
    
    if 'pm10' in df.columns and 'Rain' in df.columns:
        df['pm10_rain_ratio'] = df['pm10'] / (df['Rain'] + 1e-6)
    
    # Temperature normalized pollutants
    if 'pm25' in df.columns and 'Temp' in df.columns:
        df['pm25_temp_ratio'] = df['pm25'] / (df['Temp'] + 1e-6)
    
    return df


def create_trend_features(df: pd.DataFrame,
                         columns: List[str],
                         windows: List[int] = [3, 7, 14]) -> pd.DataFrame:
    """
    Tạo trend features (slope, change rate)
    Args:
        df: DataFrame với Date column
        columns: danh sách columns cần tính trend
        windows: danh sách window sizes
    Returns:
        DataFrame với trend features
    """
    df = df.copy()
    df = df.sort_values('Date').reset_index(drop=True)
    
    trend_features = {}
    
    for col in columns:
        if col not in df.columns:
            continue
        
        for window in windows:
            # Slope (độ dốc) - xu hướng tăng/giảm
            # Tối ưu: dùng diff thay vì polyfit để nhanh hơn
            trend_features[f'{col}_slope_{window}'] = (
                (df[col] - df[col].shift(window)) / window
            )
            
            # Change rate (% thay đổi)
            trend_features[f'{col}_change_rate_{window}'] = (
                (df[col] - df[col].shift(window)) / (df[col].shift(window) + 1e-6) * 100
            )
            
            # Difference (chênh lệch tuyệt đối)
            trend_features[f'{col}_diff_{window}'] = df[col] - df[col].shift(window)
    
    if trend_features:
        trend_df = pd.DataFrame(trend_features, index=df.index)
        df = pd.concat([df, trend_df], axis=1)
    
    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo interaction features giữa weather và pollutants
    Args:
        df: DataFrame với weather và pollutant columns
    Returns:
        DataFrame với interaction features
    """
    df = df.copy()
    
    # Wind * Pressure (ảnh hưởng đến dispersion)
    if 'Wind' in df.columns and 'Pressure' in df.columns:
        df['wind_pressure_interaction'] = df['Wind'] * df['Pressure'] / 1000
    
    # Temp * Humidity proxy (Cloud có thể làm proxy)
    if 'Temp' in df.columns and 'Cloud' in df.columns:
        df['temp_cloud_interaction'] = df['Temp'] * df['Cloud'] / 100
    
    # Rain * Wind (ảnh hưởng đến cleaning effect)
    if 'Rain' in df.columns and 'Wind' in df.columns:
        df['rain_wind_interaction'] = df['Rain'] * df['Wind']
    
    # PM * Wind (dispersion effect)
    if 'pm25' in df.columns and 'Wind' in df.columns:
        df['pm25_wind_interaction'] = df['pm25'] * df['Wind']
    
    if 'pm10' in df.columns and 'Wind' in df.columns:
        df['pm10_wind_interaction'] = df['pm10'] * df['Wind']
    
    return df


def create_all_features(df: pd.DataFrame,
                       pollutant_cols: Optional[List[str]] = None,
                       weather_cols: Optional[List[str]] = None,
                       lag_steps: List[int] = [1, 2, 3, 7, 14, 30],
                       rolling_windows: List[int] = [3, 7, 14, 30],
                       remove_original: bool = False) -> pd.DataFrame:
    """
    Tạo tất cả features một lúc
    Args:
        df: DataFrame với Date column
        pollutant_cols: danh sách pollutant columns (mặc định: tự động detect)
        weather_cols: danh sách weather columns (mặc định: tự động detect)
        lag_steps: danh sách lag steps
        rolling_windows: danh sách rolling windows
        remove_original: có xóa original columns không (False để giữ lại)
    Returns:
        DataFrame với tất cả features
    """
    if pollutant_cols is None:
        pollutant_cols = ['co', 'no2', 'o3', 'pm10', 'pm25', 'so2']
        pollutant_cols = [col for col in pollutant_cols if col in df.columns]
    
    if weather_cols is None:
        weather_cols = ['Temp', 'Rain', 'Cloud', 'Pressure', 'Wind', 'Gust']
        weather_cols = [col for col in weather_cols if col in df.columns]
    
    all_cols = pollutant_cols + weather_cols
    
    print("🔄 Đang tạo lag features...")
    df = create_lag_features(df, all_cols, lag_steps=lag_steps)
    
    print("🔄 Đang tạo rolling features...")
    df = create_rolling_features(df, all_cols, windows=rolling_windows)
    
    print("🔄 Đang tạo seasonal features...")
    df = create_seasonal_features(df)
    
    print("🔄 Đang tạo ratio features...")
    df = create_ratio_features(df)
    
    print("🔄 Đang tạo trend features...")
    df = create_trend_features(df, pollutant_cols, windows=[3, 7, 14])
    
    print("🔄 Đang tạo interaction features...")
    df = create_interaction_features(df)
    
    print(f"✅ Đã tạo features. Tổng số features: {len(df.columns)}")
    
    return df


def prepare_rf_features(df: pd.DataFrame,
                        target_cols: List[str],
                        remove_date: bool = True) -> tuple:
    """
    Chuẩn bị features và targets cho Random Forest
    Args:
        df: DataFrame đã có tất cả features
        target_cols: danh sách target columns
        remove_date: có xóa Date column không
    Returns:
        X (features), y (targets) DataFrames, dates (optional)
    """
    df = df.copy()
    
    # Loại bỏ rows có NaN (từ lag/rolling features)
    initial_rows = len(df)
    df = df.dropna().reset_index(drop=True)
    dropped_rows = initial_rows - len(df)
    if dropped_rows > 0:
        print(f"⚠️  Đã loại bỏ {dropped_rows} rows có NaN")
    
    # Tách features và targets
    exclude_cols = set(target_cols + ['Date'])
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    if not feature_cols:
        raise ValueError("Không có features nào sau khi loại bỏ target columns")
    
    X = df[feature_cols].copy()
    y = df[target_cols].copy()
    
    dates = df['Date'].copy() if 'Date' in df.columns else None
    
    print(f"✅ Features shape: {X.shape}")
    print(f"✅ Targets shape: {y.shape}")
    print(f"✅ Số features: {len(feature_cols)}")
    
    return X, y, dates

