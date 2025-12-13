"""
Data Preprocessing Module - Xử lý và làm sạch dữ liệu
Tối ưu với các kỹ thuật mới:
- IQR-based outlier detection
- Z-score outlier detection
- Advanced imputation (KNN, seasonal)
- Data quality validation
- Time series specific techniques
"""

import pandas as pd
import numpy as np
import datetime
import os
import warnings
from typing import List, Dict, Optional, Tuple
warnings.filterwarnings('ignore')

from ..utils.paths import get_data_path

# Set random seed để đảm bảo reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def _detect_outliers_iqr(df: pd.DataFrame, col: str, factor: float = 1.5) -> pd.Series:
    """
    Phát hiện outliers bằng IQR method (Interquartile Range)
    Args:
        df: DataFrame
        col: column name
        factor: IQR multiplier (default 1.5)
    Returns:
        Boolean Series: True = outlier
    """
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    return (df[col] < lower_bound) | (df[col] > upper_bound)

def _detect_outliers_zscore(df: pd.DataFrame, col: str, threshold: float = 3.0) -> pd.Series:
    """
    Phát hiện outliers bằng Z-score method
    Args:
        df: DataFrame
        col: column name
        threshold: Z-score threshold (default 3.0)
    Returns:
        Boolean Series: True = outlier
    """
    if df[col].std() == 0:
        return pd.Series([False] * len(df), index=df.index)
    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
    return z_scores > threshold

def validate_data_quality(df: pd.DataFrame, columns: List[str]) -> Dict:
    """
    Validate data quality và trả về report
    Args:
        df: DataFrame cần validate
        columns: danh sách columns cần check
    Returns:
        Dictionary với quality metrics
    """
    quality_report = {
        'total_rows': len(df),
        'missing_rates': {},
        'outlier_counts': {},
        'data_types': {},
        'ranges': {}
    }
    
    for col in columns:
        if col not in df.columns:
            continue
        
        # Missing rate
        missing_rate = df[col].isna().sum() / len(df) * 100
        quality_report['missing_rates'][col] = missing_rate
        
        # Data type
        quality_report['data_types'][col] = str(df[col].dtype)
        
        # Range (nếu numeric)
        if df[col].dtype in [np.float64, np.int64]:
            valid_values = df[col].dropna()
            if len(valid_values) > 0:
                quality_report['ranges'][col] = {
                    'min': float(valid_values.min()),
                    'max': float(valid_values.max()),
                    'mean': float(valid_values.mean()),
                    'std': float(valid_values.std())
                }
                
                # Outlier count (IQR)
                if len(valid_values) > 20:
                    outliers = _detect_outliers_iqr(df, col, factor=1.5)
                    quality_report['outlier_counts'][col] = int(outliers.sum())
    
    return quality_report

def _impute_missing_timeseries(df, columns, method='interpolate'):
    """
    Impute missing values cho time series data
    method: 'interpolate', 'forward_fill', 'backward_fill', 'seasonal'
    """
    df = df.copy()
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Set Date as index temporarily for time-based interpolation
    df_indexed = df.set_index('Date')
    
    for col in columns:
        if col not in df.columns:
            continue
            
        # Thay thế 0 và NaN bằng NaN để xử lý (trừ Rain có thể = 0 hợp lý)
        if col != 'Rain':
            mask = (df_indexed[col] == 0) | (df_indexed[col].isna())
            df_indexed.loc[mask, col] = np.nan
        else:
            # Rain: chỉ thay NaN, giữ 0
            df_indexed.loc[df_indexed[col].isna(), col] = np.nan
        
        if method == 'interpolate':
            # Interpolation cho time series (tốt nhất cho dữ liệu liên tục)
            try:
                # Thử time-based interpolation trước
                df_indexed[col] = df_indexed[col].interpolate(method='time', limit_direction='both', limit_area='inside')
            except:
                # Fallback: linear interpolation
                df_indexed[col] = df_indexed[col].interpolate(method='linear', limit_direction='both', limit_area='inside')
        elif method == 'forward_fill':
            df_indexed[col] = df_indexed[col].ffill(limit=7)  # Limit forward fill
        elif method == 'backward_fill':
            df_indexed[col] = df_indexed[col].bfill(limit=7)  # Limit backward fill
        elif method == 'seasonal':
            # Sử dụng giá trị trung bình theo day of year (seasonal pattern)
            df_temp = df_indexed.reset_index()
            df_temp['dayofyear'] = df_temp['Date'].dt.dayofyear
            df_temp['month'] = df_temp['Date'].dt.month
            # Dùng cả dayofyear và month để chính xác hơn
            seasonal_avg = df_temp.groupby(['month', 'dayofyear'])[col].transform('mean')
            # Fallback về month nếu không có dayofyear match
            if seasonal_avg.isna().any():
                monthly_avg = df_temp.groupby('month')[col].transform('mean')
                seasonal_avg = seasonal_avg.fillna(monthly_avg)
            df_indexed[col] = df_indexed[col].fillna(pd.Series(seasonal_avg.values, index=df_indexed.index))
        elif method == 'knn':
            # KNN imputation (cần sklearn)
            try:
                from sklearn.impute import KNNImputer
                # Chỉ dùng cho numeric columns tương tự
                similar_cols = [c for c in df_indexed.columns if df_indexed[c].dtype in [np.float64, np.int64]]
                if len(similar_cols) > 1:
                    temp_df = df_indexed[similar_cols].copy()
                    imputer = KNNImputer(n_neighbors=5)
                    temp_df_imputed = pd.DataFrame(
                        imputer.fit_transform(temp_df),
                        index=temp_df.index,
                        columns=temp_df.columns
                    )
                    df_indexed[col] = temp_df_imputed[col]
            except ImportError:
                # Fallback to interpolate if sklearn not available
                df_indexed[col] = df_indexed[col].interpolate(method='time', limit_direction='both')
        
        # Nếu vẫn còn NaN (ở đầu/cuối), dùng forward/backward fill
        df_indexed[col] = df_indexed[col].ffill().bfill()
        
        # Đảm bảo không có giá trị âm
        df_indexed[col] = df_indexed[col].clip(lower=0)
    
    # Reset index
    df = df_indexed.reset_index()
    
    return df


def process_AQI_data(source_df):
    """
    Xử lý dữ liệu AQI với các cải thiện:
    - Xử lý outliers tốt hơn
    - Imputation thông minh cho time series
    - Vectorized operations
    - Data validation
    """
    if source_df is None or source_df.empty:
        raise ValueError("source_df không được rỗng")
    
    # Clean column names
    source_df.columns = source_df.columns.str.strip()
    source_df = source_df.replace(['', ' '], np.nan)
    
    # Rename date column
    if 'date' in source_df.columns:
        source_df = source_df.rename(columns={'date': 'Date'})
    
    # Load old data
    aqi_path = get_data_path("AQI.csv", "raw")
    if not os.path.exists(aqi_path):
        old_df = pd.DataFrame(columns=['Date', 'co', 'no2', 'o3', 'pm10', 'pm25', 'so2'])
    else:
        old_df = pd.read_csv(aqi_path)
    
    # Convert data types
    pollutant_cols = ['co', 'no2', 'o3', 'pm10', 'pm25', 'so2']
    
    source_df['Date'] = pd.to_datetime(source_df['Date'], errors='coerce')
    for col in pollutant_cols:
        if col in source_df.columns:
            source_df[col] = pd.to_numeric(source_df[col], errors='coerce')
        else:
            source_df[col] = np.nan
    
    if not old_df.empty:
        old_df['Date'] = pd.to_datetime(old_df['Date'], errors='coerce')
        for col in pollutant_cols:
            if col in old_df.columns:
                old_df[col] = pd.to_numeric(old_df[col], errors='coerce')
    
    # Filter new data (chỉ lấy data mới hơn old_df)
    if not old_df.empty and 'Date' in old_df.columns:
        max_old_date = old_df['Date'].max()
        source_df = source_df[source_df['Date'] > max_old_date].copy()
    
    # Chỉ lấy data trong quá khứ (không lấy tương lai)
    source_df = source_df[source_df['Date'] < datetime.datetime.now()].copy()
    
    # Merge với old data
    if not old_df.empty:
        source_df = pd.concat([old_df, source_df], axis=0, ignore_index=True)
    
    # Sort và remove duplicates
    source_df = source_df.sort_values(by='Date', ascending=True).reset_index(drop=True)
    source_df = source_df.drop_duplicates(subset='Date', keep='last').reset_index(drop=True)
    
    # Xử lý outliers với nhiều phương pháp
    outlier_stats = {}
    
    # 1. Domain knowledge-based outliers (giá trị không hợp lý về mặt vật lý)
    domain_limits = {
        'o3': (0, 200),    # O3: 0-200 µg/m³
        'no2': (0, 120),   # NO2: 0-120 µg/m³
        'pm10': (0, 500),  # PM10: 0-500 µg/m³
        'pm25': (0, 500),  # PM2.5: 0-500 µg/m³
        'so2': (0, 500),   # SO2: 0-500 µg/m³
        'co': (0, 50)      # CO: 0-50 mg/m³
    }
    
    for col in pollutant_cols:
        if col in source_df.columns and col in domain_limits:
            lower, upper = domain_limits[col]
            outliers = (source_df[col] < lower) | (source_df[col] > upper)
            outlier_count = outliers.sum()
            if outlier_count > 0:
                source_df.loc[outliers, col] = np.nan
                outlier_stats[f'{col}_domain'] = outlier_count
    
    # 2. IQR-based outlier detection (cho extreme outliers)
    for col in pollutant_cols:
        if col in source_df.columns:
            valid_values = source_df[col].dropna()
            if len(valid_values) > 20:  # Cần đủ data
                outliers_iqr = _detect_outliers_iqr(source_df, col, factor=3.0)  # Stricter for extreme
                outlier_count = outliers_iqr.sum()
                if outlier_count > 0:
                    source_df.loc[outliers_iqr, col] = np.nan
                    outlier_stats[f'{col}_iqr'] = outlier_count
    
    # 3. Z-score based (cho values quá xa mean)
    for col in pollutant_cols:
        if col in source_df.columns:
            valid_values = source_df[col].dropna()
            if len(valid_values) > 20:
                outliers_z = _detect_outliers_zscore(source_df, col, threshold=4.0)  # Stricter threshold
                outlier_count = outliers_z.sum()
                if outlier_count > 0:
                    # Chỉ đánh dấu nếu chưa bị đánh dấu bởi IQR
                    mask = outliers_z & source_df[col].notna()
                    source_df.loc[mask, col] = np.nan
                    outlier_stats[f'{col}_zscore'] = mask.sum()
    
    if outlier_stats:
        print(f"⚠️  Đã phát hiện và xử lý outliers: {outlier_stats}")
    
    # Xử lý missing values và giá trị 0 bằng time series imputation
    if len(source_df) > 0:
        # Kiểm tra missing rate trước imputation
        missing_before = {}
        for col in pollutant_cols:
            if col in source_df.columns:
                missing_before[col] = source_df[col].isna().sum() / len(source_df) * 100
        
        # Chọn method imputation dựa trên missing rate
        for col in pollutant_cols:
            if col in source_df.columns:
                missing_rate = missing_before.get(col, 0)
                if missing_rate > 30:
                    # Nếu missing nhiều, dùng seasonal
                    method = 'seasonal'
                elif missing_rate > 10:
                    # Missing vừa, dùng interpolate
                    method = 'interpolate'
                else:
                    # Missing ít, dùng interpolate
                    method = 'interpolate'
                
                # Impute từng column để có thể dùng method khác nhau
                source_df = _impute_missing_timeseries(source_df, [col], method=method)
        
        # Round values
        for col in pollutant_cols:
            if col in source_df.columns:
                source_df[col] = source_df[col].round(2)
        
        # Kiểm tra missing rate sau imputation
        missing_after = {}
        for col in pollutant_cols:
            if col in source_df.columns:
                missing_after[col] = source_df[col].isna().sum() / len(source_df) * 100
        
        if any(v > 0 for v in missing_after.values()):
            print(f"⚠️  Vẫn còn missing values sau imputation: {missing_after}")
    else:
        print("⚠️  Warning: Không có dữ liệu mới để xử lý")
    
    # Save updated AQI data
    source_df.to_csv(aqi_path, index=False)
    
    # Data quality validation
    quality_report = validate_data_quality(source_df, pollutant_cols)
    if quality_report['missing_rates']:
        high_missing = {k: v for k, v in quality_report['missing_rates'].items() if v > 5}
        if high_missing:
            print(f"⚠️  Cột có missing rate cao: {high_missing}")
    
    return source_df
                

def process_weather_data(source_df):
    """
    Xử lý dữ liệu thời tiết với các cải thiện:
    - Xử lý lỗi tốt hơn
    - Validation data
    - Xử lý missing values
    """
    if source_df is None or source_df.empty:
        raise ValueError("source_df không được rỗng")
    
    # Bỏ cột không cần thiết
    if 'Weather' in source_df.columns:
        source_df = source_df.drop(columns=['Weather'])
    
    # Định dạng Date
    source_df['Date'] = pd.to_datetime(source_df['Date'], errors='coerce')
    
    # Loại bỏ rows có Date không hợp lệ
    source_df = source_df[source_df['Date'].notna()].copy()
    
    # Weather columns mapping
    weather_cols = {
        'Temp': ('°c', '°C', 'C'),
        'Rain': ('\nmm', 'mm'),
        'Cloud': ('%',),
        'Pressure': ('mb', 'hPa'),
        'Wind': ('km/h', 'kmh'),
        'Gust': ('km/h', 'kmh')
    }
    
    # Clean và convert weather columns
    for col, suffixes in weather_cols.items():
        if col in source_df.columns:
            # Remove suffixes và strip
            for suffix in suffixes:
                source_df[col] = source_df[col].astype(str).str.replace(suffix, '', regex=False)
            source_df[col] = source_df[col].str.strip()
            
            # Convert to numeric, coerce errors to NaN
            source_df[col] = pd.to_numeric(source_df[col], errors='coerce')
        else:
            source_df[col] = np.nan
    
    # Xử lý outliers cho weather data với IQR method
    weather_limits = {
        'Temp': (-10, 50),      # Hà Nội: -10°C đến 50°C
        'Pressure': (950, 1100), # 950-1100 mb
        'Wind': (0, 200),        # 0-200 km/h
        'Gust': (0, 200),        # 0-200 km/h
        'Cloud': (0, 100),       # 0-100%
        'Rain': (0, 500)         # 0-500 mm (cho extreme cases)
    }
    
    outlier_stats_weather = {}
    
    for col, (lower, upper) in weather_limits.items():
        if col in source_df.columns:
            # Domain-based filtering
            domain_outliers = (source_df[col] < lower) | (source_df[col] > upper)
            if domain_outliers.any():
                if col in ['Rain']:
                    # Rain: chỉ clip, không set NaN
                    source_df.loc[source_df[col] < 0, col] = 0
                    source_df.loc[source_df[col] > upper, col] = upper
                else:
                    source_df.loc[domain_outliers, col] = np.nan
                    outlier_stats_weather[f'{col}_domain'] = domain_outliers.sum()
            
            # IQR-based cho extreme outliers (nếu có đủ data)
            valid_values = source_df[col].dropna()
            if len(valid_values) > 20:
                outliers_iqr = _detect_outliers_iqr(source_df, col, factor=3.0)
                if outliers_iqr.any():
                    source_df.loc[outliers_iqr, col] = np.nan
                    outlier_stats_weather[f'{col}_iqr'] = outliers_iqr.sum()
    
    if outlier_stats_weather:
        print(f"⚠️  Đã phát hiện outliers trong weather data: {outlier_stats_weather}")
    
    # Group by Date và tính mean (xử lý duplicate dates)
    weather_features = ['Temp', 'Rain', 'Cloud', 'Pressure', 'Wind', 'Gust']
    available_features = [col for col in weather_features if col in source_df.columns]
    
    if available_features:
        source_df = source_df.groupby('Date')[available_features].mean().round(2).reset_index()
        
        # Impute missing values cho weather data với method phù hợp
        # Weather data thường có seasonal patterns
        for col in available_features:
            if col in source_df.columns:
                missing_rate = source_df[col].isna().sum() / len(source_df) * 100
                if missing_rate > 20:
                    # Missing nhiều: dùng seasonal (weather có pattern theo mùa)
                    method = 'seasonal'
                else:
                    # Missing ít: dùng interpolate
                    method = 'interpolate'
                source_df = _impute_missing_timeseries(source_df, [col], method=method)
    else:
        # Nếu không có feature nào, tạo DataFrame rỗng với Date
        source_df = source_df[['Date']].drop_duplicates().reset_index(drop=True)
        for col in weather_features:
            source_df[col] = np.nan
    
    return source_df

