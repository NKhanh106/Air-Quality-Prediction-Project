"""
Prediction Module - Dự đoán chất lượng không khí
"""

import os
import pandas as pd
import joblib

from ..features.engineering import create_all_features, prepare_rf_features
from ..utils.paths import get_data_path, get_model_path
from ..utils.aqi_calculator import calculate_aqi

def predict():
    """
    Dự đoán chất lượng không khí cho ngày tiếp theo
    Sử dụng Random Forest model đã được train
    """
    # Load model và feature names đã lưu từ training
    model_path = get_model_path('random_forest_model.pkl', 'rf')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model không tồn tại tại {model_path}. Vui lòng train model trước.")
    
    feature_names_path = get_model_path('rf_feature_names.pkl', 'rf')
    if not os.path.exists(feature_names_path):
        raise FileNotFoundError(f"Feature names không tồn tại tại {feature_names_path}. Vui lòng train model trước.")
    
    model = joblib.load(model_path)
    saved_feature_names = joblib.load(feature_names_path)
    
    # Đọc dữ liệu
    csv_path = get_data_path("FinalData.csv", "processed")
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values('Date').reset_index(drop=True)
    
    target_cols = ['co', 'no2', 'o3', 'pm10', 'pm25', 'so2', 
                   'Temp', 'Rain', 'Cloud', 'Pressure', 'Wind', 'Gust']
    
    # Kiểm tra xem có đủ features không
    missing_features = [f for f in target_cols if f not in df.columns]
    if missing_features:
        raise ValueError(f"Thiếu các features: {missing_features}")
    
    # Tạo features giống như khi training (quan trọng!)
    print("🔧 Đang tạo features cho prediction...")
    df_features = create_all_features(
        df,
        lag_steps=[1, 2, 3, 7, 14, 30],
        rolling_windows=[3, 7, 14, 30]
    )
    
    # Chuẩn bị features (loại bỏ NaN rows)
    X, _, _ = prepare_rf_features(df_features, target_cols)
    
    # Lấy row cuối cùng để predict (ngày gần nhất)
    if len(X) == 0:
        raise ValueError("Không có dữ liệu hợp lệ để predict. Cần ít nhất 30 ngày dữ liệu.")
    
    # Lấy row cuối cùng
    last_row = X.iloc[[-1]].copy()
    
    # Đảm bảo thứ tự features khớp với model
    # Model được train với feature names đã lưu
    missing_cols = set(saved_feature_names) - set(last_row.columns)
    if missing_cols:
        # Thêm các cột thiếu với giá trị 0 (hoặc có thể dùng giá trị mặc định khác)
        for col in missing_cols:
            last_row[col] = 0
        print(f"⚠️  Cảnh báo: Thiếu {len(missing_cols)} features, đã set = 0")
    
    # Chỉ lấy các features mà model đã được train
    last_row = last_row[saved_feature_names]
    
    # Dự đoán
    pred = model.predict(last_row)  # shape (1, 12)
    pred = pred.flatten()  # shape (12,)

    answer = {}
    for feat, val in zip(target_cols, pred):
        answer[feat] = val

    # Tính AQI cho các chất ô nhiễm
    aqi, main_pollutant, sub_indices = calculate_aqi(pred[0], pred[1], pred[2], pred[3], pred[4], pred[5])

    attention = ""

    if aqi <= 50:
        attention = "Cảnh báo: Tình trạng thời tiết **'Tốt'**. Chất lượng không khí tốt, không ảnh hưởng tới sức khỏe"
    elif aqi <= 100:
        attention = "Cảnh báo: Tình trạng thời tiết **'Trung bình'**. Chất lượng không khí ở mức chấp nhận được. Tuy nhiên, đối với những người nhạy cảm (người già, trẻ em, người mắc các bệnh hô hấp, tim mạch…) có thể chịu những tác động nhất định tới sức khỏe."
    elif aqi <= 150:
        attention = "Cảnh báo: Tình trạng thời tiết **'Kém'**. Những người nhạy cảm gặp phải các vấn đề về sức khỏe, những người bình thường ít ảnh hưởng."
    elif aqi <= 200:
        attention = "Cảnh báo: Tình trạng thời tiết **'Xấu'**. Những người bình thường bắt đầu có các ảnh hưởng tới sức khỏe, nhóm người nhạy cảm có thể gặp những vấn đề sức khỏe nghiêm trọng hơn."
    elif aqi <= 300:
        attention = "Cảnh báo hưởng tới sức khỏe: Tình trạng thời tiết **'Rất xấu'**. Mọi người bị ảnh hưởng tới sức khỏe nghiêm trọng hơn."
    else:
        attention = "Cảnh báo khẩn cấp về sức khỏe: Tình trạng thời tiết **'Nguy hại'**. Toàn bộ dân số bị ảnh hưởng tới sức khỏe tới mức nghiêm trọng."

    return answer, aqi, main_pollutant, attention

def call_chart(category, start_date, end_date):
    """
    Lấy dữ liệu để vẽ biểu đồ
    Args:
        category: tên feature cần vẽ
        start_date: ngày bắt đầu
        end_date: ngày kết thúc
    Returns:
        DataFrame với Date và category
    """
    csv_path = get_data_path("FinalData.csv", "processed")
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Convert các cột số về float
    numeric_cols = ['co', 'no2', 'o3', 'pm10', 'pm25', 'so2', 'Temp', 'Rain', 'Cloud', 'Pressure', 'Wind', 'Gust']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Filter theo date range
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    
    # Kiểm tra category có tồn tại không
    if category not in df.columns:
        raise ValueError(f"Category '{category}' không tồn tại trong dữ liệu")
    
    df = df[['Date', category]].copy()
    df = df.sort_values('Date').reset_index(drop=True)
    
    return df

