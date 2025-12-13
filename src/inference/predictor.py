"""
Prediction Module - Dự đoán chất lượng không khí
"""

import os
import torch
import pandas as pd
import datetime
import joblib

from ..models.lstm import AirQualityLSTM, create_sequences
from ..utils.paths import get_data_path, get_model_path
from ..utils.aqi_calculator import calculate_aqi

def predict():
    """
    Dự đoán chất lượng không khí cho ngày tiếp theo
    Sử dụng scaler đã lưu từ training (không fit lại để tránh data leakage)
    """
    # Load scaler đã lưu từ training
    scaler_path = get_model_path('scaler.pkl', 'lstm')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler không tồn tại tại {scaler_path}. Vui lòng train model trước.")
    
    scaler = joblib.load(scaler_path)
    
    # Đọc dữ liệu
    csv_path = get_data_path("FinalData.csv", "processed")
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values('Date').reset_index(drop=True)
    
    features = ['co','no2','o3','pm10','pm25','so2','Temp','Rain','Cloud','Pressure','Wind','Gust']
    
    # Kiểm tra xem có đủ features không
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Thiếu các features: {missing_features}")
    
    data = df[features].values
    
    # CHỈ transform, KHÔNG fit lại (quan trọng!)
    data_norm = scaler.transform(data)

    # Khởi tạo model và load weights
    model_path = get_model_path("lstm_model.pth", "lstm")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model không tồn tại tại {model_path}. Vui lòng train model trước.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size, hidden_size, num_layers, output_size = 12, 50, 2, 12
    dropout = 0.2  # Phải khớp với dropout khi training
    
    model = AirQualityLSTM(input_size, hidden_size, num_layers, output_size, dropout=dropout).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Chuẩn bị input sequence mới (14 mốc cuối)
    seq_length = 14
    last_seq = data_norm[-seq_length:]                      # shape (14,12)
    input_tensor = torch.FloatTensor(last_seq).unsqueeze(0) # shape (1,14,12)
    input_tensor = input_tensor.to(device)

    # Dự đoán next step
    with torch.no_grad():
        pred_norm = model(input_tensor)        # shape (1,12)
    pred_norm = pred_norm.cpu().numpy()       # shape (1,12)

    # Inverse transform để ra giá trị gốc
    pred = scaler.inverse_transform(pred_norm)  # shape (1,12)
    pred = pred.flatten()

    answer = {}
    for feat, val in zip(features, pred):
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

