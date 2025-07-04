import Calculate_AQI
import os
import torch
import pandas as pd
import datetime
from sklearn.preprocessing import MinMaxScaler
from Model import AirQualityLSTM, create_sequences
from Data_mining import *

base_dir = os.path.dirname(__file__)
csv_path = os.path.join(base_dir, "../Data/FinalData.csv")
model_path = os.path.join(base_dir, "../Model/Built_Model/lstm_model1.pth")

def predict():
    #Đọc lại dữ liệu và chuẩn hóa như lúc train
    df = pd.read_csv(csv_path)
    features = ['co','no2','o3','pm10','pm25','so2','Temp','Rain','Cloud','Pressure','Wind','Gust']
    data = df[features].values

    scaler = MinMaxScaler()
    data_norm = scaler.fit_transform(data)

    #Khởi tạo model và load weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size, hidden_size, num_layers, output_size = 12, 50, 2, 12
    model = AirQualityLSTM(input_size, hidden_size, num_layers, output_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    #Chuẩn bị input sequence mới (14 mốc cuối)
    seq_length = 14
    last_seq = data_norm[-seq_length:]                      # shape (14,12)
    input_tensor = torch.FloatTensor(last_seq).unsqueeze(0) # shape (1,14,12)
    input_tensor = input_tensor.to(device)

    #Dự đoán next step
    with torch.no_grad():
        pred_norm = model(input_tensor)        # shape (1,12)
    pred_norm = pred_norm.cpu().numpy()       # shape (1,12)

    #Inverse transform để ra giá trị gốc
    pred = scaler.inverse_transform(pred_norm)  # shape (1,12)
    pred = pred.flatten()

    answer = {}
    for feat, val in zip(features, pred):
        answer[feat] = val

    #Tính AQI cho các chất ô nhiễm
    aqi, main_pollutant, sub_indices = Calculate_AQI.calculate_aqi(pred[0], pred[1], pred[2], pred[3], pred[4], pred[5])

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
    df = pd.read_csv(csv_path)
    df['Date'] = df['Date'].astype('datetime64[ns]')
    df['co'] = df['co'].astype(float)
    df['no2'] = df['no2'].astype(float)
    df['o3'] = df['o3'].astype(float)
    df['pm10'] = df['pm10'].astype(float)
    df['pm25'] = df['pm25'].astype(float)
    df['so2'] = df['so2'].astype(float)
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    df = df[['Date', category]]
    return df
