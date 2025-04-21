import Data_mining
import Data_preprocess
import Train_model
import Calculate_AQI
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import Model


#Đọc lại dữ liệu và chuẩn hóa như lúc train
df = pd.read_csv('../Data/FinalData.csv')
features = ['co','no2','o3','pm10','pm25','so2','Temp','Rain','Cloud','Pressure','Wind','Gust']
data = df[features].values

scaler = MinMaxScaler()
data_norm = scaler.fit_transform(data)

#Khởi tạo model và load weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size, hidden_size, num_layers, output_size = 12, 50, 2, 12
model = Model.AirQualityLSTM(input_size, hidden_size, num_layers, output_size).to(device)
model.load_state_dict(torch.load('./Built_Model/lstm_model.pth', map_location=device))
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

#In ra kết quả
for feat, val in zip(features, pred):
    print(f"{feat}: {val:.3f}")