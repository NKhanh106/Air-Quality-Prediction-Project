import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import pandas as pd
import Model
import Data_preprocess
import Data_mining


file_path = '../Data/FinalData.csv'
df = pd.read_csv(file_path)

features = ['co', 'no2', 'o3', 'pm10', 'pm25', 'so2', 'Temp', 'Rain', 'Cloud', 'Pressure', 'Wind', 'Gust']
data = df[features].values

scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)

torch.manual_seed(42)
np.random.seed(42)


# Tham số mô hình
input_size = 12
hidden_size = 50
num_layers = 2
output_size = 12
seq_length = 14
batch_size = 64
num_epochs = 150
learning_rate = 0.0005


# Tạo sequences
X, y = Model.create_sequences(data_normalized, seq_length)


# Chia dữ liệu thành tập train và test
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# Chuyển sang tensor
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)


# Tạo DataLoader
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# Khởi tạo mô hình, loss function và optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model.AirQualityLSTM(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Huấn luyện mô hình
model.train()
for epoch in range(num_epochs):
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')


# Dự đoán trên tập test
model.eval()
with torch.no_grad():
    X_test = X_test.to(device)
    predictions = model(X_test).cpu().numpy()
    y_test = y_test.numpy()


# Tính R² Score cho từng chỉ số
r2_scores = {}
feature_names = ['co', 'no2', 'o3', 'pm10', 'pm25', 'so2', 'Temp', 'Rain', 'Cloud', 'Pressure', 'Wind', 'Gust']
for i, feature in enumerate(feature_names):
    r2 = r2_score(y_test[:, i], predictions[:, i])
    r2_scores[feature] = r2
    print(f'R² Score for {feature}: {r2:.4f}')


#Lưu lại mô hình LSTM tốt nhất
torch.save(model.state_dict(), './Built_Model/lstm_model.pth')