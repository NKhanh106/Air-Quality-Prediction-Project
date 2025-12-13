import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


# Chuẩn bị dữ liệu cho LSTM - tối ưu với pre-allocation
def create_sequences(data, seq_length):
    """
    Tạo sequences cho LSTM từ time series data (tối ưu với pre-allocation)
    Args:
        data: numpy array shape (n_samples, n_features)
        seq_length: độ dài sequence
    Returns:
        X: sequences shape (n_samples - seq_length, seq_length, n_features)
        y: targets shape (n_samples - seq_length, n_features)
    """
    n_samples = len(data) - seq_length
    n_features = data.shape[1]
    
    # Pre-allocate arrays để tránh memory reallocation (nhanh hơn)
    X = np.zeros((n_samples, seq_length, n_features), dtype=data.dtype)
    y = np.zeros((n_samples, n_features), dtype=data.dtype)
    
    for i in range(n_samples):
        X[i] = data[i:i + seq_length]
        y[i] = data[i + seq_length]
    
    return X, y


# Định nghĩa mô hình LSTM với cải thiện
class AirQualityLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        """
        LSTM model cho dự đoán chất lượng không khí
        Args:
            input_size: số features đầu vào
            hidden_size: số hidden units trong LSTM
            num_layers: số layers LSTM
            output_size: số features đầu ra
            dropout: dropout rate (chỉ áp dụng nếu num_layers > 1)
        """
        super(AirQualityLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer với dropout
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers với dropout
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: input tensor shape (batch_size, seq_length, input_size)
        Returns:
            output: tensor shape (batch_size, output_size)
        """
        # Initialize hidden states - tối ưu: không cần khởi tạo nếu không dùng
        # PyTorch sẽ tự động khởi tạo nếu không truyền vào
        # out, _ = self.lstm(x)  # Simpler và nhanh hơn
        
        # Hoặc nếu muốn explicit control:
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        
        # LSTM forward
        out, _ = self.lstm(x, (h0, c0))
        
        # Lấy output của timestep cuối cùng
        out = out[:, -1, :]
        
        # Fully connected layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out
