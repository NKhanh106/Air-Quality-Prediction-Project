# Air Quality Prediction Project - Hà Nội

Dự án dự đoán chất lượng không khí thành phố Hà Nội sử dụng mô hình LSTM và Random Forest với các kỹ thuật tối ưu hiện đại.

## 📁 Cấu Trúc Dự Án

```
Air-Quality-Prediction-Project/
│
├── src/                           # 💻 Source code chính
│   ├── data/                      # Data processing modules
│   │   ├── mining.py              # Web scraping (Selenium)
│   │   └── preprocessing.py       # Data cleaning & imputation
│   ├── features/                  # Feature engineering
│   │   └── engineering.py         # Feature creation (lag, rolling, seasonal, etc.)
│   ├── models/                    # Model definitions & training
│   │   ├── lstm.py                # LSTM architecture
│   │   ├── train_lstm.py          # LSTM training với optimizations
│   │   └── train_rf.py            # RF training với Optuna
│   ├── utils/                     # Utilities
│   │   ├── aqi_calculator.py     # AQI calculation
│   │   └── paths.py              # Path management
│   └── inference/                 # Prediction
│       └── predictor.py           # Prediction module
│
├── scripts/                       # 🚀 Entry point scripts
│   ├── train_lstm.py              # Train LSTM model
│   ├── train_rf.py                # Train Random Forest model
│   └── update_data.py             # Update data from web
│
├── data/                          # 📊 Dữ liệu
│   ├── raw/                       # Dữ liệu thô
│   ├── processed/                 # Dữ liệu đã xử lý
│   └── external/                  # Dữ liệu từ crawl
│
├── models/                        # 🤖 Trained models
│   ├── lstm/                      # LSTM models & artifacts
│   │   ├── lstm_model.pth
│   │   ├── scaler.pkl
│   │   ├── metrics.json
│   │   └── lstm_hyperparams.json
│   └── random_forest/             # RF models & artifacts
│       ├── random_forest_model.pkl
│       ├── rf_feature_names.pkl
│       ├── rf_metrics.json
│       └── rf_feature_importance.csv
│
├── config/                        # ⚙️ Configuration
│   └── config.yaml
│
├── notebooks/                     # 📓 Jupyter notebooks
│
├── app/                           # 🌐 Web application
│   └── main.py                    # Streamlit app
│
└── Visualization/                 # 📈 Visualizations
```

## 🚀 Cài Đặt

### 1. Clone repository

```bash
git clone <repository-url>
cd Air-Quality-Prediction-Project
```

### 2. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 3. Cài đặt ChromeDriver (cho web scraping)

- Tải ChromeDriver phù hợp với Chrome version của bạn
- Đặt vào PATH hoặc project root
- Hoặc sử dụng `webdriver-manager` (tự động download)

## 📖 Sử Dụng

### Cập Nhật Dữ Liệu

Cập nhật dữ liệu từ web (AQI và Weather):

```bash
python scripts/update_data.py
```

Hoặc trong Python:

```python
from src.data.mining import update_weather_data
update_weather_data()
```

### Train Models

**LSTM Model:**
```bash
python scripts/train_lstm.py
```

**Random Forest Model:**
```bash
python scripts/train_rf.py
```

**Lưu ý:** 
- LSTM training sẽ hiển thị progress bar
- Random Forest sử dụng Optuna với progress bar
- Models sẽ được lưu tự động vào `models/` folder

### Dự Đoán

Sử dụng trong Python:

```python
from src.inference.predictor import predict

answer, aqi, main_pollutant, attention = predict()
print(f"AQI: {aqi}")
print(f"Main Pollutant: {main_pollutant}")
print(f"Attention: {attention}")
```

### Deploy Web App

Chạy Streamlit web application:

```bash
streamlit run app/main.py
```

App sẽ mở tại `http://localhost:8501`

## 🔧 Modules Chi Tiết

### Data Processing (`src/data/`)

- **mining.py**: 
  - Web scraping với Selenium
  - Crawl AQI data từ aqicn.org
  - Crawl Weather data từ worldweatheronline.com
  - Tự động merge và lưu dữ liệu

- **preprocessing.py**: 
  - Advanced outlier detection (IQR, Z-score, domain knowledge)
  - Adaptive imputation strategies (seasonal, interpolate, KNN)
  - Data quality validation
  - Time series handling

### Feature Engineering (`src/features/`)

- **engineering.py**: 
  - **Lag features**: 1, 2, 3, 7, 14, 30 days
  - **Rolling statistics**: mean, std, min, max (windows: 3, 7, 14, 30)
  - **Seasonal features**: dayofweek, month
  - **Ratio features**: ratios between pollutants
  - **Trend features**: rolling trends
  - **Interaction features**: interactions between features

### Models (`src/models/`)

- **lstm.py**: 
  - Multi-layer LSTM architecture
  - Dropout regularization
  - FC layers với ReLU activation

- **train_lstm.py**: 
  - Mixed precision training (FP16) cho GPU
  - Early stopping với patience
  - Learning rate scheduling (ReduceLROnPlateau)
  - Gradient clipping
  - Optimized DataLoader (pin_memory, num_workers, prefetch)
  - Progress bar với tqdm
  - 60-20-20 train/val/test split

- **train_rf.py**: 
  - Optuna hyperparameter optimization
  - MedianPruner cho early stopping
  - TimeSeriesSplit cross-validation
  - Feature selection (top 150 features by importance)
  - Progress bar
  - Advanced feature engineering option

### Utils (`src/utils/`)

- **aqi_calculator.py**: Tính toán AQI từ 6 chất ô nhiễm
- **paths.py**: Centralized path management

### Inference (`src/inference/`)

- **predictor.py**: 
  - Load trained models
  - Dự đoán 12 chỉ số (6 chất ô nhiễm + 6 chỉ số thời tiết)
  - Tính AQI từ predictions
  - Tránh data leakage (sử dụng saved scaler)

## 📊 Dữ Liệu

### Features

**Pollutants (6):** co, no2, o3, pm10, pm25, so2  
**Weather (6):** Temp, Rain, Cloud, Pressure, Wind, Gust

**Total:** 12 features

### Data Sources

- **AQI**: [aqicn.org](https://aqicn.org/historical/vn/#!city:vietnam/hanoi)
- **Weather**: [worldweatheronline.com](https://www.worldweatheronline.com/ha-noi-weather-history/vn.aspx)

### Data Processing Pipeline

1. **Raw Data** → Crawl từ web
2. **Preprocessing** → Outlier detection, imputation, validation
3. **Feature Engineering** → Tạo features nâng cao
4. **Model Training** → Train với optimized hyperparameters
5. **Prediction** → Dự đoán và tính AQI

## 🎯 Models

### LSTM (Long Short-Term Memory)

**Architecture:**
- Input: 14 timesteps × 12 features
- Multi-layer LSTM với dropout
- FC layers với ReLU activation
- Output: 12 features (multi-output regression)

**Training Features:**
- Mixed precision training (FP16) cho GPU acceleration
- Early stopping với patience
- Learning rate scheduling
- Gradient clipping
- Optimized DataLoader settings
- AdamW optimizer với weight decay

**Hyperparameters:**
- `hidden_size`: 50
- `num_layers`: 2
- `dropout`: 0.2
- `batch_size`: 64
- `learning_rate`: 0.001
- `seq_length`: 14

### Random Forest

**Features:**
- Feature engineering mạnh mẽ (~330 features → 150 selected)
- Optuna hyperparameter optimization
- TimeSeriesSplit cross-validation (5 folds)
- MedianPruner cho early stopping
- Feature importance analysis

**Hyperparameters (Optimized by Optuna):**
- `n_estimators`: 100-500
- `max_depth`: 5-30 hoặc None
- `max_features`: sqrt, log2, 0.5, 0.7
- `min_samples_split`: 2-20
- `min_samples_leaf`: 1-10
- `bootstrap`: True/False
- `max_samples`: 0.6-1.0 (nếu bootstrap=True)

**Output:**
- Multi-output regression (12 features)
- AQI được tính từ 6 chất ô nhiễm

## 📈 Metrics

Models được đánh giá bằng:

- **R² Score**: Coefficient of determination
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error

Metrics được lưu trong:
- LSTM: `models/lstm/metrics.json`
- Random Forest: `models/random_forest/rf_metrics.json`

## 🔄 Workflow

```
1. Data Mining (Selenium) 
   ↓
2. Preprocessing (Outlier detection, Imputation)
   ↓
3. Feature Engineering (Lag, Rolling, Seasonal, etc.)
   ↓
4. Model Training (LSTM/RF với optimization)
   ↓
5. Prediction (Load model, predict, calculate AQI)
   ↓
6. Deployment (Streamlit web app)
```

## 🎨 Web Application

Streamlit app cung cấp:

- **Dự đoán chất lượng không khí**: Dự đoán 12 chỉ số và AQI cho ngày tiếp theo
- **Cập nhật dữ liệu**: Button để crawl dữ liệu mới
- **Biểu đồ**: Xem biến động của các chỉ số theo thời gian
- **Cảnh báo**: Thông báo mức độ ô nhiễm và ảnh hưởng sức khỏe

## 🛠️ Tối Ưu Hóa

### LSTM Optimizations:
- ✅ Mixed precision training (FP16)
- ✅ Optimized DataLoader (pin_memory, num_workers, prefetch)
- ✅ Early stopping
- ✅ Learning rate scheduling
- ✅ Gradient clipping
- ✅ AdamW optimizer

### Random Forest Optimizations:
- ✅ Optuna hyperparameter optimization
- ✅ Feature selection (top 150)
- ✅ TimeSeriesSplit cross-validation
- ✅ MedianPruner for early stopping
- ✅ Advanced feature engineering

### Data Processing Optimizations:
- ✅ Advanced outlier detection
- ✅ Adaptive imputation strategies
- ✅ Data quality validation
- ✅ Efficient feature creation (vectorized operations)

## 📝 Notes

- **Data Leakage Prevention**: Scaler được lưu và load lại khi prediction, không fit lại
- **Reproducibility**: Random seeds được set (42) cho NumPy và PyTorch
- **Time Series**: Sử dụng TimeSeriesSplit để tránh data leakage trong cross-validation
- **Feature Selection**: Random Forest tự động chọn top 150 features nếu có quá nhiều

## 👤 Author

**Chu Nam Khánh**

## 📄 License

[Thêm license nếu có]
