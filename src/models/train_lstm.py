"""
LSTM Training Module
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os
import pandas as pd
import joblib
import json
from tqdm import tqdm

from .lstm import create_sequences, AirQualityLSTM
from ..utils.paths import get_data_path, get_model_path, ensure_dirs

# Ensure directories exist
ensure_dirs()

def train_lstm_model(
    hidden_size=50,
    num_layers=2,
    seq_length=14,
    batch_size=64,
    num_epochs=200,
    learning_rate=0.001,
    dropout=0.2,
    patience=15,
    min_delta=1e-6
):
    """
    Huấn luyện LSTM model
    """
    print("=" * 60)
    print("Bắt đầu huấn luyện LSTM model...")
    print("=" * 60)
    
    print("\n📂 Đang load dữ liệu...")
    csv_path = get_data_path("FinalData.csv", "processed")
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values('Date').reset_index(drop=True)
    print(f"✅ Đã load {len(df)} rows")

    features = ['co', 'no2', 'o3', 'pm10', 'pm25', 'so2', 'Temp', 'Rain', 'Cloud', 'Pressure', 'Wind', 'Gust']
    
    # Kiểm tra features
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Thiếu features: {missing_features}")
    
    data = df[features].values

    print("\n🔧 Đang normalize dữ liệu...")
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data)
    
    scaler_path = get_model_path('scaler.pkl', 'lstm')
    joblib.dump(scaler, scaler_path)
    print(f"✅ Đã lưu scaler tại: {scaler_path}")

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    input_size = len(features)
    output_size = len(features)

    # Tạo sequences
    X, y = create_sequences(data_normalized, seq_length)
    print(f"✅ Đã tạo {len(X)} sequences")

    # Chia dữ liệu: Train (60%) - Validation (20%) - Test (20%)
    train_size = int(0.6 * len(X))
    val_size = int(0.2 * len(X))
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    print(f"\n📊 Chia dữ liệu:")
    print(f"  Train: {len(X_train)}")
    print(f"  Validation: {len(X_val)}")
    print(f"  Test: {len(X_test)}")

    # Setup device và tối ưu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pin_memory = device.type == 'cuda'
    # Tối ưu num_workers: dùng 4-8 workers nếu có CPU cores
    import multiprocessing
    num_workers = min(8, multiprocessing.cpu_count()) if device.type == 'cpu' else 4
    
    # Mixed precision training cho GPU (tăng tốc 2x)
    use_amp = device.type == 'cuda'
    scaler_amp = torch.cuda.amp.GradScaler() if use_amp else None

    # Chuyển sang tensor
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)

    # Tạo DataLoader với tối ưu
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )

    print(f"\n✅ Sử dụng device: {device}")
    if use_amp:
        print("✅ Mixed precision training (FP16) enabled")
    print(f"✅ DataLoader workers: {num_workers}")
    
    model = AirQualityLSTM(input_size, hidden_size, num_layers, output_size, dropout=dropout).to(device)
    
    # Compile model với torch.compile (PyTorch 2.0+) - DISABLED vì cần triton
    # use_compile = False
    # if use_compile:
    #     try:
    #         model = torch.compile(model, mode='reduce-overhead')
    #         print("✅ Model compiled với torch.compile")
    #     except Exception as e:
    #         print(f"⚠️  Không thể compile: {e}")
    
    criterion = nn.MSELoss()
    # Dùng AdamW thay vì Adam (better weight decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=False, min_lr=1e-6
    )

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    best_model_path = get_model_path('lstm_model.pth', 'lstm')

    print("\n" + "=" * 60)
    print("Bắt đầu training...")
    print("=" * 60)
    print(f"Hyperparameters: hidden_size={hidden_size}, num_layers={num_layers}, "
          f"batch_size={batch_size}, lr={learning_rate}, dropout={dropout}")

    pbar = tqdm(range(num_epochs), desc="Training")
    for epoch in pbar:
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device, non_blocking=pin_memory)
            batch_y = batch_y.to(device, non_blocking=pin_memory)
            
            optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
            
            # Mixed precision training
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler_amp.step(optimizer)
                scaler_amp.update()
            else:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device, non_blocking=pin_memory)
                batch_y = batch_y.to(device, non_blocking=pin_memory)
                
                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                else:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)

        # Early stopping và lưu best model
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        # Update progress bar
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'train_loss': f'{train_loss:.6f}',
            'val_loss': f'{val_loss:.6f}',
            'best_val': f'{best_val_loss:.6f}',
            'lr': f'{current_lr:.6f}',
            'patience': f'{patience_counter}/{patience}'
        })
        
        if patience_counter >= patience:
            print(f"\n⏹️  Early stopping tại epoch {epoch+1}")
            break

    pbar.close()
    print(f"\n✅ Best validation loss: {best_val_loss:.6f}")

    # Load best model và đánh giá trên test set
    print("\n📊 Đang đánh giá trên Test Set...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    
    with torch.no_grad():
        X_test_tensor = X_test.to(device)
        predictions = model(X_test_tensor).cpu().numpy()
        y_test_np = y_test.numpy()

    # Tính metrics cho từng feature
    feature_names = features
    
    print("\n" + "=" * 60)
    print("Metrics trên Test Set:")
    print("=" * 60)
    print(f"{'Feature':<12} {'R² Score':<12} {'MAE':<12} {'RMSE':<12}")
    print("-" * 60)
    
    metrics_summary = {}
    for i, feature in enumerate(feature_names):
        r2 = r2_score(y_test_np[:, i], predictions[:, i])
        mae = mean_absolute_error(y_test_np[:, i], predictions[:, i])
        rmse = np.sqrt(mean_squared_error(y_test_np[:, i], predictions[:, i]))
        
        metrics_summary[feature] = {
            'r2': float(r2),
            'mae': float(mae),
            'rmse': float(rmse)
        }
        
        print(f"{feature:<12} {r2:<12.4f} {mae:<12.4f} {rmse:<12.4f}")
    
    # Lưu metrics và hyperparameters
    metrics_path = get_model_path('metrics.json', 'lstm')
    metrics_json = {k: {kk: float(vv) for kk, vv in v.items()} for k, v in metrics_summary.items()}
    with open(metrics_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    # Lưu hyperparameters
    hyperparams = {
        'input_size': input_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'output_size': output_size,
        'seq_length': seq_length,
        'dropout': dropout,
        'best_val_loss': float(best_val_loss),
        'num_epochs_trained': len(train_losses)
    }
    hyperparams_path = get_model_path('lstm_hyperparams.json', 'lstm')
    with open(hyperparams_path, 'w') as f:
        json.dump(hyperparams, f, indent=2)
    
    print(f"\n✅ Đã lưu metrics tại: {metrics_path}")
    print(f"✅ Đã lưu hyperparameters tại: {hyperparams_path}")
    print(f"✅ Đã lưu best model tại: {best_model_path}")
    print(f"✅ Đã lưu scaler tại: {scaler_path}")
    
    print("\n" + "=" * 60)
    print("Hoàn thành training!")
    print("=" * 60)
    print(f"\n📊 Tổng số epochs: {len(train_losses)}")
    print(f"📊 Best validation loss: {best_val_loss:.6f}")

