"""
Path management utilities
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_EXTERNAL = DATA_DIR / "external"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_LSTM = MODELS_DIR / "lstm"
MODELS_RF = MODELS_DIR / "random_forest"

# Config paths
CONFIG_DIR = PROJECT_ROOT / "config"

# Visualization paths
VIZ_DIR = PROJECT_ROOT / "visualization"

def get_data_path(filename: str, subfolder: str = "processed") -> Path:
    """Get path to data file"""
    if subfolder == "raw":
        return DATA_RAW / filename
    elif subfolder == "processed":
        return DATA_PROCESSED / filename
    elif subfolder == "external":
        return DATA_EXTERNAL / filename
    else:
        return DATA_DIR / filename

def get_model_path(filename: str, model_type: str = "lstm") -> Path:
    """Get path to model file"""
    if model_type == "lstm":
        return MODELS_LSTM / filename
    elif model_type == "rf" or model_type == "random_forest":
        return MODELS_RF / filename
    else:
        return MODELS_DIR / filename

def get_config_path(filename: str) -> Path:
    """Get path to config file"""
    return CONFIG_DIR / filename

def ensure_dirs():
    """Create necessary directories if they don't exist"""
    dirs = [
        DATA_RAW, DATA_PROCESSED, DATA_EXTERNAL,
        MODELS_LSTM, MODELS_RF,
        CONFIG_DIR, VIZ_DIR
    ]
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)

