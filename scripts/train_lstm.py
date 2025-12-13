"""
Entry point script để train LSTM model
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.train_lstm import train_lstm_model

if __name__ == "__main__":
    train_lstm_model()

