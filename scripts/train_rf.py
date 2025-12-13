"""
Entry point script để train Random Forest model
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.train_rf import train_random_forest

if __name__ == "__main__":
    train_random_forest(use_advanced_features=True, lag_steps=12, n_trials=50, max_features=150)

