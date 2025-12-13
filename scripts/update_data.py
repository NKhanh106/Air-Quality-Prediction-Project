"""
Entry point script để cập nhật dữ liệu
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.mining import update_weather_data

if __name__ == "__main__":
    print("🔄 Đang cập nhật dữ liệu...")
    update_weather_data()
    print("✅ Hoàn thành cập nhật dữ liệu!")

