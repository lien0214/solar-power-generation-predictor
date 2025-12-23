"""
Manual test script for training the weather model.
"""

import sys
import logging
from pathlib import Path

# Add repo to path
repo_path = Path(__file__).parent.parent
sys.path.insert(0, str(repo_path))

from app.core.config import settings
from model import train_weather_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_train_weather_model_script():
    """
    Tests the weather model training function.
    """
    logger.info("=" * 70)
    logger.info("Testing Weather Model Training")
    logger.info("=" * 70)

    try:
        train_weather_model(
            csv_path=settings.weather_hist_file,
            output_dir=settings.model_dir,
            win=settings.weather_window_size,
            mode=settings.weather_model_mode,
        )
        logger.info("✅ Weather model training successful.")
        logger.info(f"Model saved to '{settings.model_dir}'")
    except Exception as e:
        logger.error(f"❌ Weather model training failed: {e}", exc_info=True)

if __name__ == "__main__":
    test_train_weather_model_script()
