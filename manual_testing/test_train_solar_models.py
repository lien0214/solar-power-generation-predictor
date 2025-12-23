"""
Manual test script for training the solar models.
"""

import sys
import logging
from pathlib import Path

# Add repo to path
repo_path = Path(__file__).parent.parent
sys.path.insert(0, str(repo_path))

from app.core.config import settings
from model import (
    train_solar_model_merged,
    train_solar_model_seperated,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_train_solar_models_script():
    """
    Tests the solar model training functions.
    """
    logger.info("=" * 70)
    logger.info("Testing Solar Model Training")
    logger.info("=" * 70)

    solar_files = settings.solar_files
    weather_hist_file = settings.weather_hist_file
    weather_pred_file = settings.weather_pred_file
    output_dir = settings.model_dir

    logger.info("--- Testing Merged Model Training ---")
    try:
        train_solar_model_merged(
            solar_files=solar_files,
            weather_hist_file=weather_hist_file,
            weather_pred_file=weather_pred_file,
            output_dir=output_dir,
        )
        logger.info("✅ Merged model training successful.")
    except Exception as e:
        logger.error(f"❌ Merged model training failed: {e}", exc_info=True)

    logger.info("--- Testing Separated Model Training ---")
    try:
        train_solar_model_seperated(
            solar_files=solar_files,
            weather_hist_file=weather_hist_file,
            weather_pred_file=weather_pred_file,
            output_dir=output_dir,
        )
        logger.info("✅ Separated model training successful.")
    except Exception as e:
        logger.error(f"❌ Separated model training failed: {e}", exc_info=True)
        
    logger.info(f"Models saved to '{settings.model_dir}'")

if __name__ == "__main__":
    test_train_solar_models_script()
