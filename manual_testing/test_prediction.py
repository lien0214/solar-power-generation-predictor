"""
Manual test script for the prediction service.
"""

import sys
import logging
import asyncio
from pathlib import Path

# Add repo to path
repo_path = Path(__file__).parent.parent
sys.path.insert(0, str(repo_path))

from app.services import get_model_manager, get_prediction_service
from app.core.config import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_prediction_script():
    """
    Tests the prediction service.
    Assumes that the models have been trained and are available.
    """
    logger.info("=" * 70)
    logger.info("Testing Prediction Service")
    logger.info("=" * 70)

    # Initialize the model manager
    model_manager = get_model_manager()
    if not model_manager.is_ready():
        logger.info("Initializing Model Manager...")
        await model_manager.initialize(
            mode="load_models",
            model_dir=settings.model_dir,
            weather_hist_file=settings.weather_hist_file,
            weather_pred_file=settings.weather_pred_file,
            solar_files=settings.solar_files,
        )
        if model_manager.is_ready():
            logger.info("✅ Model Manager initialized successfully.")
        else:
            logger.error("❌ Model Manager failed to initialize.")
            return

    prediction_service = get_prediction_service()

    # --- Test Day Prediction ---
    logger.info("--- Testing Day Prediction ---")
    try:
        logger.info("Testing with 'merged' tactic...")
        predictions = await prediction_service.predict_day_range(
            lon=120.2, lat=23.5,
            start_date="2025-01-01", end_date="2025-01-05",
            pmp=1000, tactic="merged"
        )
        logger.info(f"  Merged prediction for 5 days: {predictions}")

        logger.info("Testing with 'seperated' tactic...")
        predictions = await prediction_service.predict_day_range(
            lon=120.2, lat=23.5,
            start_date="2025-01-01", end_date="2025-01-05",
            pmp=1000, tactic="seperated"
        )
        logger.info(f"  Separated prediction for 5 days: {predictions}")

    except Exception as e:
        logger.error(f"❌ Day prediction failed: {e}", exc_info=True)

    # --- Test Rolling Weather Forecast ---
    logger.info("--- Testing Rolling Weather Forecast ---")
    try:
        logger.info("Requesting a date range outside of pre-computed weather data...")
        # This date range is intentionally set far in the future to trigger the rolling forecast
        predictions = await prediction_service.predict_day_range(
            lon=120.2, lat=23.5,
            start_date="2026-01-01", end_date="2026-01-05",
            pmp=1000, tactic="merged"
        )
        logger.info(f"  Rolling forecast prediction for 5 days: {predictions}")
        logger.info("  Check the 'repo/data/weather_predictions' directory for the generated forecast file.")

    except Exception as e:
        logger.error(f"❌ Rolling weather forecast test failed: {e}", exc_info=True)

if __name__ == "__main__":
    # Make sure you have trained the models first by running
    # test_train_weather_model.py and test_train_solar_models.py
    asyncio.run(test_prediction_script())
