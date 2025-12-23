#!/usr/bin/env python3
"""
Manual test script for verifying different components of the solar power prediction system.
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


def test_training_functions():
    """
    Test the solar model training functions directly.
    This will create model files in the 'repo/models' directory.
    """
    logger.info("=" * 70)
    logger.info("Testing Training Functions")
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


async def test_prediction_service():
    """
    Test the prediction service.
    This assumes that the models have been trained and are available in the 'repo/models' directory.
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


async def main():
    """Run all manual tests."""
    # test_training_functions()
    await test_prediction_service()


if __name__ == "__main__":
    # To run the training tests, uncomment the following line:
    # test_training_functions()

    # To run the prediction service tests, run the script as is.
    # Make sure you have trained the models first by running `test_training_functions()`.
    
    # Due to the async nature of the prediction service, we run it in an asyncio event loop.
    asyncio.run(main())
