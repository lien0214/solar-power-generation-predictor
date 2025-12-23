#!/usr/bin/env python3
"""
Test script to verify model training works correctly.
This tests the training functions directly without starting the server.
"""

import sys
import logging
from pathlib import Path

# Add repo to path
repo_path = Path(__file__).parent
sys.path.insert(0, str(repo_path))

from model import (
    train_weather_model,
    train_solar_model_merged,
    load_weather_model,
    load_solar_model_merged,
    train_solar_model_seperated,
    load_solar_model_seperated,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_weather_training():
    """Test weather model training."""
    logger.info("=" * 70)
    logger.info("Testing Weather Model Training")
    logger.info("=" * 70)
    
    try:
        result = train_weather_model(
            csv_path="../code/data/23.530236_119.588339.csv",
            output_dir="./test_models",
            win=30,
            mode="multi"
        )
        
        logger.info("✅ Weather training successful!")
        logger.info(f"   Bundle path: {result['bundle_path']}")
        logger.info(f"   Targets: {result['targets']}")
        logger.info(f"   Window size: {result['window_size']}")
        logger.info(f"   Train samples: {result['train_samples']}")
        
        # Test loading
        logger.info("Testing model loading...")
        bundle = load_weather_model(result['bundle_path'])
        logger.info(f"✅ Model loaded successfully! Mode: {bundle['mode']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Weather training failed: {e}", exc_info=True)
        return False


def test_solar_training_merged():
    """Test solar model training."""
    logger.info("=" * 70)
    logger.info("Testing Solar Model Training - Merged")
    logger.info("=" * 70)
    
    solar_files = {
        "CT安集01": "../code/data/旭東URE發電量v7-3 2.xlsx - CT安集01-data.csv",
        "CT安集02": "../code/data/旭東URE發電量v7-3 2.xlsx - CT安集02-data.csv",
        "元晶": "../code/data/旭東URE發電量v7-3 2.xlsx - 元晶-data.csv",
        "EEC_land": "../code/data/旭東URE發電量v7-3 2.xlsx - EEC-data.csv",
    }
    
    try:
        result = train_solar_model_merged(
            solar_files=solar_files,
            weather_hist_file="../code/data/23.530236_119.588339.csv",
            weather_pred_file="../code.py/data/weather-pred.csv",
            output_dir="./test_models",
            test_months=6,
            valid_months=1
        )
        
        logger.info("✅ Solar training successful!")
        logger.info(f"   Bundle path: {result['bundle_path']}")
        logger.info(f"   Datasets: {result['datasets']}")
        logger.info(f"   Features: {len(result['feature_cols'])}")
        logger.info(f"   Train samples: {result['train_samples']}")
        
        # Test loading
        logger.info("Testing model loading...")
        bundle = load_solar_model_merged(result['bundle_path'])
        logger.info(f"✅ Model loaded successfully! Datasets: {bundle['datasets']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Solar training failed: {e}", exc_info=True)
        return False


def test_solar_training_seperated():
    """Test separated solar model training."""
    logger.info("=" * 70)
    logger.info("Testing Solar Model Training - Separated")
    logger.info("=" * 70)
    
    solar_files = {
        "CT安集01": "../code/data/旭東URE發電量v7-3 2.xlsx - CT安集01-data.csv",
        "CT安集02": "../code/data/旭東URE發電量v7-3 2.xlsx - CT安集02-data.csv",
        "元晶": "../code/data/旭東URE發電量v7-3 2.xlsx - 元晶-data.csv",
        "EEC_land": "../code/data/旭東URE發電量v7-3 2.xlsx - EEC-data.csv",
    }
    
    try:
        result = train_solar_model_seperated(
            solar_files=solar_files,
            weather_hist_file="../code/data/23.530236_119.588339.csv",
            weather_pred_file="../code/data/weather-pred.csv",
            output_dir="./test_models",
            test_months=6,
            valid_months=1
        )
        
        logger.info("✅ Separated solar training successful!")
        logger.info(f"   Bundle path: {result['bundle_path']}")
        logger.info(f"   Datasets: {result['datasets']}")
        
        # Test loading
        logger.info("Testing model loading...")
        bundle = load_solar_model_seperated(result['bundle_path'])
        logger.info(f"✅ Separated models loaded successfully! Datasets: {bundle['datasets']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Separated solar training failed: {e}", exc_info=True)
        return False


def main():
    """Run all tests."""
    logger.info("🧪 Starting Model Training Tests")
    
    weather_ok = test_weather_training()
    solar_ok = test_solar_training_merged()
    solar_sep_ok = test_solar_training_seperated()
    
    logger.info("=" * 70)
    if weather_ok and solar_ok and solar_sep_ok:
        logger.info("✅ All tests passed!")
        return 0
    else:
        logger.error("❌ Some tests failed")
        logger.error(f"   Weather: {'✅' if weather_ok else '❌'}")
        logger.error(f"   Solar (Merged): {'✅' if solar_ok else '❌'}")
        logger.error(f"   Solar (Separated): {'✅' if solar_sep_ok else '❌'}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
