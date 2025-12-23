"""
Model Management Service.
Handles training, loading, and lifecycle management of ML models.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


class ModelManagerService:
    """Service for managing ML models (weather and solar forecasting)."""
    
    def __init__(self):
        """Initialize model manager service."""
        self.weather_model_bundle: Optional[Dict[str, Any]] = None
        self.solar_model_bundle_merged: Optional[Dict[str, Any]] = None
        self.solar_model_bundle_seperated: Optional[Dict[str, Any]] = None
        self._initialized = False
    
    async def initialize(
        self,
        mode: str,
        model_dir: str,
        weather_data_dir: str,
        weather_pred_file: str,
        solar_files: Dict[str, str],
        weather_window: int = 30,
        weather_mode: str = "multi",
        solar_test_months: int = 6,
        solar_valid_months: int = 1,
    ) -> None:
        """
        Initialize models based on startup mode.
        
        Args:
            mode: "train_now" or "load_models"
            model_dir: Directory for model storage
            weather_data_dir: Directory with historical weather data
            weather_pred_file: Predicted weather data path
            solar_files: Dict of solar dataset paths
            weather_window: Window size for weather model
            weather_mode: "single" or "multi" output
            solar_test_months: Test period months for solar model
            solar_valid_months: Validation period months for solar model
        """
        if self._initialized:
            logger.warning("Model manager already initialized")
            return
        
        logger.info("=" * 70)
        logger.info("🤖 Initializing Model Manager Service")
        logger.info(f"Mode: {mode}")
        logger.info("=" * 70)
        
        try:
            # Import here to avoid circular dependencies
            from ..model import (
                train_weather_model,
                train_solar_model_merged,
                load_weather_model,
                load_solar_model_merged,
                train_solar_model_seperated,
                load_solar_model_seperated,
            )
            
            if mode == "train_now":
                await self._train_models(
                    model_dir=model_dir,
                    weather_data_dir=weather_data_dir,
                    weather_pred_file=weather_pred_file,
                    solar_files=solar_files,
                    weather_window=weather_window,
                    weather_mode=weather_mode,
                    solar_test_months=solar_test_months,
                    solar_valid_months=solar_valid_months,
                    train_weather_model=train_weather_model,
                    train_solar_model_merged=train_solar_model_merged,
                    load_weather_model=load_weather_model,
                    load_solar_model_merged=load_solar_model_merged,
                    train_solar_model_seperated=train_solar_model_seperated,
                    load_solar_model_seperated=load_solar_model_seperated,
                )
            
            elif mode == "load_models":
                await self._load_models(
                    model_dir=model_dir,
                    load_weather_model=load_weather_model,
                    load_solar_model_merged=load_solar_model_merged,
                    load_solar_model_seperated=load_solar_model_seperated,
                )
            
            else:
                logger.warning(f"Unknown startup mode: {mode}")
            
            self._initialized = True
            logger.info("✅ Model Manager Service initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Model Manager: {e}", exc_info=True)
            raise
    
    async def _train_models(
        self,
        model_dir: str,
        weather_data_dir: str,
        weather_pred_file: str,
        solar_files: Dict[str, str],
        weather_window: int,
        weather_mode: str,
        solar_test_months: int,
        solar_valid_months: int,
        train_weather_model,
        train_solar_model_merged,
        load_weather_model,
        load_solar_model_merged,
        train_solar_model_seperated,
        load_solar_model_seperated,
    ) -> None:
        """Train models from scratch."""
        logger.info("🔨 Training models from scratch...")
        
        # Train weather model
        logger.info("📊 Training weather forecasting model...")
        weather_result = train_weather_model(
            data_dir=weather_data_dir,
            output_dir=model_dir,
            win=weather_window,
            mode=weather_mode
        )
        logger.info(f"✅ Weather model trained: {weather_result['bundle_path']}")
        self.weather_model_bundle = load_weather_model(weather_result['bundle_path'])
        
        # Train solar model (merged)
        logger.info("☀️ Training solar generation model (merged)...")
        solar_result = train_solar_model_merged(
            solar_files=solar_files,
            weather_hist_file=weather_hist_file,
            weather_pred_file=weather_pred_file,
            output_dir=model_dir,
            test_months=solar_test_months,
            valid_months=solar_valid_months
        )
        logger.info(f"✅ Solar model (merged) trained: {solar_result['bundle_path']}")
        self.solar_model_bundle_merged = load_solar_model_merged(solar_result['bundle_path'])
        
        # Train solar model (seperated)
        logger.info("☀️ Training solar generation model (seperated)...")
        solar_result_seperated = train_solar_model_seperated(
            solar_files=solar_files,
            weather_hist_file=weather_hist_file,
            weather_pred_file=weather_pred_file,
            output_dir=model_dir,
            test_months=solar_test_months,
            valid_months=solar_valid_months
        )
        logger.info(f"✅ Solar model (seperated) trained: {solar_result_seperated['bundle_path']}")
        self.solar_model_bundle_seperated = load_solar_model_seperated(
            solar_result_seperated['bundle_path']
        )

    async def _load_models(
        self,
        model_dir: str,
        load_weather_model,
        load_solar_model_merged,
        load_solar_model_seperated,
    ) -> None:
        """Load pre-trained models from disk."""
        logger.info("📦 Loading pre-trained models...")
        
        # Load weather model
        weather_bundle_path = Path(model_dir) / "weather_model_bundle.pkl"
        if weather_bundle_path.exists():
            self.weather_model_bundle = load_weather_model(str(weather_bundle_path))
            logger.info(f"✅ Weather model loaded: {weather_bundle_path}")
        else:
            logger.warning(f"⚠️ Weather model not found: {weather_bundle_path}")
        
        # Load solar model (merged)
        solar_bundle_path = Path(model_dir) / "solar_model_bundle_merged.pkl"
        if solar_bundle_path.exists():
            self.solar_model_bundle_merged = load_solar_model_merged(str(solar_bundle_path))
            logger.info(f"✅ Solar model loaded: {solar_bundle_path}")
        else:
            logger.warning(f"⚠️ Solar model not found: {solar_bundle_path}")
        
        # Load solar model (seperated)
        solar_bundle_path_seperated = Path(model_dir) / "solar_model_bundle_seperated.pkl"
        if solar_bundle_path_seperated.exists():
            self.solar_model_bundle_seperated = load_solar_model_seperated(
                str(solar_bundle_path_seperated)
            )
            logger.info(f"✅ Solar model (seperated) loaded: {solar_bundle_path_seperated}")
        else:
            logger.warning(f"⚠️ Solar model (seperated) not found: {solar_bundle_path_seperated}")
    
    def is_ready(self) -> bool:
        """Check if models are loaded and ready."""
        return (
            self._initialized
            and self.weather_model_bundle is not None
            and self.solar_model_bundle_merged is not None
            and self.solar_model_bundle_seperated is not None
        )
    
    def get_weather_model(self) -> Optional[Dict[str, Any]]:
        """Get loaded weather model bundle."""
        return self.weather_model_bundle
    
    def get_solar_model_merged(self) -> Optional[Dict[str, Any]]:
        """Get loaded solar model bundle."""
        return self.solar_model_bundle_merged
        
    def get_solar_model_seperated(self) -> Optional[Dict[str, Any]]:
        """Get loaded solar model bundle."""
        return self.solar_model_bundle_seperated


# Global service instance
model_manager_service = ModelManagerService()


def get_model_manager() -> ModelManagerService:
    """Get model manager service instance."""
    return model_manager_service
