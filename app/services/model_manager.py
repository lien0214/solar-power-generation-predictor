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
        self.solar_model_bundle: Optional[Dict[str, Any]] = None
        self._initialized = False
    
    async def initialize(
        self,
        mode: str,
        model_dir: str,
        weather_hist_file: str,
        weather_pred_file: str,
        solar_files: Dict[str, str],
        weather_window: int = 30,
        weather_mode: str = "multi",
        solar_test_months: int = 6,
        solar_valid_months: int = 1
    ) -> None:
        """
        Initialize models based on startup mode.
        
        Args:
            mode: "train_now" or "load_models"
            model_dir: Directory for model storage
            weather_hist_file: Historical weather data path
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
            from ..models import train_weather_model, train_solar_model, load_weather_model, load_solar_model
            
            if mode == "train_now":
                await self._train_models(
                    model_dir=model_dir,
                    weather_hist_file=weather_hist_file,
                    weather_pred_file=weather_pred_file,
                    solar_files=solar_files,
                    weather_window=weather_window,
                    weather_mode=weather_mode,
                    solar_test_months=solar_test_months,
                    solar_valid_months=solar_valid_months,
                    train_weather_model=train_weather_model,
                    train_solar_model=train_solar_model,
                    load_weather_model=load_weather_model,
                    load_solar_model=load_solar_model
                )
            
            elif mode == "load_models":
                await self._load_models(
                    model_dir=model_dir,
                    load_weather_model=load_weather_model,
                    load_solar_model=load_solar_model
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
        weather_hist_file: str,
        weather_pred_file: str,
        solar_files: Dict[str, str],
        weather_window: int,
        weather_mode: str,
        solar_test_months: int,
        solar_valid_months: int,
        train_weather_model,
        train_solar_model,
        load_weather_model,
        load_solar_model
    ) -> None:
        """Train models from scratch."""
        logger.info("🔨 Training models from scratch...")
        
        # Train weather model
        logger.info("📊 Training weather forecasting model...")
        weather_result = train_weather_model(
            csv_path=weather_hist_file,
            output_dir=model_dir,
            win=weather_window,
            mode=weather_mode
        )
        logger.info(f"✅ Weather model trained: {weather_result['bundle_path']}")
        
        # Load trained weather model
        self.weather_model_bundle = load_weather_model(weather_result['bundle_path'])
        
        # Train solar model
        logger.info("☀️ Training solar generation model...")
        solar_result = train_solar_model(
            solar_files=solar_files,
            weather_hist_file=weather_hist_file,
            weather_pred_file=weather_pred_file,
            output_dir=model_dir,
            test_months=solar_test_months,
            valid_months=solar_valid_months
        )
        logger.info(f"✅ Solar model trained: {solar_result['bundle_path']}")
        
        # Load trained solar model
        self.solar_model_bundle = load_solar_model(solar_result['bundle_path'])
    
    async def _load_models(
        self,
        model_dir: str,
        load_weather_model,
        load_solar_model
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
        
        # Load solar model
        solar_bundle_path = Path(model_dir) / "solar_model_bundle.pkl"
        if solar_bundle_path.exists():
            self.solar_model_bundle = load_solar_model(str(solar_bundle_path))
            logger.info(f"✅ Solar model loaded: {solar_bundle_path}")
        else:
            logger.warning(f"⚠️ Solar model not found: {solar_bundle_path}")
    
    def is_ready(self) -> bool:
        """Check if models are loaded and ready."""
        return (
            self._initialized
            and self.weather_model_bundle is not None
            and self.solar_model_bundle is not None
        )
    
    def get_weather_model(self) -> Optional[Dict[str, Any]]:
        """Get loaded weather model bundle."""
        return self.weather_model_bundle
    
    def get_solar_model(self) -> Optional[Dict[str, Any]]:
        """Get loaded solar model bundle."""
        return self.solar_model_bundle


# Global service instance
model_manager_service = ModelManagerService()


def get_model_manager() -> ModelManagerService:
    """Get model manager service instance."""
    return model_manager_service
