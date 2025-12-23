"""
Core configuration management using Pydantic Settings.
Handles environment variables and application-wide settings.
"""

import os
from pathlib import Path
from typing import Dict, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings from environment variables."""
    
    # Application
    app_name: str = Field(default="Solar Power Generation Predictor", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Server
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    
    # Startup Mode
    startup_mode: str = Field(
        default="load_models",
        env="STARTUP_MODE",
        description="'train_now' to train models on startup, 'load_models' to load pre-trained"
    )
    
    # Model Directories
    model_dir: str = Field(default="./models", env="MODEL_DIR")
    weather_data_dir: str = Field(default="../code/grid-weather", env="WEATHER_DATA_DIR")
    solar_data_dir: str = Field(default="../code/data", env="SOLAR_DATA_DIR")
    
    # Weather Data Files
    weather_hist_file: str = Field(
        default="../code/data/23.530236_119.588339.csv",
        env="WEATHER_HIST_FILE"
    )
    weather_pred_file: str = Field(
        default="../code/data/weather-pred.csv",
        env="WEATHER_PRED_FILE"
    )
    
    # Solar Dataset Files
    @property
    def solar_files(self) -> Dict[str, str]:
        """Solar dataset file paths."""
        base_dir = self.solar_data_dir
        return {
            "CT安集01": os.path.join(base_dir, "旭東URE發電量v7-3 2.xlsx - CT安集01-data.csv"),
            "CT安集02": os.path.join(base_dir, "旭東URE發電量v7-3 2.xlsx - CT安集02-data.csv"),
            "元晶": os.path.join(base_dir, "旭東URE發電量v7-3 2.xlsx - 元晶-data.csv"),
            "EEC_land": os.path.join(base_dir, "旭東URE發電量v7-3 2.xlsx - EEC-data.csv"),
        }
    
    # Model Training Parameters
    weather_window_size: int = Field(default=30, env="WEATHER_WINDOW_SIZE")
    weather_model_mode: str = Field(default="multi", env="WEATHER_MODEL_MODE")
    solar_test_months: int = Field(default=6, env="SOLAR_TEST_MONTHS")
    solar_valid_months: int = Field(default=1, env="SOLAR_VALID_MONTHS")
    solar_model_tactic: str = Field(
        default="merged",
        env="SOLAR_MODEL_TACTIC",
        description="'merged' or 'seperated'"
    )
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    
    # CORS
    cors_origins: list = Field(
        default=["*"],
        env="CORS_ORIGINS",
        description="Comma-separated list of allowed origins"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings instance."""
    return settings
