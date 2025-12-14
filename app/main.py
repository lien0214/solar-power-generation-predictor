"""
FastAPI Application Main Entry Point.
Orchestrates the application lifecycle and component initialization.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core import settings
from .api import api_router
from .services import get_model_manager
from .schemas import ErrorResponse

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format=settings.log_format
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("=" * 70)
    logger.info(f"🚀 {settings.app_name} v{settings.app_version}")
    logger.info("=" * 70)
    
    try:
        # Initialize model manager
        model_manager = get_model_manager()
        await model_manager.initialize(
            mode=settings.startup_mode,
            model_dir=settings.model_dir,
            weather_hist_file=settings.weather_hist_file,
            weather_pred_file=settings.weather_pred_file,
            solar_files=settings.solar_files,
            weather_window=settings.weather_window_size,
            weather_mode=settings.weather_model_mode,
            solar_test_months=settings.solar_test_months,
            solar_valid_months=settings.solar_valid_months
        )
        
        logger.info("=" * 70)
        logger.info("✅ Application startup complete!")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}", exc_info=True)
        raise
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down application...")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="API for predicting solar power generation based on location and time range",
    lifespan=lifespan,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse}
    }
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(api_router)


# Health check endpoint
@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint."""
    model_manager = get_model_manager()
    return {
        "status": "healthy",
        "models_ready": model_manager.is_ready(),
        "version": settings.app_version
    }


# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": "Solar Power Generation Predictor API",
        "docs_url": "/docs",
        "health_url": "/health"
    }
