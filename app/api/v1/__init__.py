"""API v1 package."""

from fastapi import APIRouter
from .prediction import router as prediction_router

# Aggregate all v1 routers
api_router = APIRouter()
api_router.include_router(prediction_router, prefix="/predict", tags=["predictions"])

__all__ = ["api_router"]
