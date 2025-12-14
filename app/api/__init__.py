"""API package."""

from fastapi import APIRouter
from .v1 import api_router as v1_router

# Main API router aggregating all versions
api_router = APIRouter()
api_router.include_router(v1_router, prefix="/v1")

__all__ = ["api_router"]
