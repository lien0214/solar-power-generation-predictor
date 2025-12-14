#!/usr/bin/env python3
"""
Server entry point script.
Runs the FastAPI application using uvicorn.
"""

import uvicorn
from app.core import settings

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
