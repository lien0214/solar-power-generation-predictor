"""
Custom exceptions for the application.

These exceptions provide specific error handling for various failure scenarios.
"""


class ModelsNotReadyError(Exception):
    """Raised when ML models are not loaded and a prediction is requested."""
    pass


class InvalidDateRangeError(ValueError):
    """Raised when date range validation fails."""
    pass


class ModelTrainingError(Exception):
    """Raised when model training fails."""
    pass


class ModelLoadingError(Exception):
    """Raised when model loading from disk fails."""
    pass
