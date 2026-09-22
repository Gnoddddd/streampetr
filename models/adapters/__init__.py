"""Detector-specific adapters for detector-independent research interfaces."""

from .streampetr_prediction_adapter import PreviousPrediction, from_streampetr_result

__all__ = ["PreviousPrediction", "from_streampetr_result"]
