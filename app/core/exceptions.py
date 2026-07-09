"""Custom exceptions used across the system."""

from __future__ import annotations


class RetailMonitoringError(Exception):
    """Base exception for the retail monitoring system."""


class CameraError(RetailMonitoringError):
    """Raised when the camera source cannot be opened or read."""


class ConfigurationError(RetailMonitoringError):
    """Raised when a configuration file is missing or invalid."""


class LlmClientError(RetailMonitoringError):
    """Raised when the LLM provider cannot process a request."""


class VideoAnalysisError(RetailMonitoringError):
    """Raised when a video cannot be opened or analyzed."""

