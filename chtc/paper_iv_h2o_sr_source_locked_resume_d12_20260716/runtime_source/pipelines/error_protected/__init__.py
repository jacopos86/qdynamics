"""Error-protected sidecar replay surfaces.

This package is intentionally outside the canonical ADAPT pipeline.
"""

from pipelines.error_protected.contracts import (
    DetectedObservableEstimate,
    DetectionReplayInput,
    DetectionRunSummary,
    ErrorDetectionConfig,
    RawObservableBundle,
)

__all__ = [
    "DetectedObservableEstimate",
    "DetectionReplayInput",
    "DetectionRunSummary",
    "ErrorDetectionConfig",
    "RawObservableBundle",
]
