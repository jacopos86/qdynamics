"""Contracts for detect-only sidecar replay artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Literal, Mapping


DetectionMode = Literal["off", "sector_audit", "sector_filter", "sector_detect"]
CorrectionMode = Literal["off"]
EstimateStatus = Literal[
    "ok",
    "partial",
    "unavailable",
    "failed_no_accepted_shots",
    "failed_too_few_accepted_shots",
    "failed_missing_detector_observability",
    "failed_oracle_config",
]


@dataclass(frozen=True)
class ErrorDetectionConfig:
    mode: DetectionMode = "sector_audit"
    strict: bool = True
    min_accepted_shots: int = 256
    min_acceptance_rate: float | None = None
    sector_n_up: int | None = None
    sector_n_dn: int | None = None
    ordering: str | None = None
    symmetry_mode: Literal["off", "verify_only"] = "verify_only"
    emit_rejected_records_summary: bool = True
    postprocess_correction_mode: CorrectionMode = "off"


@dataclass(frozen=True)
class DetectionReplayInput:
    artifact_json: Path
    output_json: Path
    noise_mode: Literal["runtime", "backend_scheduled"]
    execution_surface: Literal["raw_measurement_v1"] = "raw_measurement_v1"
    backend_name: str | None = None
    use_fake_backend: bool = False
    shots: int = 2048
    oracle_repeats: int = 1
    oracle_aggregate: Literal["mean", "median"] = "mean"
    seed: int = 7
    seed_transpiler: int | None = None
    transpile_optimization_level: int = 1
    raw_transport: str = "auto"
    raw_store_memory: bool = True
    raw_grouping_mode: Literal["qwc_basis_cover_reuse"] = "qwc_basis_cover_reuse"
    runtime_profile: str | Mapping[str, Any] = "legacy_runtime_v0"
    runtime_session: str | Mapping[str, Any] = "prefer_session"
    raw_artifact_path: Path | None = None
    detection: ErrorDetectionConfig = field(default_factory=ErrorDetectionConfig)
    compare_to_canonical: bool = True
    compare_to_exact_reference: bool = True


@dataclass(frozen=True)
class RawObservableBundle:
    observable_family: str
    mean: float | None
    stderr: float | None
    total_shots: int
    record_count: int
    records: tuple[Any, ...] = field(default_factory=tuple)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DetectedObservableEstimate:
    name: str
    status: EstimateStatus
    raw_mean: float | None = None
    accepted_mean: float | None = None
    corrected_mean: float | None = None
    raw_stderr: float | None = None
    accepted_stderr: float | None = None
    corrected_stderr: float | None = None
    total_shots: int = 0
    accepted_shots: int = 0
    rejected_shots: int = 0
    acceptance_rate: float = 0.0
    accepted_minus_raw: float | None = None
    corrected_minus_raw: float | None = None
    accepted_minus_canonical_final_audit: float | None = None
    accepted_minus_exact_reference: float | None = None
    per_group: Mapping[str, Any] = field(default_factory=dict)
    rejection_reasons: Mapping[str, int] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class DetectionRunSummary:
    schema_version: str
    generated_utc: str
    route_kind: Literal["adapt_detect_only_replay"]
    base_artifact_json: str
    physics_point: Mapping[str, Any]
    ansatz_identity: Mapping[str, Any]
    oracle_request: Mapping[str, Any]
    detection: Mapping[str, Any]
    raw_summary: Mapping[str, Any]
    estimates: Mapping[str, Any]
    comparisons: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    failure: Mapping[str, Any] | None = None


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return json_ready(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    item = getattr(value, "item", None)
    if callable(item) and value.__class__.__module__.split(".", 1)[0] == "numpy":
        return json_ready(item())
    return value


__all__ = [
    "CorrectionMode",
    "DetectedObservableEstimate",
    "DetectionMode",
    "DetectionReplayInput",
    "DetectionRunSummary",
    "ErrorDetectionConfig",
    "EstimateStatus",
    "RawObservableBundle",
    "json_ready",
]
