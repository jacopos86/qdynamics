#!/usr/bin/env python3
"""Small JSON sidecar contract for comparator parity/conformance checks.

This module records parity results after a caller has actually run a matched
repo-local vs external/reference comparison.  It never computes physics on its
own and never promotes manuscript/table artifacts.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from pipelines.exact_bench.comparator_provenance import (
    REQUIRED_COMPARATOR_SOURCE_FIELDS,
    comparator_source_fields,
)

COMPARATOR_PARITY_SIDECAR_SCHEMA = "paper_i_comparator_parity_sidecar_v1"
DEFAULT_PARITY_SIDECAR_FILENAME = "comparator_parity_sidecar.json"


def build_comparator_parity_sidecar(
    *,
    algorithm_id: str,
    parity_status: str,
    parity_scope: str,
    runner_module: str | None = None,
    subject_artifact: str | Path | None = None,
    subject_artifact_sha256: str | None = None,
    parity_reference_algorithm_id: str | None = None,
    parity_reference_artifact: str | Path | None = None,
    parity_reference_artifact_sha256: str | None = None,
    parity_energy_abs_delta: float | None = None,
    parity_state_infidelity: float | None = None,
    parity_selected_generators_match: bool | None = None,
    parity_compiled_cost_match: bool | None = None,
    cutoff_pair: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a normalized parity/conformance sidecar payload.

    Use explicit ``parity_status`` values such as ``passed``, ``failed``,
    ``not_run_dependency_missing``, or ``bounded_semantic_mismatch``.  Do not set
    comparison quantities unless they came from a real matched check.
    """
    source = comparator_source_fields(
        algorithm_id,
        runner_module=runner_module,
        parity_reference_artifact=parity_reference_artifact,
        parity_energy_abs_delta=parity_energy_abs_delta,
        parity_state_infidelity=parity_state_infidelity,
        parity_selected_generators_match=parity_selected_generators_match,
        parity_compiled_cost_match=parity_compiled_cost_match,
        cutoff_pair=cutoff_pair,
    )
    source["parity_status"] = str(parity_status)
    source["parity_scope"] = str(parity_scope)
    if parity_reference_algorithm_id is not None:
        source["parity_reference_algorithm_id"] = str(parity_reference_algorithm_id)
    payload: dict[str, Any] = {
        "schema": COMPARATOR_PARITY_SIDECAR_SCHEMA,
        "algorithm_id": str(algorithm_id),
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "subject_artifact": None if subject_artifact is None else str(subject_artifact),
        "subject_artifact_sha256": subject_artifact_sha256,
        "parity_reference_artifact_sha256": parity_reference_artifact_sha256,
        **source,
    }
    for field in REQUIRED_COMPARATOR_SOURCE_FIELDS:
        payload.setdefault(field, None)
    if extra:
        payload["extra"] = dict(extra)
    return payload


def write_comparator_parity_sidecar(
    output_dir: str | Path,
    payload: Mapping[str, Any],
    *,
    filename: str = DEFAULT_PARITY_SIDECAR_FILENAME,
) -> Path:
    """Write a parity sidecar without touching result/manifest/table files."""
    path = Path(output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path


__all__ = [
    "COMPARATOR_PARITY_SIDECAR_SCHEMA",
    "DEFAULT_PARITY_SIDECAR_FILENAME",
    "build_comparator_parity_sidecar",
    "write_comparator_parity_sidecar",
]
