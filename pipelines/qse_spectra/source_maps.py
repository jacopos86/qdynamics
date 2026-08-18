#!/usr/bin/env python3
"""Paper III QSE source maps: build, load, audit, and validate.

RECONSTRUCTION (2026-08-18): the original module was never committed and was
lost when snapshot commit 6442fbb5 captured its importers without it. This
implementation is reconstructed against the committed behavioral spec in
``test/test_paper_iii_qse_source_maps.py`` and the consuming wrapper
``pipelines/reporting/paper_iii_qse_audit.py``.

A source map records, for each Paper III evidence artifact (QSE manifest,
table aggregate, Optuna study summary, compatibility matrix manifest,
evidence report, report input), the file's role, relative path, sha256,
schema version, and extracted provenance (run class, compatibility tier,
method id, regime, approval status) together with a controller-boundary
check. The audit re-reads every source and fails closed on missing files,
hash or schema drift, duplicate entries, missing approval status,
controller-boundary gaps or violations, and blocked production gates.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

PAPER_III_QSE_SOURCE_MAP_SCHEMA_VERSION = "paper_iii_qse_source_map_v1"
PAPER_III_QSE_SOURCE_MAP_AUDIT_SCHEMA_VERSION = "paper_iii_qse_source_map_audit_v1"

_BOUNDARY_REQUIRED_SECTIONS = (
    "static_record_selection",
    "paper_iii_contract",
    "qse_response_functions_v1",
    "qse_conductivity_response_v1",
    "qse_green_function_v1",
)
_BOUNDARY_VIOLATION_FLAGS = (
    "feeds_controller_decisions",
    "controller_usable",
    "uses_exact_reference_for_decision",
    "uses_future_exact_forecast_for_decision",
    "reference_comparisons_feed_controller_decisions",
    "decision_path_allowed",
    "controller_decision_input",
    "uses_reference_for_decision",
)


class PaperIIIQSESourceMapError(ValueError):
    """Raised when a Paper III QSE source map is invalid or fails its audit."""


@dataclass(frozen=True)
class PaperIIIQSESourceSpec:
    """One source-map entry request: a role name and the artifact path."""

    role: str
    path: Any
    expected_schema_version: str | None = None


def sha256_file(path: Any) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PaperIIIQSESourceMapError(f"unreadable_source: {path} ({exc})") from exc
    if not isinstance(payload, Mapping):
        raise PaperIIIQSESourceMapError(f"unreadable_source: {path} is not a JSON object")
    return dict(payload)


def _first_string(*candidates: Any) -> str | None:
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value
    return None


def _mapping_or_empty(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _first_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    rows = payload.get("rows")
    if isinstance(rows, list) and rows and isinstance(rows[0], Mapping):
        return dict(rows[0])
    return {}


def _boundary_flag_violations(boundary: Mapping[str, Any]) -> list[str]:
    return [flag for flag in _BOUNDARY_VIOLATION_FLAGS if boundary.get(flag) is True]


def _evaluate_controller_boundary(
    payload: Mapping[str, Any],
) -> tuple[bool, list[str], list[dict[str, str]]]:
    """Return (passed, checked_sections, problems) for one source payload."""

    checked: list[str] = []
    problems: list[dict[str, str]] = []

    def _check_boundary(section_name: str, boundary: Mapping[str, Any]) -> None:
        checked.append(section_name)
        for flag in _boundary_flag_violations(boundary):
            problems.append(
                {
                    "code": "controller_boundary_violation",
                    "message": f"{section_name} sets {flag}=true",
                }
            )

    top = payload.get("controller_boundary")
    if isinstance(top, Mapping):
        _check_boundary("controller_boundary", top)

    for key in _BOUNDARY_REQUIRED_SECTIONS:
        section = payload.get(key)
        if not isinstance(section, Mapping):
            continue
        boundary = section.get("controller_boundary")
        if not isinstance(boundary, Mapping):
            problems.append(
                {
                    "code": "controller_boundary_missing",
                    "message": f"{key}.controller_boundary is missing",
                }
            )
            continue
        _check_boundary(f"{key}.controller_boundary", boundary)

    strict_policy = payload.get("strict_policy")
    if isinstance(strict_policy, Mapping):
        _check_boundary("strict_policy", strict_policy)

    rows = payload.get("rows")
    if isinstance(rows, list):
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping) or "controller_boundary_passed" not in row:
                continue
            checked.append(f"rows[{index}].controller_boundary_passed")
            if row.get("controller_boundary_passed") is not True:
                problems.append(
                    {
                        "code": "controller_boundary_violation",
                        "message": f"rows[{index}].controller_boundary_passed is not true",
                    }
                )

    return (not problems), checked, problems


def _evaluate_production_gates(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    problems: list[dict[str, str]] = []
    gate = payload.get("paper_iii_production_gate")
    if isinstance(gate, Mapping):
        blocked = (
            gate.get("ok") is False
            or gate.get("production_ready") is False
            or str(gate.get("n_ph2_production_readiness") or "") == "blocked"
        )
        if blocked:
            problems.append(
                {
                    "code": "production_gate_blocked",
                    "message": "paper_iii_production_gate is not production ready",
                }
            )
    rows = payload.get("rows")
    if isinstance(rows, list):
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping) or not row.get("production_gate_present"):
                continue
            if row.get("production_gate_ok") is False or row.get("production_gate_production_ready") is False:
                problems.append(
                    {
                        "code": "production_gate_blocked",
                        "message": f"rows[{index}] production gate is blocked",
                    }
                )
    return problems


def _extract_provenance(payload: Mapping[str, Any]) -> dict[str, Any]:
    contract = _mapping_or_empty(payload.get("paper_iii_contract"))
    row = _first_row(payload)
    selection = _mapping_or_empty(payload.get("static_record_selection"))
    selection_config = _mapping_or_empty(selection.get("selection_config"))

    method_id = _first_string(payload.get("method_id"), row.get("method_id"))
    if method_id is None:
        mode = _first_string(selection_config.get("mode"))
        if mode is not None:
            method_id = f"qse_selection::{mode}"

    approval_status = _first_string(
        payload.get("approval_status"),
        contract.get("approval_status"),
        row.get("approval_status"),
    )
    if approval_status is None:
        approval_status = "not_applicable"

    return {
        "run_class": _first_string(
            payload.get("run_class"), contract.get("run_class"), row.get("run_class")
        ),
        "compatibility_tier": _first_string(
            payload.get("compatibility_tier"),
            contract.get("compatibility_tier"),
            row.get("compatibility_tier"),
        ),
        "method_id": method_id,
        "regime_or_case": _first_string(
            payload.get("regime"),
            payload.get("regime_label"),
            payload.get("run_tag"),
            row.get("row_id"),
        ),
        "approval_status": approval_status,
    }


def _relative_source_path(path: Path, source_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path(source_root).resolve()))
    except ValueError:
        return os.path.relpath(str(path.resolve()), str(Path(source_root).resolve()))


def build_paper_iii_qse_source_map(
    specs: Sequence[PaperIIIQSESourceSpec],
    *,
    source_root: Any,
    map_id: str | None = None,
) -> dict[str, Any]:
    """Build a source-map payload from artifact specs.

    The builder records boundary-check outcomes without raising on boundary
    problems; enforcement is the audit/validate step.
    """

    root = Path(source_root)
    records: list[dict[str, Any]] = []
    for spec in specs:
        role = str(spec.role)
        path = Path(spec.path)
        if not path.is_file():
            raise PaperIIIQSESourceMapError(f"missing_source_file: {path}")
        payload = _load_json_mapping(path)
        relative = _relative_source_path(path, root)
        passed, checked, problems = _evaluate_controller_boundary(payload)
        record: dict[str, Any] = {
            "source_id": f"{role}::{relative}",
            "source_map_role": role,
            "source_path": relative,
            "source_sha256": sha256_file(path),
            "schema_version": payload.get("schema_version"),
            "expected_schema_version": (
                spec.expected_schema_version
                if spec.expected_schema_version is not None
                else payload.get("schema_version")
            ),
            "pipeline": payload.get("pipeline"),
            **_extract_provenance(payload),
            "controller_boundary": {
                "passed": bool(passed),
                "checked_sections": list(checked),
                "problems": list(problems),
            },
        }
        records.append(record)

    summary = {
        "roles": sorted(record["source_map_role"] for record in records),
        "all_sources_have_approval_status": all(
            bool(str(record.get("approval_status") or "").strip()) for record in records
        ),
        "all_sources_controller_boundary_passed": all(
            bool(record["controller_boundary"]["passed"]) for record in records
        ),
    }
    return {
        "schema_version": PAPER_III_QSE_SOURCE_MAP_SCHEMA_VERSION,
        "map_id": map_id,
        "source_root": str(root),
        "source_count": len(records),
        "sources": records,
        "summary": summary,
    }


def write_paper_iii_qse_source_map(source_map: Mapping[str, Any], path: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(source_map), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_paper_iii_qse_source_map(path: Any) -> dict[str, Any]:
    payload = _load_json_mapping(Path(path))
    if payload.get("schema_version") != PAPER_III_QSE_SOURCE_MAP_SCHEMA_VERSION:
        raise PaperIIIQSESourceMapError(
            "source_map_schema_mismatch: expected "
            f"{PAPER_III_QSE_SOURCE_MAP_SCHEMA_VERSION!r}, got "
            f"{payload.get('schema_version')!r}."
        )
    return payload


def audit_paper_iii_qse_source_map(
    source_map: Mapping[str, Any],
    *,
    base_dir: Any | None = None,
) -> dict[str, Any]:
    """Re-read every source and return a non-raising audit report."""

    base = Path(base_dir) if base_dir is not None else Path(".")
    sources = source_map.get("sources")
    records = [dict(record) for record in sources] if isinstance(sources, list) else []
    failures: list[dict[str, str]] = []
    seen_ids: set[str] = set()

    for record in records:
        role = str(record.get("source_map_role") or "")
        source_path = str(record.get("source_path") or "")
        source_id = str(record.get("source_id") or f"{role}::{source_path}")
        label = f"{role} ({source_path})"
        if source_id in seen_ids:
            failures.append(
                {"code": "duplicate_source_id", "message": f"duplicate source id {source_id!r}"}
            )
            continue
        seen_ids.add(source_id)

        path = base / source_path
        if not path.is_file():
            failures.append(
                {"code": "missing_source_file", "message": f"{label}: file not found at {path}"}
            )
            continue
        actual_sha = sha256_file(path)
        if actual_sha != str(record.get("source_sha256") or ""):
            failures.append(
                {"code": "source_sha256_mismatch", "message": f"{label}: sha256 changed"}
            )
        try:
            payload = _load_json_mapping(path)
        except PaperIIIQSESourceMapError as exc:
            failures.append({"code": "unreadable_source", "message": str(exc)})
            continue

        expected_schema = record.get("expected_schema_version")
        if expected_schema is not None and payload.get("schema_version") != expected_schema:
            failures.append(
                {
                    "code": "source_schema_mismatch",
                    "message": (
                        f"{label}: schema_version {payload.get('schema_version')!r} "
                        f"!= expected {expected_schema!r}"
                    ),
                }
            )

        if not str(record.get("approval_status") or "").strip():
            failures.append(
                {"code": "missing_approval_status", "message": f"{label}: approval_status is empty"}
            )

        _, _, boundary_problems = _evaluate_controller_boundary(payload)
        for problem in boundary_problems:
            failures.append({**problem, "message": f"{label}: {problem['message']}"})
        for problem in _evaluate_production_gates(payload):
            failures.append({**problem, "message": f"{label}: {problem['message']}"})

    return {
        "schema_version": PAPER_III_QSE_SOURCE_MAP_AUDIT_SCHEMA_VERSION,
        "ok": not failures,
        "source_count": len(records),
        "error_count": len(failures),
        "failures": failures,
    }


def validate_paper_iii_qse_source_map(
    source_map: Mapping[str, Any],
    *,
    base_dir: Any | None = None,
) -> dict[str, Any]:
    """Audit and raise ``PaperIIIQSESourceMapError`` on any failure."""

    report = audit_paper_iii_qse_source_map(source_map, base_dir=base_dir)
    if not report["ok"]:
        rendered = "; ".join(
            f"{failure['code']}: {failure['message']}" for failure in report["failures"]
        )
        raise PaperIIIQSESourceMapError(f"Paper III QSE source-map audit failed: {rendered}")
    return report
