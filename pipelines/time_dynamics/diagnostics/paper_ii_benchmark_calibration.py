"""Pre-execution Paper-II benchmark calibration audit.

This is the script half of ``agent_guidance/skills/time-dynamics-benchmark-calibration``.
It runs **before** CHTC submission, aggregation, evidence-PDF generation, or a
visible table update, and emits a machine-readable calibration receipt that
``agent_guidance/skills/paper-ii-results/SKILL.md`` consumes.

Fail-closed contract
--------------------
``ERROR`` means stop.  ``paper-ii-run`` SKILL.md:45 forbids submission,
aggregation, evidence-PDF generation, and table updates while a calibration
``ERROR`` stands, and ``paper-ii-results`` SKILL.md:33 additionally blocks on
"required checks were not run" -- which is why this runner reports
``not_checked`` explicitly instead of silently passing a check whose evidence is
absent.

Every threshold used here is read from a tracked source, never invented:

* the promoted class-settings lock manifest itself
  (``chtc/generic_time_dynamics_table/input/class_settings/paper_ii_class_settings_lock_v1.json``);
* the live campaign policy constants in
  ``pipelines/time_dynamics/campaigns/paper_ii_staged_accuracy_v1.py``
  (5% individual response target, 15% response guard, factor-of-two matched
  aggregate burden, mean-infidelity ceiling 1e-4), which is the executable form
  of ``agent_guidance/time-dynamics/paper-ii-comparison-protocol.md``;
* the frozen-trajectory motion tolerance ``1e-12`` used by
  ``pipelines/time_dynamics/benchmarks/common.py`` in its
  ``non_frozen_when_exact_diagnostic_moves`` check;
* the QPU-faithful decision-flag key set used by the same module's
  ``exact_reference_diagnostic_only`` check.

Judgment that this runner deliberately does not automate -- observable-set and
compile-target declaration, promotion of a newer lock, acceptance of an
author override -- is routed to the SKILL.md and the comparison protocol.

Usage
-----
::

    PYTHONPATH=. python3 -m pipelines.time_dynamics.diagnostics.paper_ii_benchmark_calibration \\
        --artifact MATH/paper_facing/paper_II_dynamics/t20_provisional_author_override_receipt.json \\
        --run-root output/paper_ii_staged \\
        --receipt-json output/diagnostics/paper_ii_calibration/calibration_receipt.json

    PYTHONPATH=. python3 -m pipelines.time_dynamics.diagnostics.paper_ii_benchmark_calibration --self-test

Receipt status is ``ERROR`` / ``WARN`` / ``INCOMPLETE`` / ``OK``. Exit status is
``0`` when no check errored, ``1`` on at least one ``ERROR``, ``2`` on a usage
error -- ``INCOMPLETE`` does not fail the process, because the unrun-check
blocker belongs to ``paper-ii-results``, but it can never be read as a pass.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

RECEIPT_SCHEMA = "paper_ii_benchmark_calibration_receipt_v1"

STATUS_OK = "ok"
STATUS_WARN = "warn"
STATUS_ERROR = "error"
STATUS_NOT_CHECKED = "not_checked"

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

PROMOTED_LOCK_PATH = (
    "chtc/generic_time_dynamics_table/input/class_settings/"
    "paper_ii_class_settings_lock_v1.json"
)
SMOKE_LOCK_PATH = (
    "chtc/generic_time_dynamics_table/input/class_settings/"
    "paper_ii_class_settings_smoke_lock_v1.json"
)
LOCK_MANIFEST_SCHEMA = "dynamics_class_settings_lock_manifest_v1"

RESPONSE_SCALE_SIDECAR_PATH = (
    "pipelines/time_dynamics/campaigns/paper_ii_exact_response_scales_v1.json"
)

# common.py: motion below this magnitude is "not moving" for the frozen-trajectory
# invariant.  Reused verbatim so the row-level signature and the payload-level
# sidecar agree.
FROZEN_MOTION_TOLERANCE = 1.0e-12

# common.py: exact-reference leakage flags for the QPU-faithful boundary.
EXACT_DECISION_FLAG_KEYS = (
    "uses_reference_for_decision",
    "uses_future_exact_forecast_for_decision",
    "exact_reference_controller_inputs",
)
DECISION_FLAG_CONTAINERS = ("runtime_contract", "summary", "route_config", "provenance")

# paper-ii-run SKILL.md:60 -- forbidden seed-artifact provenance markers.
FORBIDDEN_SEED_MARKERS = ("staged", "fallback", "pending", "recovery")
SEED_IDENTITY_FIELDS = (
    "seed_path",
    "seed_source",
    "seed_source_path",
    "seed_artifact",
    "seed_artifact_path",
    "static_seed_path",
    "source_seed",
    "source_seed_json",
    "runtime_seed_path",
    "seed_manifest_path",
)
SEED_STATUS_FIELDS = (
    "seed_kind",
    "seed_status",
    "seed_role",
    "seed_provenance",
    "seed_state",
)

# paper-ii-run SKILL.md:59 -- axes every method inside one benchmark point shares.
SAMENESS_AXES: dict[str, tuple[str, ...]] = {
    "static_seed_hash": ("seed_sha256", "static_seed_hash", "seed_hash", "seed"),
    "drive": ("drive", "drive_id"),
    "time_grid": (
        "time_grid",
        "time_grid_id",
        "reporting_grid",
        "horizon_time_step",
        "checkpoint_count",
    ),
    "phonon_cutoff": ("n_ph_max", "phonon_cutoff", "nph"),
    "reference_diagnostic_policy": (
        "reference_policy",
        "diagnostic_policy",
        "reference_diagnostic_policy",
        "accuracy_status",
    ),
    "observable_set": ("observable_set", "observables", "observable_set_id"),
    "diagnostic_exact_reference": (
        "reference_key",
        "exact_reference_key",
        "reference_data_sha256",
    ),
    "compile_target": (
        "compile_target",
        "compile_profile",
        "backend",
        "transpile_profile",
    ),
}

# A benchmark point is everything except the algorithmic method: methods inside
# one point are what the sameness audit compares.
BENCHMARK_POINT_FIELDS = ("seed", "regime", "drive", "horizon", "controller")

METHOD_FIELDS = ("route", "method")
_METHOD_ALIASES = {"apm": "exchange", "exchange": "exchange", "avqds": "avqds"}


class CalibrationInputError(ValueError):
    """The runner was pointed at inputs it cannot audit."""


# --------------------------------------------------------------------------- io


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_importable(root: Path) -> None:
    entry = str(root)
    if entry not in sys.path:
        sys.path.insert(0, entry)


def _repo_relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _check(
    check_id: str,
    status: str,
    summary: str,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "status": status,
        "summary": summary,
        "details": dict(details or {}),
    }


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "on", "1"}
    return False


# ------------------------------------------------------------------ artifacts


class Artifact:
    """One JSON input plus the shapes the calibration checks can read from it."""

    def __init__(self, path: Path, payload: Any, root: Path) -> None:
        self.path = path
        self.payload = payload
        self.relpath = _repo_relative(path, root)
        self.sha256 = _sha256(path)

    @property
    def mapping(self) -> Mapping[str, Any]:
        return self.payload if isinstance(self.payload, Mapping) else {}

    @property
    def benchmark_rows(self) -> list[Mapping[str, Any]]:
        rows = self.mapping.get("rows")
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            return []
        return [
            row
            for row in rows
            if isinstance(row, Mapping)
            and any(row.get(field) for field in METHOD_FIELDS)
            and row.get("drive")
        ]

    @property
    def trajectory_rows(self) -> list[Mapping[str, Any]]:
        raw = self.mapping.get("trajectory")
        if isinstance(raw, Mapping):
            raw = raw.get("points")
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            return []
        return [row for row in raw if isinstance(row, Mapping)]

    @property
    def declares_override(self) -> bool:
        text = " ".join(
            str(self.mapping.get(field, ""))
            for field in ("admission", "author_override", "promotion_status")
        ).lower()
        promotion = self.mapping.get("promotion")
        if isinstance(promotion, Mapping):
            text += " " + str(promotion.get("status", "")).lower()
        return "override" in text

    @property
    def is_auditable(self) -> bool:
        return bool(self.benchmark_rows or self.trajectory_rows)


def _load_artifacts(paths: Iterable[Path], root: Path) -> tuple[list[Artifact], list[dict[str, Any]]]:
    artifacts: list[Artifact] = []
    unreadable: list[dict[str, Any]] = []
    for path in paths:
        try:
            payload = _read_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            unreadable.append({"path": _repo_relative(path, root), "error": str(exc)})
            continue
        artifacts.append(Artifact(path, payload, root))
    return artifacts, unreadable


def _resolve_inputs(
    *,
    artifact_paths: Sequence[str],
    run_roots: Sequence[str],
    root: Path,
) -> list[Path]:
    resolved: list[Path] = []
    for raw in artifact_paths:
        path = Path(raw)
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            raise CalibrationInputError(f"calibration artifact is missing: {raw}")
        resolved.append(path)
    for raw in run_roots:
        base = Path(raw)
        if not base.is_absolute():
            base = root / base
        if not base.is_dir():
            raise CalibrationInputError(f"calibration run root is missing: {raw}")
        resolved.extend(sorted(base.rglob("run.json")))
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in resolved:
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


# ---------------------------------------------------------- check 1: the lock


def check_promoted_class_settings_lock(root: Path, lock_override: str | None) -> dict[str, Any]:
    """Enforce the promoted class-settings lock named by paper-ii-run SKILL.md:24."""

    lock_path = Path(lock_override) if lock_override else root / PROMOTED_LOCK_PATH
    if not lock_path.is_absolute():
        lock_path = root / lock_path
    smoke_path = root / SMOKE_LOCK_PATH
    details: dict[str, Any] = {
        "expected_path": PROMOTED_LOCK_PATH,
        "resolved_path": _repo_relative(lock_path, root),
        "smoke_variant_present": smoke_path.is_file(),
    }
    if not lock_path.is_file():
        details["remediation"] = (
            "fetch the promoted lock from CHTC "
            "(EVIDENCE_PARITY_AUDIT_20260831.md section 3 item 8); the smoke "
            "variant is not a substitute for paper-facing work"
        )
        return _check(
            "promoted_class_settings_lock",
            STATUS_ERROR,
            "the promoted class-settings lock is absent"
            + (" (only the smoke variant is present)" if smoke_path.is_file() else ""),
            details,
        )

    details["sha256"] = _sha256(lock_path)
    try:
        payload = _read_json(lock_path)
    except (OSError, json.JSONDecodeError) as exc:
        details["error"] = str(exc)
        return _check(
            "promoted_class_settings_lock",
            STATUS_ERROR,
            "the promoted class-settings lock is unreadable",
            details,
        )
    if not isinstance(payload, Mapping):
        return _check(
            "promoted_class_settings_lock",
            STATUS_ERROR,
            "the promoted class-settings lock is not a JSON object",
            details,
        )

    failures: list[str] = []
    details["schema"] = payload.get("schema")
    details["lock_status"] = payload.get("lock_status")
    details["lock_policy"] = payload.get("lock_policy")
    details["require_canonical_controller_classes"] = payload.get(
        "require_canonical_controller_classes"
    )
    if payload.get("schema") != LOCK_MANIFEST_SCHEMA:
        failures.append(f"schema is not {LOCK_MANIFEST_SCHEMA}")
    if str(payload.get("lock_status") or "").lower() != "locked":
        failures.append("lock_status is not 'locked'")
    if not _truthy(payload.get("require_canonical_controller_classes")):
        failures.append("require_canonical_controller_classes is not set")

    settings = payload.get("settings")
    if not isinstance(settings, Sequence) or isinstance(settings, (str, bytes)) or not settings:
        failures.append("the lock carries no settings entries")
        settings = []
    entries: list[dict[str, Any]] = []
    for entry in settings:
        if not isinstance(entry, Mapping):
            failures.append("a settings entry is not a JSON object")
            continue
        row = {
            "settings_id": entry.get("settings_id"),
            "tuning_class": entry.get("tuning_class"),
            "algorithm_id": entry.get("algorithm_id"),
            "class_tuned_result_locked": entry.get("class_tuned_result_locked"),
            "strict_online_feedback_exact_free": entry.get(
                "strict_online_feedback_exact_free"
            ),
        }
        entries.append(row)
        label = row["settings_id"] or row["tuning_class"] or "<unnamed>"
        if not _truthy(row["class_tuned_result_locked"]):
            failures.append(f"{label}: class_tuned_result_locked is not set")
        if not _truthy(row["strict_online_feedback_exact_free"]):
            failures.append(f"{label}: strict_online_feedback_exact_free is not set")
    details["settings_entries"] = entries
    details["settings_entry_count"] = len(entries)
    details["failures"] = failures

    if failures:
        return _check(
            "promoted_class_settings_lock",
            STATUS_ERROR,
            f"the promoted lock fails {len(failures)} enforcement condition(s)",
            details,
        )
    return _check(
        "promoted_class_settings_lock",
        STATUS_OK,
        f"promoted lock enforced ({len(entries)} locked class settings, "
        f"sha256 {details['sha256'][:12]})",
        details,
    )


# ------------------------------------------------- check 2: sameness per point


def _benchmark_point_key(row: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (field, str(row[field]))
        for field in BENCHMARK_POINT_FIELDS
        if row.get(field) is not None
    )


def _method_of(row: Mapping[str, Any]) -> str:
    for field in METHOD_FIELDS:
        value = row.get(field)
        if value:
            return _METHOD_ALIASES.get(str(value), str(value))
    return "<unknown>"


def _axis_value(row: Mapping[str, Any], candidates: Sequence[str]) -> tuple[str, Any] | None:
    for field in candidates:
        if field in row and row[field] is not None:
            return field, row[field]
    return None


def check_benchmark_point_sameness(artifacts: Sequence[Artifact]) -> dict[str, Any]:
    """Every method inside one benchmark point shares the declared axes."""

    rows: list[tuple[str, Mapping[str, Any]]] = []
    for artifact in artifacts:
        for row in artifact.benchmark_rows:
            rows.append((artifact.relpath, row))
    if not rows:
        return _check(
            "benchmark_point_sameness",
            STATUS_NOT_CHECKED,
            "no artifact carried per-benchmark-point rows",
            {"axes": sorted(SAMENESS_AXES)},
        )

    points: dict[tuple[tuple[str, str], ...], list[tuple[str, Mapping[str, Any]]]] = {}
    for source, row in rows:
        points.setdefault(_benchmark_point_key(row), []).append((source, row))

    mismatches: list[dict[str, Any]] = []
    undeclared: dict[str, int] = {}
    compared_points = 0
    for key, members in sorted(points.items()):
        methods = {_method_of(row) for _, row in members}
        if len(methods) < 2:
            continue
        compared_points += 1
        for axis, candidates in SAMENESS_AXES.items():
            observed: dict[str, list[str]] = {}
            missing_for: list[str] = []
            for source, row in members:
                found = _axis_value(row, candidates)
                if found is None:
                    missing_for.append(f"{_method_of(row)} @ {source}")
                    continue
                observed.setdefault(
                    json.dumps(found[1], sort_keys=True, default=str), []
                ).append(f"{_method_of(row)} @ {source}")
            if missing_for:
                undeclared[axis] = undeclared.get(axis, 0) + len(missing_for)
            if len(observed) > 1:
                mismatches.append(
                    {
                        "benchmark_point": dict(key),
                        "axis": axis,
                        "values": observed,
                    }
                )

    details = {
        "row_count": len(rows),
        "benchmark_point_count": len(points),
        "multi_method_point_count": compared_points,
        "mismatches": mismatches,
        "axes_not_declared_by_rows": undeclared,
        "axes": sorted(SAMENESS_AXES),
        "note": (
            "axes absent from the rows are reported as undeclared, never as "
            "passing; declare them in the artifact or verify them against the "
            "run payloads named by the rows"
        ),
    }
    if mismatches:
        return _check(
            "benchmark_point_sameness",
            STATUS_ERROR,
            f"{len(mismatches)} axis mismatch(es) inside a benchmark point",
            details,
        )
    if compared_points == 0:
        return _check(
            "benchmark_point_sameness",
            STATUS_NOT_CHECKED,
            "no benchmark point carried two or more methods to compare",
            details,
        )
    if undeclared:
        return _check(
            "benchmark_point_sameness",
            STATUS_WARN,
            f"{compared_points} multi-method point(s) agree on every declared "
            f"axis; {len(undeclared)} axis/axes are undeclared by the rows",
            details,
        )
    return _check(
        "benchmark_point_sameness",
        STATUS_OK,
        f"{compared_points} multi-method benchmark point(s) agree on all "
        f"{len(SAMENESS_AXES)} axes",
        details,
    )


# ------------------------------------------------ check 3: seed provenance


def _walk_mappings(node: Any, depth: int = 0) -> Iterable[Mapping[str, Any]]:
    if depth > 6:
        return
    if isinstance(node, Mapping):
        yield node
        for value in node.values():
            yield from _walk_mappings(value, depth + 1)
    elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
        for value in node:
            yield from _walk_mappings(value, depth + 1)


def _seed_marker_hits(value: str) -> list[str]:
    stem = Path(str(value)).stem.lower()
    tokens = {token for token in _split_tokens(stem)}
    return sorted(marker for marker in FORBIDDEN_SEED_MARKERS if marker in tokens)


def _split_tokens(text: str) -> list[str]:
    token = ""
    tokens: list[str] = []
    for char in text:
        if char.isalnum():
            token += char
        elif token:
            tokens.append(token)
            token = ""
    if token:
        tokens.append(token)
    return tokens


def check_seed_provenance(artifacts: Sequence[Artifact]) -> dict[str, Any]:
    """No staged, fallback, pending, or recovery seed artifact backs the evidence."""

    violations: list[dict[str, Any]] = []
    inspected = 0
    seed_hashes: dict[str, list[str]] = {}
    for artifact in artifacts:
        for node in _walk_mappings(artifact.payload):
            for field in SEED_IDENTITY_FIELDS:
                value = node.get(field)
                if not isinstance(value, str) or not value:
                    continue
                inspected += 1
                hits = _seed_marker_hits(value)
                if hits:
                    violations.append(
                        {
                            "artifact": artifact.relpath,
                            "field": field,
                            "value": value,
                            "markers": hits,
                        }
                    )
            for field in SEED_STATUS_FIELDS:
                value = node.get(field)
                if not isinstance(value, str) or not value:
                    continue
                inspected += 1
                hits = sorted(
                    marker
                    for marker in FORBIDDEN_SEED_MARKERS
                    if marker in _split_tokens(value.lower())
                )
                if hits:
                    violations.append(
                        {
                            "artifact": artifact.relpath,
                            "field": field,
                            "value": value,
                            "markers": hits,
                        }
                    )
            digest = node.get("seed_sha256")
            if isinstance(digest, str) and len(digest) == 64:
                seed_hashes.setdefault(digest, []).append(artifact.relpath)

    details = {
        "inspected_seed_field_count": inspected,
        "forbidden_markers": list(FORBIDDEN_SEED_MARKERS),
        "violations": violations,
        "distinct_seed_sha256": {
            digest: sorted(set(sources)) for digest, sources in seed_hashes.items()
        },
        "scope_note": (
            "markers are matched against the seed artifact's own basename and "
            "explicit seed-status fields only. Parent directories are excluded "
            "on purpose: the canonical campaign directory is literally named "
            "paper_ii_staged_accuracy_euler_v1, so a path-token rule would "
            "misreport it. Confirming the seed source artifact by hash against "
            "its manifest stays an operator step (see SKILL.md)."
        ),
    }
    if violations:
        return _check(
            "seed_provenance",
            STATUS_ERROR,
            f"{len(violations)} seed reference(s) name a forbidden provenance class",
            details,
        )
    if inspected == 0:
        return _check(
            "seed_provenance",
            STATUS_NOT_CHECKED,
            "no artifact declared a seed identity or status field",
            details,
        )
    if len(seed_hashes) > 1:
        return _check(
            "seed_provenance",
            STATUS_WARN,
            f"{inspected} seed reference(s) are clean, but {len(seed_hashes)} "
            "distinct seed digests appear across the artifact set",
            details,
        )
    return _check(
        "seed_provenance",
        STATUS_OK,
        f"{inspected} seed reference(s) carry no staged/fallback/pending/recovery marker",
        details,
    )


# --------------------------------------- check 4: trajectory-freeze invalid rows


def _geometry_candidates(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The row itself plus its nested geometry blocks (e.g. ``baseline_geometry``)."""

    candidates: list[Mapping[str, Any]] = []
    # depth 4 of the shared 6-level budget leaves the row and two nesting levels,
    # which covers both the flat AP-runner point and the nested benchmark row.
    for node in _walk_mappings(row, depth=4):
        if any(key in node for key in ("theta_dot_l2", "rho_num", "rho_miss")):
            candidates.append(node)
    return candidates


def _frozen_signature(node: Mapping[str, Any]) -> dict[str, float] | None:
    theta = _float_or_none(node.get("theta_dot_l2"))
    rho_num = _float_or_none(node.get("rho_num"))
    rho_miss = _float_or_none(node.get("rho_miss"))
    if theta is None or rho_num is None or rho_miss is None:
        return None
    if (
        abs(theta) <= FROZEN_MOTION_TOLERANCE
        and abs(rho_num) <= FROZEN_MOTION_TOLERANCE
        and abs(rho_miss - 1.0) <= FROZEN_MOTION_TOLERANCE
    ):
        return {"theta_dot_l2": theta, "rho_num": rho_num, "rho_miss": rho_miss}
    return None


def check_trajectory_freeze(artifacts: Sequence[Artifact]) -> dict[str, Any]:
    """Detect the theta_dot_l2=0 / rho_num=0 / rho_miss~1 invalid-row signature."""

    scanned_payloads = 0
    scanned_rows = 0
    invalid: list[dict[str, Any]] = []
    sidecar_failures: list[dict[str, Any]] = []
    for artifact in artifacts:
        rows = artifact.trajectory_rows
        if rows:
            scanned_payloads += 1
        for index, row in enumerate(rows):
            scanned_rows += 1
            for node in _geometry_candidates(row):
                signature = _frozen_signature(node)
                if signature is None:
                    continue
                invalid.append(
                    {
                        "artifact": artifact.relpath,
                        "row_index": int(row.get("checkpoint_index", row.get("index", index))),
                        "signature": signature,
                    }
                )
                break
        for node in _walk_mappings(artifact.payload):
            if node.get("check_id") != "non_frozen_when_exact_diagnostic_moves":
                continue
            if node.get("passed") is False:
                sidecar_failures.append(
                    {
                        "artifact": artifact.relpath,
                        "status": node.get("status"),
                        "details": node.get("details"),
                    }
                )

    details = {
        "trajectory_payload_count": scanned_payloads,
        "trajectory_row_count": scanned_rows,
        "motion_tolerance": FROZEN_MOTION_TOLERANCE,
        "invalid_rows": invalid,
        "sidecar_freeze_failures": sidecar_failures,
        "signature": "theta_dot_l2==0 and rho_num==0 and rho_miss==1",
    }
    if invalid or sidecar_failures:
        return _check(
            "trajectory_freeze_invalid_rows",
            STATUS_ERROR,
            f"{len(invalid)} frozen trajectory row(s) and "
            f"{len(sidecar_failures)} sidecar freeze failure(s) detected",
            details,
        )
    if scanned_rows == 0:
        return _check(
            "trajectory_freeze_invalid_rows",
            STATUS_NOT_CHECKED,
            "no trajectory payload was available locally to scan",
            details,
        )
    return _check(
        "trajectory_freeze_invalid_rows",
        STATUS_OK,
        f"{scanned_rows} trajectory row(s) across {scanned_payloads} payload(s) "
        "show no freeze signature",
        details,
    )


# ------------------------------- check 5: QPU-faithful decision-data separation


def check_qpu_faithful_separation(artifacts: Sequence[Artifact]) -> dict[str, Any]:
    """Exact/reference data must not reach controller decision telemetry."""

    leaks: list[dict[str, Any]] = []
    undeclared: list[dict[str, Any]] = []
    inspected = 0
    for artifact in artifacts:
        if not artifact.trajectory_rows:
            continue
        inspected += 1
        payload = artifact.mapping
        observed: dict[str, bool] = {}
        for key in EXACT_DECISION_FLAG_KEYS:
            seen = False
            for container in DECISION_FLAG_CONTAINERS:
                node = payload.get(container)
                if not isinstance(node, Mapping) or key not in node:
                    continue
                seen = True
                if _truthy(node.get(key)):
                    leaks.append(
                        {
                            "artifact": artifact.relpath,
                            "container": container,
                            "field": key,
                            "value": node.get(key),
                        }
                    )
            observed[key] = seen
        missing = [key for key, seen in observed.items() if not seen]
        if missing:
            undeclared.append({"artifact": artifact.relpath, "missing_flags": missing})

    details = {
        "run_payload_count": inspected,
        "decision_flag_keys": list(EXACT_DECISION_FLAG_KEYS),
        "flag_containers": list(DECISION_FLAG_CONTAINERS),
        "leaks": leaks,
        "payloads_without_declared_flags": undeclared,
    }
    if leaks:
        return _check(
            "qpu_faithful_decision_separation",
            STATUS_ERROR,
            f"{len(leaks)} exact-reference decision flag(s) are truthy",
            details,
        )
    if inspected == 0:
        return _check(
            "qpu_faithful_decision_separation",
            STATUS_NOT_CHECKED,
            "no run payload with decision telemetry was available locally",
            details,
        )
    if undeclared:
        return _check(
            "qpu_faithful_decision_separation",
            STATUS_ERROR,
            f"{len(undeclared)} run payload(s) do not declare the exact-decision "
            "flags; absence is not separation",
            details,
        )
    return _check(
        "qpu_faithful_decision_separation",
        STATUS_OK,
        f"{inspected} run payload(s) declare every exact-decision flag as false",
        details,
    )


# -------------------------------- checks 6 and 7: matched-accuracy policy


def _load_live_policy(root: Path, *, allow_unverified_scales: bool) -> tuple[Any, dict[str, Any]]:
    """Return the live campaign accuracy policy plus how it was obtained.

    The canonical construction verifies the tracked response-scale sidecar
    against the exact-reference store.  That store is regenerable-local and is
    routinely absent (EVIDENCE_PARITY_AUDIT_20260831.md section 2.2), so the
    degraded path -- the sidecar's recorded scales without exact cross-check --
    is opt-in and never reports ``ok``.
    """

    _ensure_importable(root)
    from pipelines.time_dynamics.campaigns import (  # noqa: PLC0415
        paper_ii_staged_accuracy_v1 as campaign_module,
    )
    from pipelines.time_dynamics.staged_campaign import (  # noqa: PLC0415
        AccuracyPolicy,
    )

    constants = {
        "response_relative_target": campaign_module.RESPONSE_RELATIVE_TARGET,
        "response_relative_guard": campaign_module.RESPONSE_RELATIVE_GUARD,
        "matched_accuracy_factor": campaign_module.MATCHED_ACCURACY_FACTOR,
        "infidelity_ceiling": campaign_module.ACCURACY_CEILING,
        "absolute_energy_error_ceiling": campaign_module.ABSOLUTE_ENERGY_ERROR_CEILING,
        "absolute_doublon_error_ceiling": campaign_module.ABSOLUTE_DOUBLON_ERROR_CEILING,
        "absolute_site_occupation_error_ceiling": (
            campaign_module.ABSOLUTE_SITE_OCCUPATION_ERROR_CEILING
        ),
    }
    try:
        policy = campaign_module.build_campaign(prior_root=None).accuracy_policy
        return policy, {
            "source": "campaign_verified_exact_reference_store",
            "degraded": False,
            "constants": constants,
        }
    except Exception as exc:  # noqa: BLE001 - reported, never swallowed
        provenance = {
            "source": "campaign_verified_exact_reference_store",
            "degraded": True,
            "constants": constants,
            "verified_construction_error": f"{type(exc).__name__}: {exc}",
        }
    if not allow_unverified_scales:
        return None, provenance
    sidecar = root / RESPONSE_SCALE_SIDECAR_PATH
    payload = _read_json(sidecar)
    policy = AccuracyPolicy.response_normalized_matched_pair(
        response_scales=payload["drives"],
        **constants,
    )
    provenance.update(
        {
            "source": "tracked_response_scale_sidecar_unverified",
            "sidecar_path": RESPONSE_SCALE_SIDECAR_PATH,
            "sidecar_sha256": _sha256(sidecar),
        }
    )
    return policy, provenance


def check_matched_accuracy_policy_binding(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """The live policy constants must equal the comparison protocol's values."""

    constants = dict(provenance.get("constants") or {})
    expected = {
        "response_relative_target": 0.05,
        "response_relative_guard": 0.15,
        "matched_accuracy_factor": 2.0,
        "infidelity_ceiling": 1.0e-4,
    }
    drifted = {
        name: {"live": constants.get(name), "protocol": value}
        for name, value in expected.items()
        if _float_or_none(constants.get(name)) != value
    }
    details = {
        "live_constants": constants,
        "protocol_source": "agent_guidance/time-dynamics/paper-ii-comparison-protocol.md",
        "live_source": (
            "pipelines/time_dynamics/campaigns/paper_ii_staged_accuracy_v1.py"
        ),
        "drifted": drifted,
    }
    if drifted:
        return _check(
            "matched_accuracy_policy_binding",
            STATUS_ERROR,
            f"{len(drifted)} live policy constant(s) diverge from the protocol",
            details,
        )
    return _check(
        "matched_accuracy_policy_binding",
        STATUS_OK,
        "live policy constants match the protocol (5% target, 15% guard, "
        "factor 2, infidelity ceiling 1e-4)",
        details,
    )


def check_response_normalized_matched_accuracy(
    artifacts: Sequence[Artifact],
    *,
    policy: Any,
    provenance: Mapping[str, Any],
    root: Path,
) -> dict[str, Any]:
    """Apply the live matched-accuracy policy to every benchmark row and pair."""

    details: dict[str, Any] = {"policy_provenance": dict(provenance)}
    rows: list[tuple[Artifact, Mapping[str, Any]]] = []
    for artifact in artifacts:
        for row in artifact.benchmark_rows:
            rows.append((artifact, row))
    if policy is None:
        details["remediation"] = (
            "regenerate the staged-lane exact-reference store "
            "(pipelines/time_dynamics/diagnostics/exact_reference_store.py) or "
            "rerun with --allow-unverified-response-scales to evaluate against "
            "the tracked sidecar's recorded scales"
        )
        return _check(
            "response_normalized_matched_accuracy",
            STATUS_NOT_CHECKED,
            "the live matched-accuracy policy could not be constructed",
            details,
        )
    if not rows:
        return _check(
            "response_normalized_matched_accuracy",
            STATUS_NOT_CHECKED,
            "no artifact carried benchmark rows with accuracy metrics",
            details,
        )

    _ensure_importable(root)
    from pipelines.time_dynamics.staged_campaign import (  # noqa: PLC0415
        AccuracyMetrics,
        CellKey,
    )

    evaluated: list[dict[str, Any]] = []
    unevaluable: list[dict[str, Any]] = []
    pairs: dict[tuple[str, ...], list[tuple[Any, Any, str]]] = {}
    override_sources: set[str] = set()
    for artifact, row in rows:
        horizon = str(row.get("horizon") or "")
        regime = str(row.get("regime") or row.get("seed") or "")
        drive = str(row.get("drive") or "")
        controller = str(row.get("controller") or "")
        method = _method_of(row)
        metric_values = [
            _float_or_none(row.get("mean_abs_energy_error")),
            _float_or_none(row.get("mean_abs_doublon_error")),
            _float_or_none(row.get("mean_max_abs_site_occupation_error")),
        ]
        infidelity = _float_or_none(row.get("mean_infidelity"))
        if not all((horizon, regime, drive, controller)) or any(
            value is None for value in metric_values
        ):
            unevaluable.append(
                {
                    "artifact": artifact.relpath,
                    "row": {k: row.get(k) for k in ("drive", "horizon", "controller", "method")},
                    "reason": "incomplete cell identity or accuracy metrics",
                }
            )
            continue
        try:
            key = CellKey(regime=regime, drive=drive, method=method, controller=controller)
            metrics = AccuracyMetrics(*metric_values, infidelity)
            components = policy.response_components(metrics, key=key, horizon=horizon)
            record = {
                "artifact": artifact.relpath,
                "drive": drive,
                "horizon": horizon,
                "controller": controller,
                "method": method,
                "qualifies_individually": bool(
                    policy.qualifies(metrics, key=key, horizon=horizon)
                ),
                "comparison_eligible": bool(
                    policy.comparison_eligible(metrics, key=key, horizon=horizon)
                ),
                "accuracy_burden": float(
                    policy.accuracy_burden(metrics, key=key, horizon=horizon)
                ),
                "response_components": {
                    name: float(value) for name, value in components.items()
                },
                "mean_infidelity": infidelity,
            }
        except Exception as exc:  # noqa: BLE001
            unevaluable.append(
                {
                    "artifact": artifact.relpath,
                    "row": {"drive": drive, "horizon": horizon, "controller": controller},
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        evaluated.append(record)
        if artifact.declares_override:
            override_sources.add(artifact.relpath)
        pairs.setdefault((regime, drive, controller, horizon), []).append(
            (key, metrics, horizon)
        )

    pair_results: list[dict[str, Any]] = []
    for cell, members in sorted(pairs.items()):
        methods = {key.method for key, _, _ in members}
        if len(members) != 2 or len(methods) != 2:
            continue
        (left_key, left_metrics, horizon), (right_key, right_metrics, _) = members
        pair_results.append(
            {
                "regime": cell[0],
                "drive": cell[1],
                "controller": cell[2],
                "horizon": cell[3],
                "methods": sorted(methods),
                "pair_matches": bool(
                    policy.pair_matches(
                        left_metrics,
                        left_key=left_key,
                        right_metrics=right_metrics,
                        right_key=right_key,
                        horizon=horizon,
                    )
                ),
            }
        )

    failing_rows = [row for row in evaluated if not row["qualifies_individually"]]
    failing_pairs = [pair for pair in pair_results if not pair["pair_matches"]]
    details.update(
        {
            "evaluated_row_count": len(evaluated),
            "rows": evaluated,
            "unevaluable_rows": unevaluable,
            "complete_pairs": pair_results,
            "rows_failing_individual_target": len(failing_rows),
            "pairs_failing_factor_of_two_test": len(failing_pairs),
            "artifacts_declaring_author_override": sorted(override_sources),
        }
    )

    degraded = bool(provenance.get("degraded"))
    if failing_rows or failing_pairs:
        summary = (
            f"{len(failing_rows)} of {len(evaluated)} row(s) miss the individual "
            f"target and {len(failing_pairs)} of {len(pair_results)} complete "
            "pair(s) miss the factor-of-two burden test"
        )
        if override_sources:
            return _check(
                "response_normalized_matched_accuracy",
                STATUS_WARN,
                summary + "; the artifact declares an explicit author override, "
                "so promotion is a user decision, not an automated pass",
                details,
            )
        return _check(
            "response_normalized_matched_accuracy",
            STATUS_ERROR,
            summary,
            details,
        )
    if degraded:
        return _check(
            "response_normalized_matched_accuracy",
            STATUS_WARN,
            f"{len(evaluated)} row(s) and {len(pair_results)} pair(s) pass, but "
            "the response scales were not cross-verified against the exact-"
            "reference store",
            details,
        )
    return _check(
        "response_normalized_matched_accuracy",
        STATUS_OK,
        f"{len(evaluated)} row(s) meet the individual target and "
        f"{len(pair_results)} complete pair(s) pass the factor-of-two test",
        details,
    )


# ------------------------------------------------------------------- receipt


def run_calibration(
    *,
    artifact_paths: Sequence[str] = (),
    run_roots: Sequence[str] = (),
    lock_path: str | None = None,
    allow_unverified_scales: bool = False,
    root: Path | None = None,
) -> dict[str, Any]:
    """Run every calibration check and return the machine-readable receipt."""

    repo_root = Path(root) if root is not None else _REPOSITORY_ROOT
    resolved = _resolve_inputs(
        artifact_paths=artifact_paths, run_roots=run_roots, root=repo_root
    )
    artifacts, unreadable = _load_artifacts(resolved, repo_root)

    policy, provenance = _load_live_policy(
        repo_root, allow_unverified_scales=allow_unverified_scales
    )

    checks = [
        check_promoted_class_settings_lock(repo_root, lock_path),
        check_benchmark_point_sameness(artifacts),
        check_seed_provenance(artifacts),
        check_trajectory_freeze(artifacts),
        check_qpu_faithful_separation(artifacts),
        check_matched_accuracy_policy_binding(provenance),
        check_response_normalized_matched_accuracy(
            artifacts, policy=policy, provenance=provenance, root=repo_root
        ),
    ]
    if unreadable:
        checks.append(
            _check(
                "artifact_readability",
                STATUS_ERROR,
                f"{len(unreadable)} calibration input(s) are unreadable",
                {"unreadable": unreadable},
            )
        )

    counts = {
        status: sum(1 for check in checks if check["status"] == status)
        for status in (STATUS_OK, STATUS_WARN, STATUS_ERROR, STATUS_NOT_CHECKED)
    }
    # ERROR is the fail-closed gate paper-ii-run SKILL.md:45 names. INCOMPLETE is
    # reported separately so a run whose evidence is simply absent can never read
    # as a clean pass; paper-ii-results SKILL.md:33 blocks on it independently.
    if counts[STATUS_ERROR]:
        overall = "ERROR"
    elif counts[STATUS_WARN]:
        overall = "WARN"
    elif counts[STATUS_NOT_CHECKED]:
        overall = "INCOMPLETE"
    else:
        overall = "OK"

    return {
        "schema": RECEIPT_SCHEMA,
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "tool": "pipelines/time_dynamics/diagnostics/paper_ii_benchmark_calibration.py",
        "repository_root": str(repo_root),
        "status": overall,
        "fail_closed": overall == "ERROR",
        "counts": counts,
        "inputs": {
            "artifact_count": len(artifacts),
            "artifacts": [
                {"path": artifact.relpath, "sha256": artifact.sha256}
                for artifact in artifacts
            ],
            "unreadable": unreadable,
            "run_roots": list(run_roots),
        },
        "checks": checks,
        "not_checked": [
            check["check_id"] for check in checks if check["status"] == STATUS_NOT_CHECKED
        ],
        "routed_to_documents": [
            {
                "topic": "matched-accuracy admission and pair reasoning",
                "document": "agent_guidance/time-dynamics/paper-ii-comparison-protocol.md",
            },
            {
                "topic": "promoting a newer class-settings lock",
                "document": "agent_guidance/skills/paper-ii-run/SKILL.md",
            },
            {
                "topic": "table/source-map transfer once calibration is clean",
                "document": "agent_guidance/skills/paper-ii-results/SKILL.md",
            },
        ],
    }


def _render_text(receipt: Mapping[str, Any]) -> str:
    lines = [
        f"paper-II benchmark calibration: {receipt['status']}",
        f"  artifacts: {receipt['inputs']['artifact_count']}",
    ]
    for check in receipt["checks"]:
        lines.append(f"  [{check['status'].upper():>11}] {check['check_id']}: {check['summary']}")
    if receipt["not_checked"]:
        lines.append(
            "  not run: " + ", ".join(receipt["not_checked"])
            + "  (paper-ii-results SKILL.md:33 blocks on unrun required checks)"
        )
    return "\n".join(lines)


# ------------------------------------------------------------------ self-test


def _self_test(root: Path) -> int:
    """Exercise every check against the restored lock and synthetic fixtures."""

    import tempfile

    failures: list[str] = []

    def expect(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    lock_check = check_promoted_class_settings_lock(root, None)
    expect(
        lock_check["status"] == STATUS_OK,
        f"restored lock should pass enforcement, got {lock_check['status']}: "
        f"{lock_check['summary']}",
    )
    expect(
        lock_check["details"]["settings_entry_count"] > 0,
        "restored lock should carry settings entries",
    )

    missing_lock = check_promoted_class_settings_lock(root, "does/not/exist.json")
    expect(
        missing_lock["status"] == STATUS_ERROR,
        "an absent lock must fail closed",
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)

        frozen = tmp_root / "frozen_run.json"
        frozen.write_text(
            json.dumps(
                {
                    "runtime_contract": {key: False for key in EXACT_DECISION_FLAG_KEYS},
                    "trajectory": [
                        {
                            "checkpoint_index": 0,
                            "baseline_geometry": {
                                "theta_dot_l2": 0.0,
                                "rho_num": 0.0,
                                "rho_miss": 1.0,
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        leaky = tmp_root / "leaky_run.json"
        leaky.write_text(
            json.dumps(
                {
                    "runtime_contract": {
                        "uses_reference_for_decision": True,
                        "uses_future_exact_forecast_for_decision": False,
                        "exact_reference_controller_inputs": False,
                    },
                    "trajectory": {
                        "points": [
                            {
                                "index": 0,
                                "theta_dot_l2": 0.25,
                                "rho_num": 0.5,
                                "rho_miss": 0.01,
                            }
                        ]
                    },
                }
            ),
            encoding="utf-8",
        )
        bad_seed = tmp_root / "bad_seed_receipt.json"
        bad_seed.write_text(
            json.dumps(
                {
                    "rows": [
                        {
                            "method": "avqds",
                            "drive": "weakslow",
                            "horizon": "t20",
                            "controller": "delta_theta_5e-5",
                            "seed": "hh_snake_nph1",
                            "seed_path": "output/seeds/fallback_seed.json",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        mismatch = tmp_root / "mismatched_point.json"
        mismatch.write_text(
            json.dumps(
                {
                    "rows": [
                        {
                            "method": "avqds",
                            "drive": "weakslow",
                            "horizon": "t20",
                            "controller": "delta_theta_5e-5",
                            "seed": "hh_snake_nph1",
                            "reference_key": "a" * 64,
                            "n_ph_max": 1,
                        },
                        {
                            "method": "apm",
                            "route": "exchange",
                            "drive": "weakslow",
                            "horizon": "t20",
                            "controller": "delta_theta_5e-5",
                            "seed": "hh_snake_nph1",
                            "reference_key": "b" * 64,
                            "n_ph_max": 2,
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )

        artifacts, unreadable = _load_artifacts(
            [frozen, leaky, bad_seed, mismatch], tmp_root
        )
        expect(not unreadable, "self-test fixtures should all parse")

        freeze = check_trajectory_freeze(artifacts)
        expect(
            freeze["status"] == STATUS_ERROR and len(freeze["details"]["invalid_rows"]) == 1,
            f"freeze detection should flag exactly one row, got {freeze['summary']}",
        )

        qpu = check_qpu_faithful_separation(artifacts)
        expect(
            qpu["status"] == STATUS_ERROR and len(qpu["details"]["leaks"]) == 1,
            f"QPU separation should flag one leak, got {qpu['summary']}",
        )

        seeds = check_seed_provenance(artifacts)
        expect(
            seeds["status"] == STATUS_ERROR
            and seeds["details"]["violations"][0]["markers"] == ["fallback"],
            f"seed provenance should flag the fallback seed, got {seeds['summary']}",
        )

        sameness = check_benchmark_point_sameness(artifacts)
        mismatched_axes = {row["axis"] for row in sameness["details"]["mismatches"]}
        expect(
            sameness["status"] == STATUS_ERROR
            and {"phonon_cutoff", "diagnostic_exact_reference"} <= mismatched_axes,
            f"sameness audit should flag cutoff and reference drift, got {sameness['summary']}",
        )

        clean_seed = tmp_root / "clean_receipt.json"
        clean_seed.write_text(
            json.dumps(
                {
                    "rows": [
                        {
                            "method": "avqds",
                            "drive": "weakslow",
                            "horizon": "t20",
                            "controller": "delta_theta_5e-5",
                            "seed": "hh_snake_nph1",
                            "seed_path": "chtc/paper_ii_staged_accuracy_euler_v1_t20_resume_repair_d3301581_20260826/input/seed.json",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        clean_artifacts, _ = _load_artifacts([clean_seed], tmp_root)
        clean = check_seed_provenance(clean_artifacts)
        expect(
            clean["status"] == STATUS_OK,
            "a seed under the canonical staged-campaign directory must not be "
            f"reported as a staged seed artifact, got {clean['summary']}",
        )

    _, provenance = _load_live_policy(root, allow_unverified_scales=False)
    binding = check_matched_accuracy_policy_binding(provenance)
    expect(
        binding["status"] == STATUS_OK,
        f"live policy constants should match the protocol, got {binding['summary']}",
    )

    if failures:
        print("self-test FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("self-test OK (lock enforcement, freeze, QPU separation, seed provenance,")
    print("  sameness audit, policy binding)")
    return 0


# ----------------------------------------------------------------------- main


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="paper_ii_benchmark_calibration",
        description=(
            "Pre-execution Paper-II benchmark calibration audit. Run before "
            "CHTC submission, aggregation, evidence-PDF generation, or a table "
            "update; fail closed on ERROR."
        ),
    )
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="PATH",
        help="JSON artifact to audit (frontier/promotion receipt or run payload). Repeatable.",
    )
    parser.add_argument(
        "--run-root",
        action="append",
        default=[],
        metavar="DIR",
        help="Directory searched recursively for run.json payloads. Repeatable.",
    )
    parser.add_argument(
        "--lock",
        default=None,
        help=f"Override the promoted class-settings lock path (default {PROMOTED_LOCK_PATH}).",
    )
    parser.add_argument(
        "--allow-unverified-response-scales",
        action="store_true",
        help=(
            "Evaluate matched accuracy against the tracked response-scale "
            "sidecar when the exact-reference store is absent. Never reports OK."
        ),
    )
    parser.add_argument(
        "--receipt-json",
        default=None,
        metavar="PATH",
        help="Write the calibration receipt to this path.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full receipt as JSON instead of the text summary.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Exercise every check against the restored lock and synthetic fixtures.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.self_test:
        return _self_test(_REPOSITORY_ROOT)

    try:
        receipt = run_calibration(
            artifact_paths=args.artifact,
            run_roots=args.run_root,
            lock_path=args.lock,
            allow_unverified_scales=args.allow_unverified_response_scales,
        )
    except CalibrationInputError as exc:
        parser.error(str(exc))
        return 2

    if args.receipt_json:
        destination = Path(args.receipt_json)
        if not destination.is_absolute():
            destination = _REPOSITORY_ROOT / destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        receipt["receipt_path"] = _repo_relative(destination, _REPOSITORY_ROOT)

    if args.json:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    else:
        print(_render_text(receipt))
        if args.receipt_json:
            print(f"  receipt: {receipt['receipt_path']}")

    return 1 if receipt["status"] == "ERROR" else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
