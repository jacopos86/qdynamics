#!/usr/bin/env python3
"""Build the weak-weak FM outer-reuse error-versus-cost support report.

This builder is intentionally evidence-explicit.  Its command-line defaults
lock the audited 2026-07-17 baseline/FM bundle; callers may instead provide the
baseline source and accounting audit, corrected FM accounting sidecar, matched
Qiskit sidecars, and settings-drift audit explicitly.  The generated report is
a contextual route-level comparison, not a one-variable outer-reuse ablation.

The module is import-light.  Matplotlib, artifact loading, and LaTeX execution
occur only inside :func:`build_report` or :func:`main`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "paper_i_hh_weak_weak_fm_outer_reuse_error_cost_report_v1"
STEM = "paper_i_hh_weak_weak_fm_outer_reuse_error_cost"
COMPILE_CONVENTION = "table_i_basis_gate_transpile_v1"
QISKIT_OPTIMIZATION_LEVEL = 0
QISKIT_TRANSPILE_SEED = 7
QUERY_CATEGORIES = ("N_E", "N_grad", "N_G", "N_Hv", "N_Q", "N_cross")
DEFAULT_BASELINE_LABEL = "SR-SNAKE no ordinary novelty (2026-07-17 evidence)"
DEFAULT_BASELINE_REFERENCE_PDF = REPO_ROOT / "Paper_I_no_ordinary_novelty_sr_snake_20260717.pdf"
DEFAULT_BASELINE_SOURCE = REPO_ROOT / (
    "raw_outputs/paper_i_hh_sr_snake_weak_weak_undamped_no_prune_no_beam_"
    "no_ordinary_novelty_fallback_on_20260715/json/result.json"
)
DEFAULT_BASELINE_EVIDENCE_AUDIT = REPO_ROOT / (
    "MATH/paper_facing/paper_I_static_scaffold/"
    "paper_i_no_ordinary_novelty_sr_snake_evidence_copy_20260717.json"
)
DEFAULT_BASELINE_ACCOUNTING_AUDIT = DEFAULT_BASELINE_SOURCE.with_name(
    "formal_query_accounting_reclosed.json"
)
DEFAULT_BASELINE_QISKIT = REPO_ROOT / (
    "raw_outputs/paper_i_hh_no_ordinary_novelty_sr_snake_plateau_qiskit_20260717/"
    "weak_weak/k30_terminal/qiskit_cost_sidecar.json"
)
DEFAULT_FM_ACCOUNTING = REPO_ROOT / (
    "raw_outputs/paper_i_hh_fm_sr_v3_outer_information_active_weak_weak_depth30_"
    "20260716/full_depth30/json/formal_manifold_query_accounting_corrected.json"
)
DEFAULT_FM_QISKIT = DEFAULT_FM_ACCOUNTING.with_name("qiskit_cost_basis_gate_opt0_seed7.json")
DEFAULT_OUTPUT_DIR = REPO_ROOT / (
    "output/pdf/paper_i_hh_weak_weak_fm_outer_reuse_error_cost_20260717"
)
DEFAULT_SETTINGS_DRIFT_AUDIT = DEFAULT_OUTPUT_DIR / "settings_drift_audit.json"


@dataclass(frozen=True)
class ErrorPoint:
    round: int
    error: float


@dataclass(frozen=True)
class CostPoint:
    round: int
    error: float
    winning_s_alg: int


@dataclass(frozen=True)
class QiskitCost:
    history_position: int
    n2q: int
    d2q: int
    dcirc: int
    compile_convention: str
    optimization_level: int
    transpile_seed: int
    primary_error_at_prefix: float | None


@dataclass(frozen=True)
class MethodEvidence:
    label: str
    error_points: tuple[ErrorPoint, ...]
    cost_points: tuple[CostPoint, ...]
    qiskit: QiskitCost

    def reported_point(self) -> CostPoint:
        matches = [
            point
            for point in self.cost_points
            if point.round == self.qiskit.history_position
        ]
        if len(matches) != 1:
            raise ValueError(
                f"{self.label}: Qiskit prefix k={self.qiskit.history_position} "
                "does not identify exactly one closed-accounting trajectory point"
            )
        point = matches[0]
        if self.qiskit.primary_error_at_prefix is not None and not math.isclose(
            point.error,
            self.qiskit.primary_error_at_prefix,
            rel_tol=1e-10,
            abs_tol=1e-14,
        ):
            raise ValueError(
                f"{self.label}: Qiskit prefix error {self.qiskit.primary_error_at_prefix} "
                f"does not match accounting/source error {point.error}"
            )
        return point


def _read_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": _rel(resolved),
        "sha256": _sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result


def _nested(payload: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _first_nested(payload: Mapping[str, Any], paths: Iterable[Sequence[str]]) -> Any:
    for path in paths:
        value = _nested(payload, path)
        if value is not None:
            return value
    return None


ROUND_PATHS = (
    ("round",),
    ("depth",),
    ("history_position",),
    ("k",),
    ("prefix",),
)
ERROR_PATHS = (
    ("absolute_same_cutoff_error",),
    ("same_cutoff_abs_delta_e",),
    ("abs_delta_e",),
    ("delta_abs_current",),
    ("error",),
    ("primary_error",),
)
S_ALG_PATHS = (
    ("winning_s_alg",),
    ("closed_winning_s_alg",),
    ("winning_lineage_s_alg",),
    ("s_alg_winning",),
    ("S_alg",),
    ("s_alg",),
    ("S",),
    ("winning_lineage", "S_alg"),
    ("winning_lineage", "s_alg"),
    ("query_closure", "winning_lineage", "S_alg"),
    ("formal_query_closure", "winning_lineage", "S_alg"),
    ("formal_manifold_query_closure", "winning_lineage", "S_alg"),
)


def _looks_like_trajectory(value: Any, *, require_s_alg: bool) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        return False
    rows = [row for row in value if isinstance(row, Mapping)]
    if not rows:
        return False
    has_round = any(_integer(_first_nested(row, ROUND_PATHS)) is not None for row in rows)
    has_error = any(_finite_float(_first_nested(row, ERROR_PATHS)) is not None for row in rows)
    has_s_alg = any(_integer(_first_nested(row, S_ALG_PATHS)) is not None for row in rows)
    return has_round and (has_s_alg if require_s_alg else has_error)


def _find_trajectory_rows(
    payload: Mapping[str, Any],
    *,
    require_s_alg: bool,
    recursive: bool = True,
) -> list[Mapping[str, Any]]:
    preferred_paths = (
        ("trajectory",),
        ("prefix_trajectory",),
        ("winning_lineage_trajectory",),
        ("accounting_trajectory",),
        ("query_accounting", "trajectory"),
        ("query_accounting", "prefix_trajectory"),
        ("query_closure", "trajectory"),
        ("query_closure", "prefix_trajectory"),
        ("formal_query_closure", "trajectory"),
        ("formal_query_closure", "prefix_trajectory"),
        ("formal_manifold_query_closure", "trajectory"),
        ("formal_manifold_query_closure", "prefix_trajectory"),
        ("corrected_accounting", "trajectory"),
        ("corrected_accounting", "prefix_trajectory"),
        ("report", "trajectory"),
    )
    for path in preferred_paths:
        value = _nested(payload, path)
        if _looks_like_trajectory(value, require_s_alg=require_s_alg):
            return [row for row in value if isinstance(row, Mapping)]

    if not recursive:
        need = "closed winning S_alg" if require_s_alg else "same-cutoff error"
        raise ValueError(f"could not find a direct trajectory carrying {need}")

    queue: list[Any] = [payload]
    visited = 0
    while queue and visited < 400:
        value = queue.pop(0)
        visited += 1
        if _looks_like_trajectory(value, require_s_alg=require_s_alg):
            return [row for row in value if isinstance(row, Mapping)]
        if isinstance(value, Mapping):
            queue.extend(value.values())
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            queue.extend(value)
    need = "closed winning S_alg" if require_s_alg else "same-cutoff error"
    raise ValueError(f"could not find a trajectory carrying {need}")


def _adapt_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    adapt = payload.get("adapt_vqe")
    if isinstance(adapt, Mapping):
        return adapt
    result = payload.get("result")
    if isinstance(result, Mapping):
        nested = result.get("adapt_vqe")
        if isinstance(nested, Mapping):
            return nested
    return payload


def _extract_source_error_points(payload: Mapping[str, Any]) -> tuple[ErrorPoint, ...]:
    adapt = _adapt_payload(payload)
    history = adapt.get("history")
    points: list[ErrorPoint] = []
    exact = _finite_float(
        _first_nested(
            adapt,
            (("exact_gs_energy",), ("ground_state", "exact_gs_energy")),
        )
    )
    if isinstance(history, Sequence) and not isinstance(history, (str, bytes)):
        history_rows = [row for row in history if isinstance(row, Mapping)]
        if history_rows and exact is not None:
            initial_energy = _finite_float(history_rows[0].get("energy_before_opt"))
            if initial_energy is not None:
                points.append(ErrorPoint(round=0, error=abs(initial_energy - exact)))
        for index, row in enumerate(history_rows, start=1):
            round_id = _integer(_first_nested(row, ROUND_PATHS)) or index
            error = _finite_float(_first_nested(row, ERROR_PATHS))
            if error is None and exact is not None:
                energy = _finite_float(
                    _first_nested(row, (("energy_after_opt",), ("energy",)))
                )
                if energy is not None:
                    error = abs(energy - exact)
            if error is not None:
                points.append(ErrorPoint(round=round_id, error=error))
    if not points:
        rows = _find_trajectory_rows(adapt, require_s_alg=False)
        for row in rows:
            round_id = _integer(_first_nested(row, ROUND_PATHS))
            error = _finite_float(_first_nested(row, ERROR_PATHS))
            if round_id is not None and error is not None:
                points.append(ErrorPoint(round=round_id, error=error))
    terminal_error = _finite_float(
        _first_nested(
            adapt,
            (
                ("abs_delta_e",),
                ("absolute_same_cutoff_error",),
                ("same_cutoff_abs_delta_e",),
            ),
        )
    )
    terminal_round = max((point.round for point in points), default=None)
    if terminal_round is None:
        terminal_round = _integer(
            _first_nested(
                adapt,
                (("history_count",), ("depth",), ("ansatz_depth",)),
            )
        )
    if terminal_round is None and points:
        terminal_round = max(point.round for point in points)
    if terminal_error is not None and terminal_round is not None:
        points.append(ErrorPoint(round=terminal_round, error=terminal_error))
    if not points:
        raise ValueError("baseline source does not expose a same-cutoff error trajectory")
    dedup = {point.round: point for point in points}
    ordered = tuple(dedup[key] for key in sorted(dedup))
    if any(point.error <= 0.0 for point in ordered):
        raise ValueError("same-cutoff errors must be strictly positive for log plotting")
    return ordered


def _closure_complete(payload: Mapping[str, Any]) -> bool:
    boolean_paths = (
        ("formal_manifold_query_accounting_complete",),
        ("query_accounting_complete",),
        ("accounting_complete",),
        ("closure_complete",),
        ("closure", "complete"),
        ("query_closure", "complete"),
        ("formal_query_closure", "complete"),
        ("formal_manifold_query_closure", "complete"),
        ("validation", "query_accounting_complete"),
        ("validation", "formal_manifold_query_accounting_complete"),
    )
    values = [_nested(payload, path) for path in boolean_paths]
    explicit = [value for value in values if isinstance(value, bool)]
    if explicit:
        return all(explicit)

    status_paths = (
        ("closure", "status"),
        ("query_closure", "status"),
        ("formal_query_closure", "status"),
        ("formal_manifold_query_closure", "status"),
        ("query_accounting", "status"),
    )
    for path in status_paths:
        status = _nested(payload, path)
        if isinstance(status, str):
            return status.lower() in {"ok", "passed", "complete", "closed"}
    schema = str(payload.get("schema") or "").lower()
    validation_passed = _nested(payload, ("validation", "passed"))
    if "query_accounting_correction" in schema and validation_passed is True:
        return True
    return False


def _cost_points_from_accounting(
    payload: Mapping[str, Any],
    *,
    source_errors: Sequence[ErrorPoint] | None,
    role: str,
    terminal_s_alg: int | None = None,
) -> tuple[CostPoint, ...]:
    if not _closure_complete(payload):
        raise ValueError(f"{role}: query accounting does not declare complete closure")
    try:
        rows = _find_trajectory_rows(
            payload,
            require_s_alg=True,
            recursive=terminal_s_alg is None,
        )
    except ValueError:
        rows = []
    error_by_round = {point.round: point.error for point in source_errors or ()}
    points: list[CostPoint] = []
    for row in rows:
        round_id = _integer(_first_nested(row, ROUND_PATHS))
        s_alg = _integer(_first_nested(row, S_ALG_PATHS))
        row_error = _finite_float(_first_nested(row, ERROR_PATHS))
        if round_id is None or s_alg is None:
            continue
        source_error = error_by_round.get(round_id)
        if row_error is not None and source_error is not None and not math.isclose(
            row_error,
            source_error,
            rel_tol=1e-8,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"{role}: accounting/source error mismatch at k={round_id}: "
                f"{row_error} != {source_error}"
            )
        error = row_error if row_error is not None else source_error
        if error is None:
            continue
        if error <= 0.0 or s_alg <= 0:
            raise ValueError(f"{role}: non-positive error or winning S_alg at k={round_id}")
        points.append(CostPoint(round=round_id, error=error, winning_s_alg=s_alg))
    if not points and terminal_s_alg is not None and source_errors:
        terminal = max(source_errors, key=lambda point: point.round)
        points.append(
            CostPoint(
                round=terminal.round,
                error=terminal.error,
                winning_s_alg=terminal_s_alg,
            )
        )
    if not points:
        raise ValueError(f"{role}: no closed winning-lineage endpoint or trajectory points")
    dedup = {point.round: point for point in points}
    ordered = tuple(dedup[key] for key in sorted(dedup))
    if any(
        right.winning_s_alg < left.winning_s_alg
        for left, right in zip(ordered, ordered[1:])
    ):
        raise ValueError(f"{role}: winning S_alg must be cumulative and nondecreasing")
    if source_errors and ordered[-1].round != max(error_by_round):
        raise ValueError(
            f"{role}: accounting trajectory does not close the terminal source round"
        )
    return ordered


def _error_points_from_cost_points(points: Sequence[CostPoint]) -> tuple[ErrorPoint, ...]:
    return tuple(ErrorPoint(round=point.round, error=point.error) for point in points)


def _query_closure_summary(payload: Mapping[str, Any], *, role: str) -> dict[str, Any]:
    """Extract and reconcile the small paper-facing closure summary.

    Primitive-ID sets remain in the authoritative correction sidecar.  The
    report records only the six category totals, winning/discarded S_alg, and
    the independently corrected optimizer/guard nfev partition.
    """

    closure = payload.get("query_closure")
    if not isinstance(closure, Mapping):
        raise ValueError(f"{role}: missing query_closure")

    def branch(key: str) -> dict[str, Any]:
        value = closure.get(key)
        if not isinstance(value, Mapping):
            raise ValueError(f"{role}: missing query_closure.{key}")
        raw_counts = value.get("counts")
        counts_source = raw_counts if isinstance(raw_counts, Mapping) else value
        counts: dict[str, int] = {}
        for category in QUERY_CATEGORIES:
            count = _integer(counts_source.get(category))
            if count is None or count < 0:
                raise ValueError(f"{role}: invalid {key}.{category}")
            counts[category] = count
        s_alg = _integer(value.get("S_alg"))
        if s_alg is None or s_alg < 0:
            raise ValueError(f"{role}: invalid {key}.S_alg")
        if sum(counts.values()) != s_alg:
            raise ValueError(
                f"{role}: {key} category sum {sum(counts.values())} != S_alg {s_alg}"
            )
        return {"counts": counts, "S_alg": s_alg}

    winning = branch("winning_branch")
    discarded = branch("discarded_branch_operational_overhead")
    nfev = {
        "stored_total": _integer(closure.get("stored_nfev_total")),
        "corrected_total": _integer(closure.get("corrected_nfev_total")),
        "correction": _integer(closure.get("nfev_correction")),
        "winning_lineage": _integer(closure.get("nfev_winning_lineage")),
        "discarded_operational_overhead": _integer(
            closure.get("nfev_discarded_operational_overhead")
        ),
    }
    if any(value is None or value < 0 for value in nfev.values()):
        raise ValueError(f"{role}: incomplete or negative nfev correction summary")
    stored = int(nfev["stored_total"])
    corrected = int(nfev["corrected_total"])
    correction = int(nfev["correction"])
    winning_nfev = int(nfev["winning_lineage"])
    discarded_nfev = int(nfev["discarded_operational_overhead"])
    if stored + correction != corrected:
        raise ValueError(f"{role}: stored nfev plus correction does not equal corrected nfev")
    if winning_nfev + discarded_nfev != corrected:
        raise ValueError(f"{role}: winning plus discarded nfev does not close corrected nfev")

    receipt = payload.get("correction_receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError(f"{role}: missing nfev correction receipt")
    receipt_tuple = (
        _integer(receipt.get("stored_optimizer_and_guard_nfev")),
        _integer(receipt.get("corrected_optimizer_and_guard_nfev")),
        _integer(receipt.get("nfev_correction")),
    )
    if receipt_tuple != (stored, corrected, correction):
        raise ValueError(f"{role}: nfev correction receipt disagrees with query closure")
    if receipt.get("unique_query_oracle_work_changed") is not False:
        raise ValueError(f"{role}: nfev correction must not change unique query-oracle work")

    return {
        "winning_branch": winning,
        "discarded_branch_operational_overhead": discarded,
        "nfev": nfev,
        "nfev_correction_reason": str(receipt.get("correction_reason") or ""),
        "unique_query_oracle_work_changed_by_nfev_correction": False,
    }


def _resolve_recorded_path(value: Any) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _verify_recorded_artifact(record: Mapping[str, Any], *, role: str) -> Path:
    path_value = record.get("path")
    expected_hash = str(record.get("sha256") or "")
    if not path_value or len(expected_hash) != 64:
        raise ValueError(f"{role}: missing recorded path or SHA-256")
    path = _resolve_recorded_path(path_value)
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_hash = _sha256(path)
    if actual_hash != expected_hash:
        raise ValueError(f"{role}: SHA-256 mismatch for {path}")
    return path


def _baseline_terminal_cost_from_evidence_copy(
    payload: Mapping[str, Any],
    *,
    source_errors: Sequence[ErrorPoint],
    baseline_source_path: Path,
    baseline_qiskit_path: Path,
) -> tuple[CostPoint, Path]:
    if payload.get("schema") != "paper_i_no_ordinary_novelty_sr_snake_evidence_copy_v1":
        raise ValueError("baseline accounting audit is not the 2026-07-17 SR evidence copy")
    rows = payload.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("baseline evidence copy has no rows")
    matches = [
        row
        for row in rows
        if isinstance(row, Mapping) and str(row.get("regime")) in {"weak_weak", "weak-weak"}
    ]
    if len(matches) != 1:
        raise ValueError("baseline evidence copy must contain exactly one weak-weak row")
    row = matches[0]
    source_record = row.get("source")
    qiskit_record = row.get("qiskit_sidecar")
    s_source = row.get("s_alg_source")
    if not all(isinstance(record, Mapping) for record in (source_record, qiskit_record, s_source)):
        raise ValueError("baseline weak-weak row lacks source, Qiskit, or S_alg provenance")
    assert isinstance(source_record, Mapping)
    assert isinstance(qiskit_record, Mapping)
    assert isinstance(s_source, Mapping)
    recorded_source = _verify_recorded_artifact(source_record, role="baseline source")
    recorded_qiskit = _verify_recorded_artifact(qiskit_record, role="baseline Qiskit")
    ledger = _verify_recorded_artifact(s_source, role="baseline estimator ledger")
    if recorded_source != baseline_source_path.resolve():
        raise ValueError("supplied baseline source is not the evidence-copy source")
    if recorded_qiskit != baseline_qiskit_path.resolve():
        raise ValueError("supplied baseline Qiskit sidecar is not the evidence-copy sidecar")
    if s_source.get("policy") != "canonical_same_state_unique_primitive_v1":
        raise ValueError("baseline S_alg is not canonical same-state unique-primitive work")
    round_id = _integer(row.get("history_position") or row.get("k_eval"))
    s_alg = _integer(row.get("s_alg"))
    error = _finite_float(row.get("absolute_same_cutoff_error"))
    if None in {round_id, s_alg} or error is None:
        raise ValueError("baseline weak-weak terminal tuple is incomplete")
    assert round_id is not None and s_alg is not None
    source_terminal = max(source_errors, key=lambda point: point.round)
    if source_terminal.round != round_id or not math.isclose(
        source_terminal.error,
        error,
        rel_tol=1e-8,
        abs_tol=1e-14,
    ):
        raise ValueError("baseline evidence-copy endpoint does not match the source trajectory")
    return CostPoint(round=round_id, error=error, winning_s_alg=s_alg), ledger


def _fm_source_from_correction(payload: Mapping[str, Any]) -> Path:
    source_record = _nested(payload, ("source", "result"))
    if not isinstance(source_record, Mapping):
        raise ValueError("FM corrected accounting lacks its source-result record")
    return _verify_recorded_artifact(source_record, role="FM source result")


def _extract_qiskit_cost(payload: Mapping[str, Any], *, role: str) -> QiskitCost:
    candidate = payload
    required = {"compiled_count_2q_total", "compiled_depth_2q_total", "compiled_depth_total"}
    if not required <= set(candidate):
        queue: list[Any] = [payload]
        while queue:
            value = queue.pop(0)
            if isinstance(value, Mapping):
                if required <= set(value):
                    candidate = value
                    break
                queue.extend(value.values())
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                queue.extend(value)
    convention = str(candidate.get("compile_convention") or "")
    opt = _integer(candidate.get("qiskit_transpile_optimization_level"))
    seed = _integer(candidate.get("qiskit_transpile_seed"))
    position = _integer(
        candidate.get("history_position")
        if candidate.get("history_position") is not None
        else candidate.get("k_pl")
    )
    if position is None:
        position = _integer(
            _first_nested(
                payload,
                (
                    ("source", "outer_iteration"),
                    ("prefix", "history_position"),
                    ("prefix", "k_eval"),
                    ("history_position",),
                    ("k_pl",),
                ),
            )
        )
    n2q = _integer(candidate.get("compiled_count_2q_total"))
    d2q = _integer(candidate.get("compiled_depth_2q_total"))
    dcirc = _integer(candidate.get("compiled_depth_total"))
    primary_error = _finite_float(
        candidate.get("primary_error_at_prefix")
        if candidate.get("primary_error_at_prefix") is not None
        else payload.get("primary_error_at_prefix")
    )
    validated = candidate.get("compiled_resource_qiskit_validated")
    status = candidate.get("compiled_circuit_stats_status")
    if convention != COMPILE_CONVENTION:
        raise ValueError(f"{role}: expected compile convention {COMPILE_CONVENTION!r}")
    if opt != QISKIT_OPTIMIZATION_LEVEL or seed != QISKIT_TRANSPILE_SEED:
        raise ValueError(f"{role}: Qiskit sidecar must use opt0 and transpile seed 7")
    if validated is not True or status not in {None, "ok"}:
        raise ValueError(f"{role}: Qiskit compiled-resource validation is not complete")
    if None in {position, n2q, d2q, dcirc}:
        raise ValueError(f"{role}: incomplete compiled Qiskit cost tuple")
    assert position is not None and n2q is not None and d2q is not None and dcirc is not None
    if min(position, n2q, d2q, dcirc) < 0:
        raise ValueError(f"{role}: negative Qiskit prefix or resource value")
    return QiskitCost(
        history_position=position,
        n2q=n2q,
        d2q=d2q,
        dcirc=dcirc,
        compile_convention=convention,
        optimization_level=opt,
        transpile_seed=seed,
        primary_error_at_prefix=primary_error,
    )


def _extract_drift_rows(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    for key in ("settings_differences", "changed_settings", "differences", "drift_rows"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            rows = []
            for field, record in value.items():
                if isinstance(record, Mapping):
                    rows.append(
                        {
                            "field": str(field),
                            "baseline": str(record.get("baseline", record.get("source", "--"))),
                            "fm": str(record.get("fm", record.get("target", "--"))),
                            "classification": str(record.get("classification", "changed")),
                        }
                    )
                else:
                    rows.append(
                        {
                            "field": str(field),
                            "baseline": "--",
                            "fm": str(record),
                            "classification": "changed",
                        }
                    )
            return rows
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            rows = []
            for record in value:
                if not isinstance(record, Mapping):
                    continue
                rows.append(
                    {
                        "field": str(record.get("field", record.get("setting", "unnamed"))),
                        "baseline": str(record.get("baseline", record.get("source", "--"))),
                        "fm": str(record.get("fm", record.get("target", "--"))),
                        "classification": str(record.get("classification", "changed")),
                    }
                )
            return rows
    return []


def _audit_missing_causal_control(payload: Mapping[str, Any]) -> bool:
    value = _first_nested(
        payload,
        (
            ("causal_reuse_off_control_present",),
            ("reuse_off_control_present",),
            ("comparison", "causal_reuse_off_control_present"),
        ),
    )
    if value is True:
        raise ValueError(
            "this dedicated report is for the missing-reuse-off-control comparison; "
            "the drift audit claims such a control is present"
        )
    return True


def _plot_trajectory(
    methods: Sequence[MethodEvidence],
    *,
    x_axis: str,
    output_stem: Path,
) -> dict[str, Path]:
    os.environ.setdefault("MPLCONFIGDIR", str(output_stem.parent / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    palette = ("#E45756", "#4C78A8")
    markers = ("*", "o")
    fig = Figure(figsize=(4.15, 2.48), dpi=190)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    legend_handles = []
    for method, color, marker in zip(methods, palette, markers):
        if x_axis == "round":
            points = method.error_points
            x = [point.round for point in points]
            y = [point.error for point in points]
        elif x_axis == "winning_s_alg":
            points = method.cost_points
            x = [point.winning_s_alg for point in points]
            y = [point.error for point in points]
        else:
            raise ValueError(x_axis)
        ax.plot(x, y, color=color, linewidth=1.7, solid_capstyle="round")
        reported = method.reported_point()
        marker_x = (
            reported.round if x_axis == "round" else reported.winning_s_alg
        )
        ax.scatter(
            [marker_x],
            [reported.error],
            color=color,
            marker=marker,
            s=52,
            edgecolors="white",
            linewidths=0.5,
            zorder=4,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=1.7,
                marker=marker,
                markersize=6,
                markerfacecolor=color,
                markeredgecolor="white",
                label=method.label,
            )
        )
    ax.set_yscale("log")
    ax.set_xlabel(
        "ADAPT round"
        if x_axis == "round"
        else r"Terminal closed winning $S_{\rm alg}$"
    )
    ax.set_ylabel(r"Same-cutoff $|\Delta E|$")
    if x_axis == "round":
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
        ax.set_xlim(left=0)
    else:
        ax.set_xlim(left=0)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.grid(True, which="major", color="#D7D7D7", linewidth=0.55, alpha=0.9)
    ax.grid(True, which="minor", axis="y", color="#ECECEC", linewidth=0.35, alpha=0.8)
    ax.legend(
        handles=legend_handles,
        fontsize=6.4,
        loc="best",
        frameon=False,
        title=(
            "marker: compiled prefix"
            if x_axis == "round"
            else "terminal closed points only"
        ),
        title_fontsize=6.1,
    )
    ax.tick_params(labelsize=7)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)
    fig.tight_layout(pad=0.55)
    png = output_stem.with_suffix(".png")
    pdf = output_stem.with_suffix(".pdf")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    return {"png": png, "pdf": pdf}


def _tex(value: Any) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_\allowbreak{}",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def _path_tex(value: Any) -> str:
    text = str(value)
    if any(char in text for char in ("{", "}", "%", "#")):
        raise ValueError(f"path contains a TeX-unsafe delimiter: {text!r}")
    return r"\path{" + text + "}"


def _fmt_error(value: float) -> str:
    return f"{value:.4e}"


def _fmt_int(value: int) -> str:
    return f"{value:,}"


def _render_tex(report: Mapping[str, Any]) -> str:
    baseline = report["methods"]["baseline"]
    fm = report["methods"]["fm"]
    rows = report["comparison_rows"]
    drift_rows = report["settings_drift"]["rows"]
    accounting = report["query_accounting"]
    fm_nfev = accounting["fm"]["nfev"]
    compact_classes = {
        "scientific settings drift": "settings drift",
        "intended FM mechanism": "FM mechanism",
        "FM route setting": "FM setting",
    }
    drift_tex = "\n".join(
        "{} & {} & {} & {} \\\\".format(
            _tex(row["field"]),
            _tex(row["baseline"]),
            _tex(row["fm"]),
            _tex(compact_classes.get(row["classification"], row["classification"])),
        )
        for row in drift_rows[:18]
    )
    if not drift_tex:
        drift_tex = r"No serialized field-level rows & -- & -- & audit retained \\"
    comparison_tex = "\n".join(
        "{} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
            _fmt_error(float(row["abs_delta_e"])),
            _tex(row["display_method"]),
            row["k"],
            _fmt_int(int(row["winning_s_alg"])),
            _fmt_int(int(row["discarded_s_alg"])),
            row["N2q"],
            row["D2q"],
            row["Dcirc"],
        )
        for row in rows
    )
    return rf"""% MACHINE_READABLE_REPORT_JSON: {STEM}.json
% MACHINE_READABLE_MANIFEST_JSON: report_manifest.json
% MACHINE_READABLE_MANIFEST_CSV: report_manifest.csv
% COMPARISON_CLASS: contextual_route_level_comparison_v1
% CAUSAL_REUSE_OFF_CONTROL_PRESENT: false
\documentclass[10pt,twocolumn]{{article}}
\usepackage[letterpaper,margin=0.55in,columnsep=0.24in]{{geometry}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern,booktabs,graphicx,tabularx,array,xcolor,microtype,hyperref}}
\hypersetup{{colorlinks=true,linkcolor=blue,urlcolor=blue}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{0.25em}}
\setlength{{\tabcolsep}}{{3.2pt}}
\renewcommand{{\arraystretch}}{{1.04}}
\newcolumntype{{Y}}{{>{{\raggedright\arraybackslash}}X}}
\urlstyle{{tt}}
\pagestyle{{plain}}
\begin{{document}}
\twocolumn[{{%
\begin{{center}}
{{\Large\bfseries Weak--weak FM outer-information reuse: error versus cost}}\\[-1pt]
{{\small Contextual comparison against {_tex(report['baseline_label'])}}}
\end{{center}}
\vspace{{-4pt}}

\begin{{center}}
\fcolorbox{{black}}{{gray!4}}{{\begin{{minipage}}{{0.965\textwidth}}
\textbf{{Compact parameter and provenance manifest.}}\par\smallskip
\footnotesize
\begin{{tabularx}}{{\linewidth}}{{@{{}}p{{0.18\linewidth}}X@{{}}}}
\textbf{{Regime / error}} & weak--weak; absolute same-cutoff ED error \\
\textbf{{Evidence class}} & contextual route-level comparison; no causal reuse claim \\
\textbf{{Baseline / FM}} & {_tex(report['baseline_label'])} / {_tex(fm['route_label'])} \\
\textbf{{Query / Qiskit}} & terminal closed winning-lineage $S_{{\rm alg}}$ / basis-gate opt0, seed 7 \\
\textbf{{Source sidecars}} & {{\fontsize{{5.8}}{{6.5}}\selectfont {_path_tex(Path(report['sources']['baseline_accounting']['path']).name)}; {_path_tex(Path(report['sources']['fm_accounting']['path']).name)}}} \\
\textbf{{Causal control}} & \textbf{{missing: no matched FM outer-reuse-off control is supplied}} \\
\end{{tabularx}}
\end{{minipage}}}}
\end{{center}}
\vspace{{2pt}}

\begin{{minipage}}[t]{{0.485\textwidth}}
\centering
\includegraphics[width=\linewidth]{{error_vs_round.png}}
\end{{minipage}}\hfill
\begin{{minipage}}[t]{{0.485\textwidth}}
\centering
\includegraphics[width=\linewidth]{{error_vs_closed_winning_s_alg.png}}
\end{{minipage}}

\vspace{{1pt}}
\begin{{center}}
\small
\begin{{tabular}}{{@{{}}r p{{0.26\textwidth}} r r r r r r@{{}}}}
\toprule
$|\Delta E|$ & Route / compiled point & $k$ & win $S_{{\rm alg}}$ & discard $S_{{\rm alg}}$ & $N_{{2q}}$ & $D_{{2q}}$ & $D_c$ \\
\midrule
{comparison_tex}
\bottomrule
\end{{tabular}}

\vspace{{1pt}}
{{\footnotesize\itshape Error is listed first. Winning $S_{{\rm alg}}$ is the scientific coordinate; discarded-branch work is operational overhead. Markers identify the exact opt0/seed-7 compiled prefixes.}}
\end{{center}}

\fcolorbox{{black}}{{yellow!10}}{{\parbox{{0.96\textwidth}}{{\footnotesize
\textbf{{Interpretation boundary.}} This is a route-level contextual comparison. The settings-drift audit records controller and optimization differences in addition to outer-information reuse. Without a matched FM reuse-off control, neither the error gap nor the query-work gap is a causal estimate of reuse alone.
}}}}
\par
\vspace{{3pt}}

\begin{{minipage}}{{0.96\textwidth}}
\scriptsize
\textbf{{Settings-drift audit (exact executed-command differences).}}\par\smallskip
\begin{{tabularx}}{{\linewidth}}{{@{{}}>{{\raggedright\arraybackslash}}p{{0.20\linewidth}}YYY@{{}}}}
\toprule
Field & Baseline & FM & Class \\
\midrule
{drift_tex}
\bottomrule
\end{{tabularx}}
FM optimizer/guard $n_{{\rm fev}}$: {_fmt_int(int(fm_nfev['stored_total']))}
$\rightarrow$ {_fmt_int(int(fm_nfev['corrected_total']))}
(+{_fmt_int(int(fm_nfev['correction']))}); the correction changes no unique
query-oracle work.\par
\textit{{Terminal query points only:}}
{_path_tex('prefix_query_trajectory_status=unavailable_no_round_boundary_ledger_checkpoints')}.
No prefix $S_{{\rm alg}}$ curve is inferred or interpolated. Source paths,
hashes, and complete drift rows are retained in
{_path_tex('report_manifest.json')}, {_path_tex('report_manifest.csv')}, and
{_path_tex('trajectory.csv')}.
\end{{minipage}}
\vspace{{3pt}}
}}]
\end{{document}}
"""


def _compile_pdf(tex_path: Path) -> tuple[Path, str]:
    latexmk = shutil.which("latexmk")
    tectonic = shutil.which("tectonic")
    attempts: list[tuple[str, list[str]]] = []
    if latexmk:
        attempts.append(
            (
                "latexmk",
                [latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            )
        )
    if tectonic:
        attempts.append(
            ("tectonic", [tectonic, "--keep-logs", "--reruns", "2", tex_path.name])
        )
    if not attempts:
        raise RuntimeError("neither latexmk nor tectonic is available")
    errors: list[str] = []
    for engine, command in attempts:
        try:
            subprocess.run(command, cwd=tex_path.parent, check=True)
        except subprocess.CalledProcessError as exc:
            errors.append(f"{engine}: exit {exc.returncode}")
            continue
        pdf = tex_path.with_suffix(".pdf")
        if pdf.is_file():
            return pdf, engine
        errors.append(f"{engine}: no PDF produced")
    raise RuntimeError("LaTeX compilation failed (" + "; ".join(errors) + ")")


def _write_trajectory_csv(path: Path, methods: Sequence[MethodEvidence]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("method", "round", "abs_delta_e", "closed_winning_s_alg"),
        )
        writer.writeheader()
        for method in methods:
            cost_by_round = {point.round: point.winning_s_alg for point in method.cost_points}
            for point in method.error_points:
                writer.writerow(
                    {
                        "method": method.label,
                        "round": point.round,
                        "abs_delta_e": f"{point.error:.17g}",
                        "closed_winning_s_alg": cost_by_round.get(point.round, ""),
                    }
                )


def _write_comparison_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = (
        "abs_delta_e",
        "method",
        "k",
        "winning_s_alg",
        "discarded_s_alg",
        "N2q",
        "D2q",
        "Dcirc",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def _write_manifest_csv(path: Path, manifest: Mapping[str, Any]) -> None:
    rows: list[tuple[str, str]] = []

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for key in sorted(value):
                visit(f"{prefix}.{key}" if prefix else str(key), value[key])
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for index, item in enumerate(value):
                visit(f"{prefix}[{index}]", item)
        else:
            rows.append((prefix, json.dumps(value, sort_keys=True)))

    visit("", manifest)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("key", "value"))
        writer.writerows(rows)


def build_report(
    *,
    baseline_label: str,
    baseline_reference_pdf: Path,
    baseline_source_json: Path,
    baseline_accounting_audit_json: Path,
    baseline_evidence_audit_json: Path,
    baseline_qiskit_sidecar_json: Path,
    fm_corrected_accounting_json: Path,
    fm_qiskit_sidecar_json: Path,
    settings_drift_audit_json: Path,
    output_dir: Path,
    compile_pdf: bool = True,
) -> dict[str, Path]:
    if not baseline_label.strip():
        raise ValueError("baseline_label must be explicit and non-empty")
    input_paths = {
        "baseline_reference_pdf": baseline_reference_pdf.resolve(),
        "baseline_source": baseline_source_json.resolve(),
        "baseline_accounting": baseline_accounting_audit_json.resolve(),
        "baseline_evidence_audit": baseline_evidence_audit_json.resolve(),
        "baseline_qiskit": baseline_qiskit_sidecar_json.resolve(),
        "fm_accounting": fm_corrected_accounting_json.resolve(),
        "fm_qiskit": fm_qiskit_sidecar_json.resolve(),
        "settings_drift_audit": settings_drift_audit_json.resolve(),
    }
    sources = {role: _artifact(path) for role, path in input_paths.items()}

    baseline_source = _read_json(input_paths["baseline_source"])
    baseline_accounting = _read_json(input_paths["baseline_accounting"])
    baseline_evidence_audit = _read_json(input_paths["baseline_evidence_audit"])
    baseline_qiskit_payload = _read_json(input_paths["baseline_qiskit"])
    fm_accounting = _read_json(input_paths["fm_accounting"])
    fm_qiskit_payload = _read_json(input_paths["fm_qiskit"])
    settings_drift = _read_json(input_paths["settings_drift_audit"])

    baseline_errors = _extract_source_error_points(baseline_source)
    evidence_terminal, baseline_ledger_path = _baseline_terminal_cost_from_evidence_copy(
        baseline_evidence_audit,
        source_errors=baseline_errors,
        baseline_source_path=input_paths["baseline_source"],
        baseline_qiskit_path=input_paths["baseline_qiskit"],
    )
    baseline_terminal_s_alg = _integer(
        _nested(baseline_accounting, ("query_closure", "winning_branch", "S_alg"))
    )
    if baseline_terminal_s_alg is None:
        raise ValueError("reclosed baseline accounting lacks terminal winning-branch S_alg")
    baseline_cost_points = _cost_points_from_accounting(
        baseline_accounting,
        source_errors=baseline_errors,
        role="reclosed baseline accounting",
        terminal_s_alg=baseline_terminal_s_alg,
    )
    baseline_closure_summary = _query_closure_summary(
        baseline_accounting,
        role="reclosed baseline accounting",
    )
    if len(baseline_cost_points) != 1 or baseline_cost_points[0] != evidence_terminal:
        raise ValueError("reclosed baseline terminal tuple disagrees with the evidence copy")
    baseline_reclosed_source = _nested(
        baseline_accounting, ("source", "estimator_ledger_sidecar")
    )
    if not isinstance(baseline_reclosed_source, Mapping):
        raise ValueError("reclosed baseline accounting lacks estimator-ledger provenance")
    if _verify_recorded_artifact(
        baseline_reclosed_source,
        role="reclosed baseline estimator ledger",
    ) != baseline_ledger_path:
        raise ValueError("reclosed baseline and evidence copy reference different ledgers")
    sources["baseline_estimator_ledger"] = _artifact(baseline_ledger_path)

    fm_source_path = _fm_source_from_correction(fm_accounting)
    sources["fm_source_result"] = _artifact(fm_source_path)
    fm_source = _read_json(fm_source_path)
    fm_errors = _extract_source_error_points(fm_source)
    fm_terminal_s_alg = _integer(
        _nested(fm_accounting, ("query_closure", "winning_branch", "S_alg"))
    )
    if fm_terminal_s_alg is None:
        raise ValueError("FM corrected accounting lacks terminal winning-branch S_alg")
    fm_cost_points = _cost_points_from_accounting(
        fm_accounting,
        source_errors=fm_errors,
        role="FM corrected accounting",
        terminal_s_alg=fm_terminal_s_alg,
    )
    fm_closure_summary = _query_closure_summary(
        fm_accounting,
        role="FM corrected accounting",
    )
    baseline = MethodEvidence(
        label=baseline_label.strip(),
        error_points=baseline_errors,
        cost_points=baseline_cost_points,
        qiskit=_extract_qiskit_cost(baseline_qiskit_payload, role="baseline"),
    )
    fm_route_label = str(
        _first_nested(
            _adapt_payload(fm_source),
            (
                ("adapt_reoptimization_route",),
                ("adapt_formal_manifold_route_profile",),
                ("route",),
            ),
        )
        or "FM-SNAKE outer-information reuse"
    )
    fm = MethodEvidence(
        label="FM-SNAKE outer reuse",
        error_points=fm_errors,
        cost_points=fm_cost_points,
        qiskit=_extract_qiskit_cost(fm_qiskit_payload, role="FM"),
    )
    baseline_reported = baseline.reported_point()
    fm_reported = fm.reported_point()
    _audit_missing_causal_control(settings_drift)
    drift_rows = _extract_drift_rows(settings_drift)

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    round_plots = _plot_trajectory(
        (baseline, fm),
        x_axis="round",
        output_stem=output_dir / "error_vs_round",
    )
    cost_plots = _plot_trajectory(
        (baseline, fm),
        x_axis="winning_s_alg",
        output_stem=output_dir / "error_vs_closed_winning_s_alg",
    )

    comparison_rows = [
        {
            "abs_delta_e": baseline_reported.error,
            "method": baseline.label,
            "display_method": "SR-SNAKE no ordinary novelty",
            "k": baseline_reported.round,
            "winning_s_alg": baseline_reported.winning_s_alg,
            "discarded_s_alg": baseline_closure_summary[
                "discarded_branch_operational_overhead"
            ]["S_alg"],
            "N2q": baseline.qiskit.n2q,
            "D2q": baseline.qiskit.d2q,
            "Dcirc": baseline.qiskit.dcirc,
        },
        {
            "abs_delta_e": fm_reported.error,
            "method": fm.label,
            "display_method": fm.label,
            "k": fm_reported.round,
            "winning_s_alg": fm_reported.winning_s_alg,
            "discarded_s_alg": fm_closure_summary[
                "discarded_branch_operational_overhead"
            ]["S_alg"],
            "N2q": fm.qiskit.n2q,
            "D2q": fm.qiskit.d2q,
            "Dcirc": fm.qiskit.dcirc,
        },
    ]
    created_utc = datetime.now(timezone.utc).isoformat()
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "created_utc": created_utc,
        "scope": {
            "paper": "Paper I",
            "hamiltonian": "Hubbard-Holstein",
            "regime": "weak-weak",
            "error_definition": "absolute same-cutoff energy error",
            "query_coordinate": "closed winning-lineage S_alg",
            "comparison_classification": "contextual_route_level_comparison_v1",
            "causal_reuse_off_control_present": False,
            "causal_claim_authorized": False,
            "prefix_query_trajectory_status": (
                "unavailable_no_round_boundary_ledger_checkpoints"
            ),
            "query_plot_mode": "terminal_closed_points_only_v1",
        },
        "baseline_label": baseline.label,
        "sources": sources,
        "methods": {
            "baseline": {
                "label": baseline.label,
                "reported_k": baseline_reported.round,
                "reported_abs_delta_e": baseline_reported.error,
                "reported_winning_s_alg": baseline_reported.winning_s_alg,
                "reported_discarded_s_alg": baseline_closure_summary[
                    "discarded_branch_operational_overhead"
                ]["S_alg"],
                "error_trajectory_point_count": len(baseline.error_points),
                "closed_query_point_count": len(baseline.cost_points),
            },
            "fm": {
                "label": fm.label,
                "route_label": fm_route_label,
                "reported_k": fm_reported.round,
                "reported_abs_delta_e": fm_reported.error,
                "reported_winning_s_alg": fm_reported.winning_s_alg,
                "reported_discarded_s_alg": fm_closure_summary[
                    "discarded_branch_operational_overhead"
                ]["S_alg"],
                "error_trajectory_point_count": len(fm.error_points),
                "closed_query_point_count": len(fm.cost_points),
            },
        },
        "qiskit_contract": {
            "compile_convention": COMPILE_CONVENTION,
            "optimization_level": QISKIT_OPTIMIZATION_LEVEL,
            "transpile_seed": QISKIT_TRANSPILE_SEED,
        },
        "query_accounting": {
            "category_order": list(QUERY_CATEGORIES),
            "scientific_x_coordinate": "winning_branch.S_alg",
            "discarded_branch_policy": (
                "reported_separately_as_operational_overhead_not_added_to_scientific_x"
            ),
            "baseline": baseline_closure_summary,
            "fm": fm_closure_summary,
        },
        "comparison_rows": comparison_rows,
        "settings_drift": {
            "source_schema": settings_drift.get("schema"),
            "rows": drift_rows,
            "changed_field_count": len(drift_rows),
            "interpretation": (
                "route-level contextual comparison; missing matched FM outer-reuse-off control"
            ),
        },
        "plots": {
            "error_vs_round": {key: _artifact(value) for key, value in round_plots.items()},
            "error_vs_closed_winning_s_alg": {
                key: _artifact(value) for key, value in cost_plots.items()
            },
        },
    }

    trajectory_csv = output_dir / "trajectory.csv"
    comparison_csv = output_dir / "comparison_table.csv"
    _write_trajectory_csv(trajectory_csv, (baseline, fm))
    _write_comparison_csv(comparison_csv, comparison_rows)
    report_json = output_dir / f"{STEM}.json"
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tex_path = output_dir / f"{STEM}.tex"
    tex_path.write_text(_render_tex(report), encoding="utf-8")

    generated: dict[str, dict[str, Any]] = {
        "report_json": _artifact(report_json),
        "trajectory_csv": _artifact(trajectory_csv),
        "comparison_csv": _artifact(comparison_csv),
        "tex": _artifact(tex_path),
        "error_vs_round_png": _artifact(round_plots["png"]),
        "error_vs_round_pdf": _artifact(round_plots["pdf"]),
        "error_vs_closed_winning_s_alg_png": _artifact(cost_plots["png"]),
        "error_vs_closed_winning_s_alg_pdf": _artifact(cost_plots["pdf"]),
    }
    pdf_path: Path | None = None
    latex_engine: str | None = None
    if compile_pdf:
        pdf_path, latex_engine = _compile_pdf(tex_path)
        generated["pdf"] = _artifact(pdf_path)

    manifest = {
        "schema": f"{SCHEMA}_artifact_manifest",
        "created_utc": created_utc,
        "comparison_classification": "contextual_route_level_comparison_v1",
        "causal_reuse_off_control_present": False,
        "prefix_query_trajectory_status": (
            "unavailable_no_round_boundary_ledger_checkpoints"
        ),
        "query_plot_mode": "terminal_closed_points_only_v1",
        "latex_engine": latex_engine,
        "consumed_artifacts": sources,
        "generated_artifacts": generated,
        "qiskit_contract": report["qiskit_contract"],
        "query_accounting": report["query_accounting"],
        "comparison_rows": comparison_rows,
    }
    manifest_json = output_dir / "report_manifest.json"
    manifest_json.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_csv = output_dir / "report_manifest.csv"
    _write_manifest_csv(manifest_csv, manifest)

    outputs = {
        "report_json": report_json,
        "manifest_json": manifest_json,
        "manifest_csv": manifest_csv,
        "trajectory_csv": trajectory_csv,
        "comparison_csv": comparison_csv,
        "tex": tex_path,
        "error_vs_round_png": round_plots["png"],
        "error_vs_round_pdf": round_plots["pdf"],
        "error_vs_closed_winning_s_alg_png": cost_plots["png"],
        "error_vs_closed_winning_s_alg_pdf": cost_plots["pdf"],
    }
    if pdf_path is not None:
        outputs["pdf"] = pdf_path
    return outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-label", default=DEFAULT_BASELINE_LABEL)
    parser.add_argument(
        "--baseline-reference-pdf", type=Path, default=DEFAULT_BASELINE_REFERENCE_PDF
    )
    parser.add_argument("--baseline-source-json", type=Path, default=DEFAULT_BASELINE_SOURCE)
    parser.add_argument(
        "--baseline-accounting-audit-json",
        type=Path,
        default=DEFAULT_BASELINE_ACCOUNTING_AUDIT,
    )
    parser.add_argument(
        "--baseline-evidence-audit-json",
        type=Path,
        default=DEFAULT_BASELINE_EVIDENCE_AUDIT,
    )
    parser.add_argument(
        "--baseline-qiskit-sidecar-json", type=Path, default=DEFAULT_BASELINE_QISKIT
    )
    parser.add_argument(
        "--fm-corrected-accounting-json", type=Path, default=DEFAULT_FM_ACCOUNTING
    )
    parser.add_argument("--fm-qiskit-sidecar-json", type=Path, default=DEFAULT_FM_QISKIT)
    parser.add_argument(
        "--settings-drift-audit-json", type=Path, default=DEFAULT_SETTINGS_DRIFT_AUDIT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-compile", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    outputs = build_report(
        baseline_label=args.baseline_label,
        baseline_reference_pdf=args.baseline_reference_pdf,
        baseline_source_json=args.baseline_source_json,
        baseline_accounting_audit_json=args.baseline_accounting_audit_json,
        baseline_evidence_audit_json=args.baseline_evidence_audit_json,
        baseline_qiskit_sidecar_json=args.baseline_qiskit_sidecar_json,
        fm_corrected_accounting_json=args.fm_corrected_accounting_json,
        fm_qiskit_sidecar_json=args.fm_qiskit_sidecar_json,
        settings_drift_audit_json=args.settings_drift_audit_json,
        output_dir=args.output_dir,
        compile_pdf=not args.no_compile,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
