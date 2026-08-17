#!/usr/bin/env python3
"""Build the provisional Paper-I local-versus-CHTC replacement report.

This is a reporting-only diagnostic.  It never launches science, mutates a
campaign, edits the manuscript, or grants paper-evidence adoption.  Every row
uses the fixed k=50 prefix and the same-cutoff energy error.  Local rows are
admitted only after a per-cell closure is readable; pending and interrupted
cells leave the recorded CHTC result intact.
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUTPUT_DIR = REPO_ROOT / (
    "output/pdf/paper_i_local_vs_chtc_replacement_decision_20260815"
)
REVISIONS_DIR = OUTPUT_DIR / "revisions"
CACHE_DIR = OUTPUT_DIR / "metric_cache"
STABLE_STEM = "paper_i_local_vs_chtc_replacement_decision_20260815"
STABLE_PDF = OUTPUT_DIR / f"{STABLE_STEM}.pdf"
LATEST_JSON = OUTPUT_DIR / "latest.json"
WATCH_STATUS = OUTPUT_DIR / "watch_status.json"
LOCK_PATH = OUTPUT_DIR / "report.lock"

GLOBAL_CHTC_ADAPTER = REPO_ROOT / (
    "output/pdf/paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving/"
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_"
    "global_singleton_gradient_phase0_page12_adapter.json"
)
POLICY_ADAPTER = REPO_ROOT / (
    "output/pdf/paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving/"
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_"
    "page12_singleton_insertion_comparator_snapshot_adapter.json"
)
POLICY_CLOSURE_DIR = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "page12_insertion_comparator_closure_receipts"
)
STRONG_POLICY_RUNTIME = REPO_ROOT / (
    "output/local_runs/paper_i_page12_strong_holstein_sector5_local_repair_"
    "20260814_v1"
)
PAPER_I_PLOT_BUILDER = REPO_ROOT / (
    "pipelines/reporting/clean_paper_i_ra_append_singleton_matched_plot.py"
)
ED_REFERENCE = REPO_ROOT / (
    "MATH/paper_facing/paper_I_static_scaffold/"
    "paper_i_hh_ed_cutoff_reference_six_regime_20260727.json"
)

MATCHED_RUNTIME = REPO_ROOT / (
    "output/local_runs/paper_i_page12_matched_singleton12_r50_20260815_v1"
)
WEAK_POLICY_RUNTIME = REPO_ROOT / (
    "output/local_runs/paper_i_page12_weak_holstein_ra6_priority_20260815_v1"
)
ORCHESTRATOR_STATUS = REPO_ROOT / (
    "output/local_runs/paper_i_weak12_priority_then_matched_unique6_"
    "20260815_v1/status.json"
)
MATCHED_STATUS = MATCHED_RUNTIME / "status/campaign.json"
POSITION_AWARE_PHASE0_RUNTIME = REPO_ROOT / (
    "output/local_runs/"
    "paper_i_position_aware_phase0_sw_always_k15_20260816_v1"
)
POSITION_AWARE_PHASE0_EXECUTION_ID = (
    "position_aware_phase0__strong_weak_u8__nph3__"
    "ra_always_commutation_reduced__k15"
)
REMOTE_RUNNER_STATE_DIR = Path.home() / ".remote-runner/state/runs"
REMOTE_JOB_ID = (
    "holstein-paper-i-weak12-priority-then-matched-unique6-20260815-v1"
)

WEAK_RUNNER = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "run_local_page12_weak_holstein_priority6_20260815.py"
)
ORCHESTRATOR_NAME = (
    "run_local_paper_i_weak12_priority_then_matched_unique6_20260815.py"
)
MATCHED_RUNNER_NAME = (
    "run_local_paper_i_page12_matched_singleton12_r50_20260815.py"
)

REPORT_SCHEMA = "paper_i_local_vs_chtc_replacement_decision_report_v4"
LATEST_SCHEMA = "paper_i_local_vs_chtc_replacement_latest_pointer_v3"
WATCH_SCHEMA = "paper_i_local_vs_chtc_replacement_watch_status_v1"
RA_SUMMARY_SCHEMA = "paper_i_run_summary_v1"
APPEND_SUMMARY_SCHEMA = "paper_i_append_run_summary_v1"
COMPILE_CONVENTION = "table_i_basis_gate_transpile_v1"
TARGET_ROUND = 50
MIN_COMPILE_AVAILABLE_MEMORY_BYTES = 4 * 1024**3
MAX_SUMMARY_BYTES = 2 * 1024**2
PLOT_ERROR_FLOOR = 1.0e-14

METHOD_STYLES = {
    "ra_plateau": {"color": "#E45756", "marker": "*", "label": "RA plateau"},
    "ra_append_only": {
        "color": "#9467BD",
        "marker": "s",
        "label": "RA append-only",
    },
    "ra_always_cr": {
        "color": "#E69F00",
        "marker": "D",
        "label": "RA always-open",
    },
    "append_conventional": {
        "color": "#4C78A8",
        "marker": "o",
        "label": "Append-ADAPT",
    },
}

REGIMES = (
    ("weak_weak", "Weak--weak", 3),
    ("intermediate_weak", "Intermediate--weak", 3),
    ("strong_weak_u8", "Strong--weak", 3),
    ("weak_strong", "Weak--strong", 7),
    ("intermediate_strong", "Intermediate--strong", 7),
    ("strong_strong_u8", "Strong--strong", 7),
)
WEAK_REGIMES = REGIMES[:3]

POLICY_PROCS = {
    ("weak_weak", "always_commutation_reduced"): 0,
    ("intermediate_weak", "always_commutation_reduced"): 1,
    ("strong_weak_u8", "always_commutation_reduced"): 2,
    ("weak_weak", "append_only"): 6,
    ("intermediate_weak", "append_only"): 7,
    ("strong_weak_u8", "append_only"): 8,
    ("weak_strong", "append_only"): 9,
}

STRONG_LOCAL_POLICY_EXECUTIONS = {
    ("weak_strong", "always_commutation_reduced"),
    ("intermediate_strong", "always_commutation_reduced"),
    ("intermediate_strong", "append_only"),
    ("strong_strong_u8", "always_commutation_reduced"),
    ("strong_strong_u8", "append_only"),
}

# Exact reader-facing marker coordinates from
# clean_paper_i_ra_append_singleton_matched_plot.py / the 2026-08-12 matched
# comparison: (RA common crossing, Append common crossing, RA shared-S_alg
# best, Append shared-S_alg best).  These are presentation coordinates over
# the authenticated trajectories, not new scientific calculations.
PAPER_I_MATCHED_MARKERS = {
    "weak_weak": (37, 37, 45, 27),
    "intermediate_weak": (34, 32, 42, 25),
    "strong_weak_u8": (11, 11, 11, 11),
    "weak_strong": (35, 50, 50, 34),
    "intermediate_strong": (39, 49, 49, 35),
    "strong_strong_u8": (45, 45, 45, 28),
}

METRIC_KEYS = ("delta_e", "N2q", "D2q", "Dc", "W1q", "S_alg")
COST_KEYS = ("N2q", "D2q", "Dc", "W1q", "S_alg")


class ReportError(ValueError):
    pass


@dataclass(frozen=True)
class CellSpec:
    key: str
    group: str
    method: str
    method_label: str
    regime: str
    regime_label: str
    nph: int
    local_execution_id: str
    local_kind: str
    policy: str | None = None


def _plateau_id(regime: str, nph: int) -> str:
    return (
        "global_singleton_gradient_phase0_phase23_qiskit_no_lanes__"
        f"{regime}__nph{nph}__ra_global_singleton_gradient_phase0_phase123_"
        "qiskit_phase23_plateau"
    )


def _policy_id(regime: str, nph: int, policy: str) -> str:
    return (
        "global_singleton_gradient_phase0_phase23_qiskit_no_lanes__"
        f"{regime}__nph{nph}__ra_global_singleton_gradient_phase0_phase123_"
        f"qiskit_phase23_{policy}"
    )


def _append_id(regime: str, nph: int) -> str:
    return f"matched_singleton12__{regime}__nph{nph}__append_conventional_unwhitened"


def cell_specs() -> tuple[CellSpec, ...]:
    rows: list[CellSpec] = []
    for regime, label, nph in REGIMES:
        rows.append(
            CellSpec(
                key=f"plateau::{regime}",
                group="RA plateau insertion",
                method="ra_plateau",
                method_label="RA plateau",
                regime=regime,
                regime_label=label,
                nph=nph,
                local_execution_id=_plateau_id(regime, nph),
                local_kind="matched_archive",
            )
        )
    for policy, method, label in (
        ("append_only", "ra_append_only", "RA append-only"),
        (
            "always_commutation_reduced",
            "ra_always_cr",
            "RA always-open",
        ),
    ):
        for regime, regime_label, nph in REGIMES:
            rows.append(
                CellSpec(
                    key=f"{method}::{regime}",
                    group="RA insertion policies",
                    method=method,
                    method_label=label,
                    regime=regime,
                    regime_label=regime_label,
                    nph=nph,
                    local_execution_id=_policy_id(regime, nph, policy),
                    local_kind=(
                        "weak_direct"
                        if nph == 3
                        else (
                            "paper_i_local_baseline"
                            if (regime, policy) in STRONG_LOCAL_POLICY_EXECUTIONS
                            else "no_distinct_rerun"
                        )
                    ),
                    policy=policy,
                )
            )
    for regime, label, nph in REGIMES:
        rows.append(
            CellSpec(
                key=f"append::{regime}",
                group="Conventional Append-ADAPT VQE",
                method="append_conventional",
                method_label="Append-ADAPT",
                regime=regime,
                regime_label=label,
                nph=nph,
                local_execution_id=_append_id(regime, nph),
                local_kind="matched_archive",
            )
        )
    if len(rows) != 24 or len({row.key for row in rows}) != 24:
        raise AssertionError("replacement-report cell matrix is not exactly 24")
    return tuple(rows)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _digested(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = copy.deepcopy(dict(value))
    if "sha256" in unsigned:
        raise ReportError("cannot digest a payload that already has sha256")
    return {**unsigned, "sha256": _canonical_sha256(unsigned)}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding(path: Path, *, canonical: bool = False) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ReportError(f"unsafe or missing file: {path}")
    row: dict[str, Any] = {
        "path": path.resolve().as_posix(),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if canonical:
        row["canonical_sha256"] = _load_digested(path, label=str(path))["sha256"]
    return row


def _relative_binding(
    path: Path, *, root: Path, canonical: bool = False
) -> dict[str, Any]:
    row = _binding(path, canonical=canonical)
    try:
        row["path"] = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ReportError(f"artifact is outside its revision root: {path}") from exc
    return row


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ReportError(f"{label} is absent or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ReportError(f"{label} is not a JSON object: {path}")
    return value


def _load_digested(path: Path, *, label: str) -> dict[str, Any]:
    value = _load_json(path, label=label)
    claimed = value.get("sha256")
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    if not isinstance(claimed, str) or claimed != _canonical_sha256(unsigned):
        raise ReportError(f"{label} self digest drifted: {path}")
    return value


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_bytes(
        path,
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
        + b"\n",
    )


def _verify_binding(
    root: Path, value: Mapping[str, Any], *, label: str, canonical: bool = False
) -> tuple[Path, dict[str, Any] | None]:
    raw = value.get("path")
    if not isinstance(raw, str) or not raw:
        raise ReportError(f"{label} path is absent")
    relative = Path(raw)
    path = relative if relative.is_absolute() else root / relative
    if ".." in relative.parts or not path.is_file() or path.is_symlink():
        raise ReportError(f"{label} path is unsafe or absent: {path}")
    if path.stat().st_size != value.get("size_bytes") or _sha256_file(path) != value.get(
        "sha256"
    ):
        raise ReportError(f"{label} file binding drifted: {path}")
    if not canonical:
        return path, None
    payload = _load_digested(path, label=label)
    if payload["sha256"] != value.get("canonical_sha256"):
        raise ReportError(f"{label} canonical binding drifted: {path}")
    return path, payload


def _finite(value: Any, *, label: str, nonnegative: bool = False) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ReportError(f"{label} is not finite")
    number = float(value)
    if nonnegative and number < 0.0:
        raise ReportError(f"{label} is negative")
    return number


def _integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReportError(f"{label} is not a nonnegative integer")
    return value


def _trajectory(
    rows: Any,
    *,
    label: str,
    round_key: str,
    error_key: str | None = None,
    energy_key: str | None = None,
    exact_energy: float | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise ReportError(f"{label} trajectory is absent")
    points: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ReportError(f"{label} trajectory row is malformed")
        controller_round = _integer(raw.get(round_key), label=f"{label} round")
        if controller_round > TARGET_ROUND:
            continue
        if error_key is not None:
            delta_e = _finite(
                raw.get(error_key), label=f"{label} error", nonnegative=True
            )
        else:
            if energy_key is None or exact_energy is None:
                raise AssertionError("energy-derived trajectory lacks its exact reference")
            energy = _finite(raw.get(energy_key), label=f"{label} energy")
            delta_e = abs(energy - exact_energy)
        points.append({"k": controller_round, "delta_e": delta_e})
    if not points:
        raise ReportError(f"{label} trajectory has no k<=50 points")
    rounds = [point["k"] for point in points]
    expected = list(range(rounds[0], TARGET_ROUND + 1))
    if rounds[0] not in {0, 1} or rounds != expected:
        raise ReportError(f"{label} trajectory is not contiguous through k=50")
    return points


def _trajectory_marker(
    trajectory: Sequence[Mapping[str, Any]],
    *,
    marker: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    by_round = {int(point["k"]): float(point["delta_e"]) for point in trajectory}
    if marker is None:
        marker_round = TARGET_ROUND
        policy = "terminal_plotted_prefix"
    else:
        marker_round = _integer(marker.get("k"), label="trajectory marker round")
        policy = str(marker.get("policy", ""))
        if policy != "first_effective_plateau_prefix":
            raise ReportError("trajectory marker policy drifted")
    if marker_round not in by_round:
        raise ReportError("trajectory marker is outside the plotted trajectory")
    marker_error = by_round[marker_round]
    if marker is not None:
        declared = _finite(
            marker.get("error"), label="trajectory marker error", nonnegative=True
        )
        tolerance = 128.0 * math.ulp(max(1.0, abs(declared), abs(marker_error)))
        if not math.isclose(declared, marker_error, rel_tol=0.0, abs_tol=tolerance):
            raise ReportError("trajectory marker error drifted")
    return {"k": marker_round, "delta_e": marker_error, "policy": policy}


def _metric_row(
    *,
    delta_e: Any,
    costs: Mapping[str, Any],
    exact: Any,
    energy: Any | None,
    origin: str,
    execution_id: str,
    qiskit_version: str | None,
    source: Mapping[str, Any],
    trajectory: Sequence[Mapping[str, Any]],
    marker: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "delta_e": _finite(delta_e, label="delta_e", nonnegative=True),
        **{key: _integer(costs.get(key), label=key) for key in COST_KEYS},
    }
    exact_number = _finite(exact, label="same-cutoff exact energy")
    energy_number = None if energy is None else _finite(energy, label="energy")
    if energy_number is not None:
        tolerance = 128.0 * math.ulp(max(1.0, abs(exact_number), abs(energy_number)))
        if not math.isclose(
            abs(energy_number - exact_number),
            row["delta_e"],
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise ReportError("same-cutoff delta_e does not close against energy")
    normalized_trajectory = [
        {
            "k": _integer(point.get("k"), label="normalized trajectory round"),
            "delta_e": _finite(
                point.get("delta_e"),
                label="normalized trajectory error",
                nonnegative=True,
            ),
        }
        for point in trajectory
    ]
    normalized_rounds = [point["k"] for point in normalized_trajectory]
    if (
        not normalized_trajectory
        or normalized_rounds[0] not in {0, 1}
        or normalized_rounds != list(range(normalized_rounds[0], TARGET_ROUND + 1))
    ):
        raise ReportError("normalized trajectory is not contiguous through k=50")
    terminal_error = normalized_trajectory[-1]["delta_e"]
    trajectory_tolerance = 128.0 * math.ulp(
        max(1.0, abs(terminal_error), abs(row["delta_e"]))
    )
    if not math.isclose(
        terminal_error, row["delta_e"], rel_tol=0.0, abs_tol=trajectory_tolerance
    ):
        raise ReportError("trajectory terminal does not close against k=50 delta_e")
    return {
        **row,
        "exact_same_cutoff_energy": exact_number,
        "energy": energy_number,
        "origin": origin,
        "execution_id": execution_id,
        "qiskit_version": qiskit_version,
        "compile_convention": COMPILE_CONVENTION,
        "source": copy.deepcopy(dict(source)),
        "trajectory": normalized_trajectory,
        "plot_marker": _trajectory_marker(normalized_trajectory, marker=marker),
    }


def _historical_global_rows() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    adapter = _load_digested(GLOBAL_CHTC_ADAPTER, label="Phase-0 Page-12 adapter")
    if (
        adapter.get("schema") != "paper_i_phase0_route_progress_adapter_v1"
        or adapter.get("status") != "completed_six_regime_evidence_ready"
        or adapter.get("paper_evidence_adopted") is not False
        or not isinstance(adapter.get("cells"), list)
        or len(adapter["cells"]) != 6
    ):
        raise ReportError("Phase-0 Page-12 adapter identity drifted")
    rows: dict[str, dict[str, Any]] = {}
    for cell in adapter["cells"]:
        if not isinstance(cell, Mapping):
            raise ReportError("Phase-0 adapter cell is malformed")
        regime = str(cell.get("regime_id"))
        nph = int(cell.get("nph", -1))
        expected = next((row for row in REGIMES if row[0] == regime), None)
        if expected is None or expected[2] != nph:
            raise ReportError(f"Phase-0 adapter regime drifted: {regime}")
        exact = cell.get("exact_same_cutoff_energy")
        phase0 = cell.get("phase0_route")
        append = cell.get("append_adapt")
        if not isinstance(phase0, Mapping) or not isinstance(append, Mapping):
            raise ReportError(f"Phase-0 adapter method rows are absent: {regime}")
        point = next(
            (
                row
                for row in phase0.get("points", [])
                if isinstance(row, Mapping) and row.get("k") == TARGET_ROUND
            ),
            None,
        )
        costs = phase0.get("costs")
        compile_row = phase0.get("compile")
        phase0_source = phase0.get("source")
        if (
            not isinstance(point, Mapping)
            or not isinstance(costs, Mapping)
            or not isinstance(compile_row, Mapping)
            or not isinstance(phase0_source, Mapping)
            or compile_row.get("compile_convention") != COMPILE_CONVENTION
        ):
            raise ReportError(f"Phase-0 k=50 terminal is incomplete: {regime}")
        completed_binding = phase0_source.get("completed_adapter")
        if not isinstance(completed_binding, Mapping):
            raise ReportError(f"Phase-0 completed adapter binding is absent: {regime}")
        completed_path, _ = _verify_binding(
            REPO_ROOT,
            completed_binding,
            label=f"Phase-0 completed adapter {regime}",
            canonical=False,
        )
        completed_adapter = _load_digested(
            completed_path, label=f"Phase-0 completed adapter {regime}"
        )
        completed_terminal = completed_adapter.get("terminal")
        if (
            completed_adapter.get("regime_id") != regime
            or completed_adapter.get("nph") != nph
            or not isinstance(completed_terminal, Mapping)
            or any(
                completed_terminal.get(key) != point.get(key)
                for key in ("k", "energy", "error")
            )
        ):
            raise ReportError(f"Phase-0 completed adapter row drifted: {regime}")
        phase0_exact = completed_adapter.get("exact_same_cutoff_energy")
        rows[f"plateau::{regime}"] = _metric_row(
            delta_e=point.get("error"),
            costs=costs,
            exact=phase0_exact,
            energy=point.get("energy"),
            origin="CHTC paper",
            execution_id=_plateau_id(regime, nph),
            qiskit_version=str(compile_row.get("qiskit_version", "unknown")),
            source={
                "adapter": _binding(GLOBAL_CHTC_ADAPTER, canonical=True),
                "cell_source": copy.deepcopy(phase0_source),
            },
            trajectory=_trajectory(
                list(phase0.get("points", [])),
                label=f"CHTC plateau {regime}",
                round_key="k",
                error_key="error",
            ),
        )
        terminal = append.get("terminal")
        if not isinstance(terminal, Mapping) or terminal.get("k") != TARGET_ROUND:
            raise ReportError(f"Append k=50 terminal is absent: {regime}")
        rows[f"append::{regime}"] = _metric_row(
            delta_e=terminal.get("error"),
            costs=terminal,
            exact=append.get("exact_same_cutoff_energy", exact),
            energy=None,
            origin="CHTC paper",
            execution_id=str(append.get("execution_id", "")),
            qiskit_version=str(
                append.get("source", {})
                .get("compile", {})
                .get("qiskit_version", "2.3.1")
            ),
            source={
                "adapter": _binding(GLOBAL_CHTC_ADAPTER, canonical=True),
                "cell_source": copy.deepcopy(append.get("source")),
            },
            trajectory=_trajectory(
                list(append.get("points", [])),
                label=f"CHTC Append {regime}",
                round_key="k",
                error_key="error",
            ),
        )
    return rows, _binding(GLOBAL_CHTC_ADAPTER, canonical=True)


def _available_memory_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def _safe_member_name(raw: Any, *, label: str) -> str:
    if not isinstance(raw, str) or not raw:
        raise ReportError(f"{label} member path is absent")
    normalized = raw
    while normalized.startswith("./"):
        normalized = normalized[2:]
    path = PurePosixPath(normalized)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ReportError(f"{label} member path is unsafe")
    return path.as_posix()


def _read_archive_member(
    archive_path: Path,
    *,
    member_name: str,
    expected_sha256: str,
    expected_size: int,
) -> bytes:
    if expected_size < 1 or expected_size > MAX_SUMMARY_BYTES:
        raise ReportError("summary member size is outside the reporting bound")
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            matches = [
                member
                for member in archive
                if _safe_member_name(member.name, label="archive") == member_name
            ]
            if len(matches) != 1 or not matches[0].isfile():
                raise ReportError(f"summary member is absent or non-unique: {member_name}")
            member = matches[0]
            if member.size != expected_size:
                raise ReportError("summary member declared size drifted")
            stream = archive.extractfile(member)
            if stream is None:
                raise ReportError("cannot open summary archive member")
            payload = stream.read(MAX_SUMMARY_BYTES + 1)
    except (OSError, tarfile.TarError) as exc:
        raise ReportError(f"cannot read archive: {archive_path}") from exc
    if len(payload) != expected_size or hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ReportError("summary member hash/size drifted")
    return payload


def _summary_json(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReportError(f"{label} is not JSON") from exc
    if not isinstance(value, dict) or value.get("schema") not in {
        RA_SUMMARY_SCHEMA,
        APPEND_SUMMARY_SCHEMA,
    }:
        raise ReportError(f"{label} schema drifted")
    return value


def _origin_from_source(source: Mapping[str, Any]) -> str:
    source_class = source.get("source_class")
    if source_class == "local":
        return "local rerun"
    if source_class == "paper_i_local_baseline":
        return "Paper-I local candidate"
    return "CHTC paper"


def _ra_summary_metrics(
    summary: Mapping[str, Any], *, execution_id: str, source: Mapping[str, Any]
) -> dict[str, Any]:
    requested = summary.get("requested_rounds")
    trace = summary.get("accepted_error_trace")
    provenance = summary.get("provenance")
    if (
        not isinstance(requested, list)
        or not isinstance(trace, list)
        or not isinstance(provenance, Mapping)
    ):
        raise ReportError(f"summary reporting sections are absent: {execution_id}")
    row = next(
        (
            item
            for item in requested
            if isinstance(item, Mapping)
            and item.get("controller_round") == TARGET_ROUND
            and item.get("status") == "available"
        ),
        None,
    )
    if row is None or len(trace) < TARGET_ROUND:
        raise ReportError(f"summary does not close k=50: {execution_id}")
    resources = row.get("resources")
    work = row.get("algorithmic_work")
    prefix = row.get("prefix")
    if (
        not isinstance(resources, Mapping)
        or not isinstance(work, Mapping)
        or not isinstance(prefix, Mapping)
        or resources.get("compile_convention") != COMPILE_CONVENTION
    ):
        raise ReportError(f"summary k=50 resources are incomplete: {execution_id}")
    available = _available_memory_bytes()
    if available is not None and available < MIN_COMPILE_AVAILABLE_MEMORY_BYTES:
        raise ReportError(
            "postprocessing_deferred_low_memory: "
            f"available={available}, required={MIN_COMPILE_AVAILABLE_MEMORY_BYTES}"
        )
    from pipelines.reporting.ingest_paper_i_phase0_completed_archive import (
        _prefix_compile_input,
    )
    from pipelines.reporting.paper_i_run_summary import (
        compile_paper_i_prefix_qiskit_payload,
    )

    compiled = compile_paper_i_prefix_qiskit_payload(_prefix_compile_input(prefix))
    costs = {
        "N2q": _integer(
            compiled.get("compiled_count_2q_total"), label="compiled N2q"
        ),
        "D2q": _integer(
            compiled.get("compiled_depth_2q_total"), label="compiled D2q"
        ),
        "Dc": _integer(compiled.get("compiled_depth_total"), label="compiled Dc"),
        "W1q": _integer(
            compiled.get("qiskit_pretranspile_pauli_1q_work_total"),
            label="compiled W1q",
        ),
        "S_alg": _integer(work.get("s_alg"), label="S_alg"),
    }
    serialized = {
        "N2q": resources.get("compiled_two_qubit_count"),
        "D2q": resources.get("compiled_two_qubit_depth"),
        "Dc": resources.get("compiled_total_depth"),
    }
    if (
        compiled.get("compile_convention") != COMPILE_CONVENTION
        or compiled.get("qiskit_basis_work_status") != "ok"
        or any(costs[key] != serialized[key] for key in serialized)
    ):
        raise ReportError(f"shared compiler cross-check failed: {execution_id}")
    terminal = trace[TARGET_ROUND - 1]
    if not isinstance(terminal, Mapping) or terminal.get("controller_round") != TARGET_ROUND:
        raise ReportError(f"summary accepted trace is not contiguous: {execution_id}")
    return _metric_row(
        delta_e=row.get("absolute_energy_error"),
        costs=costs,
        exact=provenance.get("exact_same_cutoff_energy"),
        energy=terminal.get("accepted_energy"),
        origin=_origin_from_source(source),
        execution_id=execution_id,
        qiskit_version=str(compiled.get("qiskit_version", "unknown")),
        source=source,
        trajectory=_trajectory(
            trace,
            label=f"local/CHTC RA {execution_id}",
            round_key="controller_round",
            error_key="absolute_energy_error",
        ),
        marker={
            "k": summary.get("effective_plateau", {}).get("controller_round"),
            "error": summary.get("effective_plateau", {}).get(
                "absolute_energy_error"
            ),
            "policy": "first_effective_plateau_prefix",
        }
        if isinstance(summary.get("effective_plateau"), Mapping)
        and summary.get("effective_plateau", {}).get("status") == "available"
        else None,
    )


def _append_summary_metrics(
    summary: Mapping[str, Any],
    *,
    execution_id: str,
    source: Mapping[str, Any],
    exact_same_cutoff_energy: float,
) -> dict[str, Any]:
    history = summary.get("accepted_history")
    resources = summary.get("resources")
    accounting = summary.get("estimator_accounting")
    if (
        summary.get("controller_rounds_completed") != TARGET_ROUND
        or not isinstance(history, list)
        or not isinstance(resources, Mapping)
        or not isinstance(accounting, Mapping)
    ):
        raise ReportError(f"Append summary does not close k=50: {execution_id}")
    observations = resources.get("compiled_resources_by_round")
    if not isinstance(observations, list):
        raise ReportError(f"Append compiled observations are absent: {execution_id}")
    requested = next(
        (
            row
            for row in observations
            if isinstance(row, Mapping)
            and row.get("controller_round") == TARGET_ROUND
            and row.get("observation_status") == "available"
        ),
        None,
    )
    if requested is None:
        raise ReportError(f"Append k=50 compiled observation is absent: {execution_id}")
    compiled = requested.get("compiled_resources")
    if (
        not isinstance(compiled, Mapping)
        or compiled.get("compile_convention") != COMPILE_CONVENTION
    ):
        raise ReportError(f"Append compile convention drifted: {execution_id}")
    costs = {
        "N2q": _integer(compiled.get("compiled_count_2q_total"), label="Append N2q"),
        "D2q": _integer(compiled.get("compiled_depth_2q_total"), label="Append D2q"),
        "Dc": _integer(compiled.get("compiled_depth_total"), label="Append Dc"),
        "W1q": _integer(
            compiled.get("qiskit_pretranspile_pauli_1q_work_total"),
            label="Append W1q",
        ),
        "S_alg": _integer(accounting.get("S_alg"), label="Append S_alg"),
    }
    trajectory = _trajectory(
        history,
        label=f"local Append {execution_id}",
        round_key="controller_round",
        energy_key="energy_after",
        exact_energy=exact_same_cutoff_energy,
    )
    terminal = history[-1]
    if not isinstance(terminal, Mapping):
        raise ReportError(f"Append terminal history row drifted: {execution_id}")
    return _metric_row(
        delta_e=trajectory[-1]["delta_e"],
        costs=costs,
        exact=exact_same_cutoff_energy,
        energy=terminal.get("energy_after"),
        origin=_origin_from_source(source),
        execution_id=execution_id,
        qiskit_version=str(compiled.get("qiskit_version", "unknown")),
        source=source,
        trajectory=trajectory,
    )


def _summary_metrics(
    summary: Mapping[str, Any],
    *,
    execution_id: str,
    source: Mapping[str, Any],
    exact_same_cutoff_energy: float | None = None,
) -> dict[str, Any]:
    schema = summary.get("schema")
    if schema == RA_SUMMARY_SCHEMA:
        return _ra_summary_metrics(summary, execution_id=execution_id, source=source)
    if schema == APPEND_SUMMARY_SCHEMA:
        if exact_same_cutoff_energy is None:
            raise ReportError(f"Append same-cutoff exact energy is absent: {execution_id}")
        return _append_summary_metrics(
            summary,
            execution_id=execution_id,
            source=source,
            exact_same_cutoff_energy=float(exact_same_cutoff_energy),
        )
    raise ReportError(f"unknown summary schema: {execution_id}")


def _cache_path(summary_sha256: str) -> Path:
    return CACHE_DIR / f"summary_{summary_sha256}.json"


def _load_metric_cache(
    *,
    summary_sha256: str,
    evidence_key: Mapping[str, Any],
    allow_legacy_without_trajectory: bool = False,
) -> dict[str, Any] | None:
    path = _cache_path(summary_sha256)
    if not path.exists() and not path.is_symlink():
        return None
    value = _load_digested(path, label="metric cache")
    if value.get("schema") == "paper_i_k50_metric_projection_cache_v1":
        if (
            value.get("summary_sha256") != summary_sha256
            or value.get("evidence_key") != evidence_key
            or not isinstance(value.get("metrics"), Mapping)
        ):
            raise ReportError(f"legacy metric cache binding drifted: {path}")
        return (
            copy.deepcopy(dict(value["metrics"]))
            if allow_legacy_without_trajectory
            else None
        )
    if (
        value.get("schema") != "paper_i_k50_metric_and_trajectory_cache_v2"
        or value.get("summary_sha256") != summary_sha256
        or value.get("evidence_key") != evidence_key
        or not isinstance(value.get("metrics"), Mapping)
        or not isinstance(value.get("metrics", {}).get("trajectory"), list)
    ):
        raise ReportError(f"metric cache binding drifted: {path}")
    return copy.deepcopy(dict(value["metrics"]))


def _write_metric_cache(
    *, summary_sha256: str, evidence_key: Mapping[str, Any], metrics: Mapping[str, Any]
) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = _digested(
        {
            "schema": "paper_i_k50_metric_and_trajectory_cache_v2",
            "status": "passed_authenticated_summary_compiled_k50_with_trajectory",
            "summary_sha256": summary_sha256,
            "evidence_key": copy.deepcopy(dict(evidence_key)),
            "metrics": copy.deepcopy(dict(metrics)),
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )
    _atomic_json(_cache_path(summary_sha256), payload)


def _historical_policy_rows() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    adapter = _load_digested(POLICY_ADAPTER, label="Page-18 policy adapter")
    if (
        adapter.get("schema")
        != "paper_i_ra_adapt_page12_insertion_comparator_progress_adapter_v1"
        or adapter.get("paper_evidence_adopted") is not False
    ):
        raise ReportError("Page-18 policy adapter identity drifted")
    completed_comparators = adapter.get("completed_comparators")
    if not isinstance(completed_comparators, Mapping):
        raise ReportError("Page-18 completed comparator matrix is absent")
    rows: dict[str, dict[str, Any]] = {}
    for (regime, policy), proc in POLICY_PROCS.items():
        nph = next(
            expected_nph
            for expected_regime, _label, expected_nph in REGIMES
            if expected_regime == regime
        )
        pattern = f"*proc{proc:02d}_*_{policy}_closure_receipt_20260813.json"
        matches = sorted(POLICY_CLOSURE_DIR.glob(pattern))
        if len(matches) != 1:
            raise ReportError(f"expected one policy closure for proc {proc}: {matches}")
        path = matches[0]
        receipt = _load_digested(path, label=f"policy closure proc {proc}")
        checks = receipt.get("authentication_checks")
        summary_binding = receipt.get("summary_json")
        archive_binding = receipt.get("archive")
        execution_id = _policy_id(regime, nph, policy)
        if (
            receipt.get("schema")
            != "paper_i_ra_adapt_page12_insertion_comparator_closure_receipt_v1"
            or receipt.get("status")
            != "passed_authenticated_page12_insertion_comparator_closure"
            or receipt.get("run_id") != execution_id
            or receipt.get("regime_id") != regime
            or receipt.get("nph") != nph
            or receipt.get("comparator_policy") != policy
            or receipt.get("controller_rounds_completed") != TARGET_ROUND
            or receipt.get("paper_evidence_adopted") is not False
            or not isinstance(checks, Mapping)
            or any(value is not True for value in checks.values())
            or not isinstance(summary_binding, Mapping)
            or not isinstance(archive_binding, Mapping)
        ):
            raise ReportError(f"policy closure identity drifted: {path}")
        summary_sha = str(summary_binding.get("sha256"))
        evidence_key = {
            "closure_canonical_sha256": receipt["sha256"],
            "archive_sha256": archive_binding.get("sha256"),
            "archive_size_bytes": archive_binding.get("size_bytes"),
            "summary_member": summary_binding.get("path_inside_archive"),
            "summary_size_bytes": summary_binding.get("size_bytes"),
            "source_class": "chtc",
        }
        regime_comparators = completed_comparators.get(regime)
        comparator = (
            regime_comparators.get(policy)
            if isinstance(regime_comparators, Mapping)
            else None
        )
        if (
            not isinstance(comparator, Mapping)
            or comparator.get("run_id") != execution_id
            or comparator.get("controller_rounds_completed") != TARGET_ROUND
            or comparator.get("comparator_policy") != policy
            or not isinstance(comparator.get("marker"), Mapping)
        ):
            raise ReportError(f"policy adapter trajectory drifted: {execution_id}")
        adapter_trajectory = _trajectory(
            list(comparator.get("points", [])),
            label=f"CHTC policy {execution_id}",
            round_key="k",
            error_key="error",
        )
        adapter_marker = _trajectory_marker(
            adapter_trajectory, marker=comparator.get("marker")
        )
        metrics = _load_metric_cache(
            summary_sha256=summary_sha,
            evidence_key=evidence_key,
            allow_legacy_without_trajectory=True,
        )
        if metrics is None:
            raw_archive = archive_binding.get("path")
            if not isinstance(raw_archive, str):
                raise ReportError("policy archive path is absent")
            archive_path = Path(raw_archive)
            if not archive_path.is_absolute():
                archive_path = REPO_ROOT / archive_path
            if (
                not archive_path.is_file()
                or archive_path.is_symlink()
                or archive_path.stat().st_size != archive_binding.get("size_bytes")
                or _sha256_file(archive_path) != archive_binding.get("sha256")
            ):
                raise ReportError(f"policy archive binding drifted: {archive_path}")
            member = _safe_member_name(
                summary_binding.get("path_inside_archive"), label="policy summary"
            )
            payload = _read_archive_member(
                archive_path,
                member_name=member,
                expected_sha256=summary_sha,
                expected_size=int(summary_binding.get("size_bytes", -1)),
            )
            metrics = _summary_metrics(
                _summary_json(payload, label="policy summary"),
                execution_id=execution_id,
                source={
                    "source_class": "chtc",
                    "closure_receipt": _binding(path, canonical=True),
                    "archive": copy.deepcopy(dict(archive_binding)),
                    "summary_member": copy.deepcopy(dict(summary_binding)),
                },
            )
            _write_metric_cache(
                summary_sha256=summary_sha,
                evidence_key=evidence_key,
                metrics=metrics,
            )
        terminal_tolerance = 128.0 * math.ulp(
            max(1.0, abs(adapter_trajectory[-1]["delta_e"]), abs(metrics["delta_e"]))
        )
        if not math.isclose(
            adapter_trajectory[-1]["delta_e"],
            float(metrics["delta_e"]),
            rel_tol=0.0,
            abs_tol=terminal_tolerance,
        ):
            raise ReportError(f"policy adapter terminal drifted: {execution_id}")
        metrics = {
            **metrics,
            "trajectory": adapter_trajectory,
            "plot_marker": adapter_marker,
        }
        _write_metric_cache(
            summary_sha256=summary_sha,
            evidence_key=evidence_key,
            metrics=metrics,
        )
        method = "ra_append_only" if policy == "append_only" else "ra_always_cr"
        rows[f"{method}::{regime}"] = metrics
    local_rows, local_binding = _paper_i_local_policy_rows()
    overlap = set(rows) & set(local_rows)
    if overlap:
        raise ReportError(f"Paper-I policy source overlap drifted: {sorted(overlap)}")
    rows.update(local_rows)
    return rows, {
        "chtc_policy_adapter": _binding(POLICY_ADAPTER, canonical=True),
        "strong_holstein_local_policy_closure": local_binding,
    }


def _load_module(path: Path, name: str) -> Any:
    if not path.is_file() or path.is_symlink():
        raise ReportError(f"module path is absent or unsafe: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ReportError(f"cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _paper_i_local_policy_rows() -> tuple[
    dict[str, dict[str, Any]], dict[str, Any]
]:
    """Load the five local policy curves already displayed by Paper I.

    These are a mixed-origin part of the current Paper-I figure, not CHTC
    trajectories and not new reruns to compare against themselves.
    """

    module = _load_module(PAPER_I_PLOT_BUILDER, "paper_i_report_plot_builder")
    completed, source_paths, campaign_closure = module.load_local_completed_policies()
    expected_ids = {
        _policy_id(regime, 7, policy)
        for regime, policy in STRONG_LOCAL_POLICY_EXECUTIONS
    }
    rows: dict[str, dict[str, Any]] = {}
    for regime, policy in sorted(STRONG_LOCAL_POLICY_EXECUTIONS):
        execution_id = _policy_id(regime, 7, policy)
        policy_row = completed.get(regime, {}).get(policy)
        if (
            not isinstance(policy_row, Mapping)
            or policy_row.get("controller_rounds_completed") != TARGET_ROUND
            or policy_row.get("comparator_policy") != policy
        ):
            raise ReportError(f"Paper-I local policy trajectory drifted: {execution_id}")
        summary_path = (
            STRONG_POLICY_RUNTIME / "runs" / execution_id / "summary/summary.json"
        )
        manifest_path = (
            STRONG_POLICY_RUNTIME / "runs" / execution_id / "execution_manifest.json"
        )
        receipt_path = STRONG_POLICY_RUNTIME / "worker_receipts" / f"{execution_id}.json"
        guard_path = STRONG_POLICY_RUNTIME / "guard_receipts" / f"{execution_id}.json"
        summary_sha = _sha256_file(summary_path)
        evidence_key = {
            "summary": _binding(summary_path),
            "execution_manifest": _binding(manifest_path, canonical=True),
            "worker_receipt": _binding(receipt_path, canonical=True),
            "guard_receipt": _binding(guard_path, canonical=True),
            "campaign_terminal_canonical_sha256": campaign_closure.get(
                "terminal_receipt_sha256"
            ),
            "source_class": "paper_i_local_baseline",
        }
        metrics = _load_metric_cache(
            summary_sha256=summary_sha,
            evidence_key=evidence_key,
        )
        if metrics is None:
            summary = _summary_json(
                summary_path.read_bytes(), label=f"Paper-I local policy {execution_id}"
            )
            metrics = _summary_metrics(
                summary,
                execution_id=execution_id,
                source={
                    "source_class": "paper_i_local_baseline",
                    **copy.deepcopy(evidence_key),
                },
            )
            _write_metric_cache(
                summary_sha256=summary_sha,
                evidence_key=evidence_key,
                metrics=metrics,
            )
        points = _trajectory(
            list(policy_row.get("points", [])),
            label=f"Paper-I local policy {execution_id}",
            round_key="k",
            error_key="error",
        )
        terminal_tolerance = 128.0 * math.ulp(
            max(1.0, abs(points[-1]["delta_e"]), abs(float(metrics["delta_e"])))
        )
        if not math.isclose(
            points[-1]["delta_e"],
            float(metrics["delta_e"]),
            rel_tol=0.0,
            abs_tol=terminal_tolerance,
        ):
            raise ReportError(f"Paper-I local policy endpoint drifted: {execution_id}")
        method = "ra_append_only" if policy == "append_only" else "ra_always_cr"
        rows[f"{method}::{regime}"] = {
            **metrics,
            "origin": "Paper-I local candidate",
            "trajectory": points,
            "plot_marker": _trajectory_marker(points),
        }
    if set(campaign_closure.get("completed_execution_ids", ())) != expected_ids:
        raise ReportError("Paper-I local strong-policy terminal inventory drifted")
    return rows, {
        "campaign_closure": copy.deepcopy(dict(campaign_closure)),
        "source_files": [_binding(path) for path in source_paths],
        "plot_builder": _binding(PAPER_I_PLOT_BUILDER),
    }


def _local_matched_metric(
    spec: CellSpec, *, exact_same_cutoff_energy: float
) -> tuple[dict[str, Any] | None, str, str | None]:
    execution_id = spec.local_execution_id
    cleanup_path = MATCHED_RUNTIME / "rotation_cleanup_receipts" / f"{execution_id}.json"
    if not cleanup_path.exists() and not cleanup_path.is_symlink():
        return None, "pending", None
    cleanup = _load_digested(cleanup_path, label=f"local cleanup {execution_id}")
    archive_binding = cleanup.get("archive")
    closure_binding = cleanup.get("archive_closure")
    authority = cleanup.get("authority_metadata")
    cell = cleanup.get("cell_metadata")
    if (
        cleanup.get("schema") != "paper_i_matched_singleton12_archive_cleanup_v1"
        or cleanup.get("status")
        != "passed_exact_safe_tree_removed_archive_retained"
        or cleanup.get("execution_id") != execution_id
        or cleanup.get("direct_source_absent") is not True
        or not isinstance(archive_binding, Mapping)
        or not isinstance(closure_binding, Mapping)
        or not isinstance(authority, Mapping)
        or authority.get("paper_adoption_authorized") is not False
        or authority.get("paper_evidence_adoption_authorized") is not False
        or authority.get("submission_authorized") is not False
        or not isinstance(cell, Mapping)
        or cell.get("execution_id") != execution_id
    ):
        raise ReportError(f"local cleanup closure drifted: {execution_id}")
    closure_path, closure = _verify_binding(
        MATCHED_RUNTIME,
        closure_binding,
        label=f"local archive closure {execution_id}",
        canonical=True,
    )
    assert closure is not None
    if (
        closure.get("schema") != "paper_i_matched_singleton12_archive_closure_v1"
        or closure.get("status") != "passed_archive_and_direct_tree_byte_closure"
        or closure.get("execution_id") != execution_id
        or closure.get("archive") != archive_binding
    ):
        raise ReportError(f"local archive closure identity drifted: {execution_id}")
    manifest_binding = closure.get("archive_manifest")
    if not isinstance(manifest_binding, Mapping):
        raise ReportError("local archive manifest binding is absent")
    _manifest_path, manifest = _verify_binding(
        MATCHED_RUNTIME,
        manifest_binding,
        label=f"local archive manifest {execution_id}",
        canonical=True,
    )
    assert manifest is not None
    payload_files = manifest.get("payload_files")
    if not isinstance(payload_files, list):
        raise ReportError("local archive payload inventory is absent")
    suffix = f"runs/{execution_id}/summary/summary.json"
    summaries = [
        row
        for row in payload_files
        if isinstance(row, Mapping) and row.get("path") == suffix
    ]
    if len(summaries) != 1:
        raise ReportError(f"local archive summary inventory drifted: {execution_id}")
    summary_binding = summaries[0]
    summary_sha = str(summary_binding.get("sha256"))
    evidence_key = {
        "cleanup_canonical_sha256": cleanup["sha256"],
        "closure_canonical_sha256": closure["sha256"],
        "archive_manifest_canonical_sha256": manifest["sha256"],
        "archive_sha256": archive_binding.get("sha256"),
        "archive_size_bytes": archive_binding.get("size_bytes"),
        "summary_member": suffix,
        "summary_size_bytes": summary_binding.get("size_bytes"),
        "source_class": "local",
    }
    metrics = _load_metric_cache(summary_sha256=summary_sha, evidence_key=evidence_key)
    if metrics is None:
        raw_archive = archive_binding.get("path")
        if not isinstance(raw_archive, str):
            raise ReportError("local archive path is absent")
        archive_path = MATCHED_RUNTIME / raw_archive
        if (
            not archive_path.is_file()
            or archive_path.is_symlink()
            or archive_path.stat().st_size != archive_binding.get("size_bytes")
            or _sha256_file(archive_path) != archive_binding.get("sha256")
        ):
            raise ReportError(f"local archive binding drifted: {archive_path}")
        payload = _read_archive_member(
            archive_path,
            member_name=suffix,
            expected_sha256=summary_sha,
            expected_size=int(summary_binding.get("size_bytes", -1)),
        )
        try:
            metrics = _summary_metrics(
                _summary_json(payload, label="local matched summary"),
                execution_id=execution_id,
                exact_same_cutoff_energy=exact_same_cutoff_energy,
                source={
                    "source_class": "local",
                    "cleanup_receipt": _binding(cleanup_path, canonical=True),
                    "archive_closure": _binding(closure_path, canonical=True),
                    "archive": copy.deepcopy(dict(archive_binding)),
                    "summary_member": copy.deepcopy(dict(summary_binding)),
                },
            )
        except ReportError as exc:
            if str(exc).startswith("postprocessing_deferred_low_memory"):
                return None, "postprocessing_deferred", str(exc)
            raise
        _write_metric_cache(
            summary_sha256=summary_sha, evidence_key=evidence_key, metrics=metrics
        )
    return metrics, "completed", None


def _local_weak_metric(spec: CellSpec) -> tuple[dict[str, Any] | None, str, str | None]:
    if not WEAK_POLICY_RUNTIME.exists() and not WEAK_POLICY_RUNTIME.is_symlink():
        return None, "pending", None
    execution_id = spec.local_execution_id
    run_root = WEAK_POLICY_RUNTIME / "runs" / execution_id
    receipt_path = WEAK_POLICY_RUNTIME / "worker_receipts" / f"{execution_id}.json"
    guard_path = WEAK_POLICY_RUNTIME / "guard_receipts" / f"{execution_id}.json"
    manifest_path = run_root / "execution_manifest.json"
    in_progress = WEAK_POLICY_RUNTIME / "in_progress" / execution_id
    quarantine = WEAK_POLICY_RUNTIME / "quarantine" / execution_id
    if in_progress.exists() or in_progress.is_symlink():
        return None, "in_progress", None
    if quarantine.exists() or quarantine.is_symlink():
        return None, "interrupted_or_blocked", "quarantined attempt requires inspection"
    primary = (run_root, receipt_path, guard_path, manifest_path)
    if not any(path.exists() or path.is_symlink() for path in primary):
        return None, "pending", None
    if (
        not run_root.is_dir()
        or run_root.is_symlink()
        or not receipt_path.is_file()
        or receipt_path.is_symlink()
        or not guard_path.is_file()
        or guard_path.is_symlink()
        or not manifest_path.is_file()
        or manifest_path.is_symlink()
    ):
        return None, "interrupted_or_blocked", "incomplete direct closure"
    receipt = _load_digested(receipt_path, label="local weak worker receipt")
    guard = _load_digested(guard_path, label="local weak guard receipt")
    manifest = _load_digested(manifest_path, label="local weak execution manifest")
    if (
        receipt.get("status") != "passed"
        or receipt.get("execution_id") != execution_id
        or receipt.get("controller_rounds_completed") != TARGET_ROUND
        or manifest.get("status") != "passed"
        or manifest.get("execution_id") != execution_id
        or manifest.get("controller_rounds_completed") != TARGET_ROUND
        or receipt.get("execution_manifest_sha256") != manifest.get("sha256")
        or guard.get("status") != "passed"
        or guard.get("execution_id") != execution_id
        or guard.get("child_returncode") != 0
        or guard.get("guard_stop_reason") is not None
        or guard.get("execution_manifest_sha256") != manifest.get("sha256")
    ):
        return None, "interrupted_or_blocked", "direct closure receipts are not passed"
    artifacts = receipt.get("artifacts")
    suffix = f"runs/{execution_id}/summary/summary.json"
    matches = [
        row
        for row in artifacts or []
        if isinstance(row, Mapping) and row.get("path") == suffix
    ]
    if len(matches) != 1:
        raise ReportError(f"local weak summary binding drifted: {execution_id}")
    binding = matches[0]
    summary_path = WEAK_POLICY_RUNTIME / suffix
    if (
        not summary_path.is_file()
        or summary_path.is_symlink()
        or summary_path.stat().st_size != binding.get("size_bytes")
        or _sha256_file(summary_path) != binding.get("sha256")
    ):
        raise ReportError(f"local weak summary file drifted: {execution_id}")
    summary_sha = str(binding.get("sha256"))
    evidence_key = {
        "worker_receipt_canonical_sha256": receipt["sha256"],
        "guard_receipt_canonical_sha256": guard["sha256"],
        "execution_manifest_canonical_sha256": manifest["sha256"],
        "summary_path": suffix,
        "summary_size_bytes": binding.get("size_bytes"),
        "source_class": "local",
    }
    metrics = _load_metric_cache(summary_sha256=summary_sha, evidence_key=evidence_key)
    if metrics is None:
        module = _load_module(WEAK_RUNNER, "paper_i_report_weak_runner")
        try:
            closed = bool(module._closed_cell(WEAK_POLICY_RUNTIME, execution_id))
        except Exception as exc:
            return None, "interrupted_or_blocked", f"{type(exc).__name__}: {exc}"
        if not closed:
            return None, "pending", None
        payload = summary_path.read_bytes()
        try:
            metrics = _summary_metrics(
                _summary_json(payload, label="local weak summary"),
                execution_id=execution_id,
                source={
                    "source_class": "local",
                    "worker_receipt": _binding(receipt_path, canonical=True),
                    "guard_receipt": _binding(guard_path, canonical=True),
                    "execution_manifest": _binding(manifest_path, canonical=True),
                    "summary": _binding(summary_path),
                },
            )
        except ReportError as exc:
            if str(exc).startswith("postprocessing_deferred_low_memory"):
                return None, "postprocessing_deferred", str(exc)
            raise
        _write_metric_cache(
            summary_sha256=summary_sha, evidence_key=evidence_key, metrics=metrics
        )
    return metrics, "completed", None


def _position_aware_phase0_overlay() -> dict[str, Any] | None:
    """Load the closed k=15 position-aware Phase-0 diagnostic.

    This curve is supplemental only.  It is deliberately not returned through
    ``_local_weak_metric`` because that seam requires a complete k=50
    replacement candidate and feeds the decision table.
    """

    root = POSITION_AWARE_PHASE0_RUNTIME
    if not root.exists() and not root.is_symlink():
        return None
    execution_id = POSITION_AWARE_PHASE0_EXECUTION_ID
    terminal_path = root / "terminal_receipt.json"
    status_path = root / "status.json"
    receipt_path = root / "receipts" / f"{execution_id}.json"
    guard_path = root / "guard_receipts" / f"{execution_id}.json"
    run_root = root / "runs" / execution_id
    manifest_path = run_root / "execution_manifest.json"
    overlay_path = run_root / "route_overlay.json"
    summary_path = run_root / "summary/summary.json"
    required = (
        terminal_path,
        status_path,
        receipt_path,
        guard_path,
        manifest_path,
        overlay_path,
        summary_path,
    )
    if not terminal_path.exists() and not terminal_path.is_symlink():
        return None
    if any(not path.is_file() or path.is_symlink() for path in required):
        raise ReportError("position-aware Phase-0 diagnostic closure is incomplete")

    terminal = _load_digested(terminal_path, label="position-aware terminal")
    status = _load_digested(status_path, label="position-aware status")
    receipt = _load_digested(receipt_path, label="position-aware worker receipt")
    guard = _load_digested(guard_path, label="position-aware guard receipt")
    manifest = _load_digested(manifest_path, label="position-aware manifest")
    overlay = _load_digested(overlay_path, label="position-aware route overlay")
    authorization_sha = terminal.get("authorization_sha256")
    plan_sha = terminal.get("plan_sha256")
    adoption_documents = (terminal, receipt, manifest, overlay)
    if (
        terminal.get("schema")
        != "paper_i_position_aware_phase0_canary_terminal_receipt_v1"
        or terminal.get("status") != "passed_k15"
        or terminal.get("execution_id") != execution_id
        or terminal.get("controller_rounds_completed") != 15
        or terminal.get("phase0_policy")
        != "global_singleton_insertion_position_absolute_gradient_shortlist_v1"
        or status.get("status") != "passed_k15"
        or status.get("execution_id") != execution_id
        or status.get("terminal_sha256") != terminal.get("sha256")
        or receipt.get("status") != "passed"
        or receipt.get("execution_id") != execution_id
        or receipt.get("controller_rounds_completed") != 15
        or receipt.get("authorization_sha256") != authorization_sha
        or receipt.get("plan_sha256") != plan_sha
        or terminal.get("worker_receipt_sha256") != receipt.get("sha256")
        or guard.get("status") != "passed"
        or guard.get("execution_id") != execution_id
        or guard.get("child_returncode") != 0
        or guard.get("stop_reason") is not None
        or terminal.get("guard_receipt_sha256") != guard.get("sha256")
        or manifest.get("status") != "passed"
        or manifest.get("execution_id") != execution_id
        or manifest.get("controller_rounds_completed") != 15
        or manifest.get("target_horizon") != 15
        or manifest.get("authorization_sha256") != authorization_sha
        or manifest.get("plan_sha256") != plan_sha
        or receipt.get("execution_manifest_sha256") != manifest.get("sha256")
        or overlay.get("execution_id") != execution_id
        or overlay.get("target_horizon") != 15
        or overlay.get("phase0_policy")
        != "global_singleton_insertion_position_absolute_gradient_shortlist_v1"
        or overlay.get("phase0_insertion_position_scope")
        != "all_commutation_reduced_insertion_position_records_before_shortlist_v1"
        or any(document.get("submission_authorized") is not False for document in adoption_documents)
        or any(document.get("paper_adoption_authorized") is not False for document in adoption_documents)
        or any(
            document.get("paper_evidence_adoption_authorized") is not False
            for document in adoption_documents
        )
    ):
        raise ReportError("position-aware Phase-0 authority chain drifted")

    worker_artifacts = receipt.get("artifacts")
    if not isinstance(worker_artifacts, list):
        raise ReportError("position-aware worker inventory is absent")

    def worker_artifact(path: Path) -> dict[str, Any]:
        relative = path.relative_to(root).as_posix()
        matches = [
            row
            for row in worker_artifacts
            if isinstance(row, Mapping) and row.get("path") == relative
        ]
        if len(matches) != 1:
            raise ReportError(f"position-aware artifact binding drifted: {relative}")
        binding = matches[0]
        if (
            binding.get("size_bytes") != path.stat().st_size
            or binding.get("sha256") != _sha256_file(path)
        ):
            raise ReportError(f"position-aware artifact bytes drifted: {relative}")
        return copy.deepcopy(dict(binding))

    summary_binding = worker_artifact(summary_path)
    worker_artifact(manifest_path)
    worker_artifact(overlay_path)
    manifest_artifacts = manifest.get("artifacts")
    if not isinstance(manifest_artifacts, Mapping):
        raise ReportError("position-aware manifest inventory is absent")
    for label, path in (("summary", summary_path), ("route_overlay", overlay_path)):
        binding = manifest_artifacts.get(label)
        if (
            not isinstance(binding, Mapping)
            or binding.get("path") != path.relative_to(run_root).as_posix()
            or binding.get("size_bytes") != path.stat().st_size
            or binding.get("sha256") != _sha256_file(path)
        ):
            raise ReportError(f"position-aware manifest binding drifted: {label}")

    summary = _summary_json(summary_path.read_bytes(), label="position-aware summary")
    provenance = summary.get("provenance")
    trace = summary.get("accepted_error_trace")
    if (
        summary.get("schema") != RA_SUMMARY_SCHEMA
        or not isinstance(provenance, Mapping)
        or provenance.get("qiskit_compile_convention") != COMPILE_CONVENTION
        or not isinstance(trace, list)
        or len(trace) != 15
    ):
        raise ReportError("position-aware summary identity drifted")
    trajectory: list[dict[str, Any]] = []
    for expected_round, row in enumerate(trace, start=1):
        if not isinstance(row, Mapping):
            raise ReportError("position-aware accepted trace is malformed")
        controller_round = _integer(
            row.get("controller_round"), label="position-aware controller round"
        )
        if controller_round != expected_round:
            raise ReportError("position-aware accepted trace is not contiguous")
        trajectory.append(
            {
                "k": controller_round,
                "delta_e": _finite(
                    row.get("absolute_energy_error"),
                    label="position-aware accepted error",
                    nonnegative=True,
                ),
            }
        )
    requested = summary.get("requested_rounds")
    requested_rows = [
        row
        for row in requested or []
        if isinstance(row, Mapping) and row.get("controller_round") == 15
    ]
    effective = summary.get("effective_plateau")
    if len(requested_rows) != 1 or not isinstance(effective, Mapping):
        raise ReportError("position-aware terminal or plateau row is absent")
    terminal_row = requested_rows[0]
    resources = terminal_row.get("resources")
    work = terminal_row.get("algorithmic_work")
    plateau_round = _integer(
        effective.get("controller_round"), label="position-aware plateau round"
    )
    if (
        terminal_row.get("failure") is not None
        or terminal_row.get("active_ansatz_depth") != 15
        or not isinstance(resources, Mapping)
        or resources.get("compile_convention") != COMPILE_CONVENTION
        or not isinstance(work, Mapping)
        or effective.get("status") != "available"
        or plateau_round not in range(1, 16)
    ):
        raise ReportError("position-aware terminal metrics drifted")
    terminal_error = trajectory[-1]["delta_e"]
    declared_terminal_error = _finite(
        terminal_row.get("absolute_energy_error"),
        label="position-aware terminal error",
        nonnegative=True,
    )
    tolerance = 128.0 * math.ulp(max(1.0, terminal_error, declared_terminal_error))
    if not math.isclose(
        terminal_error, declared_terminal_error, rel_tol=0.0, abs_tol=tolerance
    ):
        raise ReportError("position-aware terminal error does not close")
    marker_error = trajectory[plateau_round - 1]["delta_e"]
    declared_plateau_error = _finite(
        effective.get("absolute_energy_error"),
        label="position-aware plateau error",
        nonnegative=True,
    )
    if not math.isclose(
        marker_error, declared_plateau_error, rel_tol=0.0, abs_tol=tolerance
    ):
        raise ReportError("position-aware plateau marker does not close")
    return {
        "key": "position_aware_phase0::ra_always_cr::strong_weak_u8",
        "method": "ra_always_cr",
        "method_label": "RA always-open",
        "variant_label": "position-aware Phase 0",
        "regime": "strong_weak_u8",
        "regime_label": "Strong--weak",
        "nph": 3,
        "origin": "new local diagnostic",
        "execution_id": execution_id,
        "run_class": "diagnostic",
        "target_horizon": 15,
        "trajectory": trajectory,
        "plot_marker": {
            "k": plateau_round,
            "delta_e": marker_error,
            "policy": "first_effective_plateau_prefix",
        },
        "terminal": {
            "k": 15,
            "active_ansatz_depth": 15,
            "delta_e": terminal_error,
            "energy": _finite(
                trace[-1].get("accepted_energy"),
                label="position-aware terminal energy",
            ),
            "N2q": _integer(
                resources.get("compiled_two_qubit_count"), label="position-aware N2q"
            ),
            "D2q": _integer(
                resources.get("compiled_two_qubit_depth"), label="position-aware D2q"
            ),
            "Dc": _integer(
                resources.get("compiled_total_depth"), label="position-aware Dc"
            ),
            "W1q": None,
            "S_alg": _integer(work.get("s_alg"), label="position-aware S_alg"),
        },
        "source": {
            "status": _binding(status_path, canonical=True),
            "terminal_receipt": _binding(terminal_path, canonical=True),
            "worker_receipt": _binding(receipt_path, canonical=True),
            "guard_receipt": _binding(guard_path, canonical=True),
            "execution_manifest": _binding(manifest_path, canonical=True),
            "route_overlay": _binding(overlay_path, canonical=True),
            "summary": {**_binding(summary_path), "worker_binding": summary_binding},
        },
        "submission_authorized": False,
        "paper_adoption_authorized": False,
        "paper_evidence_adoption_authorized": False,
    }


def _latest_remote_run() -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not REMOTE_RUNNER_STATE_DIR.is_dir() or REMOTE_RUNNER_STATE_DIR.is_symlink():
        return None, None
    candidates: list[tuple[str, Path, dict[str, Any]]] = []
    for path in REMOTE_RUNNER_STATE_DIR.glob("*.json"):
        try:
            value = _load_json(path, label="remote-runner state")
        except ReportError:
            continue
        if value.get("job_id") == REMOTE_JOB_ID:
            candidates.append((str(value.get("created_at", "")), path, value))
    if not candidates:
        return None, None
    _, path, value = max(candidates, key=lambda row: row[0])
    binding = _binding(path)
    log_path = value.get("log_path")
    log_binding = None
    reason = None
    if isinstance(log_path, str):
        log = Path(log_path)
        if log.is_file() and not log.is_symlink():
            log_binding = _binding(log)
            with log.open("rb") as stream:
                try:
                    stream.seek(max(0, log.stat().st_size - 256 * 1024))
                except OSError:
                    pass
                tail = stream.read().decode("utf-8", errors="replace")
            if "available_memory_floor_breached" in tail:
                reason = "available_memory_floor_breached"
    return {
        "id": value.get("id"),
        "job_id": value.get("job_id"),
        "status": value.get("status"),
        "created_at": value.get("created_at"),
        "finished_at": value.get("finished_at"),
        "returncode": value.get("returncode"),
        "failure_reason": reason,
        "state_binding": binding,
        "log_binding": log_binding,
    }, value


def _active_science_processes() -> list[dict[str, Any]]:
    try:
        output = subprocess.run(
            ["ps", "-axo", "pid=,command=", "-ww"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    matches: list[dict[str, Any]] = []
    for raw in output.splitlines():
        line = raw.strip()
        pid, _, command = line.partition(" ")
        if ORCHESTRATOR_NAME in command or MATCHED_RUNNER_NAME in command:
            try:
                matches.append({"pid": int(pid), "command": command})
            except ValueError:
                continue
    return matches


def _campaign_snapshot(completed: int) -> dict[str, Any]:
    orchestrator = None
    orchestrator_binding = None
    if ORCHESTRATOR_STATUS.is_file() and not ORCHESTRATOR_STATUS.is_symlink():
        orchestrator = _load_digested(
            ORCHESTRATOR_STATUS, label="priority orchestrator status"
        )
        orchestrator_binding = _binding(ORCHESTRATOR_STATUS, canonical=True)
    matched_status = None
    matched_status_binding = None
    if MATCHED_STATUS.is_file() and not MATCHED_STATUS.is_symlink():
        matched_status = _load_digested(
            MATCHED_STATUS, label="matched campaign status"
        )
        matched_status_binding = _binding(MATCHED_STATUS, canonical=True)
    remote, _raw = _latest_remote_run()
    active = _active_science_processes()
    if completed == 24:
        state = "completed_all_24"
    elif active:
        state = "running"
    elif remote and remote.get("status") == "failed":
        state = (
            "interrupted_guard_memory"
            if remote.get("failure_reason") == "available_memory_floor_breached"
            else "interrupted_failed"
        )
    elif remote and remote.get("status") == "succeeded":
        state = "runner_succeeded_partial_report_refresh_pending"
    elif orchestrator is not None:
        state = "stale_or_inactive_status"
    else:
        state = "not_started"
    return {
        "state": state,
        "completed_cells": completed,
        "total_cells": 24,
        "active_processes": active,
        "orchestrator_status": None
        if orchestrator is None
        else {
            "status": orchestrator.get("status"),
            "current_execution_ids": copy.deepcopy(
                orchestrator.get("current_execution_ids", [])
            ),
            "binding": orchestrator_binding,
        },
        "matched_status": None
        if matched_status is None
        else {
            "status": matched_status.get("status"),
            "current_execution_id": matched_status.get("current_execution_id"),
            "completed_execution_ids": copy.deepcopy(
                matched_status.get("completed_execution_ids", [])
            ),
            "binding": matched_status_binding,
        },
        "remote_run": remote,
    }


def _error_tolerance(local: Mapping[str, Any], historical: Mapping[str, Any]) -> float:
    scale = max(
        1.0,
        abs(float(local["delta_e"])),
        abs(float(historical["delta_e"])),
        abs(float(local["exact_same_cutoff_energy"])),
        abs(float(historical["exact_same_cutoff_energy"])),
    )
    return 128.0 * math.ulp(scale)


def compare_metrics(
    local: Mapping[str, Any] | None, historical: Mapping[str, Any]
) -> dict[str, Any]:
    if local is None:
        return {
            "classification": "pending",
            "energy_comparison": "pending",
            "cost_comparison": "pending",
            "metric_directions": {},
        }
    tolerance = _error_tolerance(local, historical)
    directions: dict[str, str] = {}
    delta = float(local["delta_e"]) - float(historical["delta_e"])
    directions["delta_e"] = (
        "tie" if abs(delta) <= tolerance else ("better" if delta < 0 else "worse")
    )
    for key in COST_KEYS:
        directions[key] = (
            "tie"
            if int(local[key]) == int(historical[key])
            else ("better" if int(local[key]) < int(historical[key]) else "worse")
        )
    cost_values = [directions[key] for key in COST_KEYS]
    if "worse" not in cost_values and "better" in cost_values:
        cost_comparison = "local_cost_pareto_better"
    elif "better" not in cost_values and "worse" in cost_values:
        cost_comparison = "chtc_cost_dominates"
    elif all(value == "tie" for value in cost_values):
        cost_comparison = "cost_equivalent"
    else:
        cost_comparison = "cost_tradeoff"
    energy_comparison = {
        "better": "local_delta_e_better",
        "worse": "chtc_delta_e_better",
        "tie": "delta_e_equivalent_within_tolerance",
    }[directions["delta_e"]]
    values = list(directions.values())
    if "worse" not in values and "better" in values:
        classification = "local_pareto_better"
    elif "better" not in values and "worse" in values:
        classification = "chtc_dominates"
    elif all(value == "tie" for value in values):
        classification = "equivalent"
    else:
        classification = "tradeoff"
    return {
        "classification": classification,
        "energy_comparison": energy_comparison,
        "cost_comparison": cost_comparison,
        "error_tolerance": tolerance,
        "metric_directions": directions,
        "local_minus_chtc": {
            key: float(local[key]) - float(historical[key]) for key in METRIC_KEYS
        },
    }


def _ed_reference_binding() -> dict[str, Any]:
    value = _load_json(ED_REFERENCE, label="same-cutoff ED reference")
    validation = value.get("validation")
    if (
        value.get("schema") != "paper_i_hh_ed_cutoff_reference_six_regime_v1"
        or not isinstance(validation, Mapping)
        or validation.get("status") != "pass"
        or value.get("execution_authorized") is not True
    ):
        raise ReportError("same-cutoff ED reference validation drifted")
    return _binding(ED_REFERENCE)


def collect_report() -> dict[str, Any]:
    historical, global_binding = _historical_global_rows()
    policy_rows, policy_binding = _historical_policy_rows()
    historical.update(policy_rows)
    position_aware = _position_aware_phase0_overlay()
    specs = cell_specs()
    if set(historical) != {spec.key for spec in specs}:
        missing = sorted({spec.key for spec in specs} - set(historical))
        raise ReportError(f"Paper-I 24-cell curve matrix is incomplete: {missing}")
    rows: list[dict[str, Any]] = []
    completed = 0
    for spec in specs:
        if spec.local_kind == "matched_archive":
            local, local_state, detail = _local_matched_metric(
                spec,
                exact_same_cutoff_energy=float(
                    historical[spec.key]["exact_same_cutoff_energy"]
                ),
            )
        elif spec.local_kind == "weak_direct":
            local, local_state, detail = _local_weak_metric(spec)
        elif spec.local_kind == "paper_i_local_baseline":
            local, local_state, detail = (
                None,
                "already_current_paper_i_local_no_distinct_rerun",
                "This displayed Paper-I curve is the completed local strong-sector candidate; no separate newer rerun exists.",
            )
        elif spec.local_kind == "no_distinct_rerun":
            local, local_state, detail = (
                None,
                "no_distinct_local_rerun_scheduled",
                "The current priority chain does not contain a separate rerun for this Paper-I curve.",
            )
        else:
            raise AssertionError(f"unknown local metric source: {spec.local_kind}")
        if local is not None or historical[spec.key].get("origin") == "Paper-I local candidate":
            completed += 1
        rows.append(
            {
                "key": spec.key,
                "group": spec.group,
                "method": spec.method,
                "method_label": spec.method_label,
                "regime": spec.regime,
                "regime_label": spec.regime_label,
                "nph": spec.nph,
                "local_execution_id": spec.local_execution_id,
                "local_state": local_state,
                "local_state_detail": detail,
                "historical": historical[spec.key],
                "local": local,
                "comparison": compare_metrics(local, historical[spec.key]),
            }
        )
    campaign = _campaign_snapshot(completed)
    if campaign["state"].startswith("interrupted"):
        failed_execution_id = (
            campaign.get("matched_status", {}) or {}
        ).get("current_execution_id")
        for row in rows:
            if (
                row["local_execution_id"] == failed_execution_id
                and row["local"] is None
            ):
                row["local_state"] = campaign["state"]
                row["local_state_detail"] = (
                    campaign.get("remote_run", {}) or {}
                ).get("failure_reason")
    counts = {
        "completed": completed,
        "pending": 24 - completed,
        "paper_i_displayed_curves": len(rows),
        "paper_i_chtc_curves": sum(
            row["historical"].get("origin") == "CHTC paper" for row in rows
        ),
        "paper_i_local_candidate_curves": sum(
            row["historical"].get("origin") == "Paper-I local candidate"
            for row in rows
        ),
        "distinct_new_local_reruns": sum(row["local"] is not None for row in rows),
        "local_pareto_better": sum(
            row["comparison"]["classification"] == "local_pareto_better"
            for row in rows
        ),
        "tradeoff": sum(
            row["comparison"]["classification"] == "tradeoff" for row in rows
        ),
        "chtc_dominates": sum(
            row["comparison"]["classification"] == "chtc_dominates"
            for row in rows
        ),
        "equivalent": sum(
            row["comparison"]["classification"] == "equivalent" for row in rows
        ),
        "supplemental_diagnostic_curves": int(position_aware is not None),
    }
    supplemental_curves: list[dict[str, Any]] = []
    if position_aware is not None:
        baseline = historical["ra_always_cr::strong_weak_u8"]
        baseline_by_round = {
            int(point["k"]): float(point["delta_e"])
            for point in baseline["trajectory"]
        }
        diagnostic_terminal = float(position_aware["terminal"]["delta_e"])
        baseline_k15 = baseline_by_round[15]
        baseline_k50 = baseline_by_round[50]
        position_aware["comparison_context"] = {
            "baseline_execution_id": baseline["execution_id"],
            "baseline_origin": baseline["origin"],
            "same_k": 15,
            "baseline_k15_delta_e": baseline_k15,
            "diagnostic_k15_delta_e": diagnostic_terminal,
            "relative_error_reduction_vs_baseline_k15": (
                (baseline_k15 - diagnostic_terminal) / baseline_k15
            ),
            "baseline_k50_delta_e": baseline_k50,
            "relative_error_reduction_vs_baseline_k50": (
                (baseline_k50 - diagnostic_terminal) / baseline_k50
            ),
            "decision_table_inclusion": False,
            "reason": "diagnostic horizon is k=15, not the report's fixed k=50 decision horizon",
        }
        supplemental_curves.append(position_aware)
    distinct_new_local_reruns = counts["distinct_new_local_reruns"]
    evidence_projection = {
        "rows": [
            {
                "key": row["key"],
                "local_state": row["local_state"],
                "historical": row["historical"],
                "local": row["local"],
                "comparison": row["comparison"],
            }
            for row in rows
        ],
        "supplemental_curves": supplemental_curves,
        "campaign": campaign,
        "sources": {
            "report_builder": _binding(Path(__file__).resolve()),
            "global_chtc_adapter": global_binding,
            "policy_adapter": policy_binding,
            "ed_reference": _ed_reference_binding(),
        },
    }
    return {
        "schema": REPORT_SCHEMA,
        "status": (
            "complete_full_24_curve_paper_i_matrix_with_all_distinct_reruns"
            if completed == 24
            else f"live_provisional_{distinct_new_local_reruns}_of_19_distinct_chtc_replacements"
        ),
        "generated_at_utc": _utc_now(),
        "evidence_revision_sha256": _canonical_sha256(evidence_projection),
        "scope": {
            "cell_count": 24,
            "target_controller_round": TARGET_ROUND,
            "weak_sector": "nph=3; 3 Hubbard regimes; RA plateau, RA append-only, RA always-open, conventional Append",
            "strong_sector": "nph=7; 3 Hubbard regimes; RA plateau, RA append-only, RA always-open, conventional Append",
            "paper_i_displayed_curve_origin": "19 authenticated CHTC trajectories plus 5 completed local strong-Holstein policy candidates",
            "metrics": list(METRIC_KEYS),
            "qiskit_cost_tuple": ["N2q", "D2q", "Dc", "W1q"],
            "supplemental_diagnostic_overlay": (
                "authenticated strong--weak position-aware Phase-0 RA always-open k=15"
                if supplemental_curves
                else "absent"
            ),
        },
        "decision_policy": {
            "lower_is_better": list(METRIC_KEYS),
            "integer_cost_tolerance": 0,
            "delta_e_tolerance": "128*ulp(max(1, |local|, |CHTC|, |same-cutoff ED|))",
            "classification_is_diagnostic_not_adoption": True,
        },
        "campaign": campaign,
        "counts": counts,
        "cells": rows,
        "supplemental_curves": supplemental_curves,
        "sources": evidence_projection["sources"],
        "submission_authorized": False,
        "paper_adoption_authorized": False,
        "paper_evidence_adoption_authorized": False,
        "manuscript_modified": False,
    }


def _plot_error(value: float) -> float:
    return max(PLOT_ERROR_FLOOR, float(value))


def _render_combined_plot(
    report: Mapping[str, Any], *, output_dir: Path
) -> dict[str, Any]:
    mpl_cache = OUTPUT_DIR / ".matplotlib-cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ["MPLCONFIGDIR"] = mpl_cache.as_posix()
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.ticker import LogLocator, NullFormatter
    except Exception as exc:
        raise ReportError("Matplotlib is required for the convergence report") from exc

    with plt.rc_context(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "axes.linewidth": 0.7,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.3,
            "legend.fontsize": 6.9,
        }
    ):
        fig, axes = plt.subplots(2, 3, figsize=(11.0, 5.15), sharex=True, sharey="row")
        legend_handles: dict[str, Any] = {}
        panel_rows: list[dict[str, Any]] = []
        row_values: dict[int, list[float]] = {0: [], 1: []}
        for panel_index, (ax, (regime, regime_label, nph)) in enumerate(
            zip(axes.flat, REGIMES)
        ):
            rows = [row for row in report["cells"] if row["regime"] == regime]
            if len(rows) != 4:
                raise ReportError(f"report regime matrix drifted: {regime}")
            rows.sort(
                key=lambda row: (
                    "ra_plateau",
                    "ra_append_only",
                    "ra_always_cr",
                    "append_conventional",
                ).index(str(row["method"]))
            )
            curve_rows: list[dict[str, Any]] = []
            holstein_row = 0 if nph == 3 else 1
            for row in rows:
                style = METHOD_STYLES[str(row["method"])]
                baseline = row["historical"]
                baseline_is_local = baseline.get("origin") == "Paper-I local candidate"
                for origin, metrics, local, is_new_rerun in (
                    (
                        "Paper-I local",
                        baseline,
                        baseline_is_local,
                        False,
                    )
                    if baseline_is_local
                    else ("CHTC", baseline, False, False),
                    ("new local", row.get("local"), True, True),
                ):
                    if metrics is None:
                        continue
                    trajectory = metrics.get("trajectory")
                    marker = metrics.get("plot_marker")
                    if not isinstance(trajectory, list) or not isinstance(marker, Mapping):
                        raise ReportError(
                            f"plot trajectory/marker is absent: {row['key']} {origin}"
                        )
                    x = [int(point["k"]) for point in trajectory]
                    y_exact = [float(point["delta_e"]) for point in trajectory]
                    y = [_plot_error(value) for value in y_exact]
                    marker_round = int(marker["k"])
                    try:
                        marker_index = x.index(marker_round)
                    except ValueError as exc:
                        raise ReportError(
                            f"plot marker is outside trajectory: {row['key']} {origin}"
                        ) from exc
                    line_width = 2.0 if is_new_rerun else (1.65 if local else 1.15)
                    alpha = 1.0 if is_new_rerun else (0.82 if local else 0.48)
                    label = f"{style['label']} - {origin}"
                    primary_paper_i_baseline = (
                        origin == "CHTC"
                        and row["method"] in {"ra_plateau", "append_conventional"}
                    )
                    ax.plot(
                        x,
                        y,
                        color=style["color"],
                        linewidth=line_width,
                        alpha=alpha,
                        solid_capstyle="round",
                        zorder=4 if is_new_rerun else (3 if local else 2),
                    )
                    display_markers: list[dict[str, Any]] = []
                    if primary_paper_i_baseline:
                        ra_cross, append_cross, ra_budget, append_budget = (
                            PAPER_I_MATCHED_MARKERS[regime]
                        )
                        crossing_round = (
                            ra_cross if row["method"] == "ra_plateau" else append_cross
                        )
                        budget_round = (
                            ra_budget if row["method"] == "ra_plateau" else append_budget
                        )
                        for special_round, special_marker, special_policy, filled in (
                            (
                                crossing_round,
                                "o",
                                "first_common_accuracy_crossing",
                                False,
                            ),
                            (
                                budget_round,
                                "^",
                                "best_error_within_shared_s_alg",
                                True,
                            ),
                        ):
                            try:
                                special_index = x.index(special_round)
                            except ValueError as exc:
                                raise ReportError(
                                    f"Paper-I matched marker is outside trajectory: {row['key']}"
                                ) from exc
                            ax.scatter(
                                [special_round],
                                [y[special_index]],
                                s=31,
                                marker=special_marker,
                                facecolors=style["color"] if filled else "white",
                                edgecolors=style["color"],
                                linewidths=1.25,
                                zorder=6,
                            )
                            display_markers.append(
                                {
                                    "k": special_round,
                                    "delta_e": y_exact[special_index],
                                    "policy": special_policy,
                                }
                            )
                    else:
                        marker_size = 48 if style["marker"] == "*" else 24
                        ax.scatter(
                            [marker_round],
                            [y[marker_index]],
                            s=marker_size,
                            marker=style["marker"],
                            facecolors=style["color"] if local else "white",
                            edgecolors=style["color"],
                            linewidths=1.0,
                            alpha=1.0 if local else 0.75,
                            zorder=5,
                        )
                        display_markers.append(copy.deepcopy(dict(marker)))
                    if label not in legend_handles:
                        legend_handles[label] = Line2D(
                            [0],
                            [0],
                            color=style["color"],
                            linewidth=line_width,
                            alpha=alpha,
                            marker=("" if primary_paper_i_baseline else style["marker"]),
                            markerfacecolor=style["color"] if local else "white",
                            markeredgecolor=style["color"],
                            markersize=6.2 if style["marker"] == "*" else 4.5,
                            label=label,
                        )
                    row_values[holstein_row].extend(y)
                    curve_rows.append(
                        {
                            "key": row["key"],
                            "method": row["method"],
                            "origin": origin,
                            "execution_id": metrics["execution_id"],
                            "point_count": len(trajectory),
                            "first_k": x[0],
                            "last_k": x[-1],
                            "terminal_delta_e_exact": y_exact[-1],
                            "marker": copy.deepcopy(dict(marker)),
                            "display_markers": display_markers,
                            "trajectory_sha256": _canonical_sha256(trajectory),
                            "source": copy.deepcopy(dict(metrics["source"])),
                        }
                    )
            supplemental = [
                curve
                for curve in report.get("supplemental_curves", [])
                if isinstance(curve, Mapping) and curve.get("regime") == regime
            ]
            for curve in supplemental:
                if curve.get("method") != "ra_always_cr":
                    raise ReportError("unknown supplemental plot method")
                style = METHOD_STYLES["ra_always_cr"]
                trajectory = curve.get("trajectory")
                marker = curve.get("plot_marker")
                if not isinstance(trajectory, list) or not isinstance(marker, Mapping):
                    raise ReportError("supplemental trajectory/marker is absent")
                x = [int(point["k"]) for point in trajectory]
                y_exact = [float(point["delta_e"]) for point in trajectory]
                if x != list(range(1, int(curve.get("target_horizon", -1)) + 1)):
                    raise ReportError("supplemental trajectory is not contiguous")
                y = [_plot_error(value) for value in y_exact]
                marker_round = int(marker["k"])
                try:
                    marker_index = x.index(marker_round)
                except ValueError as exc:
                    raise ReportError("supplemental marker is outside trajectory") from exc
                label = "RA always-open - position-aware Phase 0 (local k=15)"
                ax.plot(
                    x,
                    y,
                    color=style["color"],
                    linewidth=2.45,
                    alpha=1.0,
                    solid_capstyle="round",
                    zorder=8,
                )
                ax.scatter(
                    [marker_round],
                    [y[marker_index]],
                    s=31,
                    marker=style["marker"],
                    facecolors=style["color"],
                    edgecolors="#4A2F00",
                    linewidths=1.15,
                    zorder=9,
                )
                legend_handles[label] = Line2D(
                    [0],
                    [0],
                    color=style["color"],
                    linewidth=2.45,
                    marker=style["marker"],
                    markerfacecolor=style["color"],
                    markeredgecolor="#4A2F00",
                    markersize=4.8,
                    label=label,
                )
                row_values[holstein_row].extend(y)
                curve_rows.append(
                    {
                        "key": curve["key"],
                        "method": curve["method"],
                        "variant": curve["variant_label"],
                        "origin": curve["origin"],
                        "execution_id": curve["execution_id"],
                        "run_class": curve["run_class"],
                        "decision_table_inclusion": False,
                        "point_count": len(trajectory),
                        "first_k": x[0],
                        "last_k": x[-1],
                        "terminal_delta_e_exact": y_exact[-1],
                        "marker": copy.deepcopy(dict(marker)),
                        "display_markers": [copy.deepcopy(dict(marker))],
                        "trajectory_sha256": _canonical_sha256(trajectory),
                        "source": copy.deepcopy(dict(curve["source"])),
                    }
                )
            ax.set_yscale("log")
            ax.set_xlim(0, TARGET_ROUND)
            ax.set_xticks(tuple(range(0, TARGET_ROUND + 1, 10)))
            ax.yaxis.set_major_locator(LogLocator(base=10.0))
            ax.yaxis.set_minor_locator(
                LogLocator(base=10.0, subs=tuple(range(2, 10)), numticks=100)
            )
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.grid(True, which="major", color="#98A2B3", alpha=0.32, linewidth=0.55)
            ax.grid(
                True,
                which="minor",
                color="#D0D5DD",
                alpha=0.28,
                linewidth=0.4,
                linestyle=":",
            )
            ax.set_title(rf"{regime_label} ($n_{{\rm ph}}={nph}$)")
            ax.set_xlabel("ADAPT iteration")
            if panel_index in {0, 3}:
                ax.set_ylabel(r"same-cutoff $|\Delta E|$")
            panel_rows.append(
                {
                    "panel_index": panel_index,
                    "regime": regime,
                    "regime_label": regime_label,
                    "nph": nph,
                    "curves": curve_rows,
                }
            )
        for holstein_row in (0, 1):
            values = row_values[holstein_row]
            if not values:
                raise ReportError(f"combined plot row is empty: {holstein_row}")
            lower_decade = 10.0 ** math.floor(math.log10(min(values)))
            upper_decade = 10.0 ** math.ceil(math.log10(max(values)))
            if lower_decade == upper_decade:
                lower_decade /= 10.0
                upper_decade *= 10.0
            for ax in axes[holstein_row, :]:
                ax.set_ylim(lower_decade, upper_decade)
        legend_handles["first common-accuracy crossing"] = Line2D(
            [0],
            [0],
            color="#3F3F46",
            linewidth=0,
            marker="o",
            markerfacecolor="white",
            markeredgecolor="#3F3F46",
            markersize=4.8,
            label="first common-accuracy crossing",
        )
        legend_handles["best error within shared S_alg"] = Line2D(
            [0],
            [0],
            color="#3F3F46",
            linewidth=0,
            marker="^",
            markerfacecolor="#3F3F46",
            markeredgecolor="#3F3F46",
            markersize=5.0,
            label=r"best error within shared $S_{\rm alg}$",
        )
        fig.legend(
            handles=list(legend_handles.values()),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.985),
            ncol=4,
            frameon=False,
            title="Paper-I crossing/budget markers retained; policy markers are terminal k=50; the filled orange diamond marks the position-aware local effective plateau",
            title_fontsize=7.0,
            handlelength=2.0,
            columnspacing=1.2,
        )
        fig.subplots_adjust(
            left=0.07,
            right=0.995,
            bottom=0.09,
            top=0.80,
            hspace=0.40,
            wspace=0.17,
        )
        stem = f"{STABLE_STEM}_six_regime_combined_convergence"
        pdf_path = output_dir / f"{stem}.pdf"
        png_path = output_dir / f"{stem}.png"
        fig.savefig(
            pdf_path,
            bbox_inches="tight",
            metadata={"CreationDate": None, "ModDate": None},
        )
        fig.savefig(
            png_path,
            dpi=180,
            bbox_inches="tight",
            metadata={"Software": "Matplotlib"},
        )
        plt.close(fig)
    combined = {
        "layout": "two_holstein_rows_by_three_hubbard_columns",
        "share_y_by_holstein_row": True,
        "pdf": _relative_binding(pdf_path, root=output_dir),
        "png": _relative_binding(png_path, root=output_dir),
        "panels": panel_rows,
    }
    provenance = _digested(
        {
            "schema": "paper_i_local_vs_chtc_replacement_plot_provenance_v1",
            "status": "passed_paper_i_six_regime_combined_convergence_plot",
            "metric": "same_cutoff_absolute_energy_error",
            "display_horizon": {"minimum_k": 0, "maximum_k": TARGET_ROUND},
            "display_error_floor": PLOT_ERROR_FLOOR,
            "plot_policy": {
                "explicit_user_requested_combined_six_regime_layout": True,
                "layout": "two_holstein_rows_by_three_hubbard_columns",
                "shared_y_axis_by_holstein_row": True,
                "log_y": True,
                "solid_curves_only": True,
                "repeated_curve_markers": False,
                "marker_policy": "Paper-I primary curves retain first-common-accuracy and shared-S_alg markers; policy curves mark terminal k=50; the supplemental position-aware local curve marks its effective plateau",
                "paper_i_common_accuracy_markers": True,
                "paper_i_shared_s_alg_markers": True,
                "target_accuracy_lines": False,
                "pending_local_curves_plotted": False,
                "supplemental_authenticated_k15_diagnostic_plotted": bool(
                    report.get("supplemental_curves")
                ),
            },
            "combined_plot": copy.deepcopy(combined),
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )
    provenance_path = output_dir / f"{STABLE_STEM}_plot_provenance.json"
    _atomic_json(provenance_path, provenance)
    return {
        "combined_plot": combined,
        "provenance": _relative_binding(
            provenance_path, root=output_dir, canonical=True
        ),
        "compile_root": output_dir.as_posix(),
    }


def _tex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _fmt_error(value: Any) -> str:
    if value is None:
        return "--"
    number = float(value)
    if number == 0.0:
        return r"$0$"
    exponent = int(math.floor(math.log10(abs(number))))
    mantissa = number / 10**exponent
    return rf"${mantissa:.3f}\!\times\!10^{{{exponent}}}$"


def _fmt_int(value: Any) -> str:
    return "--" if value is None else f"{int(value):,}"


def _color_metric(
    rendered: str, *, comparison: Mapping[str, Any], key: str, local: bool
) -> str:
    if not local:
        return rendered
    direction = comparison.get("metric_directions", {}).get(key)
    color = {"better": "LocalGreen", "worse": "LocalRed", "tie": "TieGray"}.get(
        direction
    )
    return rendered if color is None else rf"\textcolor{{{color}}}{{{rendered}}}"


def _verdict_label(value: str) -> str:
    return {
        "local_pareto_better": r"\textcolor{LocalGreen}{local Pareto-better}",
        "chtc_dominates": r"\textcolor{LocalRed}{CHTC dominates}",
        "tradeoff": r"\textcolor{TradeBlue}{trade-off}",
        "equivalent": r"\textcolor{TieGray}{equivalent}",
        "pending": r"\textcolor{PendingBlue}{pending}",
    }.get(value, _tex_escape(value))


def _table_rows(rows: Sequence[Mapping[str, Any]]) -> str:
    lines: list[str] = []
    for row in rows:
        historical = row["historical"]
        comparison = row["comparison"]
        label = rf"{_tex_escape(row['regime_label'])} / {_tex_escape(row['method_label'])}"
        historical_origin = (
            "Paper-I local"
            if historical.get("origin") == "Paper-I local candidate"
            else "CHTC"
        )
        historical_signal = (
            "current Paper-I local"
            if historical_origin == "Paper-I local"
            else "recorded CHTC"
        )
        lines.append(
            " & ".join(
                [
                    label,
                    rf"\cellcolor{{gray!12}}{historical_origin}",
                    _fmt_error(historical["delta_e"]),
                    *[_fmt_int(historical[key]) for key in COST_KEYS],
                    rf"\cellcolor{{gray!12}}{historical_signal}",
                ]
            )
            + r" \\"
        )
        local = row.get("local")
        if local is None:
            if row.get("local_state") == "already_current_paper_i_local_no_distinct_rerun":
                lines.append(r"\addlinespace[1.5pt]")
                continue
            state = _tex_escape(row.get("local_state", "pending")).replace(
                r"interrupted\_guard\_memory", "interrupted: memory guard"
            )
            state = state.replace(
                r"already\_current\_paper\_i\_local\_no\_distinct\_rerun",
                "already current Paper-I local",
            )
            state = state.replace(
                r"no\_distinct\_local\_rerun\_scheduled",
                "no distinct rerun scheduled",
            )
            lines.append(
                " & ".join(
                    [
                        "",
                        r"\textcolor{PendingBlue}{new local}",
                        "--",
                        "--",
                        "--",
                        "--",
                        "--",
                        "--",
                        rf"\textcolor{{PendingBlue}}{{{state}}}",
                    ]
                )
                + r" \\"
            )
        else:
            values = [
                _color_metric(
                    _fmt_error(local["delta_e"]),
                    comparison=comparison,
                    key="delta_e",
                    local=True,
                )
            ] + [
                _color_metric(
                    _fmt_int(local[key]),
                    comparison=comparison,
                    key=key,
                    local=True,
                )
                for key in COST_KEYS
            ]
            lines.append(
                " & ".join(
                    [
                        "",
                        r"\textbf{local}",
                        *values,
                        _verdict_label(comparison["classification"]),
                    ]
                )
                + r" \\"
            )
        lines.append(r"\addlinespace[1.5pt]")
    return "\n".join(lines)


def _combined_plot_tex_path(plot_artifacts: Mapping[str, Any] | None) -> str:
    if plot_artifacts is None:
        raw = f"{STABLE_STEM}_six_regime_combined_convergence.pdf"
    else:
        plot = plot_artifacts.get("combined_plot")
        pdf = plot.get("pdf") if isinstance(plot, Mapping) else None
        raw = pdf.get("path") if isinstance(pdf, Mapping) else None
        if not isinstance(raw, str) or not raw:
            raise ReportError("combined plot PDF binding is absent")
        if not Path(raw).is_absolute():
            compile_root = plot_artifacts.get("compile_root")
            if not isinstance(compile_root, str) or not compile_root:
                raise ReportError("combined plot compile root is absent")
            raw = (Path(compile_root) / raw).as_posix()
    return rf"\detokenize{{{raw}}}"


def render_latex(
    report: Mapping[str, Any], *, plot_artifacts: Mapping[str, Any] | None = None
) -> str:
    method_order = (
        "ra_plateau",
        "ra_append_only",
        "ra_always_cr",
        "append_conventional",
    )
    regime_order = {regime: index for index, (regime, _label, _nph) in enumerate(REGIMES)}
    rows = list(report["cells"])
    rows.sort(
        key=lambda row: (
            regime_order[str(row["regime"])],
            method_order.index(str(row["method"])),
        )
    )
    if len(rows) != 24:
        raise ReportError("LaTeX consolidated table is not exactly 24 cells")
    return rf"""\documentclass[9pt]{{article}}
\usepackage[letterpaper,landscape,margin=0.45in]{{geometry}}
\usepackage{{booktabs,array,xcolor,colortbl,graphicx,hyperref,microtype}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern}}
\definecolor{{LocalGreen}}{{HTML}}{{177245}}
\definecolor{{LocalRed}}{{HTML}}{{B42318}}
\definecolor{{TradeBlue}}{{HTML}}{{175CD3}}
\definecolor{{PendingBlue}}{{HTML}}{{026AA2}}
\definecolor{{TieGray}}{{HTML}}{{667085}}
\hypersetup{{hidelinks}}
\setlength{{\parindent}}{{0pt}}
\pagestyle{{plain}}
\begin{{document}}
\section*{{Current Paper-I curves vs. new local reruns: six-regime convergence}}
\begin{{center}}
\footnotesize\textbf{{PROVISIONAL DIAGNOSTIC --- no automatic manuscript replacement}}\\[-1pt]
$L=2$, open boundary, binary phonons; Powell/200; ADAPT and transpiler seeds 7; decision rows use fixed accepted prefix $k=50$; the supplemental position-aware strong--weak curve closes at $k=15$; same-cutoff ED; Qiskit \texttt{{table\_i\_basis\_gate\_transpile\_v1}}.
\end{{center}}
\vspace{{-0.35em}}
\begin{{center}}
\includegraphics[width=0.99\textwidth,height=5.45in,keepaspectratio]{{{_combined_plot_tex_path(plot_artifacts)}}}
\end{{center}}
\vspace{{-0.45em}}
\noindent\tiny Top row: weak Holstein sector ($n_{{\rm ph}}=3$); bottom row: strong Holstein sector ($n_{{\rm ph}}=7$). Columns are weak, intermediate, and strong Hubbard regimes. Every panel contains the complete four-method Paper-I set: plateau-triggered RA-ADAPT, RA append-only, RA always-insertion, and conventional Append-ADAPT. Curves use same-cutoff $|\Delta E|$; the Paper-I comparison curves extend through $k=50$. CHTC curves use lighter lines; authenticated local closures use heavier lines. In the strong--weak panel, the heavy orange $k\leq15$ overlay is the authenticated position-aware Phase-0 RA always-open diagnostic; its filled orange diamond marks the effective plateau prefix and it is excluded from the $k=50$ decision table. Open circles retain the Paper-I first common-accuracy crossings and filled triangles retain the best errors within shared $S_{{\rm alg}}$.

\clearpage
\section*{{$k=50$ local-versus-Paper-I decision table}}
\begin{{center}}
\fontsize{{5.45}}{{5.9}}\selectfont
\setlength{{\tabcolsep}}{{2.3pt}}
\renewcommand{{\arraystretch}}{{0.78}}
\begin{{tabular*}}{{0.985\textwidth}}{{@{{\extracolsep{{\fill}}}}>{{\raggedright\arraybackslash}}p{{4.0cm}}lrrrrrr>{{\raggedright\arraybackslash}}p{{3.0cm}}@{{}}}}
\toprule
Regime / method & Origin & $|\Delta E|$ & $N_{{2q}}$ & $D_{{2q}}$ & $D_c$ & $W_{{1q}}$ & $S_{{\rm alg}}$ & Diagnostic signal \\
\midrule
{_table_rows(rows)}
\bottomrule
\end{{tabular*}}
\end{{center}}
\vspace{{0.15em}}
\noindent\footnotesize Green new-local entries are lower, red are higher, and gray are tied under the preregistered tolerance. The displayed Paper-I baseline contains 19 CHTC trajectories and five completed local strong-Holstein policy candidates; those five are not mislabeled as CHTC or compared against themselves. Pending or interrupted reruns retain the current Paper-I curve. Route-level signals are descriptive only; replacement remains a user decision.

\clearpage
\section*{{Parameter and provenance manifest}}
\small
\begin{{tabular}}{{@{{}}p{{5.0cm}}p{{11.7cm}}@{{}}}}
\toprule
Model & Hubbard--Holstein, $L=2$, open boundary, binary phonon encoding \\
Optimizer / seeds & Powell, maximum 200 iterations; ADAPT seed 7; transpiler seed 7 \\
Reporting horizon & Decision table: accepted controller round $k=50$; supplemental strong--weak position-aware Phase-0 overlay: authenticated $k=15$ diagnostic \\
Energy reference & Exact diagonalization at the same phonon cutoff ($n_{{\rm ph}}=3$ weak sector; $n_{{\rm ph}}=7$ strong sector) \\
Qiskit convention & \texttt{{table\_i\_basis\_gate\_transpile\_v1}}, optimization level 0 \\
Evidence policy & Full 24-curve Paper-I display matrix plus one provenance-bound diagnostic overlay; per-cell authenticated closure; the $k=15$ overlay is excluded from endpoint replacement decisions \\
Adoption policy & Report is provisional; no paper/manuscript evidence is adopted or replaced automatically \\
\bottomrule
\end{{tabular}}

\vspace{{0.8em}}
\textbf{{Mixed-origin Paper-I baseline.}} Nineteen displayed trajectories are authenticated CHTC results. The five strong-Holstein policy trajectories that completed the current Paper-I figure are authenticated local candidates and are labeled separately; they are never presented as CHTC results or compared against themselves. The weak--weak historical conventional-Append record has complete sealed archive authority but lacks an independent remote/local retrieval receipt.

\vspace{{0.8em}}
\textbf{{Machine-readable companion.}} The revision directory contains the complete JSON manifest and CSV table, including execution IDs, source hashes, local states, exact metric deltas, comparison tolerance, and dependency identity. Explicit authority remains \texttt{{paper\_adoption\_authorized=false}} and \texttt{{paper\_evidence\_adoption\_authorized=false}}.
\end{{document}}
"""


def _csv_bytes(report: Mapping[str, Any]) -> bytes:
    buffer = io.StringIO(newline="")
    fields = [
        "key",
        "group",
        "method",
        "regime",
        "nph",
        "origin",
        "local_state",
        *METRIC_KEYS,
        "classification",
        "execution_id",
        "qiskit_version",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for row in report["cells"]:
        baseline_origin = (
            "Paper-I local"
            if row["historical"].get("origin") == "Paper-I local candidate"
            else "CHTC"
        )
        for origin, metrics in (
            (baseline_origin, row["historical"]),
            ("new local", row["local"]),
        ):
            output = {
                "key": row["key"],
                "group": row["group"],
                "method": row["method"],
                "regime": row["regime"],
                "nph": row["nph"],
                "origin": origin,
                "local_state": (
                    row["local_state"]
                    if origin == "new local"
                    else "current_paper_i_curve"
                ),
                "classification": row["comparison"]["classification"],
            }
            if metrics is not None:
                for key in METRIC_KEYS:
                    output[key] = metrics[key]
                output["execution_id"] = metrics["execution_id"]
                output["qiskit_version"] = metrics.get("qiskit_version")
            writer.writerow(output)
    for curve in report.get("supplemental_curves", []):
        terminal = curve["terminal"]
        writer.writerow(
            {
                "key": curve["key"],
                "group": "Supplemental diagnostic overlay",
                "method": curve["method"],
                "regime": curve["regime"],
                "nph": curve["nph"],
                "origin": curve["origin"],
                "local_state": "authenticated_k15_diagnostic_not_k50_decision",
                "delta_e": terminal["delta_e"],
                "N2q": terminal["N2q"],
                "D2q": terminal["D2q"],
                "Dc": terminal["Dc"],
                "W1q": "",
                "S_alg": terminal["S_alg"],
                "classification": "supplemental_not_in_k50_decision",
                "execution_id": curve["execution_id"],
                "qiskit_version": "",
            }
        )
    return buffer.getvalue().encode("utf-8")


def _compile_tex(tex_path: Path, *, output_dir: Path) -> Path:
    system_latexmk = Path("/Library/TeX/texbin/latexmk")
    latexmk = (
        system_latexmk.as_posix()
        if system_latexmk.is_file()
        else (shutil.which("latexmk") or "/Users/jakestrobel/.local/bin/latexmk")
    )
    command = [
        latexmk,
        "-pdf",
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-outdir={output_dir}",
        tex_path.as_posix(),
    ]
    env = os.environ.copy()
    # MacTeX's latexmk driver invokes pdflatex by basename.  Remote-runner
    # launchd jobs intentionally receive a narrow PATH, so finding latexmk by
    # its absolute path is not enough: its child would still fail with
    # ``pdflatex: command not found``.  Keep the inherited PATH but prepend the
    # fixed MacTeX tool directory whenever it is installed.
    system_tex_bin = Path("/Library/TeX/texbin")
    if system_tex_bin.is_dir():
        inherited_path = env.get("PATH", "")
        path_entries = [
            entry for entry in inherited_path.split(os.pathsep) if entry
        ]
        tex_bin = system_tex_bin.as_posix()
        env["PATH"] = os.pathsep.join(
            [tex_bin, *(entry for entry in path_entries if entry != tex_bin)]
        )
    env["TEXMFOUTPUT"] = output_dir.as_posix()
    isolated_texmf = output_dir / ".isolated-texmf-home"
    isolated_texmf.mkdir()
    env["TEXMFHOME"] = isolated_texmf.as_posix()
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
    finally:
        shutil.rmtree(isolated_texmf, ignore_errors=True)
    pdf = output_dir / f"{tex_path.stem}.pdf"
    if completed.returncode != 0 or not pdf.is_file() or pdf.stat().st_size < 10_000:
        tail = "\n".join((completed.stdout + "\n" + completed.stderr).splitlines()[-80:])
        raise ReportError(f"LaTeX report build failed:\n{tail}")
    pdfinfo = shutil.which("pdfinfo")
    if pdfinfo:
        check = subprocess.run(
            [pdfinfo, pdf.as_posix()], capture_output=True, text=True
        )
        if check.returncode != 0 or "Pages:" not in check.stdout:
            raise ReportError("generated PDF did not pass pdfinfo")
    return pdf


def _same_published_revision(evidence_sha: str) -> bool:
    if not LATEST_JSON.exists() and not LATEST_JSON.is_symlink():
        return False
    try:
        latest = _load_digested(LATEST_JSON, label="latest report pointer")
    except ReportError:
        return False
    return (
        latest.get("schema") == LATEST_SCHEMA
        and latest.get("evidence_revision_sha256") == evidence_sha
        and STABLE_PDF.is_file()
        and not STABLE_PDF.is_symlink()
        and latest.get("stable_pdf", {}).get("sha256") == _sha256_file(STABLE_PDF)
    )


def publish_report(report: Mapping[str, Any], *, force: bool = False) -> dict[str, Any]:
    evidence_sha = str(report["evidence_revision_sha256"])
    if not force and _same_published_revision(evidence_sha):
        return {
            "status": "unchanged",
            "evidence_revision_sha256": evidence_sha,
            "pdf": STABLE_PDF.as_posix(),
        }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REVISIONS_DIR.mkdir(parents=True, exist_ok=True)
    revision_dir = REVISIONS_DIR / evidence_sha
    if not revision_dir.exists():
        staging = REVISIONS_DIR / f".{evidence_sha}.in_progress.{os.getpid()}"
        if staging.exists() or staging.is_symlink():
            raise ReportError(f"stale report staging directory exists: {staging}")
        staging.mkdir()
        try:
            plot_bundle = _render_combined_plot(report, output_dir=staging)
            tex_path = staging / f"{STABLE_STEM}.tex"
            tex_path.write_text(
                render_latex(report, plot_artifacts=plot_bundle), encoding="utf-8"
            )
            csv_path = staging / f"{STABLE_STEM}.csv"
            csv_path.write_bytes(_csv_bytes(report))
            built_pdf = _compile_tex(tex_path, output_dir=staging)
            manifest_unsigned = {
                **copy.deepcopy(dict(report)),
                "revision_files": {
                    "pdf": _relative_binding(built_pdf, root=staging),
                    "tex": _relative_binding(tex_path, root=staging),
                    "csv": _relative_binding(csv_path, root=staging),
                    "combined_plot": copy.deepcopy(plot_bundle["combined_plot"]),
                    "plot_provenance": copy.deepcopy(plot_bundle["provenance"]),
                },
            }
            manifest = _digested(manifest_unsigned)
            manifest_path = staging / f"{STABLE_STEM}.json"
            _atomic_json(manifest_path, manifest)
            plot_paths = [
                staging / str(plot_bundle["combined_plot"][kind]["path"])
                for kind in ("pdf", "png")
            ]
            plot_paths.append(
                staging / str(plot_bundle["provenance"]["path"])
            )
            for path in (tex_path, csv_path, built_pdf, manifest_path, *plot_paths):
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
            os.rename(staging, revision_dir)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    revision_pdf = revision_dir / f"{STABLE_STEM}.pdf"
    revision_manifest = revision_dir / f"{STABLE_STEM}.json"
    revision_tex = revision_dir / f"{STABLE_STEM}.tex"
    revision_csv = revision_dir / f"{STABLE_STEM}.csv"
    for path in (revision_pdf, revision_manifest, revision_tex, revision_csv):
        if not path.is_file() or path.is_symlink():
            raise ReportError(f"published revision is incomplete: {path}")
    revision_plot_provenance = (
        revision_dir / f"{STABLE_STEM}_plot_provenance.json"
    )
    combined_stem = f"{STABLE_STEM}_six_regime_combined_convergence"
    combined_pdf = revision_dir / f"{combined_stem}.pdf"
    combined_png = revision_dir / f"{combined_stem}.png"
    if (
        not combined_pdf.is_file()
        or combined_pdf.is_symlink()
        or not combined_png.is_file()
        or combined_png.is_symlink()
    ):
        raise ReportError("published combined plot revision is incomplete")
    revision_combined_plot = {
        "pdf": _binding(combined_pdf),
        "png": _binding(combined_png),
    }
    if not revision_plot_provenance.is_file() or revision_plot_provenance.is_symlink():
        raise ReportError("published plot provenance is incomplete")
    _atomic_bytes(STABLE_PDF, revision_pdf.read_bytes())
    for suffix, source in (
        (".tex", revision_tex),
        (".json", revision_manifest),
        (".csv", revision_csv),
    ):
        _atomic_bytes(OUTPUT_DIR / f"{STABLE_STEM}{suffix}", source.read_bytes())
    latest = _digested(
        {
            "schema": LATEST_SCHEMA,
            "status": "passed_atomic_latest_publication",
            "published_at_utc": _utc_now(),
            "evidence_revision_sha256": evidence_sha,
            "revision_directory": revision_dir.resolve().as_posix(),
            "stable_pdf": _binding(STABLE_PDF),
            "manifest": _binding(
                OUTPUT_DIR / f"{STABLE_STEM}.json", canonical=True
            ),
            "csv": _binding(OUTPUT_DIR / f"{STABLE_STEM}.csv"),
            "tex": _binding(OUTPUT_DIR / f"{STABLE_STEM}.tex"),
            "combined_plot": revision_combined_plot,
            "plot_provenance": _binding(
                revision_plot_provenance, canonical=True
            ),
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )
    _atomic_json(LATEST_JSON, latest)
    return {
        "status": "published",
        "evidence_revision_sha256": evidence_sha,
        "pdf": STABLE_PDF.as_posix(),
        "latest": LATEST_JSON.as_posix(),
        "completed": report["counts"]["completed"],
        "campaign_state": report["campaign"]["state"],
    }


def build_once(*, force: bool = False) -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return publish_report(collect_report(), force=force)


def _watch_status(payload: Mapping[str, Any]) -> None:
    status = _digested(
        {
            "schema": WATCH_SCHEMA,
            "updated_at_utc": _utc_now(),
            **copy.deepcopy(dict(payload)),
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )
    _atomic_json(WATCH_STATUS, status)
    artifact_dir = os.environ.get("REMOTE_ARTIFACT_DIR")
    if artifact_dir:
        target = Path(artifact_dir) / "paper_i_local_vs_chtc_report_watch_status.json"
        _atomic_json(target, status)


def watch(*, poll_seconds: float, max_poll_seconds: float) -> None:
    try:
        os.nice(10)
    except OSError:
        pass
    delay = poll_seconds
    while True:
        try:
            result = build_once()
            _watch_status({"status": "watching", "last_build": result})
            delay = poll_seconds
        except Exception as exc:
            _watch_status(
                {
                    "status": "report_refresh_failed_will_retry",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            delay = min(max_poll_seconds, max(poll_seconds, delay * 2.0))
        time.sleep(delay)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--max-poll-seconds", type=float, default=300.0)
    args = parser.parse_args(argv)
    if args.poll_seconds < 10.0 or args.max_poll_seconds < args.poll_seconds:
        parser.error("poll interval must be at least 10 seconds and max >= initial")
    if args.watch:
        watch(
            poll_seconds=float(args.poll_seconds),
            max_poll_seconds=float(args.max_poll_seconds),
        )
        return 0
    print(json.dumps(build_once(force=bool(args.force)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
