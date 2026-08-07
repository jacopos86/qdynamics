#!/usr/bin/env python3
"""Build the Paper-I HH Pauli-child fairness comparison PDF."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from functools import lru_cache
import os
import shutil
import struct
import subprocess
import sys
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_SUPPORT_JSON = (
    REPO_ROOT
    / "MATH"
    / "paper_facing"
    / "paper_I_static_scaffold"
    / "paper_i_hh_native200_manuscript_update_20260619.json"
)
DEFAULT_QISKIT_ROWS_JSON = REPO_ROOT / "output" / "pdf" / "paper_i_hh_native200_qiskit_poster_plateau_rows_20260622.json"
DEFAULT_CHILD_RECORDS_TSV = (
    REPO_ROOT
    / "chtc"
    / "phase3_optuna"
    / "input"
    / "paper_i_hh_native_forced_child_matrix_depth30_20260623_v1"
    / "paper_i_hh_spsa_budget_ladder_records.tsv"
)
DEFAULT_MONOTONE_RECORDS_TSV = (
    REPO_ROOT
    / "chtc"
    / "phase3_optuna"
    / "input"
    / "paper_i_hh_monotone_child_schur_depth30_20260623_v1"
    / "paper_i_hh_spsa_budget_ladder_records.tsv"
)
DEFAULT_SNAKE_SFAIR_RECORDS_TSV = (
    REPO_ROOT
    / "chtc"
    / "phase3_optuna"
    / "input"
    / "paper_i_hh_snake_sfair_depth30_20260623_v1"
    / "paper_i_hh_spsa_budget_ladder_records.tsv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "pdf" / "paper_i_hh_child_fairness_20260623"
DEFAULT_STEM = "paper_i_hh_child_fairness_incremental_20260623"
DEFAULT_COMPACT_STEM = "paper_i_hh_child_fairness_compact_child_schur_20260623"
DEFAULT_LOCAL_COSTS_JSON = DEFAULT_OUTPUT_DIR / "paper_i_hh_child_fairness_local_qiskit_costs_20260623.json"
SNAKE_TERMINAL_QISKIT_COST_SCHEMA = "paper_i_hh_child_fairness_snake_terminal_qiskit_cost_v2"
SNAKE_TERMINAL_WORK_SEMANTICS_VERSION = "snake_terminal_s_alg_winner_lineage_v1"
SNAKE_TERMINAL_BEAM_ROW_POLICY = "beam_terminal_winner_history_v1"
SNAKE_TERMINAL_BEAM_AGGREGATE_SCOPE = "all_expanded_scored_branches"
SNAKE_TERMINAL_WINNER_WORK_SCOPES = {"winner_lineage_terminal", "winner_lineage_only"}
DEFAULT_SCHUR_OVERLAY_JSON = (
    REPO_ROOT
    / "output"
    / "pdf"
    / "paper_i_hh_schur_warm_start_native200_depth30_overlay_20260623"
    / "paper_i_hh_schur_warm_start_native200_depth30_overlay_20260623.json"
)
DEFAULT_POWELL_PARTIAL_SUMMARY_JSON = (
    REPO_ROOT
    / "output"
    / "chtc_retrievals"
    / "paper_i_hh_powell_all_regimes_20260625_current_17of18_20260625T2229Z"
    / "tmp"
    / "paper_i_hh_powell_all_regimes_20260625_current_17of18_20260625T2229Z"
    / "paper_i_hh_powell_all_regimes_current_summary.json"
)
DEFAULT_POWELL_TUNING_SUMMARY_JSON = (
    REPO_ROOT
    / "output"
    / "pdf"
    / "paper_i_hh_powell_snake_tuning_20260625"
    / "paper_i_hh_powell_snake_tuning_summary_20260625.json"
)
DEFAULT_POWELL_TUNING_OVERLAY_PNG = (
    REPO_ROOT
    / "output"
    / "pdf"
    / "paper_i_hh_powell_weak_weak_overlay_20260625"
    / "weak_weak_powell_error_vs_iteration_overlay_with_tuned_snake.png"
)
DEFAULT_OPTIMIZER_CROSSCHECK_PREFLIGHT_JSON = (
    REPO_ROOT
    / "output"
    / "pdf"
    / "paper_i_hh_spsa_budget_ladder_optimizer_crosscheck_all_regimes_20260625_v1_preflight.json"
)
DEFAULT_RETRIEVED_APPEND_GEO_DIR = (
    REPO_ROOT
    / "raw_outputs"
    / "chtc_retrievals"
    / "paper_i_hh_finished_append_geo_20260626"
)
DEFAULT_LOCAL_SNAKE_ROTOSOLVE_DIR = (
    REPO_ROOT
    / "raw_outputs"
    / "local_parallel_snake_rotosolve_childpoolfix_20260626"
)

REGIME_ORDER = (
    "weak-weak",
    "intermediate-weak",
    "strong-weak",
    "weak-strong",
    "intermediate-strong",
    "strong-strong",
)
REGIME_DISPLAY = {
    "weak-weak": "weak--weak",
    "intermediate-weak": "intermediate--weak",
    "strong-weak": "strong--weak",
    "weak-strong": "weak--strong",
    "intermediate-strong": "intermediate--strong",
    "strong-strong": "strong--strong",
}
METHOD_STYLE = {
    "Append-ADAPT": {"color": "#4C78A8", "marker": "o", "markersize": 7.5},
    "Geo-ADAPT": {"color": "#54A24B", "marker": "^", "markersize": 8.0},
    "SNAKE": {"color": "#E45756", "marker": "*", "markersize": 12.0},
}
CURRENT_METHODS = ("Append-ADAPT", "Geo-ADAPT", "SNAKE")
METHOD_BY_KEY = {"append": "Append-ADAPT", "geo": "Geo-ADAPT", "snake": "SNAKE"}
METHOD_KEY_BY_DISPLAY = {value: key for key, value in METHOD_BY_KEY.items()}
SCHUR_VARIANT_KEY = "snake_schur_child"
VARIANT_ORDER = (
    "append_no_child",
    "append_child",
    "geo_no_child",
    "geo_child",
    "snake_no_child",
    "snake_child",
    SCHUR_VARIANT_KEY,
)
VARIANT_LABELS = {
    "append_no_child": "Append no child",
    "append_child": "Append child",
    "geo_no_child": "Geo no child",
    "geo_child": "Geo child",
    "snake_no_child": "SNAKE no child",
    "snake_child": "SNAKE child",
    SCHUR_VARIANT_KEY: "SNAKE child+Schur",
}
VARIANT_METHOD = {
    "append_no_child": "Append-ADAPT",
    "append_child": "Append-ADAPT",
    "geo_no_child": "Geo-ADAPT",
    "geo_child": "Geo-ADAPT",
    "snake_no_child": "SNAKE",
    "snake_child": "SNAKE",
    SCHUR_VARIANT_KEY: "SNAKE",
}
VARIANT_BY_METHOD_CHILD = {
    ("Append-ADAPT", "no_child"): "append_no_child",
    ("Append-ADAPT", "polychildren"): "append_child",
    ("Geo-ADAPT", "no_child"): "geo_no_child",
    ("Geo-ADAPT", "polychildren"): "geo_child",
    ("SNAKE", "no_child"): "snake_no_child",
    ("SNAKE", "polychildren"): "snake_child",
}
MONOTONE_VARIANT_ORDER = (
    "monotone_append_no_child",
    "monotone_append_child",
    "monotone_geo_no_child",
    "monotone_geo_child",
    "monotone_snake_no_child",
    "monotone_snake_child",
)
MONOTONE_VARIANT_LABELS = {
    "monotone_append_no_child": "Append no child",
    "monotone_append_child": "Append child",
    "monotone_geo_no_child": "Geo no child",
    "monotone_geo_child": "Geo child",
    "monotone_snake_no_child": "SNAKE no child",
    "monotone_snake_child": "SNAKE child",
}
MONOTONE_VARIANT_METHOD = {
    "monotone_append_no_child": "Append-ADAPT",
    "monotone_append_child": "Append-ADAPT",
    "monotone_geo_no_child": "Geo-ADAPT",
    "monotone_geo_child": "Geo-ADAPT",
    "monotone_snake_no_child": "SNAKE",
    "monotone_snake_child": "SNAKE",
}
MONOTONE_VARIANT_BY_METHOD_CHILD = {
    ("Append-ADAPT", "no_child"): "monotone_append_no_child",
    ("Append-ADAPT", "polychildren"): "monotone_append_child",
    ("Geo-ADAPT", "no_child"): "monotone_geo_no_child",
    ("Geo-ADAPT", "polychildren"): "monotone_geo_child",
    ("SNAKE", "no_child"): "monotone_snake_no_child",
    ("SNAKE", "polychildren"): "monotone_snake_child",
}
PAULI_CHILD_MODE = "shortlist_pauli_children_v1"
SNAKE_GLOBAL_CHILD_POOL_MODES = {"global_pauli_child_sets_v1", "pauli_child_sets_v1"}
FAIR_WORK_CURRENCY = "expanded_common_candidate_probe_event_count_v1"
SNAKE_CORRECTED_CHILD_POOL_BLOCKER = "snake_global_pauli_child_pool_rerun_required"
RETRIEVED_OPTIMIZERS = ("powell", "rotosolve")
RETRIEVED_METHODS = ("append", "geo", "snake")


@dataclass(frozen=True)
class Series:
    variant_key: str
    label: str
    method: str
    status: str
    points: tuple[tuple[int, float], ...]
    marker_k: int | None
    marker_y: float | None
    marker_policy: str
    delta_e: float | None
    source_json: str | None
    source_sha256: str | None
    blocker: str | None = None


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=512)
def _sha256_cached(path_key: str, mtime_ns: int, size: int) -> str:
    h = hashlib.sha256()
    with Path(path_key).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256(path: Path) -> str:
    resolved = path.resolve()
    stat = resolved.stat()
    return _sha256_cached(str(resolved), int(stat.st_mtime_ns), int(stat.st_size))


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _fmt_sci(value: float | str | None) -> str:
    parsed = _float_or_none(value)
    if parsed is None:
        return "--"
    return f"{parsed:.3e}"


def _fmt_int(value: Any) -> str:
    parsed = _int_or_none(value)
    if parsed is None:
        return "--"
    return f"{parsed:d}"


def _fmt_fidelity(value: Any) -> str:
    parsed = _float_or_none(value)
    if parsed is None:
        return "--"
    return f"{min(1.0, max(0.0, parsed)):.6f}"


def _fmt_compact_int(value: Any) -> str:
    parsed = _int_or_none(value)
    if parsed is None:
        return "--"
    return f"{parsed:d}"


def _tex_escape(value: object) -> str:
    text = str(value)
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def _tex_comment_block(name: str, payload: Mapping[str, Any]) -> list[str]:
    lines = [f"% BEGIN_{name}"]
    for line in json.dumps(payload, indent=2, sort_keys=True).splitlines():
        lines.append(f"% {line}")
    lines.append(f"% END_{name}")
    return lines


def _report_tex_provenance_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "paper_i_hh_child_fairness_tex_provenance_v1",
        "generated_utc": manifest.get("generated_utc"),
        "pdf": manifest.get("pdf"),
        "tex": manifest.get("tex"),
        "manifest_json": manifest.get("manifest_json"),
        "sidecars": manifest.get("sidecars"),
        "pool_exposure_policy": manifest.get("pool_exposure_policy"),
        "snake_child_pool_repair_status": manifest.get("snake_child_pool_repair_status"),
        "page_semantics": manifest.get("page_semantics"),
    }


def _latex_path(path: Path) -> str:
    return str(path).replace(os.sep, "/")


def _graphics_path(tex_path: Path, figure_path: Path) -> str:
    try:
        rel = figure_path.resolve().relative_to(tex_path.parent.resolve())
        return _latex_path(rel)
    except ValueError:
        return _latex_path(figure_path.resolve())


def _trajectory(row: Mapping[str, Any]) -> tuple[tuple[int, float], ...]:
    points: list[tuple[int, float]] = []
    raw_points = row.get("trajectory")
    if not isinstance(raw_points, Sequence):
        return ()
    for point in raw_points:
        if not isinstance(point, Sequence) or len(point) < 2:
            continue
        x = _int_or_none(point[0])
        y = _float_or_none(point[1])
        if x is not None and y is not None and y > 0.0:
            points.append((int(x), float(y)))
    return tuple(sorted(points))


def _configure_matplotlib() -> tuple[Any, Any, Any]:
    os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "tmp" / "matplotlib_config"))
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "axes.formatter.use_mathtext": False,
        }
    )
    return Figure, FigureCanvasAgg, Line2D


def _load_current_rows(path: Path) -> dict[tuple[str, str], Mapping[str, Any]]:
    payload = _read_json(path)
    out: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in payload.get("rows", []):
        if isinstance(row, Mapping):
            out[(str(row.get("regime")), str(row.get("method")))] = row
    return out


def _load_qiskit_rows(path: Path) -> dict[tuple[str, str], Mapping[str, Any]]:
    method_map = {"Append": "Append-ADAPT", "Geo": "Geo-ADAPT", "SNAKE": "SNAKE"}
    payload = _read_json(path)
    out: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in payload.get("rows", []):
        if not isinstance(row, Mapping):
            continue
        method = method_map.get(str(row.get("method")), str(row.get("method")))
        out[(str(row.get("regime")), method)] = row
    return out


def _load_child_records(path: Path) -> dict[tuple[str, str, str], Mapping[str, str]]:
    out: dict[tuple[str, str, str], Mapping[str, str]] = {}
    if not path.exists():
        return out
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            method = METHOD_BY_KEY.get(str(row.get("method_key") or ""))
            if method is None:
                continue
            normalized = {
                str(k): "" if v is None else str(v) for k, v in row.items()
            }
            out[(str(row.get("display_regime")), method, _record_child_mode(normalized))] = normalized
    return out


def _load_snake_sfair_repair_records(
    path: Path,
    *,
    engine_key: str,
) -> dict[tuple[str, str, str], Mapping[str, str]]:
    """Load completed SNAKE S-fair repair records as page-specific overrides."""

    out: dict[tuple[str, str, str], Mapping[str, str]] = {}
    if not path.exists():
        return out
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if str(row.get("method_key") or "") != "snake":
                continue
            if str(row.get("engine_key") or "") != engine_key:
                continue
            normalized = {str(k): "" if v is None else str(v) for k, v in row.items()}
            out[(str(row.get("display_regime")), "SNAKE", _record_child_mode(normalized))] = normalized
    return out


def _load_local_cost_rows(path: Path) -> dict[tuple[str, str, str, str], Mapping[str, Any]]:
    if not path.exists():
        return {}
    payload = _read_json(path)
    out: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    rows = payload.get("rows") if isinstance(payload, Mapping) else None
    if not isinstance(rows, Sequence):
        return out
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        row = _refresh_snake_local_cost_row(row)
        key = (
            str(row.get("page") or ""),
            str(row.get("regime") or ""),
            str(row.get("method") or ""),
            str(row.get("child_policy") or ""),
        )
        out[key] = row
    return out


def _resolve_local_path(raw: Any) -> Path | None:
    if raw is None or raw == "":
        return None
    path = Path(str(raw))
    if path.exists():
        return path
    candidate = REPO_ROOT / path
    if candidate.exists():
        return candidate
    return path


def _refresh_snake_local_cost_row(row: Mapping[str, Any]) -> Mapping[str, Any]:
    if str(row.get("method") or "").lower() != "snake":
        return row
    out = dict(row)
    unfair_value = out.get("S")
    out["S_unfair"] = unfair_value
    out["S_unfair_status"] = "ok:mixed_work_currency" if _int_or_none(unfair_value) is not None else str(out.get("status") or "")
    out["S_unfair_source_kind"] = out.get("S_source_kind") or "legacy_or_mixed_snake_shot_proxy"
    out["S_fair"] = None
    out["S_fair_status"] = "blocked:mixed_work_currency"
    out["S_fair_source_kind"] = "snake_expanded_child_probe_ledger_missing"
    source_path = _resolve_local_path(out.get("source_json"))
    if source_path is None or not source_path.exists():
        out.setdefault("S_alg_status", "source_json_missing")
        out["S_fair_status"] = str(out.get("status") or "source_json_missing")
        return out
    try:
        sidecar = _read_snake_grouped_sidecar(source_path)
    except Exception as exc:
        out["S_ctrl_error"] = str(exc)
        return out
    if sidecar:
        out.update(
            S_ctrl_grouped=sidecar.get("S_alg"),
            S_ctrl_grouped_status=sidecar.get("S_alg_status"),
            S_ctrl_grouped_source_json=sidecar.get("source_json"),
            S_ctrl_grouped_source_sha256=sidecar.get("source_sha256"),
            S_grouped=sidecar.get("S_alg"),
            S_grouped_status=sidecar.get("S_alg_status"),
        )
    fair_sidecar = _read_snake_fair_sidecar(source_path)
    if fair_sidecar:
        out.update(
            S_actual=fair_sidecar.get("S_actual"),
            S_actual_status=fair_sidecar.get("S_actual_status"),
            S_common_exposure=fair_sidecar.get("S_common_exposure"),
            S_common_exposure_status=fair_sidecar.get("S_common_exposure_status"),
            S_fair=fair_sidecar.get("S_fair"),
            S_fair_status=fair_sidecar.get("S_fair_status"),
            S_fair_policy=fair_sidecar.get("S_fair_policy"),
            S_fair_source=fair_sidecar.get("S_fair_source"),
            S_fair_source_kind=fair_sidecar.get("S_fair_source_kind"),
            S_fair_source_json=fair_sidecar.get("source_json"),
            S_fair_source_sha256=fair_sidecar.get("source_sha256"),
            fair_work_currency=fair_sidecar.get("fair_work_currency"),
            work_contract_id=fair_sidecar.get("work_contract_id"),
        )
    return out


def _read_snake_grouped_sidecar(source_path: Path) -> dict[str, Any] | None:
    candidates: list[Path] = []
    if source_path.is_file():
        candidates.extend(
            [
                source_path.parent / "snake_algorithmic_work.json",
                source_path.parent.parent / "snake_algorithmic_work.json",
            ]
        )
    for candidate in candidates:
        if not candidate.exists():
            continue
        payload = _read_json(candidate)
        if not isinstance(payload, Mapping):
            continue
        return {
            "S_alg": payload.get("S_alg"),
            "S_alg_status": payload.get("S_alg_status"),
            "source_json": _rel(candidate),
            "source_sha256": _sha256(candidate),
        }
    return None


def _read_snake_fair_sidecar(source_path: Path) -> dict[str, Any] | None:
    candidates: list[Path] = []
    if source_path.is_file():
        candidates.extend(
            [
                source_path.parent / "snake_fair_shot_work.json",
                source_path.parent.parent / "snake_fair_shot_work.json",
            ]
        )
    for candidate in candidates:
        if not candidate.exists():
            continue
        payload = _read_json(candidate)
        if not isinstance(payload, Mapping):
            continue
        contract_ok = (
            payload.get("work_contract_id") == "paper_i_hh_operator_probe_contract_v2"
            and payload.get("S_fair_source") == "S_common_exposure"
            and payload.get("S_fair_policy") == "trajectory_conditioned_full_child_common_exposure_v1"
        )
        fair_status = payload.get("S_fair_status")
        fair_value = payload.get("S_fair")
        if not contract_ok:
            fair_status = "blocked:legacy_fair_sidecar_revalidation_required"
            fair_value = None
        return {
            "S_actual": payload.get("S_actual"),
            "S_actual_status": payload.get("S_actual_status"),
            "S_common_exposure": payload.get("S_common_exposure"),
            "S_common_exposure_status": payload.get("S_common_exposure_status"),
            "S_fair": fair_value,
            "S_fair_status": fair_status,
            "S_fair_policy": payload.get("S_fair_policy"),
            "S_fair_source": payload.get("S_fair_source"),
            "S_fair_source_kind": payload.get("S_fair_source_kind"),
            "fair_work_currency": payload.get("fair_work_currency"),
            "work_contract_id": payload.get("work_contract_id"),
            "source_json": _rel(candidate),
            "source_sha256": _sha256(candidate),
        }
    return None


def _with_shot_columns(row: dict[str, Any], method: str) -> dict[str, Any]:
    status = str(row.get("status") or "")
    s_value = row.get("S")
    if method == "SNAKE":
        if row.get("S_unfair") is None:
            row["S_unfair"] = s_value
        row.setdefault(
            "S_unfair_status",
            "ok:mixed_work_currency" if _int_or_none(row.get("S_unfair")) is not None else status,
        )
        row.setdefault("S_unfair_source_kind", "legacy_or_mixed_snake_shot_proxy")
        row.setdefault("S_fair", None)
        if not row.get("S_fair_status"):
            row["S_fair_status"] = "blocked:mixed_work_currency" if status == "done" else status
        row.setdefault("S_fair_source_kind", "snake_expanded_child_probe_ledger_missing")
        return row
    if row.get("S_unfair") is None:
        row["S_unfair"] = s_value
    row.setdefault("S_unfair_status", "ok" if _int_or_none(row.get("S_unfair")) is not None else status)
    row.setdefault("S_unfair_source_kind", row.get("S_source_kind") or "method_reported_shot_proxy")
    if row.get("S_fair") is None:
        row["S_fair"] = s_value
    row.setdefault("S_fair_status", "ok" if _int_or_none(row.get("S_fair")) is not None else status)
    row.setdefault("S_fair_source_kind", row.get("S_source_kind") or "append_geo_expanded_probe_proxy")
    return row


def _record_child_mode(row: Mapping[str, str]) -> str:
    method_key = str(row.get("method_key") or "")
    if method_key == "snake":
        if _snake_global_child_pool_enabled(row) or _snake_runtime_child_pool_enabled(row):
            return "polychildren"
        return "no_child"
    else:
        split_mode = str(row.get("generic_adapt_runtime_split_mode") or "").strip()
        return "polychildren" if split_mode == PAULI_CHILD_MODE else "no_child"


def _snake_global_child_pool_enabled(row: Mapping[str, Any]) -> bool:
    child_pool_mode = str(
        row.get("snake_adapt_child_pool_expansion_mode")
        or row.get("adapt_child_pool_expansion_mode")
        or row.get("snake_child_pool_expansion_mode")
        or ""
    ).strip()
    return child_pool_mode in SNAKE_GLOBAL_CHILD_POOL_MODES


def _snake_runtime_child_pool_enabled(row: Mapping[str, Any]) -> bool:
    split_mode = str(
        row.get("snake_phase3_runtime_split_mode")
        or row.get("phase3_runtime_split_mode")
        or row.get("runtime_split_mode")
        or ""
    ).strip()
    return split_mode == PAULI_CHILD_MODE


def _blocked_snake_child_series(
    *,
    variant_key: str,
    label: str,
    source_json: str | None,
    source_sha256: str | None,
) -> Series:
    return Series(
        variant_key=variant_key,
        label=label,
        method="SNAKE",
        status="pending",
        points=(),
        marker_k=None,
        marker_y=None,
        marker_policy="pending_corrected_snake_child_pool",
        delta_e=None,
        source_json=source_json,
        source_sha256=source_sha256,
        blocker=SNAKE_CORRECTED_CHILD_POOL_BLOCKER,
    )


def _load_monotone_records(path: Path) -> dict[tuple[str, str, str], Mapping[str, str]]:
    out: dict[tuple[str, str, str], Mapping[str, str]] = {}
    if not path.exists():
        return out
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            method = METHOD_BY_KEY.get(str(row.get("method_key") or ""))
            if method is None:
                continue
            normalized = {str(k): "" if v is None else str(v) for k, v in row.items()}
            out[(str(row.get("display_regime")), method, _record_child_mode(normalized))] = normalized
    return out


def _load_schur_overlay(path: Path) -> dict[str, Mapping[str, Any]]:
    if not path.exists():
        return {}
    payload = _read_json(path)
    rows = payload.get("regimes") if isinstance(payload, Mapping) else None
    if not isinstance(rows, Sequence):
        return {}
    out: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        regime = str(row.get("display_regime") or row.get("internal_regime") or "")
        if regime:
            out[regime] = row
    return out


def _xy_points(payload: Mapping[str, Any]) -> tuple[tuple[int, float], ...]:
    xs = payload.get("x")
    ys = payload.get("y")
    if not isinstance(xs, Sequence) or isinstance(xs, (str, bytes)):
        return ()
    if not isinstance(ys, Sequence) or isinstance(ys, (str, bytes)):
        return ()
    points: list[tuple[int, float]] = []
    for raw_x, raw_y in zip(xs, ys):
        x = _int_or_none(raw_x)
        y = _float_or_none(raw_y)
        if x is not None and y is not None and y > 0.0:
            points.append((int(x), float(y)))
    return tuple(sorted(points))


def _first_number(mapping: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _float_or_none(mapping.get(key))
        if value is not None:
            return value
    return None


def _extract_result_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    rows = payload.get("rows")
    if isinstance(rows, Sequence) and rows and isinstance(rows[0], Mapping):
        return rows[0]
    result = payload.get("result")
    if isinstance(result, Mapping):
        return result
    adapt = payload.get("adapt_vqe")
    if isinstance(adapt, Mapping):
        return adapt
    return payload


def _child_output_root(row: Mapping[str, str], result_path: Path) -> Path:
    raw = str(row.get("record_output_dir") or "").strip()
    if raw:
        return _resolve(raw)
    return result_path.parent.parent


def _child_manifest_mismatch(row: Mapping[str, str], output_root: Path) -> str | None:
    raw_manifest = str(row.get("cell_manifest_rel") or "").strip()
    manifest_path = _resolve(raw_manifest) if raw_manifest else output_root / "cell_manifest.json"
    if not manifest_path.exists():
        return "missing_cell_manifest"
    try:
        manifest = _read_json(manifest_path)
    except Exception:
        return "unreadable_cell_manifest"
    if not isinstance(manifest, Mapping):
        return "invalid_cell_manifest"
    if str(manifest.get("status") or "") != "ok":
        return f"cell_manifest_status_{manifest.get('status') or 'unknown'}"
    env = manifest.get("env_overlay")
    if not isinstance(env, Mapping):
        return "missing_env_overlay"
    expected = {
        "generic_adapt_runtime_split_mode": "GENERIC_STATIC_TABLE_GENERIC_ADAPT_RUNTIME_SPLIT_MODE",
        "generic_adapt_runtime_split_symmetry_policy": "GENERIC_STATIC_TABLE_GENERIC_ADAPT_RUNTIME_SPLIT_SYMMETRY_POLICY",
        "generic_adapt_runtime_split_max_subset_size": "GENERIC_STATIC_TABLE_GENERIC_ADAPT_RUNTIME_SPLIT_MAX_SUBSET_SIZE",
        "generic_adapt_stop_policy": "GENERIC_STATIC_TABLE_GENERIC_ADAPT_STOP_POLICY",
        "resource_pool_term_cap": "GENERIC_STATIC_TABLE_RESOURCE_POOL_TERM_CAP",
        "resource_qubit_cap": "GENERIC_STATIC_TABLE_RESOURCE_QUBIT_CAP",
    }
    mismatches: list[str] = []
    for row_key, env_key in expected.items():
        expected_value = str(row.get(row_key) or "").strip()
        if not expected_value:
            continue
        if str(env.get(env_key) or "").strip() != expected_value:
            mismatches.append(row_key)
    if str(row.get("generic_adapt_stop_policy") or "").strip() == "fixed_horizon_no_target_v1":
        if str(env.get("GENERIC_STATIC_TABLE_ENERGY_STOP_TARGET", "unset")) != "":
            mismatches.append("energy_stop_target_not_blank")
    if mismatches:
        return "cell_manifest_env_mismatch:" + ",".join(mismatches)
    return None


def _monotone_manifest_mismatch(row: Mapping[str, str], output_root: Path) -> str | None:
    raw_manifest = str(row.get("cell_manifest_rel") or "").strip()
    manifest_path = _resolve(raw_manifest) if raw_manifest else output_root / "cell_manifest.json"
    if not manifest_path.exists():
        return "missing_cell_manifest"
    try:
        manifest = _read_json(manifest_path)
    except Exception:
        return "unreadable_cell_manifest"
    if not isinstance(manifest, Mapping):
        return "invalid_cell_manifest"
    if str(manifest.get("status") or "") != "ok":
        return f"cell_manifest_status_{manifest.get('status') or 'unknown'}"
    env = manifest.get("env_overlay")
    if not isinstance(env, Mapping):
        return "missing_env_overlay"
    mismatches: list[str] = []
    if str(env.get("ADAPT_SPSA_REFIT_ENGINE") or "").strip() != str(row.get("spsa_refit_engine") or "").strip():
        mismatches.append("spsa_refit_engine")
    method_key = str(row.get("method_key") or "")
    if method_key in {"append", "geo"}:
        expected = {
            "generic_adapt_runtime_split_mode": "GENERIC_STATIC_TABLE_GENERIC_ADAPT_RUNTIME_SPLIT_MODE",
            "generic_adapt_runtime_split_symmetry_policy": "GENERIC_STATIC_TABLE_GENERIC_ADAPT_RUNTIME_SPLIT_SYMMETRY_POLICY",
            "generic_adapt_runtime_split_max_subset_size": "GENERIC_STATIC_TABLE_GENERIC_ADAPT_RUNTIME_SPLIT_MAX_SUBSET_SIZE",
            "generic_adapt_stop_policy": "GENERIC_STATIC_TABLE_GENERIC_ADAPT_STOP_POLICY",
            "resource_pool_term_cap": "GENERIC_STATIC_TABLE_RESOURCE_POOL_TERM_CAP",
            "resource_qubit_cap": "GENERIC_STATIC_TABLE_RESOURCE_QUBIT_CAP",
        }
        for row_key, env_key in expected.items():
            expected_value = str(row.get(row_key) or "").strip()
            if not expected_value:
                continue
            if str(env.get(env_key) or "").strip() != expected_value:
                mismatches.append(row_key)
        if str(row.get("generic_adapt_stop_policy") or "").strip() == "fixed_horizon_no_target_v1":
            if str(env.get("GENERIC_STATIC_TABLE_ENERGY_STOP_TARGET", "unset")) != "":
                mismatches.append("energy_stop_target_not_blank")
    elif method_key == "snake":
        expected_schur = str(row.get("adapt_schur_warm_start_mode") or "off").strip() or "off"
        expected_split = str(row.get("snake_phase3_runtime_split_mode") or "source").strip() or "source"
        if str(manifest.get("diagnostic_schur_warm_start_mode") or "off").strip() != expected_schur:
            mismatches.append("adapt_schur_warm_start_mode")
        if str(manifest.get("diagnostic_snake_phase3_runtime_split_mode") or "source").strip() != expected_split:
            mismatches.append("snake_phase3_runtime_split_mode")
        audit = manifest.get("source_lock_command_audit")
        if not isinstance(audit, Mapping):
            mismatches.append("missing_source_lock_command_audit")
        elif str(audit.get("status") or "") != "pass":
            mismatches.append("source_lock_command_audit_not_pass")
    if mismatches:
        return "cell_manifest_mismatch:" + ",".join(mismatches)
    return None


@lru_cache(maxsize=512)
def _parse_progress_points_cached(
    path_key: str,
    mtime_ns: int,
    size: int,
    exact_energy_key: str,
) -> tuple[tuple[int, float], ...]:
    exact_energy = None if exact_energy_key == "" else float(exact_energy_key)
    points: dict[int, float] = {}
    for line in Path(path_key).read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, Mapping):
            continue
        depth = _int_or_none(item.get("depth_after") or item.get("depth"))
        if depth is None:
            iteration = _int_or_none(item.get("iteration"))
            depth = None if iteration is None else iteration + 1
        if depth is None:
            continue
        err = _first_number(
            item,
            "abs_delta_e_same_cutoff",
            "same_cutoff_abs_delta_e",
            "benchmark_target_abs_delta_e_current",
            "abs_delta_e",
        )
        energy = _first_number(item, "energy_after", "energy")
        if err is None and energy is not None and exact_energy is not None:
            err = abs(float(energy) - float(exact_energy))
        if err is not None and err > 0.0:
            points[int(depth)] = float(err)
    return tuple(sorted(points.items()))


def _parse_progress_points(path: Path, exact_energy: float | None) -> tuple[tuple[int, float], ...]:
    if not path.exists():
        return ()
    resolved = path.resolve()
    stat = resolved.stat()
    exact_energy_key = "" if exact_energy is None else repr(float(exact_energy))
    return _parse_progress_points_cached(
        str(resolved),
        int(stat.st_mtime_ns),
        int(stat.st_size),
        exact_energy_key,
    )


def _parse_history_points(result: Mapping[str, Any]) -> tuple[tuple[int, float], ...]:
    history = result.get("adapt_history") or result.get("history") or result.get("history_tail") or []
    if not isinstance(history, Sequence):
        return ()
    points: dict[int, float] = {}
    for idx, item in enumerate(history):
        if not isinstance(item, Mapping):
            continue
        depth = _int_or_none(item.get("depth_after") or item.get("depth") or idx + 1)
        err = _first_number(
            item,
            "abs_delta_e_same_cutoff",
            "same_cutoff_abs_delta_e",
            "benchmark_target_abs_delta_current",
            "benchmark_target_abs_delta_e_current",
            "delta_abs_current",
            "abs_delta_e_same_cutoff_after",
            "abs_delta_e_after",
            "abs_delta_e",
        )
        if depth is not None and err is not None and err > 0.0:
            points[int(depth)] = float(err)
    return tuple(sorted(points.items()))


def _series_from_current(regime: str, method: str, row: Mapping[str, Any], qiskit: Mapping[str, Any] | None) -> Series:
    points = _trajectory(row)
    delta_e = _float_or_none(row.get("same_cutoff_abs_delta_e"))
    marker_k = None
    marker_y = None
    if isinstance(qiskit, Mapping):
        display = qiskit.get("display")
        if isinstance(display, Mapping):
            marker_k = _int_or_none(display.get("k_pl"))
        marker_y = _float_or_none(qiskit.get("same_cutoff_abs_delta_e_at_k_pl"))
    if marker_k is None and points:
        marker_k = int(points[-1][0])
        marker_y = float(points[-1][1])
    if marker_y is None and delta_e is not None:
        marker_y = delta_e
    variant_key = {
        "Append-ADAPT": "current_append",
        "Geo-ADAPT": "current_geo",
        "SNAKE": "current_snake",
    }[method]
    source_json = str(row.get("source_json") or "")
    source_path = _resolve(source_json) if source_json else None
    return Series(
        variant_key=variant_key,
        label=VARIANT_LABELS[variant_key],
        method=method,
        status="done",
        points=points,
        marker_k=marker_k,
        marker_y=marker_y,
        marker_policy="paper_i_plateau_prefix",
        delta_e=delta_e,
        source_json=source_json or None,
        source_sha256=str(row.get("source_sha256") or (_sha256(source_path) if source_path and source_path.exists() else "")) or None,
    )


def _series_from_child_record(
    *,
    regime: str,
    method: str,
    child_mode: str,
    row: Mapping[str, str] | None,
) -> Series:
    variant_key = VARIANT_BY_METHOD_CHILD[(method, child_mode)]
    if row is None:
        return Series(
            variant_key=variant_key,
            label=VARIANT_LABELS[variant_key],
            method=method,
            status="pending",
            points=(),
            marker_k=None,
            marker_y=None,
            marker_policy="pending",
            delta_e=None,
            source_json=None,
            source_sha256=None,
            blocker="missing_child_record",
        )
    result_path = _resolve(str(row.get("result_json_rel") or ""))
    snake_runtime_child_diagnostic = (
        method == "SNAKE"
        and child_mode == "polychildren"
        and _snake_runtime_child_pool_enabled(row)
        and not _snake_global_child_pool_enabled(row)
    )
    if (
        method == "SNAKE"
        and child_mode == "polychildren"
        and not (_snake_global_child_pool_enabled(row) or _snake_runtime_child_pool_enabled(row))
    ):
        return _blocked_snake_child_series(
            variant_key=variant_key,
            label=VARIANT_LABELS[variant_key],
            source_json=_rel(result_path),
            source_sha256=_sha256(result_path) if result_path.exists() else None,
        )
    output_root = _child_output_root(row, result_path)
    progress_path = output_root / "adapt_iteration_progress.jsonl"
    if not result_path.exists():
        return Series(
            variant_key=variant_key,
            label=VARIANT_LABELS[variant_key],
            method=method,
            status="pending",
            points=(),
            marker_k=None,
            marker_y=None,
            marker_policy="pending",
            delta_e=None,
            source_json=_rel(result_path),
            source_sha256=None,
            blocker="missing_result_json",
        )
    manifest_blocker = _child_manifest_mismatch(row, output_root)
    if manifest_blocker is not None:
        return Series(
            variant_key=variant_key,
            label=VARIANT_LABELS[variant_key],
            method=method,
            status="pending",
            points=_parse_progress_points(progress_path, _float_or_none(row.get("same_cutoff_exact_gs_energy"))),
            marker_k=None,
            marker_y=None,
            marker_policy="pending",
            delta_e=None,
            source_json=_rel(result_path),
            source_sha256=_sha256(result_path),
            blocker=manifest_blocker,
        )
    payload = _read_json(result_path)
    result = _extract_result_payload(payload if isinstance(payload, Mapping) else {})
    exact = _float_or_none(row.get("same_cutoff_exact_gs_energy")) or _first_number(
        result,
        "same_cutoff_exact_gs_energy",
        "exact_energy",
        "exact_gs_energy",
    )
    points = _parse_progress_points(progress_path, exact)
    if not points:
        points = _parse_history_points(result)
    delta_e = _first_number(
        result,
        "abs_delta_e_same_cutoff",
        "same_cutoff_abs_delta_e",
        "abs_delta_e",
        "benchmark_target_abs_delta_e_current",
    )
    energy = _first_number(result, "energy")
    if delta_e is None and energy is not None and exact is not None:
        delta_e = abs(float(energy) - float(exact))
    if delta_e is not None and points and points[-1][1] != delta_e:
        points = tuple(points[:-1]) + ((int(points[-1][0]), float(delta_e)),)
    if not points and delta_e is not None:
        depth = _int_or_none(result.get("adapt_depth_reached") or result.get("ansatz_depth")) or 0
        points = ((depth, float(delta_e)),)
    marker_k = int(points[-1][0]) if points else None
    marker_y = float(points[-1][1]) if points else delta_e
    return Series(
        variant_key=variant_key,
        label=VARIANT_LABELS[variant_key],
        method=method,
        status="done" if points else "blocked",
        points=points,
        marker_k=marker_k,
        marker_y=marker_y,
        marker_policy=(
            "terminal_until_plateau_cost_replay_runtime_child_diagnostic"
            if snake_runtime_child_diagnostic
            else "terminal_until_plateau_cost_replay"
        ),
        delta_e=delta_e,
        source_json=_rel(result_path),
        source_sha256=_sha256(result_path),
        blocker=None if points else "no_trajectory_points",
    )


def _series_from_monotone_record(
    *,
    regime: str,
    method: str,
    child_mode: str,
    row: Mapping[str, str] | None,
) -> Series:
    variant_key = MONOTONE_VARIANT_BY_METHOD_CHILD[(method, child_mode)]
    label = MONOTONE_VARIANT_LABELS[variant_key]
    if row is None:
        return Series(
            variant_key=variant_key,
            label=label,
            method=method,
            status="pending",
            points=(),
            marker_k=None,
            marker_y=None,
            marker_policy="pending",
            delta_e=None,
            source_json=None,
            source_sha256=None,
            blocker="missing_monotone_record",
        )
    result_path = _resolve(str(row.get("result_json_rel") or ""))
    snake_runtime_child_diagnostic = (
        method == "SNAKE"
        and child_mode == "polychildren"
        and _snake_runtime_child_pool_enabled(row)
        and not _snake_global_child_pool_enabled(row)
    )
    if (
        method == "SNAKE"
        and child_mode == "polychildren"
        and not (_snake_global_child_pool_enabled(row) or _snake_runtime_child_pool_enabled(row))
    ):
        return _blocked_snake_child_series(
            variant_key=variant_key,
            label=label,
            source_json=_rel(result_path),
            source_sha256=_sha256(result_path) if result_path.exists() else None,
        )
    output_root = _child_output_root(row, result_path)
    progress_path = _resolve(str(row.get("current_json_rel") or "")) if str(row.get("current_json_rel") or "").strip() else output_root / "adapt_iteration_progress.jsonl"
    if not result_path.exists():
        return Series(
            variant_key=variant_key,
            label=label,
            method=method,
            status="pending",
            points=(),
            marker_k=None,
            marker_y=None,
            marker_policy="pending",
            delta_e=None,
            source_json=_rel(result_path),
            source_sha256=None,
            blocker="missing_result_json",
        )
    manifest_blocker = _monotone_manifest_mismatch(row, output_root)
    payload = _read_json(result_path)
    result = _extract_result_payload(payload if isinstance(payload, Mapping) else {})
    exact = _float_or_none(row.get("same_cutoff_exact_gs_energy")) or _first_number(
        result,
        "same_cutoff_exact_gs_energy",
        "exact_energy",
        "exact_gs_energy",
    )
    points = _parse_progress_points(progress_path, exact) if method != "SNAKE" else ()
    if not points:
        points = _parse_history_points(result)
    delta_e = _first_number(
        result,
        "abs_delta_e_same_cutoff",
        "same_cutoff_abs_delta_e",
        "abs_delta_e",
        "benchmark_target_abs_delta_e_current",
    )
    energy = _first_number(result, "energy")
    if delta_e is None and energy is not None and exact is not None:
        delta_e = abs(float(energy) - float(exact))
    if delta_e is not None and points and points[-1][1] != delta_e:
        points = tuple(points[:-1]) + ((int(points[-1][0]), float(delta_e)),)
    if not points and delta_e is not None:
        depth = _int_or_none(result.get("adapt_depth_reached") or result.get("ansatz_depth")) or 0
        points = ((depth, float(delta_e)),)
    marker_k = int(points[-1][0]) if points else None
    marker_y = float(points[-1][1]) if points else delta_e
    return Series(
        variant_key=variant_key,
        label=label,
        method=method,
        status="done" if points and manifest_blocker is None else "pending",
        points=points,
        marker_k=marker_k if manifest_blocker is None else None,
        marker_y=marker_y if manifest_blocker is None else None,
        marker_policy=(
            (
                "terminal_until_plateau_cost_replay_runtime_child_diagnostic"
                if snake_runtime_child_diagnostic
                else "terminal_until_plateau_cost_replay"
            )
            if manifest_blocker is None
            else "pending_manifest_validation"
        ),
        delta_e=delta_e if manifest_blocker is None else None,
        source_json=_rel(result_path),
        source_sha256=_sha256(result_path),
        blocker=manifest_blocker if manifest_blocker is not None else (None if points else "no_trajectory_points"),
    )


def _series_from_schur_overlay(regime: str, overlay_row: Mapping[str, Any] | None) -> Series:
    if not isinstance(overlay_row, Mapping):
        return Series(
            variant_key=SCHUR_VARIANT_KEY,
            label=VARIANT_LABELS[SCHUR_VARIANT_KEY],
            method="SNAKE",
            status="pending",
            points=(),
            marker_k=None,
            marker_y=None,
            marker_policy="pending",
            delta_e=None,
            source_json=None,
            source_sha256=None,
            blocker="missing_schur_overlay_row",
        )
    warm = overlay_row.get("warm") if isinstance(overlay_row.get("warm"), Mapping) else {}
    resource = (
        overlay_row.get("warm_resource_row")
        if isinstance(overlay_row.get("warm_resource_row"), Mapping)
        else {}
    )
    source_json = str(warm.get("path") or "")
    source_path = _resolve(source_json) if source_json else None
    source_sha256 = str(warm.get("sha256") or "")
    blocker: str | None = None
    global_child_pool = any(
        _snake_global_child_pool_enabled(section)
        for section in (overlay_row, warm, resource)
        if isinstance(section, Mapping)
    )
    schur_mode = str(warm.get("adapt_schur_warm_start_mode") or "").strip()
    runtime_child_diagnostic = not global_child_pool
    if str(warm.get("status") or "") != "ok":
        blocker = f"schur_warm_status_{warm.get('status') or 'unknown'}"
    elif schur_mode and schur_mode not in {"append-prune", "append_prune"}:
        blocker = "schur_warm_start_mode_not_append_prune"
    elif source_path is None or not source_path.exists():
        blocker = "missing_schur_result_json"
    elif source_sha256 and _sha256(source_path) != source_sha256:
        blocker = "schur_result_sha256_mismatch"
    points = _xy_points(warm)
    delta_e = _float_or_none(resource.get("same_cutoff_abs_delta_e")) or _float_or_none(warm.get("terminal_y"))
    marker_k = _int_or_none(resource.get("reported_iteration")) or _int_or_none(warm.get("terminal_x"))
    marker_y = delta_e
    if marker_k is None and points:
        marker_k = points[-1][0]
    if marker_y is None and points:
        marker_y = points[-1][1]
    return Series(
        variant_key=SCHUR_VARIANT_KEY,
        label=VARIANT_LABELS[SCHUR_VARIANT_KEY],
        method="SNAKE",
        status="done" if blocker is None and points else "pending",
        points=points if blocker is None else (),
        marker_k=marker_k if blocker is None else None,
        marker_y=marker_y if blocker is None else None,
        marker_policy=(
            (
                "terminal_schur_native_forced_depth30_import_runtime_child_diagnostic"
                if runtime_child_diagnostic
                else "terminal_schur_native_forced_depth30_import"
            )
            if blocker is None
            else "pending"
        ),
        delta_e=delta_e if blocker is None else None,
        source_json=source_json or None,
        source_sha256=source_sha256 or (_sha256(source_path) if source_path and source_path.exists() else None),
        blocker=blocker if blocker is not None else (None if points else "no_schur_points"),
    )


def _pending_snake_no_child() -> Series:
    return Series(
        variant_key="snake_no_child",
        label=VARIANT_LABELS["snake_no_child"],
        method="SNAKE",
        status="pending",
        points=(),
        marker_k=None,
        marker_y=None,
        marker_policy="pending",
        delta_e=None,
        source_json=None,
        source_sha256=None,
        blocker="snake_no_child_not_run_yet",
    )


def _series_from_manifest_item(item: Mapping[str, Any]) -> Series:
    return Series(
        variant_key=str(item.get("variant_key") or ""),
        label=str(item.get("label") or item.get("variant_key") or ""),
        method=str(item.get("method") or ""),
        status=str(item.get("status") or "pending"),
        points=(),
        marker_k=_int_or_none(item.get("marker_k")),
        marker_y=_float_or_none(item.get("marker_y")),
        marker_policy=str(item.get("marker_policy") or "reused_manifest"),
        delta_e=_float_or_none(item.get("delta_e")),
        source_json=None if item.get("source_json") is None else str(item.get("source_json")),
        source_sha256=None if item.get("source_sha256") is None else str(item.get("source_sha256")),
        blocker=None if item.get("blocker") is None else str(item.get("blocker")),
    )


def _series_by_regime_from_manifest(raw: Any) -> dict[str, list[Series]]:
    out: dict[str, list[Series]] = {regime: [] for regime in REGIME_ORDER}
    if not isinstance(raw, Mapping):
        return out
    for regime in REGIME_ORDER:
        items = raw.get(regime)
        if not isinstance(items, Sequence):
            continue
        out[regime] = [_series_from_manifest_item(item) for item in items if isinstance(item, Mapping)]
    return out


def _build_series(
    current_rows: Mapping[tuple[str, str], Mapping[str, Any]],
    qiskit_rows: Mapping[tuple[str, str], Mapping[str, Any]],
    child_records: Mapping[tuple[str, str, str], Mapping[str, str]],
    schur_overlay: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[Series]]:
    out: dict[str, list[Series]] = {}
    for regime in REGIME_ORDER:
        series: list[Series] = []
        for method in CURRENT_METHODS:
            for child_mode in ("no_child", "polychildren"):
                series.append(
                    _series_from_child_record(
                        regime=regime,
                        method=method,
                        child_mode=child_mode,
                        row=child_records.get((regime, method, child_mode)),
                    )
                )
        series.append(_series_from_schur_overlay(regime, schur_overlay.get(regime)))
        series_by_key = {item.variant_key: item for item in series}
        out[regime] = [series_by_key[key] for key in VARIANT_ORDER if key in series_by_key]
    return out


def _build_monotone_series(
    monotone_records: Mapping[tuple[str, str, str], Mapping[str, str]],
) -> dict[str, list[Series]]:
    out: dict[str, list[Series]] = {}
    for regime in REGIME_ORDER:
        series: list[Series] = []
        for method in CURRENT_METHODS:
            for child_mode in ("no_child", "polychildren"):
                series.append(
                    _series_from_monotone_record(
                        regime=regime,
                        method=method,
                        child_mode=child_mode,
                        row=monotone_records.get((regime, method, child_mode)),
                    )
                )
        by_key = {item.variant_key: item for item in series}
        out[regime] = [by_key[key] for key in MONOTONE_VARIANT_ORDER if key in by_key]
    return out


def _render_plots(
    series_by_regime: Mapping[str, Sequence[Series]],
    figures_dir: Path,
    *,
    filename_prefix: str = "",
    figsize: tuple[float, float] = (3.45, 2.02),
    axis_fontsize: float = 7.0,
    title_fontsize: float = 8.0,
    tick_fontsize: float = 6.0,
    legend_fontsize: float = 4.8,
    tight_pad: float = 0.25,
) -> dict[str, str]:
    Figure, FigureCanvasAgg, _Line2D = _configure_matplotlib()
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for regime, series_list in series_by_regime.items():
        fig = Figure(figsize=figsize, dpi=180)
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        y_values: list[float] = []
        for series in series_list:
            if not series.points:
                continue
            style = METHOD_STYLE[series.method]
            is_current = series.variant_key.startswith("current_")
            is_schur_import = series.variant_key == SCHUR_VARIANT_KEY
            is_child_variant = series.variant_key.endswith("_child") and not series.variant_key.endswith("_no_child")
            alpha = 0.9 if is_schur_import else (0.96 if is_current or not is_child_variant else 0.72)
            linewidth = 1.35 if is_schur_import else (1.45 if is_current else 1.15)
            linestyle = "-." if is_schur_import else ("--" if is_child_variant else "-")
            xs = [x for x, _ in series.points]
            ys = [max(y, 1e-14) for _, y in series.points]
            y_values.extend(ys)
            ax.plot(
                xs,
                ys,
                color=style["color"],
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=alpha,
                label=series.label,
            )
            if series.marker_k is not None and series.marker_y is not None and series.marker_y > 0.0:
                y_values.append(float(series.marker_y))
                ax.plot(
                    [series.marker_k],
                    [series.marker_y],
                    color=style["color"],
                    marker=style["marker"],
                    markersize=style["markersize"] if is_current else max(7.0, style["markersize"] * 0.75),
                    markeredgecolor="black",
                    markeredgewidth=1.0,
                    alpha=alpha,
                    linestyle="None",
                    zorder=5,
                )
        ax.set_yscale("log")
        ax.set_xlim(0, 30.5)
        if y_values:
            ax.set_ylim(max(min(y_values) * 0.55, 1e-12), max(2.0, max(y_values) * 1.25))
        ax.set_xlabel("ADAPT round", fontsize=axis_fontsize)
        ax.set_ylabel(r"$|\Delta E|$", fontsize=axis_fontsize)
        ax.set_title(REGIME_DISPLAY.get(regime, regime), fontsize=title_fontsize)
        ax.tick_params(axis="both", which="major", labelsize=tick_fontsize, pad=1.5)
        ax.grid(True, which="major", color="#d0d0d0", linewidth=0.45)
        ax.grid(True, which="minor", color="#eeeeee", linewidth=0.28)
        ax.legend(loc="best", fontsize=legend_fontsize, frameon=True, borderpad=0.25, handlelength=1.2)
        fig.tight_layout(pad=tight_pad)
        path = figures_dir / f"{filename_prefix}{regime.replace('-', '_')}_error_vs_iteration.png"
        fig.savefig(path)
        outputs[regime] = _rel(path)
    return outputs


def _expected_figure_paths(figures_dir: Path, filename_prefix: str) -> dict[str, str]:
    return {
        regime: _rel(figures_dir / f"{filename_prefix}{regime.replace('-', '_')}_error_vs_iteration.png")
        for regime in REGIME_ORDER
    }


def _hex_rgb(value: str) -> tuple[int, int, int]:
    text = value.lstrip("#")
    if len(text) != 6:
        return (0, 0, 0)
    return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))


def _draw_marker(draw: Any, x: float, y: float, color: tuple[int, int, int], marker: str, size: float) -> None:
    r = max(2, int(round(size / 2)))
    if marker == "^":
        draw.polygon([(x, y - r), (x - r, y + r), (x + r, y + r)], fill=color, outline=(0, 0, 0))
    elif marker == "*":
        draw.line([(x - r, y), (x + r, y)], fill=(0, 0, 0), width=1)
        draw.line([(x, y - r), (x, y + r)], fill=(0, 0, 0), width=1)
        draw.ellipse((x - r + 1, y - r + 1, x + r - 1, y + r - 1), fill=color, outline=(0, 0, 0))
    else:
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=(0, 0, 0))


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _write_rgb_png(path: Path, width: int, height: int, pixels: bytearray) -> None:
    stride = width * 3
    raw = b"".join(b"\x00" + bytes(pixels[y * stride : (y + 1) * stride]) for y in range(height))
    payload = b"\x89PNG\r\n\x1a\n"
    payload += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    payload += _png_chunk(b"IDAT", zlib.compress(raw, level=6))
    payload += _png_chunk(b"IEND", b"")
    path.write_bytes(payload)


def _put_pixel(pixels: bytearray, width: int, height: int, x: int, y: int, color: tuple[int, int, int]) -> None:
    if x < 0 or y < 0 or x >= width or y >= height:
        return
    idx = (y * width + x) * 3
    pixels[idx : idx + 3] = bytes(color)


def _draw_line_pixels(
    pixels: bytearray,
    width: int,
    height: int,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    x0_i, y0_i, x1_i, y1_i = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
    dx = abs(x1_i - x0_i)
    dy = -abs(y1_i - y0_i)
    sx = 1 if x0_i < x1_i else -1
    sy = 1 if y0_i < y1_i else -1
    err = dx + dy
    x, y = x0_i, y0_i
    radius = max(0, thickness // 2)
    while True:
        for ox in range(-radius, radius + 1):
            for oy in range(-radius, radius + 1):
                _put_pixel(pixels, width, height, x + ox, y + oy, color)
        if x == x1_i and y == y1_i:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


def _draw_rect_pixels(
    pixels: bytearray,
    width: int,
    height: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
) -> None:
    _draw_line_pixels(pixels, width, height, x0, y0, x1, y0, color)
    _draw_line_pixels(pixels, width, height, x1, y0, x1, y1, color)
    _draw_line_pixels(pixels, width, height, x1, y1, x0, y1, color)
    _draw_line_pixels(pixels, width, height, x0, y1, x0, y0, color)


def _draw_marker_pixels(
    pixels: bytearray,
    width: int,
    height: int,
    x: float,
    y: float,
    color: tuple[int, int, int],
    size: int = 4,
) -> None:
    cx, cy = int(round(x)), int(round(y))
    for ox in range(-size, size + 1):
        for oy in range(-size, size + 1):
            if ox * ox + oy * oy <= size * size:
                _put_pixel(pixels, width, height, cx + ox, cy + oy, color)
    _draw_rect_pixels(pixels, width, height, cx - size, cy - size, cx + size, cy + size, (0, 0, 0))


def _render_plots_pil(
    series_by_regime: Mapping[str, Sequence[Series]],
    figures_dir: Path,
    *,
    filename_prefix: str,
    width_px: int = 927,
    height_px: int = 171,
) -> dict[str, str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for regime, series_list in series_by_regime.items():
        pixels = bytearray([255]) * (width_px * height_px * 3)
        left, right, top, bottom = 36, 10, 10, 18
        plot_w = width_px - left - right
        plot_h = height_px - top - bottom
        y_values = [
            max(float(y), 1e-14)
            for series in series_list
            for _, y in series.points
            if y is not None and y > 0.0
        ]
        for series in series_list:
            if series.marker_y is not None and series.marker_y > 0.0:
                y_values.append(max(float(series.marker_y), 1e-14))
        y_min = max((min(y_values) * 0.55) if y_values else 1e-6, 1e-12)
        y_max = max(2.0, (max(y_values) * 1.25) if y_values else 2.0)
        log_min = math.log10(y_min)
        log_max = math.log10(y_max)

        def tx(x: int | float) -> float:
            return left + max(0.0, min(30.5, float(x))) / 30.5 * plot_w

        def ty(y: int | float) -> float:
            ly = math.log10(max(float(y), 1e-14))
            if log_max <= log_min:
                return top + plot_h
            return top + (log_max - ly) / (log_max - log_min) * plot_h

        _draw_rect_pixels(pixels, width_px, height_px, left, top, left + plot_w, top + plot_h, (115, 115, 115))
        for x_tick in (0, 10, 20, 30):
            x = tx(x_tick)
            _draw_line_pixels(pixels, width_px, height_px, x, top, x, top + plot_h, (224, 224, 224))
        for exp in range(math.floor(log_min), math.ceil(log_max) + 1):
            y_val = 10.0**exp
            if y_min <= y_val <= y_max:
                y = ty(y_val)
                _draw_line_pixels(pixels, width_px, height_px, left, y, left + plot_w, y, (224, 224, 224))

        for series in series_list:
            if not series.points:
                continue
            color = _hex_rgb(str(METHOD_STYLE[series.method]["color"]))
            points = [(tx(x), ty(y)) for x, y in series.points if y > 0.0]
            if len(points) >= 2:
                for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
                    _draw_line_pixels(pixels, width_px, height_px, x0, y0, x1, y1, color, thickness=2)
                for x, y in points:
                    _draw_marker_pixels(pixels, width_px, height_px, x, y, color, size=2)
            elif len(points) == 1:
                _draw_marker_pixels(pixels, width_px, height_px, points[0][0], points[0][1], color, size=4)
            if series.marker_k is not None and series.marker_y is not None and series.marker_y > 0.0:
                _draw_marker_pixels(
                    pixels,
                    width_px,
                    height_px,
                    tx(series.marker_k),
                    ty(series.marker_y),
                    color,
                    size=4,
                )

        path = figures_dir / f"{filename_prefix}{regime.replace('-', '_')}_error_vs_iteration.png"
        _write_rgb_png(path, width_px, height_px, pixels)
        outputs[regime] = _rel(path)
    return outputs


def _current_cost_row(qiskit: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(qiskit, Mapping):
        return {"status": "blocked:missing_current_qiskit_row"}
    display = qiskit.get("display") if isinstance(qiskit.get("display"), Mapping) else {}
    row = {
        "status": str(qiskit.get("status") or "done"),
        "k": display.get("k_pl"),
        "DeltaE": display.get("DeltaE"),
        "F": qiskit.get("fidelity_exact"),
        "theta_count": display.get("k_pl"),
        "N2q": display.get("N2q"),
        "D2q": display.get("D2q"),
        "Dc": display.get("Dc"),
        "S": display.get("S"),
        "source_json": qiskit.get("source_json"),
        "source_sha256": qiskit.get("source_sha256"),
    }
    return _with_shot_columns(row, str(qiskit.get("method") or ""))


def _pending_cost_row(series: Series) -> dict[str, Any]:
    if series.status == "done":
        return _with_shot_columns(
            {
            "status": "pending_qiskit_replay",
            "k": series.marker_k,
            "DeltaE": _fmt_sci(series.delta_e),
            "F": None,
            "theta_count": series.marker_k,
            "N2q": None,
            "D2q": None,
            "Dc": None,
            "S": None,
            "source_json": series.source_json,
            "source_sha256": series.source_sha256,
            },
            series.method,
        )
    return _with_shot_columns(
        {
        "status": f"pending:{series.blocker or series.status}",
        "k": None,
        "DeltaE": None,
        "F": None,
        "theta_count": None,
        "N2q": None,
        "D2q": None,
        "Dc": None,
        "S": None,
        "source_json": series.source_json,
        "source_sha256": series.source_sha256,
        },
        series.method,
    )


def _cost_row_from_local(series: Series, cost: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(cost, Mapping):
        return _pending_cost_row(series)
    status = str(cost.get("status") or "")
    if status == "done":
        row = {
            "status": "done",
            "k": cost.get("k") if cost.get("k") is not None else series.marker_k,
            "DeltaE": _fmt_sci(cost.get("DeltaE") if cost.get("DeltaE") is not None else series.delta_e),
            "F": cost.get("fidelity_exact"),
            "theta_count": cost.get("logical_operator_count") if cost.get("logical_operator_count") is not None else series.marker_k,
            "N2q": cost.get("N2q"),
            "D2q": cost.get("D2q"),
            "Dc": cost.get("Dc"),
            "S": cost.get("S"),
            "source_json": cost.get("source_json") or series.source_json,
            "source_sha256": cost.get("source_sha256") or series.source_sha256,
            "compile_convention": cost.get("compile_convention"),
            "compiled_circuit_scope": cost.get("compiled_circuit_scope"),
            "compiled_resource_source_kind": cost.get("compiled_resource_source_kind"),
            "S_unfair": cost.get("S_unfair"),
            "S_unfair_status": cost.get("S_unfair_status"),
            "S_unfair_source_kind": cost.get("S_unfair_source_kind"),
            "S_fair": cost.get("S_fair"),
            "S_fair_status": cost.get("S_fair_status"),
            "S_fair_policy": cost.get("S_fair_policy"),
            "S_fair_source": cost.get("S_fair_source"),
            "S_fair_source_kind": cost.get("S_fair_source_kind"),
            "S_actual": cost.get("S_actual"),
            "S_actual_status": cost.get("S_actual_status"),
            "S_common_exposure": cost.get("S_common_exposure"),
            "S_common_exposure_status": cost.get("S_common_exposure_status"),
            "fair_work_currency": cost.get("fair_work_currency"),
            "work_contract_id": cost.get("work_contract_id"),
            "S_ctrl_grouped": cost.get("S_ctrl_grouped"),
            "S_ctrl_grouped_status": cost.get("S_ctrl_grouped_status"),
            "S_ctrl_grouped_source_json": cost.get("S_ctrl_grouped_source_json"),
            "S_ctrl_grouped_source_sha256": cost.get("S_ctrl_grouped_source_sha256"),
        }
        return _with_shot_columns(row, series.method)
    row = _pending_cost_row(series)
    if status:
        row["status"] = status
    row["source_json"] = cost.get("source_json") or row.get("source_json")
    row["source_sha256"] = cost.get("source_sha256") or row.get("source_sha256")
    for key in (
        "S_unfair",
        "S_unfair_status",
        "S_unfair_source_kind",
        "S_fair",
        "S_fair_status",
        "S_fair_policy",
        "S_fair_source",
        "S_fair_source_kind",
        "S_actual",
        "S_actual_status",
        "S_common_exposure",
        "S_common_exposure_status",
        "fair_work_currency",
        "work_contract_id",
        "S_ctrl_grouped",
        "S_ctrl_grouped_status",
        "S_ctrl_grouped_source_json",
        "S_ctrl_grouped_source_sha256",
    ):
        if key in cost:
            row[key] = cost.get(key)
    return _with_shot_columns(row, series.method)


def _cost_row_from_schur_overlay(series: Series, overlay_row: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(overlay_row, Mapping) or series.status != "done":
        return _pending_cost_row(series)
    warm = overlay_row.get("warm") if isinstance(overlay_row.get("warm"), Mapping) else {}
    resource = (
        overlay_row.get("warm_resource_row")
        if isinstance(overlay_row.get("warm_resource_row"), Mapping)
        else {}
    )
    if str(resource.get("cost_status") or resource.get("status") or "ok") not in {"ok", "done"}:
        row = _pending_cost_row(series)
        row["status"] = f"pending:{resource.get('cost_status') or resource.get('status') or 'missing_schur_cost'}"
        return row
    source_json = str(resource.get("source_json") or warm.get("path") or series.source_json or "")
    source_path = _resolve(source_json) if source_json else None
    qiskit_cost = resource.get("qiskit_cost") if isinstance(resource.get("qiskit_cost"), Mapping) else {}
    one_minus_f = _float_or_none(resource.get("one_minus_f"))
    fidelity = None if one_minus_f is None else 1.0 - one_minus_f
    fair_sidecar = _read_snake_fair_sidecar(source_path) if source_path is not None else None
    fair_currency = fair_sidecar.get("fair_work_currency") if fair_sidecar else None
    fair_status = str(fair_sidecar.get("S_fair_status") or "missing_sidecar") if fair_sidecar else "missing_sidecar"
    fair_value = fair_sidecar.get("S_fair") if fair_status == "ok" and fair_currency == FAIR_WORK_CURRENCY else None
    if fair_status == "ok" and fair_currency != FAIR_WORK_CURRENCY:
        fair_status = "blocked:wrong_fair_work_currency"
    grouped_sidecar = _read_snake_grouped_sidecar(source_path) if source_path is not None else None
    source_sha256 = str(resource.get("source_sha256") or series.source_sha256 or "")
    row = {
        "status": "done",
        "k": resource.get("reported_iteration") or series.marker_k,
        "DeltaE": _fmt_sci(resource.get("same_cutoff_abs_delta_e") if resource.get("same_cutoff_abs_delta_e") is not None else series.delta_e),
        "F": fidelity,
        "theta_count": resource.get("final_ansatz_length"),
        "N2q": resource.get("N2q"),
        "D2q": resource.get("D2q"),
        "Dc": resource.get("D_circ"),
        "S": resource.get("S"),
        "source_json": source_json or series.source_json,
        "source_sha256": source_sha256 or None,
        "compile_convention": qiskit_cost.get("compile_convention") or qiskit_cost.get("backend") or resource.get("compile_source"),
        "compiled_circuit_scope": "ansatz_circuit_including_reference_state",
        "compiled_resource_source_kind": "qiskit_compile_cost_sidecar",
        "S_unfair": resource.get("S"),
        "S_unfair_status": resource.get("S_status") or "ok:legacy_schur_overlay",
        "S_unfair_source_kind": "legacy_or_grouped_schur_overlay_shot_proxy",
        "S_fair": fair_value,
        "S_fair_status": fair_status,
        "S_fair_source_kind": fair_sidecar.get("S_fair_source_kind") if fair_sidecar else "snake_fair_sidecar_missing",
        "S_actual": fair_sidecar.get("S_actual") if fair_sidecar else None,
        "S_actual_status": fair_sidecar.get("S_actual_status") if fair_sidecar else None,
        "S_common_exposure": fair_sidecar.get("S_common_exposure") if fair_sidecar else None,
        "S_common_exposure_status": fair_sidecar.get("S_common_exposure_status") if fair_sidecar else None,
        "S_fair_policy": fair_sidecar.get("S_fair_policy") if fair_sidecar else None,
        "S_fair_source": fair_sidecar.get("S_fair_source") if fair_sidecar else None,
        "work_contract_id": fair_sidecar.get("work_contract_id") if fair_sidecar else None,
        "S_ctrl_grouped": (grouped_sidecar or {}).get("S_alg") if grouped_sidecar else resource.get("S"),
        "S_ctrl_grouped_status": (grouped_sidecar or {}).get("S_alg_status") if grouped_sidecar else resource.get("S_status"),
        "S_ctrl_grouped_source_json": (grouped_sidecar or {}).get("source_json") if grouped_sidecar else None,
        "S_ctrl_grouped_source_sha256": (grouped_sidecar or {}).get("source_sha256") if grouped_sidecar else None,
        "fair_work_currency": fair_currency,
    }
    return _with_shot_columns(row, series.method)


def _local_cost_for_series(
    local_cost_rows: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
    *,
    page: str,
    regime: str,
    series: Series,
) -> Mapping[str, Any] | None:
    method_key = METHOD_KEY_BY_DISPLAY.get(series.method)
    if method_key is None:
        return None
    if series.variant_key == SCHUR_VARIANT_KEY:
        return None
    child_policy = "no_child" if series.variant_key.endswith("_no_child") else "polychildren"
    return local_cost_rows.get((page, regime, method_key, child_policy))


def _cost_rows_for_regime(
    regime: str,
    series_list: Sequence[Series],
    qiskit_rows: Mapping[tuple[str, str], Mapping[str, Any]],
    local_cost_rows: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
    schur_overlay: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    by_key = {series.variant_key: series for series in series_list}
    for variant in VARIANT_ORDER:
        series = by_key.get(variant)
        if series is None:
            continue
        if variant == SCHUR_VARIANT_KEY:
            row = _cost_row_from_schur_overlay(series, schur_overlay.get(regime))
        else:
            row = _cost_row_from_local(series, _local_cost_for_series(local_cost_rows, page="native", regime=regime, series=series))
        row.update(
            {
                "variant_key": variant,
                "variant_label": VARIANT_LABELS[variant],
                "method": series.method,
                "snake_child_pool_policy": _snake_child_pool_policy_for_series(series),
            }
        )
        _demote_runtime_child_snake_fair_s(row)
        out.append(row)
    return out


def _cost_rows_for_monotone_regime(
    regime: str,
    series_list: Sequence[Series],
    local_cost_rows: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    by_key = {series.variant_key: series for series in series_list}
    for variant in MONOTONE_VARIANT_ORDER:
        series = by_key.get(variant)
        if series is None:
            continue
        row = _cost_row_from_local(
            series,
            _local_cost_for_series(local_cost_rows, page="monotone", regime=regime, series=series),
        )
        row.update(
            {
                "variant_key": variant,
                "variant_label": MONOTONE_VARIANT_LABELS[variant],
                "method": series.method,
                "snake_child_pool_policy": _snake_child_pool_policy_for_series(series),
            }
        )
        _demote_runtime_child_snake_fair_s(row)
        out.append(row)
    return out


def _snake_child_pool_policy_for_series(series: Series) -> str | None:
    if series.method != "SNAKE":
        return None
    variant = str(series.variant_key or "")
    if "child" not in variant or variant.endswith("no_child"):
        return "no_child"
    if "runtime_child_diagnostic" in str(series.marker_policy or ""):
        return "runtime_child_shortlist_diagnostic"
    if series.blocker == SNAKE_CORRECTED_CHILD_POOL_BLOCKER:
        return "missing_global_pauli_child_pool"
    return "global_pauli_child_pool"


def _is_corrected_global_snake_anchor(row: Mapping[str, Any] | None) -> bool:
    if not isinstance(row, Mapping):
        return False
    if row.get("method") != "SNAKE":
        return False
    if str(row.get("variant_key") or "") != SCHUR_VARIANT_KEY:
        return False
    if row.get("status") != "done":
        return False
    return str(row.get("snake_child_pool_policy") or "") == "global_pauli_child_pool"


def _demote_runtime_child_snake_fair_s(row: dict[str, Any]) -> None:
    """Keep SNAKE runtime-child work as provenance, not visible fair-S."""

    if row.get("method") != "SNAKE":
        return
    variant = str(row.get("variant_key") or "")
    if "child" not in variant or variant.endswith("no_child"):
        return
    if str(row.get("snake_child_pool_policy") or "") != "runtime_child_shortlist_diagnostic":
        return
    if _int_or_none(row.get("S_fair")) is not None:
        row["S_runtime_child_shortlist"] = row.get("S_fair")
        row["S_runtime_child_shortlist_status"] = row.get("S_fair_status")
    row["S_fair_status"] = "blocked:runtime_child_not_full_pool_child_currency"
    row["S_fair_source_kind"] = "snake_phase3_runtime_child_shortlist_not_append_geo_full_pool_child"


def _tex_table(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lrrrrrrll}",
        r"\hline",
        r"row & $k$ & $|\Delta E|$ & $F$ & $N_{2q}$ & $D_{2q}$ & $D_c$ & $S_{\rm unfair}$ & $S_{\rm fair}$ \\",
        r"\hline",
    ]
    for row in rows:
        s_unfair_cell = _shot_cell(row, "S_unfair", "S_unfair_status")
        s_fair_cell = _shot_cell(row, "S_fair", "S_fair_status")
        lines.append(
            " & ".join(
                [
                    _tex_escape(row.get("variant_label") or ""),
                    _fmt_int(row.get("k")),
                    _tex_escape(row.get("DeltaE") or "--"),
                    _fmt_fidelity(row.get("F")),
                    _fmt_int(row.get("N2q")),
                    _fmt_int(row.get("D2q")),
                    _fmt_int(row.get("Dc")),
                    s_unfair_cell,
                    s_fair_cell,
                ]
            )
            + r" \\"
        )
    lines.extend([r"\hline", r"\end{tabular}%", r"}"])
    return "\n".join(lines)


def _shot_cell(row: Mapping[str, Any], value_key: str, status_key: str) -> str:
    value = row.get(value_key)
    status = str(row.get(status_key) or row.get("status") or "--")
    if status.startswith("blocked:"):
        if status == "blocked:mixed_work_currency":
            status = "blocked:mixed"
        elif status == "blocked:runtime_child_not_full_pool_child_currency":
            status = "blocked:runtime-child"
        return _tex_escape(status)
    parsed = _int_or_none(value)
    if parsed is not None:
        return _fmt_int(parsed)
    return _tex_escape(status)


def _rows_by_label(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(row.get("variant_label") or ""): row for row in rows}


def _ratio_or_none(numerator: Any, denominator: Any) -> float | None:
    num = _float_or_none(numerator)
    den = _float_or_none(denominator)
    if num is None or den is None or den == 0.0:
        return None
    out = float(num / den)
    return out if math.isfinite(out) else None


def _fmt_ratio(value: Any) -> str:
    parsed = _float_or_none(value)
    if parsed is None:
        return "--"
    if parsed >= 100.0:
        return f"{parsed:.0f}x"
    if parsed >= 10.0:
        return f"{parsed:.1f}x"
    if parsed >= 1.0:
        return f"{parsed:.2f}x"
    if parsed >= 0.01:
        return f"{parsed:.2f}x"
    return f"{parsed:.3g}x"


def _compact_ratio_cell(row: Mapping[str, Any]) -> str:
    if not bool(row.get("ratio_complete")):
        return str(row.get("ratio_status") or "blocked")
    return (
        f"E {_fmt_ratio(row.get('ratio_delta_e'))}; "
        f"D {_fmt_ratio(row.get('ratio_D2q'))}/{_fmt_ratio(row.get('ratio_Dc'))}"
    )


def _build_compact_child_schur_rows(
    *,
    cost_rows_by_regime: Mapping[str, Sequence[Mapping[str, Any]]],
    monotone_cost_rows_by_regime: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    comparators = (
        {
            "comparator_key": "geo_native_no_child",
            "comparator_label": "Geo native no-child",
            "geo_page": "native_forced",
            "geo_rows": cost_rows_by_regime,
            "geo_variant_label": "Geo no child",
        },
        {
            "comparator_key": "geo_native_child",
            "comparator_label": "Geo native child",
            "geo_page": "native_forced",
            "geo_rows": cost_rows_by_regime,
            "geo_variant_label": "Geo child",
        },
        {
            "comparator_key": "geo_monotone_no_child",
            "comparator_label": "Geo monotone no-child",
            "geo_page": "monotone_nonforced",
            "geo_rows": monotone_cost_rows_by_regime,
            "geo_variant_label": "Geo no child",
        },
        {
            "comparator_key": "geo_monotone_child",
            "comparator_label": "Geo monotone child",
            "geo_page": "monotone_nonforced",
            "geo_rows": monotone_cost_rows_by_regime,
            "geo_variant_label": "Geo child",
        },
    )
    out: list[dict[str, Any]] = []
    for regime in REGIME_ORDER:
        native_rows = _rows_by_label(cost_rows_by_regime.get(regime, ()))
        snake = native_rows.get("SNAKE child+Schur")
        for comparator in comparators:
            geo_rows = _rows_by_label(comparator["geo_rows"].get(regime, ()))
            geo = geo_rows.get(str(comparator["geo_variant_label"]))
            ratio_delta_e = _ratio_or_none(
                None if snake is None else snake.get("DeltaE"),
                None if geo is None else geo.get("DeltaE"),
            )
            ratio_D2q = _ratio_or_none(
                None if snake is None else snake.get("D2q"),
                None if geo is None else geo.get("D2q"),
            )
            ratio_Dc = _ratio_or_none(
                None if snake is None else snake.get("Dc"),
                None if geo is None else geo.get("Dc"),
            )
            ratio_complete = ratio_delta_e is not None and ratio_D2q is not None and ratio_Dc is not None
            if snake is None:
                ratio_status = "blocked:missing_snake_child_schur_anchor"
            elif snake.get("status") != "done":
                ratio_status = f"blocked:{snake.get('status') or SNAKE_CORRECTED_CHILD_POOL_BLOCKER}"
            elif not _is_corrected_global_snake_anchor(snake):
                ratio_status = f"blocked:{SNAKE_CORRECTED_CHILD_POOL_BLOCKER}"
                ratio_complete = False
            elif geo is None:
                ratio_status = "blocked:missing_geo_comparator"
            elif not ratio_complete:
                ratio_status = "blocked:missing_numeric_ratio_input"
            else:
                ratio_status = "ok"
            row = {
                "regime": str(regime),
                "regime_display": REGIME_DISPLAY.get(regime, regime),
                "anchor_page": "native_forced",
                "anchor_variant_label": "SNAKE child+Schur",
                "anchor_status": None if snake is None else snake.get("status"),
                "anchor_source_json": None if snake is None else snake.get("source_json"),
                "anchor_source_sha256": None if snake is None else snake.get("source_sha256"),
                "anchor_DeltaE": None if snake is None else _float_or_none(snake.get("DeltaE")),
                "anchor_D2q": None if snake is None else _int_or_none(snake.get("D2q")),
                "anchor_Dc": None if snake is None else _int_or_none(snake.get("Dc")),
                "anchor_snake_child_pool_policy": None if snake is None else snake.get("snake_child_pool_policy"),
                "comparator_key": str(comparator["comparator_key"]),
                "comparator_label": str(comparator["comparator_label"]),
                "geo_page": str(comparator["geo_page"]),
                "geo_variant_label": str(comparator["geo_variant_label"]),
                "geo_status": None if geo is None else geo.get("status"),
                "geo_source_json": None if geo is None else geo.get("source_json"),
                "geo_source_sha256": None if geo is None else geo.get("source_sha256"),
                "geo_DeltaE": None if geo is None else _float_or_none(geo.get("DeltaE")),
                "geo_D2q": None if geo is None else _int_or_none(geo.get("D2q")),
                "geo_Dc": None if geo is None else _int_or_none(geo.get("Dc")),
                "ratio_delta_e": ratio_delta_e,
                "ratio_D2q": ratio_D2q,
                "ratio_Dc": ratio_Dc,
                "ratio_complete": bool(ratio_complete),
                "ratio_status": ratio_status,
            }
            row["display_cell"] = _compact_ratio_cell(row)
            out.append(row)
    return out


def _geomean(values: Sequence[float]) -> float | None:
    positive = [float(v) for v in values if math.isfinite(float(v)) and float(v) > 0.0]
    if not positive:
        return None
    return float(math.exp(sum(math.log(v) for v in positive) / len(positive)))


def _compact_geomean_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    keys = []
    for row in rows:
        key = str(row.get("comparator_key") or "")
        if key and key not in keys:
            keys.append(key)
    for key in keys:
        subset = [row for row in rows if row.get("comparator_key") == key and bool(row.get("ratio_complete"))]
        out.append(
            {
                "comparator_key": key,
                "comparator_label": str(subset[0].get("comparator_label") if subset else key),
                "complete_regime_count": int(len(subset)),
                "geomean_ratio_delta_e": _geomean([float(row["ratio_delta_e"]) for row in subset]),
                "geomean_ratio_D2q": _geomean([float(row["ratio_D2q"]) for row in subset]),
                "geomean_ratio_Dc": _geomean([float(row["ratio_Dc"]) for row in subset]),
            }
        )
    return out


def _relabel_series(series: Series, *, label: str) -> Series:
    return Series(
        variant_key=series.variant_key,
        label=label,
        method=series.method,
        status=series.status,
        points=series.points,
        marker_k=series.marker_k,
        marker_y=series.marker_y,
        marker_policy=series.marker_policy,
        delta_e=series.delta_e,
        source_json=series.source_json,
        source_sha256=series.source_sha256,
        blocker=series.blocker,
    )


def _build_compact_overlay_series(
    *,
    series_by_regime: Mapping[str, Sequence[Series]],
    monotone_series_by_regime: Mapping[str, Sequence[Series]],
) -> dict[str, list[Series]]:
    out: dict[str, list[Series]] = {}
    for regime in REGIME_ORDER:
        native_by_key = {series.variant_key: series for series in series_by_regime.get(regime, ())}
        monotone_by_key = {series.variant_key: series for series in monotone_series_by_regime.get(regime, ())}
        selected: list[Series] = []
        snake = native_by_key.get(SCHUR_VARIANT_KEY)
        if snake is not None and _snake_child_pool_policy_for_series(snake) == "global_pauli_child_pool":
            selected.append(_relabel_series(snake, label="SNAKE corrected-global child+Schur"))
        geo = monotone_by_key.get("monotone_geo_no_child")
        if geo is not None:
            selected.append(_relabel_series(geo, label="Geo monotone no-child"))
        out[regime] = selected
    return out


def _build_compact_overlay_rows(
    *,
    cost_rows_by_regime: Mapping[str, Sequence[Mapping[str, Any]]],
    monotone_cost_rows_by_regime: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for regime in REGIME_ORDER:
        native_rows = _rows_by_label(cost_rows_by_regime.get(regime, ()))
        monotone_rows = _rows_by_label(monotone_cost_rows_by_regime.get(regime, ()))
        snake = native_rows.get("SNAKE child+Schur")
        geo = monotone_rows.get("Geo no child")
        ratio_delta_e = _ratio_or_none(None if snake is None else snake.get("DeltaE"), None if geo is None else geo.get("DeltaE"))
        ratio_D2q = _ratio_or_none(None if snake is None else snake.get("D2q"), None if geo is None else geo.get("D2q"))
        ratio_Dc = _ratio_or_none(None if snake is None else snake.get("Dc"), None if geo is None else geo.get("Dc"))
        ratio_complete = ratio_delta_e is not None and ratio_D2q is not None and ratio_Dc is not None
        if snake is None:
            ratio_status = "blocked:missing_snake_child_schur_anchor"
        elif snake.get("status") != "done":
            ratio_status = f"blocked:{snake.get('status') or SNAKE_CORRECTED_CHILD_POOL_BLOCKER}"
        elif not _is_corrected_global_snake_anchor(snake):
            ratio_status = f"blocked:{SNAKE_CORRECTED_CHILD_POOL_BLOCKER}"
            ratio_complete = False
        elif geo is None:
            ratio_status = "blocked:missing_geo_monotone_no_child"
        elif not ratio_complete:
            ratio_status = "blocked:missing_numeric_ratio_input"
        else:
            ratio_status = "ok"
        out[regime] = {
            "regime": regime,
            "regime_display": REGIME_DISPLAY.get(regime, regime),
            "snake_row": snake,
            "geo_row": geo,
            "ratio_delta_e": ratio_delta_e,
            "ratio_D2q": ratio_D2q,
            "ratio_Dc": ratio_Dc,
            "ratio_complete": bool(ratio_complete),
            "ratio_status": ratio_status,
        }
    return out


def _compact_overlay_cost_table(row: Mapping[str, Any]) -> str:
    snake = row.get("snake_row") if isinstance(row.get("snake_row"), Mapping) else {}
    geo = row.get("geo_row") if isinstance(row.get("geo_row"), Mapping) else {}
    ratio_delta_e = row.get("ratio_delta_e")
    ratio_D2q = row.get("ratio_D2q")
    ratio_Dc = row.get("ratio_Dc")
    lines = [
        r"\resizebox{0.78\linewidth}{!}{%",
        r"\begin{tabular}{lrrrr}",
        r"\hline",
        r"row & $k$ & $|\Delta E|$ & $D_{2q}$ & $D_c$ \\",
        r"\hline",
        " & ".join(
            [
                "SNAKE",
                _fmt_int(snake.get("k")),
                _tex_escape(snake.get("DeltaE") or "--"),
                _fmt_int(snake.get("D2q")),
                _fmt_int(snake.get("Dc")),
            ]
        )
        + r" \\",
        " & ".join(
            [
                "Geo",
                _fmt_int(geo.get("k")),
                _tex_escape(geo.get("DeltaE") or "--"),
                _fmt_int(geo.get("D2q")),
                _fmt_int(geo.get("Dc")),
            ]
        )
        + r" \\",
        " & ".join(
            [
                "S/G",
                "--",
                _tex_escape(_fmt_ratio(ratio_delta_e)),
                _tex_escape(_fmt_ratio(ratio_D2q)),
                _tex_escape(_fmt_ratio(ratio_Dc)),
            ]
        )
        + r" \\",
        r"\hline",
        r"\end{tabular}%",
        r"}",
    ]
    return "\n".join(lines)


def _append_compact_overlay_page(
    lines: list[str],
    *,
    tex_path: Path,
    figure_paths: Mapping[str, str],
    series_by_regime: Mapping[str, Sequence[Series]],
    overlay_rows_by_regime: Mapping[str, Mapping[str, Any]],
) -> None:
    lines.extend(
        [
            r"\newpage",
            r"\vspace*{-0.14in}",
            r"\begin{center}",
                r"\textbf{Compact trajectories: corrected-global SNAKE child+Schur versus Geo monotone no-child}\\[-0.25em]",
                r"{\scriptsize SNAKE curves and costs on this page are included only when corrected global Pauli-child evidence is present. "
                r"Runtime-child diagnostic anchors from pages 1--2 are blocked; Geo is the monotone/non-forced no-child comparator. S/G ratios below one favor SNAKE.\par}",
            r"\end{center}",
            r"\vspace{-0.06in}",
        ]
    )
    for idx, regime in enumerate(REGIME_ORDER):
        series_list = series_by_regime.get(regime, ())
        blockers = [series for series in series_list if series.status != "done"]
        blocker_text = "; ".join(f"{series.label}: {series.blocker or series.status}" for series in blockers)
        lines.extend(
            [
                r"\begin{minipage}[t]{0.488\linewidth}",
                r"\centering",
                rf"\includegraphics[width=\linewidth]{{{_graphics_path(tex_path, _resolve(figure_paths[regime]))}}}",
                r"\vspace{-0.11in}",
                r"{\tiny",
                _compact_overlay_cost_table(overlay_rows_by_regime[regime]),
                r"}",
            ]
        )
        if blocker_text:
            lines.append(rf"\par\tiny Pending: {_tex_escape(blocker_text)}")
        lines.append(r"\end{minipage}")
        if idx % 2 == 1:
            lines.append(r"\par\vspace{0.018in}")
        else:
            lines.append(r"\hfill")


def _write_compact_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "regime",
        "regime_display",
        "comparator_key",
        "comparator_label",
        "anchor_page",
        "anchor_variant_label",
        "anchor_status",
        "anchor_DeltaE",
        "anchor_D2q",
        "anchor_Dc",
        "anchor_source_json",
        "anchor_source_sha256",
        "geo_page",
        "geo_variant_label",
        "geo_status",
        "geo_DeltaE",
        "geo_D2q",
        "geo_Dc",
        "geo_source_json",
        "geo_source_sha256",
        "ratio_delta_e",
        "ratio_D2q",
        "ratio_Dc",
        "ratio_complete",
        "ratio_status",
        "display_cell",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _write_compact_tex(
    *,
    tex_path: Path,
    generated_utc: str,
    rows: Sequence[Mapping[str, Any]],
    geomean_rows: Sequence[Mapping[str, Any]],
    source_manifest_json: Path,
    csv_path: Path,
    json_path: Path,
    overlay_figure_paths: Mapping[str, str] | None = None,
    overlay_series_by_regime: Mapping[str, Sequence[Series]] | None = None,
    overlay_rows_by_regime: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    by_regime: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        by_regime.setdefault(str(row.get("regime")), {})[str(row.get("comparator_key"))] = row
    comparator_keys = (
        ("geo_native_no_child", "Geo native no-child"),
        ("geo_native_child", "Geo native child"),
        ("geo_monotone_no_child", "Geo monotone no-child"),
        ("geo_monotone_child", "Geo monotone child"),
    )
    lines = _tex_comment_block(
        "MACHINE_READABLE_REPORT_PROVENANCE",
        {
            "schema": "paper_i_hh_compact_child_schur_tex_provenance_v1",
            "generated_utc": generated_utc,
            "tex": _rel(tex_path),
            "source_manifest_json": _rel(source_manifest_json),
            "csv": _rel(csv_path),
            "json": _rel(json_path),
            "snake_child_pool_repair_status": SNAKE_CORRECTED_CHILD_POOL_BLOCKER,
            "pool_exposure_policy": "Corrected SNAKE child rows require global_pauli_child_sets_v1; old runtime child+Schur anchors are blocked.",
        },
    ) + [
        r"\documentclass[10pt]{article}",
        r"\usepackage[landscape,margin=0.35in]{geometry}",
        r"\usepackage{graphicx}",
        r"\usepackage{hyperref}",
        r"\hypersetup{colorlinks=true,linkcolor=black,urlcolor=blue}",
        r"\pagestyle{empty}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{document}",
        r"\sloppy",
        r"\begin{center}",
        r"\textbf{Compact SNAKE child+Schur versus Geo comparison}\\[-0.2em]",
        r"{\scriptsize Corrected-global SNAKE child+Schur is the fixed anchor when available; runtime-child diagnostic anchors are blocked. "
        r"Each cell reports $E=|\Delta E|_{\rm SNAKE}/|\Delta E|_{\rm Geo}$ and "
        r"$D=D_{2q}^{\rm SNAKE}/D_{2q}^{\rm Geo}/D_c^{\rm SNAKE}/D_c^{\rm Geo}$. "
        r"Values below one favor SNAKE.\par}",
        r"\end{center}",
        r"{\scriptsize",
        rf"Generated {_tex_escape(generated_utc)}. Source report: \nolinkurl{{{_rel(source_manifest_json)}}}. "
        rf"Sidecars: \nolinkurl{{{_rel(csv_path)}}}, \nolinkurl{{{_rel(json_path)}}}.",
        r"\par}",
        r"\vspace{0.08in}",
        r"{\small",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lllll}",
        r"\hline",
        r"Regime & Geo native no-child & Geo native child & Geo monotone no-child & Geo monotone child \\",
        r"\hline",
    ]
    for regime in REGIME_ORDER:
        cells = [_tex_escape(REGIME_DISPLAY.get(regime, regime))]
        for key, _label in comparator_keys:
            row = by_regime.get(regime, {}).get(key)
            cells.append(_tex_escape("--" if row is None else _compact_ratio_cell(row)))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}%",
            r"}",
            r"}",
            r"\vspace{0.12in}",
            r"{\scriptsize",
            r"\begin{tabular}{lrrrr}",
            r"\hline",
            r"Comparator & complete regimes & $E$ geomean & $D_{2q}$ geomean & $D_c$ geomean \\",
            r"\hline",
        ]
    )
    for row in geomean_rows:
        lines.append(
            " & ".join(
                [
                    _tex_escape(row.get("comparator_label") or ""),
                    _fmt_int(row.get("complete_regime_count")),
                    _tex_escape(_fmt_ratio(row.get("geomean_ratio_delta_e"))),
                    _tex_escape(_fmt_ratio(row.get("geomean_ratio_D2q"))),
                    _tex_escape(_fmt_ratio(row.get("geomean_ratio_Dc"))),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"}",
            r"\par",
            r"\vspace{0.08in}",
            r"{\scriptsize Missing numeric inputs are blocked and are not included in geometric means. "
            r"The compact view is diagnostic/report-facing; it does not modify manuscript table cells.\par}",
        ]
    )
    if (
        overlay_figure_paths is not None
        and overlay_series_by_regime is not None
        and overlay_rows_by_regime is not None
    ):
        _append_compact_overlay_page(
            lines,
            tex_path=tex_path,
            figure_paths=overlay_figure_paths,
            series_by_regime=overlay_series_by_regime,
            overlay_rows_by_regime=overlay_rows_by_regime,
        )
    lines.extend([r"\end{document}", ""])
    tex_path.write_text("\n".join(lines), encoding="utf-8")


def _append_compact_child_schur_page(
    lines: list[str],
    *,
    generated_utc: str,
    rows: Sequence[Mapping[str, Any]],
    geomean_rows: Sequence[Mapping[str, Any]],
    source_manifest_json: Path,
    csv_path: Path,
    json_path: Path,
) -> None:
    by_regime: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        by_regime.setdefault(str(row.get("regime")), {})[str(row.get("comparator_key"))] = row
    comparator_keys = (
        ("geo_native_no_child", "Geo native no-child"),
        ("geo_native_child", "Geo native child"),
        ("geo_monotone_no_child", "Geo monotone no-child"),
        ("geo_monotone_child", "Geo monotone child"),
    )
    lines.extend(
        [
            r"\newpage",
            r"\vspace*{-0.02in}",
            r"\setlength{\tabcolsep}{3pt}",
            r"\renewcommand{\arraystretch}{1.12}",
            r"\begin{center}",
            r"\textbf{Page 3: compact SNAKE child+Schur versus Geo comparison}\\[-0.2em]",
            r"{\scriptsize Corrected-global SNAKE child+Schur is the fixed anchor when available; runtime-child diagnostic anchors are blocked. "
            r"Each cell reports $E=|\Delta E|_{\rm SNAKE}/|\Delta E|_{\rm Geo}$ and "
            r"$D=D_{2q}^{\rm SNAKE}/D_{2q}^{\rm Geo}/D_c^{\rm SNAKE}/D_c^{\rm Geo}$. "
            r"Values below one favor SNAKE.\par}",
            r"\end{center}",
            r"{\scriptsize",
            rf"Generated {_tex_escape(generated_utc)}. Source report: \nolinkurl{{{_rel(source_manifest_json)}}}. "
            rf"Sidecars: \nolinkurl{{{_rel(csv_path)}}}, \nolinkurl{{{_rel(json_path)}}}.",
            r"\par}",
            r"\vspace{0.08in}",
            r"{\small",
            r"\resizebox{\linewidth}{!}{%",
            r"\begin{tabular}{lllll}",
            r"\hline",
            r"Regime & Geo native no-child & Geo native child & Geo monotone no-child & Geo monotone child \\",
            r"\hline",
        ]
    )
    for regime in REGIME_ORDER:
        cells = [_tex_escape(REGIME_DISPLAY.get(regime, regime))]
        for key, _label in comparator_keys:
            row = by_regime.get(regime, {}).get(key)
            cells.append(_tex_escape("--" if row is None else _compact_ratio_cell(row)))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}%",
            r"}",
            r"}",
            r"\vspace{0.12in}",
            r"{\scriptsize",
            r"\begin{tabular}{lrrrr}",
            r"\hline",
            r"Comparator & complete regimes & $E$ geomean & $D_{2q}$ geomean & $D_c$ geomean \\",
            r"\hline",
        ]
    )
    for row in geomean_rows:
        lines.append(
            " & ".join(
                [
                    _tex_escape(row.get("comparator_label") or ""),
                    _fmt_int(row.get("complete_regime_count")),
                    _tex_escape(_fmt_ratio(row.get("geomean_ratio_delta_e"))),
                    _tex_escape(_fmt_ratio(row.get("geomean_ratio_D2q"))),
                    _tex_escape(_fmt_ratio(row.get("geomean_ratio_Dc"))),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"}",
            r"\par",
            r"\vspace{0.08in}",
            r"{\scriptsize Missing numeric inputs are blocked and are not included in geometric means. "
            r"The compact view is diagnostic/report-facing; it does not modify manuscript table cells.\par}",
        ]
    )


def _powell_matrix_cell(row: Mapping[str, Any] | None) -> str:
    if not isinstance(row, Mapping):
        return "not fetched"
    status = str(row.get("status") or "unknown")
    if status != "done":
        return status
    n2q = row.get("n2q", row.get("N2q"))
    d2q = row.get("d2q", row.get("D2q"))
    s_value = _fmt_compact_int(row.get("s_alg", row.get("S_alg")))
    method_key = str(row.get("method_key") or row.get("method") or "").lower()
    if method_key in {"snake", "snake-adapt", "snake adapt", "snake"} and s_value == "--":
        s_value = "blocked"
    return (
        f"E {_fmt_sci(row.get('abs_delta_e'))}; "
        f"N/D {_fmt_compact_int(n2q)}/{_fmt_compact_int(d2q)}; "
        f"S {s_value}"
    )


def _record_id_regime(record_id: str) -> str:
    for regime in REGIME_ORDER:
        if f"__{regime.replace('-', '_')}__" in record_id:
            return regime
    return ""


def _record_id_method(record_id: str) -> str:
    if "__append__" in record_id:
        return "append"
    if "__geo__" in record_id:
        return "geo"
    if "__snake__" in record_id:
        return "snake"
    return ""


def _record_id_optimizer(record_id: str) -> str:
    if "__powell200__" in record_id:
        return "powell"
    if "__rotosolve200__" in record_id:
        return "rotosolve"
    if "__bfgs200__" in record_id:
        return "bfgs"
    return ""


def _terminal_progress(progress_path: Path) -> dict[str, Any] | None:
    if not progress_path.exists():
        return None
    terminal: dict[str, Any] | None = None
    for line in progress_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, Mapping) and item.get("event") == "terminal":
            terminal = dict(item)
    return terminal


def _final_complete_progress(progress_path: Path) -> dict[str, Any] | None:
    if not progress_path.exists():
        return None
    final: dict[str, Any] | None = None
    for line in progress_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, Mapping) and item.get("event") == "iteration_complete":
            final = dict(item)
    return final


def _stdout_status(stdout_path: Path) -> str | None:
    if not stdout_path.exists():
        return None
    try:
        payload = json.loads(stdout_path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return "stdout_parse_failed"
    if isinstance(payload, Mapping):
        status = payload.get("status")
        return None if status is None else str(status)
    return "stdout_not_mapping"


def _load_retrieved_append_geo_rows(retrieved_dir: Path) -> list[dict[str, Any]]:
    if not retrieved_dir.exists():
        return []
    manifest_dir = retrieved_dir / "raw_outputs" / "retrieval_manifests"
    manifest_paths = sorted(manifest_dir.glob("*.json"))
    record_dirs: dict[str, tuple[Path, str]] = {}
    for manifest_path in manifest_paths:
        try:
            manifest = _read_json(manifest_path)
        except Exception:
            continue
        if not isinstance(manifest, Mapping):
            continue
        records = manifest.get("included_records")
        if not isinstance(records, Sequence):
            continue
        for record in records:
            if not isinstance(record, Mapping):
                continue
            rel = str(record.get("relative_output_dir") or "")
            record_id = str(record.get("record_id") or Path(rel).name)
            if rel and record_id:
                record_dirs[record_id] = (retrieved_dir / rel, _rel(manifest_path))

    rows: list[dict[str, Any]] = []
    for record_id, (record_dir, manifest_rel) in sorted(record_dirs.items()):
        method = _record_id_method(record_id)
        optimizer = _record_id_optimizer(record_id)
        if method not in set(RETRIEVED_METHODS) or optimizer not in set(RETRIEVED_OPTIMIZERS):
            continue
        cell_status: str | None = None
        returncode: Any = None
        cell_manifest_path = record_dir / "cell_manifest.json"
        if cell_manifest_path.exists():
            try:
                cell_manifest = _read_json(cell_manifest_path)
                if isinstance(cell_manifest, Mapping):
                    cell_status = None if cell_manifest.get("status") is None else str(cell_manifest.get("status"))
                    returncode = cell_manifest.get("returncode")
            except Exception:
                cell_status = "cell_manifest_parse_failed"
        progress_path = record_dir / "adapt_iteration_progress.jsonl"
        stdout_path = record_dir / "stdout.log"
        terminal = _terminal_progress(progress_path)
        final_complete = _final_complete_progress(progress_path)
        stdout_status = _stdout_status(stdout_path)
        if stdout_status == "completed" and terminal is not None:
            status = "completed"
        elif stdout_status == "failed":
            status = "failed"
        elif cell_status == "ok":
            status = "cell_ok_no_terminal"
        else:
            status = cell_status or stdout_status or "missing"
        result_metrics = _retrieved_result_metrics(record_dir)
        rows.append(
            {
                "record_id": record_id,
                "regime": _record_id_regime(record_id),
                "method": method,
                "optimizer": optimizer,
                "status": status,
                "cell_status": cell_status,
                "returncode": returncode,
                "stdout_status": stdout_status,
                "abs_delta_e": None if terminal is None else terminal.get("abs_delta_e"),
                "energy": None if terminal is None else terminal.get("energy"),
                "depth": None if terminal is None else terminal.get("depth"),
                "terminal_reason": None if terminal is None else terminal.get("reason"),
                "nfev_total": None if final_complete is None else final_complete.get("nfev_total"),
                "nit_total": None if final_complete is None else final_complete.get("nit_total"),
                "source_dir": _rel(record_dir),
                "source_manifest": manifest_rel,
                "source_sha256": _sha256(stdout_path) if stdout_path.exists() else None,
                "pool_exposure_policy": "global_pauli_child_sets_v1" if method == "snake" else "pauli_child_expanded_comparator_pool",
                **result_metrics,
            }
        )
    return rows


def _with_pending_snake_retrieved_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = [dict(row) for row in rows]
    existing = {
        (str(row.get("optimizer") or ""), str(row.get("regime") or ""), str(row.get("method") or ""))
        for row in out
    }
    for optimizer in RETRIEVED_OPTIMIZERS:
        for regime in REGIME_ORDER:
            for method in RETRIEVED_METHODS:
                key = (optimizer, regime, method)
                if key in existing:
                    continue
                if method == "snake":
                    status = f"pending:{SNAKE_CORRECTED_CHILD_POOL_BLOCKER}"
                    terminal_reason = SNAKE_CORRECTED_CHILD_POOL_BLOCKER
                    pool_policy = "global_pauli_child_sets_v1_required"
                    record_id = f"pending__{regime.replace('-', '_')}__snake__{optimizer}200__global_pauli_child_sets_v1"
                else:
                    status = "pending:missing_retrieved_row"
                    terminal_reason = "missing_retrieved_row"
                    pool_policy = "pauli_child_expanded_comparator_pool"
                    record_id = f"pending__{regime.replace('-', '_')}__{method}__{optimizer}200__pauli_child_expanded"
                out.append(
                    {
                        "record_id": record_id,
                        "regime": regime,
                        "method": method,
                        "optimizer": optimizer,
                        "status": status,
                        "cell_status": None,
                        "returncode": None,
                        "stdout_status": None,
                        "abs_delta_e": None,
                        "energy": None,
                        "depth": None,
                        "terminal_reason": terminal_reason,
                        "nfev_total": None,
                        "nit_total": None,
                        "source_dir": None,
                        "source_manifest": None,
                        "source_sha256": None,
                        "result_json": None,
                        "N2q": None,
                        "D2q": None,
                        "Dc": None,
                        "S_alg": None,
                        "S_alg_status": status,
                        "shots_total_deterministic_proxy": None,
                        "shots_total_deterministic_proxy_status": None,
                        "compiled_resource_source_kind": None,
                        "compiled_circuit_scope": None,
                        "pool_exposure_policy": pool_policy,
                    }
                )
    return out


def _retrieved_result_row(record_dir: Path) -> Mapping[str, Any] | None:
    for candidate in (
        record_dir / "result" / "rows.json",
        record_dir / "result" / "result.json",
        record_dir / "result" / "generic_static_single.json",
        record_dir / "json" / "result.json",
    ):
        if not candidate.exists():
            continue
        try:
            payload = _read_json(candidate)
        except Exception:
            continue
        rows = payload.get("rows") if isinstance(payload, Mapping) else None
        if isinstance(rows, Sequence) and rows and isinstance(rows[0], Mapping):
            return rows[0]
        if isinstance(payload, Mapping):
            extracted = _extract_result_payload(payload)
            if isinstance(extracted, Mapping):
                return extracted
    return None


def _retrieved_result_metrics(record_dir: Path) -> dict[str, Any]:
    row = _retrieved_result_row(record_dir)
    if not isinstance(row, Mapping):
        return {
            "result_json": None,
            "N2q": None,
            "D2q": None,
            "Dc": None,
            "S_alg": None,
            "shots_total_deterministic_proxy": None,
        }
    result_json = next(
        (
            candidate
            for candidate in (
                record_dir / "result" / "rows.json",
                record_dir / "result" / "generic_static_single.json",
                record_dir / "result" / "result.json",
                record_dir / "json" / "result.json",
            )
            if candidate.exists()
        ),
        record_dir / "result" / "rows.json",
    )
    return {
        "result_json": _rel(result_json) if result_json.exists() else None,
        "N2q": row.get("compiled_count_2q_total") or row.get("N2q"),
        "D2q": row.get("compiled_depth_2q_total") or row.get("D2q"),
        "Dc": row.get("compiled_depth_total") or row.get("Dc") or row.get("D_circ"),
        "S_alg": row.get("S_alg"),
        "S_alg_status": "ok" if _int_or_none(row.get("S_alg")) is not None else row.get("S_alg_status"),
        "shots_total_deterministic_proxy": row.get("shots_total"),
        "shots_total_deterministic_proxy_status": row.get("static_shot_estimate_status"),
        "compiled_resource_source_kind": row.get("compiled_resource_source_kind"),
        "compiled_circuit_scope": row.get("compiled_circuit_scope"),
    }


def _complex_from_json(value: Any) -> complex | None:
    if isinstance(value, Mapping):
        real = _float_or_none(value.get("re", value.get("real")))
        imag = _float_or_none(value.get("im", value.get("imag")))
        if real is None and imag is None:
            return None
        return complex(0.0 if real is None else real, 0.0 if imag is None else imag)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2:
        real = _float_or_none(value[0])
        imag = _float_or_none(value[1])
        if real is not None or imag is not None:
            return complex(0.0 if real is None else real, 0.0 if imag is None else imag)
    scalar = _float_or_none(value)
    return None if scalar is None else complex(float(scalar), 0.0)


def _statevector_from_state_payload(state: Mapping[str, Any] | None) -> tuple[Any | None, str]:
    if not isinstance(state, Mapping):
        return None, "missing_state_payload"
    nq = _int_or_none(state.get("nq_total") or state.get("num_qubits") or state.get("nq"))
    amps = state.get("amplitudes_qn_to_q0")
    if nq is None or nq <= 0 or not isinstance(amps, Mapping):
        return None, "missing_statevector_amplitudes"
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        return None, f"numpy_unavailable:{type(exc).__name__}"
    vec = np.zeros(1 << int(nq), dtype=complex)
    populated = 0
    for bitstring, raw_amp in amps.items():
        text = str(bitstring).strip()
        if len(text) != int(nq) or set(text) - {"0", "1"}:
            continue
        amp = _complex_from_json(raw_amp)
        if amp is None:
            continue
        vec[int(text, 2)] = amp
        populated += 1
    if populated == 0:
        return None, "empty_statevector_amplitudes"
    return vec, "statevector_from_ansatz_input_state"


def _terminal_snake_pauli_label_groups(payload: Mapping[str, Any]) -> tuple[list[list[str]] | None, dict[str, Any]]:
    adapt = _extract_result_payload(payload)
    labels_raw = adapt.get("operators") if isinstance(adapt, Mapping) else None
    labels = [str(item) for item in labels_raw if str(item).strip()] if isinstance(labels_raw, Sequence) and not isinstance(labels_raw, (str, bytes)) else []
    parameterization = adapt.get("parameterization") if isinstance(adapt, Mapping) else None
    blocks_raw = parameterization.get("blocks") if isinstance(parameterization, Mapping) else None
    blocks = [block for block in blocks_raw if isinstance(block, Mapping)] if isinstance(blocks_raw, Sequence) and not isinstance(blocks_raw, (str, bytes)) else []
    if not blocks:
        return None, {"status": "missing_parameterization_blocks"}
    if not labels:
        labels = [str(block.get("candidate_label") or "") for block in blocks if str(block.get("candidate_label") or "").strip()]
    used: set[int] = set()
    groups: list[list[str]] = []
    missing: list[str] = []
    for label in labels:
        match_index: int | None = None
        for idx, block in enumerate(blocks):
            if idx in used:
                continue
            if str(block.get("candidate_label") or "") == str(label):
                match_index = idx
                break
        if match_index is None:
            missing.append(str(label))
            continue
        used.add(int(match_index))
        terms = blocks[match_index].get("runtime_terms_exyz")
        if not isinstance(terms, Sequence) or isinstance(terms, (str, bytes)):
            return None, {"status": "bad_runtime_terms", "label": str(label)}
        group: list[str] = []
        for term in terms:
            if isinstance(term, Mapping):
                pauli = str(term.get("pauli_exyz") or "").strip().lower()
                if pauli:
                    group.append(pauli)
        groups.append(group)
    if missing:
        return None, {"status": "selected_label_missing_from_parameterization", "missing_labels": missing[:10], "missing_count": len(missing)}
    return groups, {"status": "ok", "logical_operator_count": len(groups)}


def _num_qubits_from_terminal_groups(groups: Sequence[Sequence[str]], payload: Mapping[str, Any]) -> int | None:
    for group in groups:
        for label in group:
            text = str(label)
            if text:
                return len(text)
    adapt = _extract_result_payload(payload)
    for source in (adapt, payload.get("hamiltonian") if isinstance(payload.get("hamiltonian"), Mapping) else None):
        if not isinstance(source, Mapping):
            continue
        for key in ("num_qubits", "nq", "nq_total", "total_qubits"):
            parsed = _int_or_none(source.get(key))
            if parsed is not None and parsed > 0:
                return int(parsed)
    return None


def _terminal_snake_cost_cache_is_current(cached: Any, source_sha: str | None) -> bool:
    return (
        isinstance(cached, Mapping)
        and cached.get("source_sha256") == source_sha
        and cached.get("work_semantics_version") == SNAKE_TERMINAL_WORK_SEMANTICS_VERSION
    )


def _terminal_snake_s_alg_display_semantics(sidecar: Mapping[str, Any]) -> tuple[str, str | None]:
    """Return whether a terminal SNAKE sidecar may display row-level ``S_alg``.

    Beam-enabled rows carry two valid work numbers: row-aligned winner-lineage
    ``S_alg`` and aggregate all-branch ``S_beam_search_total``.  The latter is
    provenance only and must never be promoted into the row-level table cell.
    """

    if sidecar.get("work_semantics_version") != SNAKE_TERMINAL_WORK_SEMANTICS_VERSION:
        return "stale:missing_or_old_work_semantics_version", "regenerate terminal sidecar"
    s_alg_status = str(sidecar.get("S_alg_status") or "")
    if s_alg_status and s_alg_status != "ok":
        return f"blocked:{s_alg_status}", "S_alg reconstruction did not finish ok"
    beam_total_status = str(sidecar.get("S_beam_search_total_status") or "")
    beam_scope = str(sidecar.get("S_beam_search_scope") or "")
    has_beam_total = sidecar.get("S_beam_search_total") is not None or beam_scope == SNAKE_TERMINAL_BEAM_AGGREGATE_SCOPE
    if not has_beam_total and beam_total_status not in {"", "None", "none", "missing"}:
        has_beam_total = True
    if has_beam_total:
        work_scope = str(sidecar.get("S_alg_work_scope") or "")
        row_policy = str(sidecar.get("S_alg_row_policy") or "")
        if work_scope not in SNAKE_TERMINAL_WINNER_WORK_SCOPES:
            return "blocked:beam_row_s_alg_scope_not_winner_lineage", f"S_alg_work_scope={work_scope or 'missing'}"
        if row_policy != SNAKE_TERMINAL_BEAM_ROW_POLICY:
            return "blocked:beam_row_policy_missing_or_wrong", f"S_alg_row_policy={row_policy or 'missing'}"
        if beam_scope and beam_scope != SNAKE_TERMINAL_BEAM_AGGREGATE_SCOPE:
            return "blocked:unexpected_beam_search_scope", f"S_beam_search_scope={beam_scope}"
    return "ok", None


def _with_terminal_snake_s_alg_display_semantics(sidecar: dict[str, Any]) -> dict[str, Any]:
    status, detail = _terminal_snake_s_alg_display_semantics(sidecar)
    sidecar["S_alg_display_semantics_status"] = status
    sidecar["S_alg_display_semantics_detail"] = detail
    return sidecar


def _compile_local_snake_terminal_cost(record_dir: Path, result_path: Path) -> dict[str, Any]:
    sidecar_path = record_dir / "paper_i_terminal_qiskit_cost.json"
    source_sha = _sha256(result_path) if result_path.exists() else None
    if sidecar_path.exists():
        try:
            cached = _read_json(sidecar_path)
        except Exception:
            cached = None
        if _terminal_snake_cost_cache_is_current(cached, source_sha):
            return dict(cached)
    if not result_path.exists():
        sidecar = {
            "schema": SNAKE_TERMINAL_QISKIT_COST_SCHEMA,
            "work_semantics_version": SNAKE_TERMINAL_WORK_SEMANTICS_VERSION,
            "status": "pending:missing_result_json",
            "source_json": _rel(result_path),
            "source_sha256": None,
        }
        sidecar = _with_terminal_snake_s_alg_display_semantics(sidecar)
        sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return sidecar
    try:
        payload = _read_json(result_path)
    except Exception as exc:
        sidecar = {
            "schema": SNAKE_TERMINAL_QISKIT_COST_SCHEMA,
            "work_semantics_version": SNAKE_TERMINAL_WORK_SEMANTICS_VERSION,
            "status": "blocked:result_json_unreadable",
            "error": f"{type(exc).__name__}: {exc}",
            "source_json": _rel(result_path),
            "source_sha256": source_sha,
        }
        sidecar = _with_terminal_snake_s_alg_display_semantics(sidecar)
        sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return sidecar
    if not isinstance(payload, Mapping):
        sidecar = {
            "schema": SNAKE_TERMINAL_QISKIT_COST_SCHEMA,
            "work_semantics_version": SNAKE_TERMINAL_WORK_SEMANTICS_VERSION,
            "status": "blocked:result_json_not_mapping",
            "source_json": _rel(result_path),
            "source_sha256": source_sha,
        }
        sidecar = _with_terminal_snake_s_alg_display_semantics(sidecar)
        sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return sidecar
    groups, groups_meta = _terminal_snake_pauli_label_groups(payload)
    nq = _num_qubits_from_terminal_groups(groups or (), payload)
    reference_state, reference_state_status = _statevector_from_state_payload(
        payload.get("ansatz_input_state") if isinstance(payload.get("ansatz_input_state"), Mapping) else None
    )
    work: dict[str, Any] = {}
    work_audit: dict[str, Any] = {}
    try:
        from pipelines.exact_bench.snake_table_i_measurement_work import snake_algorithmic_work_from_payload

        work, work_audit = snake_algorithmic_work_from_payload(payload, scope="terminal", source_label=str(result_path))
    except Exception as exc:  # pragma: no cover - reporting should fail closed
        work = {"S_alg": None, "S_alg_status": f"blocked:{type(exc).__name__}"}
        work_audit = {"status": "blocked", "reason": repr(exc)}
    if groups is None or nq is None:
        sidecar = {
            "schema": SNAKE_TERMINAL_QISKIT_COST_SCHEMA,
            "work_semantics_version": SNAKE_TERMINAL_WORK_SEMANTICS_VERSION,
            "status": f"blocked:{groups_meta.get('status') if groups is None else 'num_qubits_missing'}",
            "source_json": _rel(result_path),
            "source_sha256": source_sha,
            "parameterization_reconstruction": groups_meta,
            "S_alg": work.get("S_alg"),
            "S_alg_status": work.get("S_alg_status"),
            "S_alg_work_scope": work.get("S_alg_work_scope"),
            "S_alg_row_policy": work.get("S_alg_row_policy"),
            "S_beam_search_total": work.get("S_beam_search_total"),
            "S_beam_search_total_status": work.get("S_beam_search_total_status"),
            "S_beam_search_scope": work.get("S_beam_search_scope"),
            "S_beam_search_components": work.get("S_beam_search_components"),
            "algorithmic_measurement_work": work.get("algorithmic_measurement_work"),
            "work_reconstruction": work_audit,
        }
        sidecar = _with_terminal_snake_s_alg_display_semantics(sidecar)
        sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        return sidecar
    try:
        from pipelines.exact_bench.table_i_qiskit_resource_compile import (
            TableICompileUnavailable,
            compile_table_i_pauli_label_groups,
        )

        compiled = compile_table_i_pauli_label_groups(
            pauli_label_groups=groups,
            num_qubits=int(nq),
            reference_state=reference_state,
            source_kind="snake_qiskit_compiled_terminal_ansatz_circuit",
        )
        status = "done"
        compile_error = None
    except Exception as exc:  # pragma: no cover - optional Qiskit path
        status = f"blocked:{getattr(exc, 'status', type(exc).__name__)}"
        compile_error = getattr(exc, "reason", str(exc))
        compiled = {}
    adapt = _extract_result_payload(payload)
    sidecar = {
        "schema": SNAKE_TERMINAL_QISKIT_COST_SCHEMA,
        "work_semantics_version": SNAKE_TERMINAL_WORK_SEMANTICS_VERSION,
        "status": status,
        "source_json": _rel(result_path),
        "source_sha256": source_sha,
        "terminal_cost_scope": "terminal_final_ansatz",
        "compile_error": compile_error,
        "N2q": compiled.get("compiled_count_2q_total"),
        "D2q": compiled.get("compiled_depth_2q_total"),
        "Dc": compiled.get("compiled_depth_total"),
        "compiled_count_2q_total": compiled.get("compiled_count_2q_total"),
        "compiled_depth_2q_total": compiled.get("compiled_depth_2q_total"),
        "compiled_depth_total": compiled.get("compiled_depth_total"),
        "compiled_resource_source_kind": compiled.get("compiled_resource_source_kind") or "snake_qiskit_compiled_terminal_ansatz_circuit",
        "compiled_circuit_scope": compiled.get("compiled_circuit_scope"),
        "compile_convention": compiled.get("compile_convention"),
        "logical_operator_count": compiled.get("logical_operator_count") or _int_or_none(adapt.get("ansatz_depth")),
        "runtime_rotation_count": compiled.get("runtime_rotation_count"),
        "num_qubits": nq,
        "reference_state_status": reference_state_status,
        "S_alg": work.get("S_alg"),
        "S_alg_status": work.get("S_alg_status"),
        "S_alg_work_scope": work.get("S_alg_work_scope"),
        "S_alg_row_policy": work.get("S_alg_row_policy"),
        "S_beam_search_total": work.get("S_beam_search_total"),
        "S_beam_search_total_status": work.get("S_beam_search_total_status"),
        "S_beam_search_scope": work.get("S_beam_search_scope"),
        "S_beam_search_components": work.get("S_beam_search_components"),
        "S_alg_N_H_outer_eval": work.get("S_alg_N_H_outer_eval"),
        "S_alg_N_grad_probe": work.get("S_alg_N_grad_probe"),
        "S_alg_N_metric_probe": work.get("S_alg_N_metric_probe"),
        "S_alg_N_H_refit_eval": work.get("S_alg_N_H_refit_eval"),
        "S_alg_N_other_quantum": work.get("S_alg_N_other_quantum"),
        "algorithmic_measurement_work": work.get("algorithmic_measurement_work"),
        "work_reconstruction": work_audit,
        "parameterization_reconstruction": groups_meta,
    }
    sidecar = _with_terminal_snake_s_alg_display_semantics(sidecar)
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return sidecar


def _local_snake_result_metrics(record_dir: Path) -> dict[str, Any]:
    result_path = record_dir / "json" / "result.json"
    cost = _compile_local_snake_terminal_cost(record_dir, result_path)
    return {
        "result_json": _rel(result_path) if result_path.exists() else None,
        "N2q": cost.get("N2q") or cost.get("compiled_count_2q_total"),
        "D2q": cost.get("D2q") or cost.get("compiled_depth_2q_total"),
        "Dc": cost.get("Dc") or cost.get("compiled_depth_total"),
        "S_alg": cost.get("S_alg"),
        "S_alg_status": cost.get("S_alg_status") or cost.get("status"),
        "shots_total_deterministic_proxy": None,
        "shots_total_deterministic_proxy_status": None,
        "compiled_resource_source_kind": cost.get("compiled_resource_source_kind"),
        "compiled_circuit_scope": cost.get("compiled_circuit_scope"),
        "terminal_qiskit_cost_json": _rel(record_dir / "paper_i_terminal_qiskit_cost.json"),
        "terminal_qiskit_cost_status": cost.get("status"),
    }


def _local_terminal_metrics(record_dir: Path, method: str) -> dict[str, Any]:
    if method == "snake":
        for current_path in (record_dir / "json" / "result.json", record_dir / "current.json"):
            if not current_path.exists():
                continue
            try:
                payload = _read_json(current_path)
            except Exception:
                payload = {}
            adapt = payload.get("adapt_vqe") if isinstance(payload, Mapping) else None
            if not isinstance(adapt, Mapping) and isinstance(payload, Mapping):
                adapt = _extract_result_payload(payload)
            if isinstance(adapt, Mapping):
                history = adapt.get("history") or adapt.get("history_tail")
                points = _parse_history_points({"history": history})
                return {
                    "abs_delta_e": _first_number(
                        adapt,
                        "benchmark_target_abs_delta_e_current",
                        "same_cutoff_abs_delta_e",
                        "abs_delta_e",
                    ),
                    "energy": _first_number(adapt, "energy"),
                    "depth": _int_or_none(adapt.get("ansatz_depth") or adapt.get("adapt_depth_reached")),
                    "terminal_reason": adapt.get("stop_reason") or "cell_manifest_ok",
                    "nfev_total": None,
                    "nit_total": None,
                    "trajectory_points": points,
                }
    progress_path = record_dir / "adapt_iteration_progress.jsonl"
    terminal = _terminal_progress(progress_path)
    final_complete = _final_complete_progress(progress_path)
    if terminal is not None:
        return {
            "abs_delta_e": terminal.get("abs_delta_e"),
            "energy": terminal.get("energy"),
            "depth": terminal.get("depth"),
            "terminal_reason": terminal.get("reason"),
            "nfev_total": None if final_complete is None else final_complete.get("nfev_total"),
            "nit_total": None if final_complete is None else final_complete.get("nit_total"),
            "trajectory_points": _parse_progress_points(progress_path, None),
        }
    row = _retrieved_result_row(record_dir)
    if not isinstance(row, Mapping):
        return {
            "abs_delta_e": None,
            "energy": None,
            "depth": None,
            "terminal_reason": None,
            "nfev_total": None,
            "nit_total": None,
            "trajectory_points": (),
        }
    return {
        "abs_delta_e": _first_number(
            row,
            "abs_delta_e_same_cutoff",
            "same_cutoff_abs_delta_e",
            "abs_delta_e",
            "benchmark_target_abs_delta_e_current",
        ),
        "energy": _first_number(row, "energy"),
        "depth": _int_or_none(row.get("adapt_depth_reached") or row.get("ansatz_depth") or row.get("depth")),
        "terminal_reason": row.get("stop_reason") or row.get("terminal_reason") or "cell_manifest_ok",
        "nfev_total": None if method == "snake" else row.get("nfev_total"),
        "nit_total": None if method == "snake" else row.get("nit_total"),
        "trajectory_points": _parse_history_points(row),
    }


def _load_local_optimizer_rows(local_dir: Path) -> list[dict[str, Any]]:
    if not local_dir.exists():
        return []
    completed_states: dict[str, Mapping[str, Any]] = {}
    events_path = local_dir / "alternating_supervisor_events.jsonl"
    if events_path.exists():
        for line in events_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, Mapping) or event.get("event") != "batch_finish":
                continue
            states = event.get("states")
            if not isinstance(states, Sequence):
                continue
            for state in states:
                if isinstance(state, Mapping) and state.get("done") is True:
                    record_id = str(state.get("record_id") or "")
                    if record_id:
                        completed_states[record_id] = state
    rows: list[dict[str, Any]] = []
    for record_dir in sorted(path for path in local_dir.iterdir() if path.is_dir()):
        record_id = record_dir.name
        method = _record_id_method(record_id)
        optimizer = _record_id_optimizer(record_id)
        if method not in set(RETRIEVED_METHODS) or optimizer not in set(RETRIEVED_OPTIMIZERS):
            continue
        cell_status: str | None = None
        returncode: Any = None
        cell_manifest_path = record_dir / "cell_manifest.json"
        if cell_manifest_path.exists():
            try:
                cell_manifest = _read_json(cell_manifest_path)
                if isinstance(cell_manifest, Mapping):
                    cell_status = None if cell_manifest.get("status") is None else str(cell_manifest.get("status"))
                    returncode = cell_manifest.get("returncode")
            except Exception:
                cell_status = "cell_manifest_parse_failed"
        stdout_status = _stdout_status(record_dir / "stdout.log")
        terminal_metrics = _local_terminal_metrics(record_dir, method)
        completed_state = completed_states.get(record_id)
        if method == "snake" and isinstance(completed_state, Mapping):
            current = completed_state.get("current")
            if isinstance(current, Mapping):
                final_depth = _int_or_none(current.get("ansatz_depth"))
                final_abs = _first_number(current, "benchmark_target_abs_delta_e_current", "same_cutoff_abs_delta_e")
                final_energy = _first_number(current, "energy")
                if final_depth is not None:
                    terminal_metrics["depth"] = final_depth
                if final_abs is not None:
                    terminal_metrics["abs_delta_e"] = final_abs
                if final_energy is not None:
                    terminal_metrics["energy"] = final_energy
                terminal_metrics["terminal_reason"] = "supervisor_batch_finish"
                raw_points = terminal_metrics.get("trajectory_points")
                if isinstance(raw_points, Sequence) and final_depth is not None and final_abs is not None:
                    filtered: list[tuple[int, float]] = []
                    for point in raw_points:
                        if not isinstance(point, Sequence) or len(point) < 2:
                            continue
                        x = _int_or_none(point[0])
                        y = _float_or_none(point[1])
                        if x is not None and y is not None and x <= final_depth:
                            filtered.append((int(x), float(y)))
                    filtered = [point for point in filtered if point[0] != final_depth]
                    filtered.append((int(final_depth), float(final_abs)))
                    terminal_metrics["trajectory_points"] = tuple(sorted(filtered))
        if method == "snake":
            result_metrics = _local_snake_result_metrics(record_dir)
        else:
            result_metrics = _retrieved_result_metrics(record_dir)
        has_result = bool(result_metrics.get("result_json"))
        if method == "snake" and not isinstance(completed_state, Mapping) and not (cell_status == "ok" and has_result):
            status = "pending:waiting_batch_finish"
        elif cell_status == "ok" and has_result:
            status = "completed"
        elif stdout_status == "failed":
            status = "failed"
        elif cell_status:
            status = cell_status
        else:
            status = stdout_status or "missing"
        source_path = _resolve(str(result_metrics.get("result_json") or "")) if result_metrics.get("result_json") else None
        rows.append(
            {
                "record_id": record_id,
                "regime": _record_id_regime(record_id),
                "method": method,
                "optimizer": optimizer,
                "status": status,
                "cell_status": cell_status,
                "returncode": returncode,
                "stdout_status": stdout_status,
                "abs_delta_e": terminal_metrics.get("abs_delta_e"),
                "energy": terminal_metrics.get("energy"),
                "depth": terminal_metrics.get("depth"),
                "terminal_reason": terminal_metrics.get("terminal_reason"),
                "nfev_total": terminal_metrics.get("nfev_total"),
                "nit_total": terminal_metrics.get("nit_total"),
                "trajectory_points": terminal_metrics.get("trajectory_points"),
                "source_dir": _rel(record_dir),
                "source_manifest": _rel(cell_manifest_path) if cell_manifest_path.exists() else None,
                "source_sha256": _sha256(source_path) if source_path is not None and source_path.exists() else None,
                "pool_exposure_policy": (
                    "global_pauli_child_sets_v1"
                    if method == "snake"
                    else "pauli_child_expanded_comparator_pool"
                ),
                **result_metrics,
            }
        )
    return rows


def _merge_optimizer_rows(
    retrieved_rows: Sequence[Mapping[str, Any]],
    local_rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in retrieved_rows:
        key = (str(row.get("optimizer") or ""), str(row.get("regime") or ""), str(row.get("method") or ""))
        merged[key] = dict(row)
    for row in summary_rows:
        key = (str(row.get("optimizer") or ""), str(row.get("regime") or ""), str(row.get("method") or ""))
        old = merged.get(key)
        if old is None or old.get("status") != "completed":
            merged[key] = dict(row)
    for row in local_rows:
        key = (str(row.get("optimizer") or ""), str(row.get("regime") or ""), str(row.get("method") or ""))
        old = merged.get(key)
        if old is None or row.get("status") == "completed" or old.get("status") != "completed":
            merged[key] = dict(row)
    return [merged[key] for key in sorted(merged)]


def _resolve_summary_result_path(summary_json: Path, result_path: str) -> Path | None:
    if not result_path:
        return None
    raw = Path(result_path)
    if raw.is_absolute():
        return raw if raw.exists() else None
    for parent in (summary_json.parent, *summary_json.parents):
        candidate = parent / raw
        if candidate.exists():
            return candidate
    candidate = REPO_ROOT / raw
    return candidate if candidate.exists() else None


def _load_powell_summary_comparator_rows(summary_json: Path) -> list[dict[str, Any]]:
    if not summary_json.exists():
        return []
    try:
        payload = _read_json(summary_json)
    except Exception:
        return []
    raw_rows = payload.get("rows") if isinstance(payload, Mapping) else payload
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
        return []
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            continue
        method = str(raw.get("method_key") or _record_id_method(str(raw.get("record_id") or "")) or "").lower()
        optimizer = str(
            raw.get("payload_optimizer_kind")
            or raw.get("record_adapt_optimizer_kind")
            or raw.get("optimizer")
            or _record_id_optimizer(str(raw.get("record_id") or ""))
            or ""
        ).lower()
        if method not in {"append", "geo"} or optimizer != "powell":
            continue
        result_path = _resolve_summary_result_path(summary_json, str(raw.get("result_path") or ""))
        record_dir = result_path.parents[1] if result_path is not None and len(result_path.parents) > 1 else None
        status = "completed" if str(raw.get("status") or "").lower() == "done" else str(raw.get("status") or "missing")
        rows.append(
            {
                "record_id": raw.get("record_id"),
                "regime": raw.get("regime") or _record_id_regime(str(raw.get("record_id") or "")),
                "method": method,
                "optimizer": "powell",
                "status": status,
                "cell_status": None,
                "returncode": 0 if status == "completed" else None,
                "stdout_status": None,
                "abs_delta_e": raw.get("abs_delta_e"),
                "energy": raw.get("energy"),
                "depth": raw.get("depth"),
                "terminal_reason": raw.get("stop"),
                "nfev_total": raw.get("nfev_total"),
                "nit_total": raw.get("nit_total"),
                "source_dir": _rel(record_dir) if record_dir is not None and record_dir.exists() else None,
                "source_manifest": _rel(summary_json),
                "source_sha256": _sha256(result_path) if result_path is not None and result_path.exists() else _sha256(summary_json),
                "pool_exposure_policy": "pauli_child_expanded_comparator_pool",
                "result_json": _rel(result_path) if result_path is not None and result_path.exists() else None,
                "N2q": raw.get("N2q"),
                "D2q": raw.get("D2q"),
                "Dc": raw.get("Dcirc") or raw.get("Dc") or raw.get("D_circ"),
                "S_alg": raw.get("S_alg"),
                "S_alg_status": raw.get("S_alg_status"),
                "shots_total_deterministic_proxy": raw.get("shots_total"),
                "shots_total_deterministic_proxy_status": raw.get("static_shot_estimate_status"),
                "compiled_resource_source_kind": raw.get("compiled_resource_source_kind"),
                "compiled_circuit_scope": raw.get("compiled_circuit_scope"),
            }
        )
    return rows


def _write_retrieved_append_geo_sidecars(
    *,
    csv_path: Path,
    json_path: Path,
    generated_utc: str,
    retrieved_dir: Path,
    local_optimizer_dir: Path,
    powell_summary_json: Path,
) -> dict[str, Any] | None:
    retrieved_rows = _load_retrieved_append_geo_rows(retrieved_dir)
    local_rows = _load_local_optimizer_rows(local_optimizer_dir)
    summary_rows = _load_powell_summary_comparator_rows(powell_summary_json)
    parsed_rows = _merge_optimizer_rows(retrieved_rows, local_rows, summary_rows)
    rows = _with_pending_snake_retrieved_rows(parsed_rows)
    if not rows:
        return None
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "regime",
        "method",
        "optimizer",
        "status",
        "cell_status",
        "stdout_status",
        "returncode",
        "abs_delta_e",
        "energy",
        "depth",
        "nfev_total",
        "nit_total",
        "terminal_reason",
        "N2q",
        "D2q",
        "Dc",
        "S_alg",
        "S_alg_status",
        "shots_total_deterministic_proxy",
        "shots_total_deterministic_proxy_status",
        "compiled_resource_source_kind",
        "compiled_circuit_scope",
        "terminal_qiskit_cost_json",
        "terminal_qiskit_cost_status",
        "pool_exposure_policy",
        "result_json",
        "source_dir",
        "source_manifest",
        "source_sha256",
        "record_id",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    by_optimizer: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for row in rows:
        optimizer = str(row.get("optimizer") or "")
        regime = str(row.get("regime") or "")
        method = str(row.get("method") or "")
        if optimizer and regime and method:
            by_optimizer.setdefault(optimizer, {}).setdefault(regime, {})[method] = row
    payload = {
        "schema": "paper_i_hh_child_fairness_retrieved_optimizer_rows_v3",
        "generated_utc": str(generated_utc),
        "diagnostic_scope": (
            "Retrieved and local Geo-ADAPT, append-only ADAPT, and SNAKE optimizer-study cells. SNAKE rows require "
            "the corrected global Pauli-child pool exposure and remain pending until those reruns complete."
        ),
        "retrieved_dir": _rel(retrieved_dir) if retrieved_dir.exists() else str(retrieved_dir),
        "local_optimizer_dir": _rel(local_optimizer_dir) if local_optimizer_dir.exists() else str(local_optimizer_dir),
        "powell_summary_json": _rel(powell_summary_json) if powell_summary_json.exists() else str(powell_summary_json),
        "csv": _rel(csv_path),
        "json": _rel(json_path),
        "retrieved_row_count": len(retrieved_rows),
        "local_row_count": len(local_rows),
        "powell_summary_row_count": len(summary_rows),
        "parsed_row_count": len(parsed_rows),
        "row_count": len(rows),
        "completed_count": sum(1 for row in rows if row.get("status") == "completed"),
        "failed_count": sum(1 for row in rows if row.get("status") == "failed"),
        "pending_snake_count": sum(
            1
            for row in rows
            if row.get("method") == "snake" and str(row.get("status") or "").startswith("pending:")
        ),
        "pool_exposure_policy": {
            "append_geo": "Pauli-child expanded comparator pool from retrieved records.",
            "snake": "Requires global_pauli_child_sets_v1; old phase-3 runtime child shortlisting is blocked.",
        },
        "rows": rows,
        "matrix": by_optimizer,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _retrieved_matrix_cell(row: Mapping[str, Any] | None) -> str:
    if not isinstance(row, Mapping):
        return "--"
    if row.get("status") == "completed":
        return (
            f"E {_fmt_sci(row.get('abs_delta_e'))}; "
            f"d {_fmt_compact_int(row.get('depth'))}; "
            f"nfev {_fmt_compact_int(row.get('nfev_total'))}"
        )
    if row.get("status") == "failed":
        return "failed: no terminal"
    return str(row.get("status") or "--")


def _retrieved_label(row: Mapping[str, Any]) -> str:
    method = str(row.get("method") or "")
    optimizer = str(row.get("optimizer") or "")
    method_label = {"append": "Append", "geo": "Geo", "snake": "SNAKE"}.get(method, method or "--")
    opt_label = {"powell": "Powell", "rotosolve": "Rotosolve"}.get(optimizer, optimizer or "--")
    return f"{method_label} {opt_label}"


def _retrieved_method(row: Mapping[str, Any]) -> str:
    return {"append": "Append-ADAPT", "geo": "Geo-ADAPT", "snake": "SNAKE"}.get(
        str(row.get("method") or ""), "Append-ADAPT"
    )


def _retrieved_series_from_row(row: Mapping[str, Any] | None) -> Series | None:
    if not isinstance(row, Mapping):
        return None
    label = _retrieved_label(row)
    method = _retrieved_method(row)
    variant_key = f"retrieved_{row.get('optimizer')}_{row.get('method')}"
    source_dir_raw = str(row.get("source_dir") or "")
    source_dir = _resolve(source_dir_raw) if source_dir_raw else None
    points: tuple[tuple[int, float], ...] = ()
    raw_points = row.get("trajectory_points")
    if isinstance(raw_points, Sequence) and not isinstance(raw_points, (str, bytes)):
        parsed_points: list[tuple[int, float]] = []
        for point in raw_points:
            if not isinstance(point, Sequence) or len(point) < 2:
                continue
            x = _int_or_none(point[0])
            y = _float_or_none(point[1])
            if x is not None and y is not None and y > 0.0:
                parsed_points.append((int(x), float(y)))
        points = tuple(sorted(parsed_points))
    if not points:
        result_row = _retrieved_result_row(source_dir) if source_dir is not None else None
        points = _parse_history_points(result_row) if isinstance(result_row, Mapping) else ()
    if source_dir is not None:
        terminal_energy = _float_or_none(row.get("energy"))
        terminal_error = _float_or_none(row.get("abs_delta_e"))
        exact_energy = None
        if terminal_energy is not None and terminal_error is not None:
            exact_energy = float(terminal_energy) - float(terminal_error)
        progress_points = _parse_progress_points(source_dir / "adapt_iteration_progress.jsonl", exact_energy)
        if len(progress_points) > len(points):
            points = progress_points
    delta_e = _float_or_none(row.get("abs_delta_e"))
    if delta_e is not None and points:
        points = tuple(points[:-1]) + ((int(points[-1][0]), float(delta_e)),)
    if not points and delta_e is not None:
        depth = _int_or_none(row.get("depth")) or 0
        points = ((depth, float(delta_e)),)
    status = str(row.get("status") or "")
    done = status == "completed" and bool(points)
    return Series(
        variant_key=variant_key,
        label=label,
        method=method,
        status="done" if done else "pending",
        points=points if done else (),
        marker_k=(int(points[-1][0]) if done else None),
        marker_y=(float(points[-1][1]) if done else None),
        marker_policy="terminal_retrieved_optimizer_row" if done else "pending",
        delta_e=delta_e if done else None,
        source_json=str(row.get("result_json") or source_dir_raw or "") or None,
        source_sha256=str(row.get("source_sha256") or "") or None,
        blocker=None if done else status or "missing_retrieved_row",
    )


def _build_retrieved_append_geo_series(
    retrieved_payload: Mapping[str, Any] | None,
) -> dict[str, list[Series]]:
    out: dict[str, list[Series]] = {regime: [] for regime in REGIME_ORDER}
    if not isinstance(retrieved_payload, Mapping):
        return out
    matrix = retrieved_payload.get("matrix") if isinstance(retrieved_payload.get("matrix"), Mapping) else {}
    for regime in REGIME_ORDER:
        series: list[Series] = []
        for optimizer in RETRIEVED_OPTIMIZERS:
            opt_rows = matrix.get(optimizer) if isinstance(matrix.get(optimizer), Mapping) else {}
            regime_rows = opt_rows.get(regime) if isinstance(opt_rows.get(regime), Mapping) else {}
            for method in RETRIEVED_METHODS:
                item = _retrieved_series_from_row(regime_rows.get(method) if isinstance(regime_rows, Mapping) else None)
                if item is not None:
                    series.append(item)
        out[regime] = series
    return out


def _retrieved_cost_row(row: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        return {
            "variant_label": "--",
            "optimizer": None,
            "method": None,
            "status": "pending:missing_retrieved_row",
            "k": None,
            "DeltaE": None,
            "F": None,
            "theta_count": None,
            "N2q": None,
            "D2q": None,
            "Dc": None,
            "S": None,
            "S_unfair": None,
            "S_unfair_status": "pending:missing_retrieved_row",
            "S_fair": None,
            "S_fair_status": "pending:missing_retrieved_row",
            "source_json": None,
            "source_sha256": None,
        }
    if row.get("status") != "completed":
        raw_status = str(row.get("status") or "pending")
        status = raw_status if raw_status.startswith("pending:") or raw_status == "failed" else f"pending:{raw_status}"
        return {
            "variant_label": _retrieved_label(row),
            "optimizer": row.get("optimizer"),
            "method": row.get("method"),
            "status": status,
            "k": None,
            "DeltaE": None,
            "F": None,
            "theta_count": None,
            "N2q": None,
            "D2q": None,
            "Dc": None,
            "S": None,
            "S_unfair": None,
            "S_unfair_status": status,
            "S_fair": None,
            "S_fair_status": status,
            "source_json": row.get("result_json") or row.get("source_dir"),
            "source_sha256": row.get("source_sha256"),
        }
    return {
        "variant_label": _retrieved_label(row),
        "optimizer": row.get("optimizer"),
        "method": row.get("method"),
        "status": "done",
        "k": row.get("depth"),
        "DeltaE": _fmt_sci(row.get("abs_delta_e")),
        "F": None,
        "theta_count": row.get("depth"),
        "N2q": row.get("N2q"),
        "D2q": row.get("D2q"),
        "Dc": row.get("Dc"),
        "S": row.get("S_alg"),
        "S_unfair": row.get("shots_total_deterministic_proxy"),
        "S_unfair_status": row.get("shots_total_deterministic_proxy_status") or "legacy:deterministic_pauli_term_proxy",
        "S_unfair_source_kind": "legacy_deterministic_pauli_term_proxy",
        "S_fair": row.get("S_alg"),
        "S_fair_status": row.get("S_alg_status") or "ok",
        "S_fair_source_kind": "algorithmic_measurement_work_S_alg",
        "source_json": row.get("result_json") or row.get("source_dir"),
        "source_sha256": row.get("source_sha256"),
        "compile_convention": row.get("compiled_resource_source_kind"),
        "compiled_circuit_scope": row.get("compiled_circuit_scope"),
    }


def _build_retrieved_append_geo_cost_rows(
    retrieved_payload: Mapping[str, Any] | None,
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {regime: [] for regime in REGIME_ORDER}
    if not isinstance(retrieved_payload, Mapping):
        return out
    matrix = retrieved_payload.get("matrix") if isinstance(retrieved_payload.get("matrix"), Mapping) else {}
    for regime in REGIME_ORDER:
        rows: list[dict[str, Any]] = []
        for optimizer in RETRIEVED_OPTIMIZERS:
            opt_rows = matrix.get(optimizer) if isinstance(matrix.get(optimizer), Mapping) else {}
            regime_rows = opt_rows.get(regime) if isinstance(opt_rows.get(regime), Mapping) else {}
            for method in RETRIEVED_METHODS:
                row = _retrieved_cost_row(regime_rows.get(method) if isinstance(regime_rows, Mapping) else None)
                if row.get("variant_label") != "--":
                    rows.append(row)
        out[regime] = rows
    return out


def _retrieved_series_by_optimizer(
    series_by_regime: Mapping[str, Sequence[Series]],
) -> dict[str, dict[str, list[Series]]]:
    return {
        optimizer: {
            regime: [
                series
                for series in series_by_regime.get(regime, ())
                if series.variant_key.startswith(f"retrieved_{optimizer}_")
            ]
            for regime in REGIME_ORDER
        }
        for optimizer in RETRIEVED_OPTIMIZERS
    }


def _retrieved_cost_rows_by_optimizer(
    cost_rows_by_regime: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, dict[str, list[Mapping[str, Any]]]]:
    return {
        optimizer: {
            regime: [
                row
                for row in cost_rows_by_regime.get(regime, ())
                if str(row.get("optimizer") or "") == optimizer
            ]
            for regime in REGIME_ORDER
        }
        for optimizer in RETRIEVED_OPTIMIZERS
    }


def _retrieved_cost_table(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lrrrrrr}",
        r"\hline",
        r"row & $k$ & $|\Delta E|$ & $N_{2q}$ & $D_{2q}$ & $D_c$ & $S_{\rm alg}$ \\",
        r"\hline",
    ]
    for row in rows:
        if row.get("status") == "done":
            values = [
                _tex_escape(row.get("variant_label") or ""),
                _fmt_int(row.get("k")),
                _tex_escape(row.get("DeltaE") or "--"),
                _fmt_int(row.get("N2q")),
                _fmt_int(row.get("D2q")),
                _fmt_int(row.get("Dc")),
                _fmt_int(row.get("S_fair")),
            ]
        else:
            status = str(row.get("status") or "pending")
            if SNAKE_CORRECTED_CHILD_POOL_BLOCKER in status:
                status = "pending SNAKE rerun"
            values = [
                _tex_escape(row.get("variant_label") or ""),
                "--",
                _tex_escape(status),
                "--",
                "--",
                "--",
                "--",
            ]
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}%", r"}"])
    return "\n".join(lines)


def _append_retrieved_append_geo_page(
    lines: list[str],
    *,
    tex_path: Path,
    generated_utc: str,
    retrieved_payload: Mapping[str, Any] | None,
    csv_path: Path,
    json_path: Path,
    figure_paths_by_optimizer: Mapping[str, Mapping[str, str]],
    series_by_optimizer: Mapping[str, Mapping[str, Sequence[Series]]],
    cost_rows_by_optimizer: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> None:
    if not isinstance(retrieved_payload, Mapping):
        return
    optimizer_titles = {"powell": "Powell", "rotosolve": "ROTOSOLVE"}
    for opt_idx, optimizer in enumerate(RETRIEVED_OPTIMIZERS):
        opt_label = optimizer_titles.get(optimizer, optimizer)
        page_label = "Page 5" if opt_idx == 0 else "Page 6"
        opt_series = series_by_optimizer.get(optimizer, {})
        opt_cost_rows = cost_rows_by_optimizer.get(optimizer, {})
        opt_figures = figure_paths_by_optimizer.get(optimizer, {})
        opt_completed = sum(
            1
            for regime in REGIME_ORDER
            for row in opt_cost_rows.get(regime, ())
            if row.get("status") == "done"
        )
        lines.extend(
            [
                r"\newpage",
                r"\vspace*{-0.14in}",
                r"\setlength{\tabcolsep}{2pt}",
                r"\renewcommand{\arraystretch}{0.82}",
                r"\begin{center}",
                rf"\textbf{{{page_label}: {opt_label} fetched/local Geo/Append/SNAKE trajectories and costs}}\\[-0.25em]",
                r"{\scriptsize Diagnostic optimizer-study rows only; these do not replace the active SPSA child-fairness cells. "
                r"Plots and cost tables are separated by inner optimizer. "
                r"Cost tables use compiled final-ansatz Qiskit resources when emitted; $S_{\rm fair}=S_{\rm alg}$ when available. "
                r"nfev is intentionally not displayed as a shot/work proxy. SNAKE entries use corrected global Pauli-child rows when available; pending rows are explicit.\par}",
                r"\end{center}",
                r"{\scriptsize",
                rf"Generated {_tex_escape(generated_utc)}. Sidecars: {_tex_escape(csv_path.name)}, {_tex_escape(json_path.name)}. "
                rf"{opt_label} completed rows displayed: {_fmt_int(opt_completed)}. "
                rf"All optimizers parsed: {_fmt_int(retrieved_payload.get('parsed_row_count'))} ({_fmt_int(retrieved_payload.get('local_row_count'))} local); "
                rf"displayed rows: {_fmt_int(retrieved_payload.get('row_count'))}; failed result rows: {_fmt_int(retrieved_payload.get('failed_count'))}; pending SNAKE rows: {_fmt_int(retrieved_payload.get('pending_snake_count'))}. "
                rf"Retrieved root: \nolinkurl{{{_tex_escape(str(retrieved_payload.get('retrieved_dir') or '--'))}}}; "
                rf"local root: \nolinkurl{{{_tex_escape(str(retrieved_payload.get('local_optimizer_dir') or '--'))}}}.",
                r"\par}",
                r"\vspace{0.035in}",
            ]
        )
        for idx, regime in enumerate(REGIME_ORDER):
            figure_path = opt_figures.get(regime)
            if not figure_path:
                continue
            series_list = opt_series.get(regime, ())
            blockers = [series for series in series_list if series.status != "done"]
            blocker_text = "; ".join(f"{series.label}: {series.blocker or series.status}" for series in blockers)
            del blocker_text
            lines.extend(
                [
                    r"\begin{minipage}[t]{0.488\linewidth}",
                    r"\centering",
                    rf"\includegraphics[width=\linewidth]{{{_graphics_path(tex_path, _resolve(figure_path))}}}",
                    r"\vspace{-0.055in}",
                    r"{\tiny",
                    _retrieved_cost_table(opt_cost_rows.get(regime, ())),
                    r"}",
                ]
            )
            lines.append(r"\end{minipage}")
            if idx % 2 == 1:
                lines.append(r"\par\vspace{0.11in}")
            else:
                lines.append(r"\hfill")


def _write_optimizer_addendum_sidecars(
    *,
    csv_path: Path,
    json_path: Path,
    generated_utc: str,
    powell_summary_json: Path,
    powell_tuning_summary_json: Path,
    powell_tuning_overlay_png: Path,
    optimizer_crosscheck_preflight_json: Path,
) -> dict[str, Any] | None:
    if not powell_summary_json.exists() and not powell_tuning_summary_json.exists():
        return None

    powell_rows: list[dict[str, Any]] = []
    powell_summary_payload: dict[str, Any] = {}
    if powell_summary_json.exists():
        payload = _read_json(powell_summary_json)
        if isinstance(payload, list):
            powell_rows = [dict(row) for row in payload if isinstance(row, Mapping)]
        elif isinstance(payload, Mapping):
            powell_summary_payload = dict(payload)
            raw_rows = payload.get("rows")
            if isinstance(raw_rows, list):
                powell_rows = [dict(row) for row in raw_rows if isinstance(row, Mapping)]

    tuning_payload: dict[str, Any] = {}
    tuning_rows: list[dict[str, Any]] = []
    if powell_tuning_summary_json.exists():
        raw_tuning = _read_json(powell_tuning_summary_json)
        if isinstance(raw_tuning, Mapping):
            tuning_payload = dict(raw_tuning)
            raw_rows = raw_tuning.get("result_rows")
            if isinstance(raw_rows, list):
                tuning_rows = [dict(row) for row in raw_rows if isinstance(row, Mapping)]

    crosscheck_payload: dict[str, Any] = {}
    if optimizer_crosscheck_preflight_json.exists():
        raw_crosscheck = _read_json(optimizer_crosscheck_preflight_json)
        if isinstance(raw_crosscheck, Mapping):
            crosscheck_payload = dict(raw_crosscheck)

    by_regime_method: dict[str, dict[str, dict[str, Any]]] = {}
    for row in powell_rows:
        regime = str(row.get("regime") or "")
        method = str(row.get("method_key") or row.get("method") or "").lower()
        method = {
            "snake": "snake",
            "snake-adapt": "snake",
            "geo-adapt": "geo",
            "append-only adapt": "append",
            "append-adapt": "append",
        }.get(method, method)
        if regime and method:
            by_regime_method.setdefault(regime, {})[method] = row

    powell_done = sum(1 for row in powell_rows if row.get("status") == "done")
    powell_pending = sum(1 for row in powell_rows if row.get("status") != "done")
    powell_fetched = len(powell_rows)
    powell_status_counts = powell_summary_payload.get("status_counts") if isinstance(powell_summary_payload.get("status_counts"), Mapping) else {
        "done": powell_done,
        "not_done": powell_pending,
    }
    powell_record_optimizers = sorted(
        {
            str(row.get("optimizer") or row.get("payload_optimizer_kind") or row.get("record_adapt_optimizer_kind") or "").lower()
            for row in powell_rows
            if row.get("status") == "done"
            and (row.get("optimizer") or row.get("payload_optimizer_kind") or row.get("record_adapt_optimizer_kind"))
        }
    )

    crosscheck_ids = crosscheck_payload.get("record_ids") if isinstance(crosscheck_payload.get("record_ids"), list) else []
    crosscheck_snake_bfgs = sum(1 for rid in crosscheck_ids if "__snake__bfgs200__" in str(rid))
    crosscheck_geo_powell = sum(1 for rid in crosscheck_ids if "__geo__powell200__" in str(rid))
    crosscheck_append_powell = sum(1 for rid in crosscheck_ids if "__append__powell200__" in str(rid))

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        fieldnames = [
            "source_group",
            "regime",
            "method",
            "label",
            "status",
            "optimizer",
            "abs_delta_e",
            "depth",
            "N2q",
            "D2q",
            "Dcirc",
            "S",
            "source_path",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in powell_rows:
            writer.writerow(
                {
                    "source_group": "powell_all_regimes_partial_fetch",
                    "regime": row.get("regime"),
                    "method": row.get("method"),
                    "label": row.get("record_id"),
                    "status": row.get("status"),
                    "optimizer": row.get("optimizer") or row.get("payload_optimizer_kind") or row.get("record_adapt_optimizer_kind"),
                    "abs_delta_e": row.get("abs_delta_e"),
                    "depth": row.get("depth"),
                    "N2q": row.get("n2q", row.get("N2q")),
                    "D2q": row.get("d2q", row.get("D2q")),
                    "Dcirc": row.get("dcirc", row.get("Dcirc")),
                    "S": row.get("s_alg", row.get("S_alg")),
                    "source_path": row.get("record_id"),
                }
            )
        for row in tuning_rows:
            writer.writerow(
                {
                    "source_group": "weak_weak_powell_snake_tuning",
                    "regime": row.get("regime"),
                    "method": row.get("method"),
                    "label": row.get("key"),
                    "status": "done",
                    "optimizer": "powell",
                    "abs_delta_e": row.get("abs_delta_e"),
                    "depth": row.get("ansatz_depth"),
                    "N2q": row.get("N2q"),
                    "D2q": row.get("D2q"),
                    "Dcirc": row.get("Dcirc"),
                    "S": row.get("S"),
                    "source_path": row.get("path"),
                }
            )

    payload = {
        "schema": "paper_i_hh_child_fairness_optimizer_addendum_v1",
        "generated_utc": str(generated_utc),
        "diagnostic_scope": (
            "Optimizer-study addendum only. These Powell/BFGS rows do not replace the active "
            "SPSA fair-comparison cells in Paper I."
        ),
        "powell_summary_json": _rel(powell_summary_json) if powell_summary_json.exists() else str(powell_summary_json),
        "powell_summary_json_sha256": _sha256(powell_summary_json) if powell_summary_json.exists() else None,
        "powell_tuning_summary_json": _rel(powell_tuning_summary_json)
        if powell_tuning_summary_json.exists()
        else str(powell_tuning_summary_json),
        "powell_tuning_summary_json_sha256": _sha256(powell_tuning_summary_json)
        if powell_tuning_summary_json.exists()
        else None,
        "powell_tuning_overlay_png": _rel(powell_tuning_overlay_png)
        if powell_tuning_overlay_png.exists()
        else str(powell_tuning_overlay_png),
        "powell_tuning_overlay_png_sha256": _sha256(powell_tuning_overlay_png)
        if powell_tuning_overlay_png.exists()
        else None,
        "optimizer_crosscheck_preflight_json": _rel(optimizer_crosscheck_preflight_json)
        if optimizer_crosscheck_preflight_json.exists()
        else str(optimizer_crosscheck_preflight_json),
        "optimizer_crosscheck_preflight_json_sha256": _sha256(optimizer_crosscheck_preflight_json)
        if optimizer_crosscheck_preflight_json.exists()
        else None,
        "csv": _rel(csv_path),
        "json": _rel(json_path),
        "powell_partial_fetch": {
            "row_count": powell_fetched,
            "done_count": powell_done,
            "pending_count": powell_pending,
            "status_counts": dict(powell_status_counts),
            "fetched_done_optimizers": powell_record_optimizers,
            "matrix": by_regime_method,
            "source_note": powell_summary_payload.get("note"),
            "optimizer_field_note": powell_summary_payload.get("optimizer_field_note"),
        },
        "weak_weak_tuning": {
            "result_rows": tuning_rows,
            "ratios_tuned_snake_vs_geo": tuning_payload.get("weak_weak_ratios_tuned_snake_vs_geo"),
            "diagnosis": tuning_payload.get("diagnosis"),
            "intermediate_transfer": tuning_payload.get("intermediate_transfer"),
        },
        "optimizer_crosscheck_preflight": {
            "ok": crosscheck_payload.get("ok"),
            "record_count": crosscheck_payload.get("record_count"),
            "snake_bfgs_record_count": crosscheck_snake_bfgs,
            "geo_powell_record_count": crosscheck_geo_powell,
            "append_powell_record_count": crosscheck_append_powell,
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _append_optimizer_diagnostic_page(
    lines: list[str],
    *,
    tex_path: Path,
    generated_utc: str,
    optimizer_payload: Mapping[str, Any] | None,
    csv_path: Path,
    json_path: Path,
) -> None:
    if not isinstance(optimizer_payload, Mapping):
        return
    powell = optimizer_payload.get("powell_partial_fetch")
    powell = powell if isinstance(powell, Mapping) else {}
    weak = optimizer_payload.get("weak_weak_tuning")
    weak = weak if isinstance(weak, Mapping) else {}
    crosscheck = optimizer_payload.get("optimizer_crosscheck_preflight")
    crosscheck = crosscheck if isinstance(crosscheck, Mapping) else {}
    matrix = powell.get("matrix") if isinstance(powell.get("matrix"), Mapping) else {}
    tuning_rows = weak.get("result_rows") if isinstance(weak.get("result_rows"), list) else []
    overlay_png = _resolve(str(optimizer_payload.get("powell_tuning_overlay_png") or ""))
    overlay_exists = overlay_png.exists()

    lines.extend(
        [
            r"\newpage",
            r"\vspace*{-0.02in}",
            r"\setlength{\tabcolsep}{2.6pt}",
            r"\renewcommand{\arraystretch}{1.03}",
            r"\begin{center}",
            r"\textbf{Page 4: Powell/BFGS optimizer diagnostic addendum}\\[-0.2em]",
            r"{\scriptsize This page is diagnostic/report-facing. Powell/BFGS rows are an optimizer study and do not replace the active SPSA fair-comparison cells. "
            r"The fetched Powell rows report Powell in both submitted records and fetched result metadata; SNAKE-BFGS appears in a separate optimizer-crosscheck preflight.\par}",
            r"\end{center}",
            r"{\scriptsize",
            rf"Generated {_tex_escape(generated_utc)}. Sidecars: {_tex_escape(csv_path.name)}, {_tex_escape(json_path.name)}. "
            rf"Powell retrieval rows: {_fmt_int(powell.get('done_count'))} done, {_fmt_int(powell.get('pending_count'))} running/missing at fetch time. "
            rf"Crosscheck preflight: {_fmt_int(crosscheck.get('snake_bfgs_record_count'))} SNAKE-BFGS records, "
            rf"{_fmt_int(crosscheck.get('geo_powell_record_count'))} Geo-Powell records, "
            rf"{_fmt_int(crosscheck.get('append_powell_record_count'))} Append-Powell records.",
            r"\par}",
            r"\vspace{0.05in}",
        ]
    )

    lines.extend([r"\begin{minipage}[t]{0.43\linewidth}", r"\vspace{0pt}", r"\centering"])
    if overlay_exists:
        lines.append(rf"\includegraphics[width=0.98\linewidth,height=2.28in,keepaspectratio]{{{_graphics_path(tex_path, overlay_png)}}}")
        lines.append(r"\vspace{0.02in}")
    lines.extend(
        [
            r"{\tiny",
            r"\resizebox{\linewidth}{!}{%",
            r"\begin{tabular}{llllll}",
            r"\hline",
            r"Row & $|\Delta E|$ & depth & $N_{2q}$ & $D_{2q}$ & $S$ \\",
            r"\hline",
        ]
    )
    label_map = {
        "weak_append_powell_native_child": "Append Powell",
        "weak_geo_powell_native_child": "Geo Powell",
        "weak_snake_baseline_powell_child_schur": "SNAKE Powell base",
        "weak_snake_wide_rho050": "SNAKE wide rho",
        "weak_snake_wide_beam6_rho050": "SNAKE wide+beam",
        "intermediate_snake_transfer_wide_beam6_rho050": "SNAKE int.-weak transfer",
    }
    for row in tuning_rows:
        if not isinstance(row, Mapping):
            continue
        key = str(row.get("key") or "")
        if key not in label_map:
            continue
        lines.append(
            " & ".join(
                [
                    _tex_escape(label_map[key]),
                    _tex_escape(_fmt_sci(row.get("abs_delta_e"))),
                    _fmt_compact_int(row.get("ansatz_depth")),
                    _fmt_compact_int(row.get("N2q")),
                    _fmt_compact_int(row.get("D2q")),
                    _fmt_compact_int(row.get("S")),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\hline", r"\end{tabular}%", r"}", r"}", r"\end{minipage}", r"\hfill"])

    lines.extend(
        [
            r"\begin{minipage}[t]{0.55\linewidth}",
            r"\vspace{0pt}",
            r"{\tiny",
            r"\resizebox{\linewidth}{!}{%",
            r"\begin{tabular}{llll}",
            r"\hline",
            r"Regime & Append Powell & Geo Powell & SNAKE Powell \\",
            r"\hline",
        ]
    )
    for regime in REGIME_ORDER:
        regime_rows = matrix.get(regime) if isinstance(matrix.get(regime), Mapping) else {}
        cells = [
            _tex_escape(REGIME_DISPLAY.get(regime, regime)),
            _tex_escape(_powell_matrix_cell(regime_rows.get("append") if isinstance(regime_rows, Mapping) else None)),
            _tex_escape(_powell_matrix_cell(regime_rows.get("geo") if isinstance(regime_rows, Mapping) else None)),
            _tex_escape(_powell_matrix_cell(regime_rows.get("snake") if isinstance(regime_rows, Mapping) else None)),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}%",
            r"}",
            r"}",
            r"\vspace{0.05in}",
            r"{\scriptsize Local interpretation: the all-regime Powell retrieval has one unfinished Geo row and SNAKE $S$ remains blocked there by the strict accounting repair. "
            r"The separate optimizer-crosscheck batch records SNAKE BFGS200+Schur warm start with Powell200 comparators, but no completed-result sidecar is included on this page.\par}",
            r"\end{minipage}",
        ]
    )


def _write_compact_child_schur_artifacts(
    *,
    output_dir: Path,
    compact_stem: str,
    generated_utc: str,
    cost_rows_by_regime: Mapping[str, Sequence[Mapping[str, Any]]],
    monotone_cost_rows_by_regime: Mapping[str, Sequence[Mapping[str, Any]]],
    source_manifest_json: Path,
    overlay_figure_paths: Mapping[str, str] | None = None,
    overlay_series_by_regime: Mapping[str, Sequence[Series]] | None = None,
    overlay_rows_by_regime: Mapping[str, Mapping[str, Any]] | None = None,
    compile_pdf: bool,
) -> dict[str, Any]:
    rows = _build_compact_child_schur_rows(
        cost_rows_by_regime=cost_rows_by_regime,
        monotone_cost_rows_by_regime=monotone_cost_rows_by_regime,
    )
    geomean_rows = _compact_geomean_rows(rows)
    tex_path = output_dir / f"{compact_stem}.tex"
    pdf_path = output_dir / f"{compact_stem}.pdf"
    csv_path = output_dir / f"{compact_stem}.csv"
    json_path = output_dir / f"{compact_stem}.json"
    _write_compact_csv(csv_path, rows)
    payload = {
        "schema": "paper_i_hh_child_fairness_compact_child_schur_v1",
        "generated_utc": str(generated_utc),
        "anchor_policy": "native_forced_snake_child_plus_schur",
        "ratio_semantics": {
            "ratio_delta_e": "abs_delta_e_snake_child_schur_native_forced / abs_delta_e_geo_comparator",
            "ratio_D2q": "D2q_snake_child_schur_native_forced / D2q_geo_comparator",
            "ratio_Dc": "Dc_snake_child_schur_native_forced / Dc_geo_comparator",
            "interpretation": "values_below_one_favor_snake",
        },
        "source_report_manifest_json": _rel(source_manifest_json),
        "tex": _rel(tex_path),
        "pdf": _rel(pdf_path),
        "csv": _rel(csv_path),
        "json": _rel(json_path),
        "overlay_policy": (
            "Second compact page overlays corrected-global SNAKE child+Schur against "
            "Geo monotone/non-forced no-child only when the corrected-global anchor is available; "
            "S/G table ratios below one favor SNAKE once complete."
        ),
        "overlay_figures": dict(overlay_figure_paths or {}),
        "overlay_rows": overlay_rows_by_regime or {},
        "overlay_series": {
            regime: [
                {
                    "variant_key": series.variant_key,
                    "label": series.label,
                    "method": series.method,
                    "status": series.status,
                    "point_count": len(series.points),
                    "marker_k": series.marker_k,
                    "marker_y": series.marker_y,
                    "marker_policy": series.marker_policy,
                    "delta_e": series.delta_e,
                    "source_json": series.source_json,
                    "source_sha256": series.source_sha256,
                    "blocker": series.blocker,
                }
                for series in series_list
            ]
            for regime, series_list in (overlay_series_by_regime or {}).items()
        },
        "rows": rows,
        "geomean_rows": geomean_rows,
    }
    _write_compact_tex(
        tex_path=tex_path,
        generated_utc=generated_utc,
        rows=rows,
        geomean_rows=geomean_rows,
        source_manifest_json=source_manifest_json,
        csv_path=csv_path,
        json_path=json_path,
        overlay_figure_paths=overlay_figure_paths,
        overlay_series_by_regime=overlay_series_by_regime,
        overlay_rows_by_regime=overlay_rows_by_regime,
    )
    if compile_pdf:
        _compile_tex(tex_path)
    payload["pdf_exists"] = bool(pdf_path.exists())
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "tex": _rel(tex_path),
        "pdf": _rel(pdf_path),
        "csv": _rel(csv_path),
        "json": _rel(json_path),
        "pdf_exists": bool(pdf_path.exists()),
        "row_count": int(len(rows)),
        "geomean_row_count": int(len(geomean_rows)),
        "overlay_page": bool(overlay_figure_paths and overlay_rows_by_regime),
    }


def _append_grid_page(
    lines: list[str],
    *,
    tex_path: Path,
    title: str,
    note: str,
    figure_paths: Mapping[str, str],
    series_by_regime: Mapping[str, Sequence[Series]],
    cost_rows_by_regime: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    lines.extend(
        [
            r"\begin{center}",
            rf"\textbf{{{_tex_escape(title)}}}\\[-0.2em]",
            rf"{{\scriptsize {_tex_escape(note)}\par}}",
            r"\end{center}",
            r"\vspace{0.03in}",
        ]
    )
    for idx, regime in enumerate(REGIME_ORDER):
        series_list = series_by_regime.get(regime, ())
        blockers = [s for s in series_list if s.status != "done"]
        blocker_text = "; ".join(f"{s.label}: {s.blocker or s.status}" for s in blockers)
        lines.extend(
            [
                r"\begin{minipage}[t]{0.318\linewidth}",
                r"\centering",
                rf"\includegraphics[width=\linewidth]{{{_graphics_path(tex_path, _resolve(figure_paths[regime]))}}}",
                r"\vspace{-0.04in}",
                r"{\tiny",
                _tex_table(cost_rows_by_regime[regime]),
                r"}",
            ]
        )
        if blocker_text:
            lines.append(rf"\par\tiny Pending: {_tex_escape(blocker_text)}")
        lines.append(r"\end{minipage}")
        if idx % 3 == 2:
            lines.append(r"\par\vspace{0.045in}")
        else:
            lines.append(r"\hfill")


def _write_tex(
    *,
    tex_path: Path,
    manifest: Mapping[str, Any],
    figure_paths: Mapping[str, str],
    series_by_regime: Mapping[str, Sequence[Series]],
    cost_rows_by_regime: Mapping[str, Sequence[Mapping[str, Any]]],
    monotone_figure_paths: Mapping[str, str] | None = None,
    monotone_series_by_regime: Mapping[str, Sequence[Series]] | None = None,
    monotone_cost_rows_by_regime: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    compact_child_schur_rows: Sequence[Mapping[str, Any]] | None = None,
    compact_child_schur_geomeans: Sequence[Mapping[str, Any]] | None = None,
    compact_csv_path: Path | None = None,
    compact_json_path: Path | None = None,
    source_manifest_json: Path | None = None,
    optimizer_payload: Mapping[str, Any] | None = None,
    optimizer_csv_path: Path | None = None,
    optimizer_json_path: Path | None = None,
    retrieved_append_geo_payload: Mapping[str, Any] | None = None,
    retrieved_append_geo_csv_path: Path | None = None,
    retrieved_append_geo_json_path: Path | None = None,
    retrieved_append_geo_figure_paths: Mapping[str, Mapping[str, str]] | None = None,
    retrieved_append_geo_series_by_optimizer: Mapping[str, Mapping[str, Sequence[Series]]] | None = None,
    retrieved_append_geo_cost_rows_by_optimizer: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]] | None = None,
) -> None:
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = _tex_comment_block(
        "MACHINE_READABLE_REPORT_PROVENANCE",
        _report_tex_provenance_payload(manifest),
    ) + [
        r"\documentclass[10pt]{article}",
        r"\usepackage[landscape,margin=0.25in]{geometry}",
        r"\usepackage{graphicx}",
        r"\usepackage{hyperref}",
        r"\hypersetup{colorlinks=true,linkcolor=black,urlcolor=blue}",
        r"\pagestyle{empty}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\tabcolsep}{2pt}",
        r"\renewcommand{\arraystretch}{0.86}",
        r"\begin{document}",
        r"\sloppy",
    ]
    lines.extend(
        [
            r"{\scriptsize",
            r"\textbf{Parameter Manifest.} ",
            rf"Generated {_tex_escape(manifest.get('generated_utc') or '--')}. ",
            rf"Scope: {_tex_escape(manifest.get('scope') or '--')} ",
            rf"Child rows: {_tex_escape(manifest.get('child_split_policy') or '--')} ",
            rf"Records: \nolinkurl{{{manifest.get('child_records_tsv') or '--'}}}. ",
            rf"Current Qiskit rows: \nolinkurl{{{manifest.get('current_qiskit_rows_json') or '--'}}}. ",
            rf"Schur overlay: \nolinkurl{{{manifest.get('schur_overlay_json') or '--'}}}. ",
            r"Markers are current Paper-I plateau prefixes or terminal rerun points until prefix Qiskit replay exists.",
            r" \(S_{\rm unfair}\) preserves the legacy/mixed shot proxy; \(S_{\rm fair}\) is reserved for common expanded-work accounting.",
            r" Pool exposure: macro rows use full-meta-minus-HVA macro generators; Append/Geo child rows use Pauli-child expanded candidate pools. SNAKE child rows on pages 1--2 are runtime-child diagnostics; corrected-global SNAKE child rows are kept separate.",
            r"\par}",
            r"\vspace{0.05in}",
        ]
    )
    _append_grid_page(
        lines,
        tex_path=tex_path,
        title="Page 1: native-forced SPSA child-fairness overlay",
        note=(
            "Current Paper-I rows are overlaid with native-forced depth-30 child/no-child follow-up rows. "
            "SNAKE runtime-child and child+Schur rows are rendered here as current-input diagnostic evidence; corrected-global evidence is kept on the strict companion page. "
            "Child curves are dashed; cost cells for unfinished Qiskit replays remain pending. "
            "F is shown only when saved prepared-state or saved exact-state fidelity passes provenance checks. "
                "SNAKE child fair-S cells are blocked when the sidecar reflects runtime Phase-3 child shortlisting rather than the Append/Geo full child-pool currency."
        ),
        figure_paths=figure_paths,
        series_by_regime=series_by_regime,
        cost_rows_by_regime=cost_rows_by_regime,
    )
    if monotone_figure_paths is not None and monotone_series_by_regime is not None and monotone_cost_rows_by_regime is not None:
        lines.extend([r"\newpage", r"\vspace*{-0.02in}"])
        _append_grid_page(
            lines,
            tex_path=tex_path,
            title="Page 2: monotone/non-forced SPSA child-fairness overlay",
            note=(
                "Append/Geo/SNAKE are shown with no-child and Pauli-child candidate routes. "
                "SNAKE phase-3 runtime-child rows are rendered here as diagnostic/current-input evidence, not corrected-global child-pool evidence. Plot curves update from optimization results; "
                "Qiskit compile-cost cells are allowed to remain pending. "
                "F is blank only when no validated saved prepared-state or saved exact-state fidelity is available. "
                "SNAKE child fair-S cells are blocked when the sidecar reflects runtime Phase-3 child shortlisting rather than the Append/Geo full child-pool currency."
            ),
            figure_paths=monotone_figure_paths,
            series_by_regime=monotone_series_by_regime,
            cost_rows_by_regime=monotone_cost_rows_by_regime,
        )
    if (
        compact_child_schur_rows is not None
        and compact_child_schur_geomeans is not None
        and compact_csv_path is not None
        and compact_json_path is not None
        and source_manifest_json is not None
    ):
        _append_compact_child_schur_page(
            lines,
            generated_utc=str(manifest.get("generated_utc") or "--"),
            rows=compact_child_schur_rows,
            geomean_rows=compact_child_schur_geomeans,
            source_manifest_json=source_manifest_json,
            csv_path=compact_csv_path,
            json_path=compact_json_path,
        )
    if optimizer_payload is not None and optimizer_csv_path is not None and optimizer_json_path is not None:
        _append_optimizer_diagnostic_page(
            lines,
            tex_path=tex_path,
            generated_utc=str(manifest.get("generated_utc") or "--"),
            optimizer_payload=optimizer_payload,
            csv_path=optimizer_csv_path,
            json_path=optimizer_json_path,
        )
    if (
        retrieved_append_geo_payload is not None
        and retrieved_append_geo_csv_path is not None
        and retrieved_append_geo_json_path is not None
        and retrieved_append_geo_figure_paths is not None
        and retrieved_append_geo_series_by_optimizer is not None
        and retrieved_append_geo_cost_rows_by_optimizer is not None
    ):
        _append_retrieved_append_geo_page(
            lines,
            tex_path=tex_path,
            generated_utc=str(manifest.get("generated_utc") or "--"),
            retrieved_payload=retrieved_append_geo_payload,
            csv_path=retrieved_append_geo_csv_path,
            json_path=retrieved_append_geo_json_path,
            figure_paths_by_optimizer=retrieved_append_geo_figure_paths,
            series_by_optimizer=retrieved_append_geo_series_by_optimizer,
            cost_rows_by_optimizer=retrieved_append_geo_cost_rows_by_optimizer,
        )
    lines.extend([r"\end{document}", ""])
    tex_path.write_text("\n".join(lines), encoding="utf-8")


def _compile_tex(tex_path: Path) -> None:
    cmd = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name]
    for _ in range(2):
        subprocess.run(cmd, cwd=str(tex_path.parent), check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def build_report(
    *,
    support_json: Path,
    qiskit_rows_json: Path,
    child_records_tsv: Path,
    monotone_records_tsv: Path,
    snake_sfair_records_tsv: Path,
    local_costs_json: Path,
    schur_overlay_json: Path,
    powell_partial_summary_json: Path,
    powell_tuning_summary_json: Path,
    powell_tuning_overlay_png: Path,
    optimizer_crosscheck_preflight_json: Path,
    retrieved_append_geo_dir: Path,
    local_optimizer_dir: Path,
    output_dir: Path,
    stem: str,
    compact_stem: str,
    compile_pdf: bool,
    reuse_existing_figures: bool = False,
    reuse_existing_sidecars: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    tex_path = output_dir / f"{stem}.tex"
    pdf_path = output_dir / f"{stem}.pdf"
    manifest_json = output_dir / f"{stem}.provenance.json"
    compact_csv_path = output_dir / f"{compact_stem}.csv"
    compact_json_path = output_dir / f"{compact_stem}.json"
    optimizer_csv_path = output_dir / "paper_i_hh_child_fairness_optimizer_diagnostic_20260625.csv"
    optimizer_json_path = output_dir / "paper_i_hh_child_fairness_optimizer_diagnostic_20260625.json"
    retrieved_append_geo_csv_path = output_dir / "paper_i_hh_child_fairness_retrieved_optimizer_rows_20260626.csv"
    retrieved_append_geo_json_path = output_dir / "paper_i_hh_child_fairness_retrieved_optimizer_rows_20260626.json"

    if reuse_existing_figures and reuse_existing_sidecars and manifest_json.exists():
        previous_manifest = _read_json(manifest_json)
        if isinstance(previous_manifest, Mapping):
            figure_paths = dict(previous_manifest.get("figures") if isinstance(previous_manifest.get("figures"), Mapping) else {})
            monotone_figure_paths = dict(
                previous_manifest.get("monotone_figures")
                if isinstance(previous_manifest.get("monotone_figures"), Mapping)
                else {}
            )
            compact_overlay_figure_paths = dict(
                previous_manifest.get("compact_overlay_figures")
                if isinstance(previous_manifest.get("compact_overlay_figures"), Mapping)
                else {}
            )
            series_by_regime = _series_by_regime_from_manifest(previous_manifest.get("series"))
            monotone_series_by_regime = _series_by_regime_from_manifest(previous_manifest.get("monotone_series"))
            compact_overlay_series_by_regime = _series_by_regime_from_manifest(previous_manifest.get("compact_overlay_series"))
            cost_rows_by_regime = {
                regime: list(rows) if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)) else []
                for regime, rows in (
                    previous_manifest.get("cost_rows") if isinstance(previous_manifest.get("cost_rows"), Mapping) else {}
                ).items()
            }
            monotone_cost_rows_by_regime = {
                regime: list(rows) if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)) else []
                for regime, rows in (
                    previous_manifest.get("monotone_cost_rows")
                    if isinstance(previous_manifest.get("monotone_cost_rows"), Mapping)
                    else {}
                ).items()
            }
            compact_overlay_rows_by_regime = dict(
                previous_manifest.get("compact_overlay_rows")
                if isinstance(previous_manifest.get("compact_overlay_rows"), Mapping)
                else {}
            )
            retrieved_append_geo_figure_paths = dict(
                previous_manifest.get("retrieved_append_geo_figures_by_optimizer")
                if isinstance(previous_manifest.get("retrieved_append_geo_figures_by_optimizer"), Mapping)
                else previous_manifest.get("retrieved_append_geo_figures")
                if isinstance(previous_manifest.get("retrieved_append_geo_figures"), Mapping)
                else {}
            )
            retrieved_append_geo_series_by_optimizer = {
                optimizer: _series_by_regime_from_manifest(by_regime)
                for optimizer, by_regime in (
                    previous_manifest.get("retrieved_append_geo_series_by_optimizer")
                    if isinstance(previous_manifest.get("retrieved_append_geo_series_by_optimizer"), Mapping)
                    else {}
                ).items()
            }
            retrieved_append_geo_cost_rows_by_optimizer = dict(
                previous_manifest.get("retrieved_append_geo_cost_rows_by_optimizer")
                if isinstance(previous_manifest.get("retrieved_append_geo_cost_rows_by_optimizer"), Mapping)
                else {}
            )
            compact_child_schur_rows = _build_compact_child_schur_rows(
                cost_rows_by_regime=cost_rows_by_regime,
                monotone_cost_rows_by_regime=monotone_cost_rows_by_regime,
            )
            compact_child_schur_geomeans = _compact_geomean_rows(compact_child_schur_rows)
            optimizer_payload = (
                dict(previous_manifest.get("optimizer_diagnostic"))
                if isinstance(previous_manifest.get("optimizer_diagnostic"), Mapping)
                else None
            )
            retrieved_append_geo_payload = (
                dict(previous_manifest.get("retrieved_append_geo"))
                if isinstance(previous_manifest.get("retrieved_append_geo"), Mapping)
                else None
            )
            generated_utc = datetime.now(timezone.utc).isoformat()
            manifest = dict(previous_manifest)
            manifest.update(
                {
                    "generated_utc": generated_utc,
                    "figure_render_policy": "reuse_existing_all_figures",
                    "sidecar_refresh_policy": "reuse_existing_sidecars",
                    "fast_update_source_manifest_json": _rel(manifest_json),
                    "tex": _rel(tex_path),
                    "pdf": _rel(pdf_path),
                    "manifest_json": _rel(manifest_json),
                }
            )
            _write_tex(
                tex_path=tex_path,
                manifest=manifest,
                figure_paths=figure_paths,
                series_by_regime=series_by_regime,
                cost_rows_by_regime=cost_rows_by_regime,
                monotone_figure_paths=monotone_figure_paths,
                monotone_series_by_regime=monotone_series_by_regime,
                monotone_cost_rows_by_regime=monotone_cost_rows_by_regime,
                compact_child_schur_rows=compact_child_schur_rows,
                compact_child_schur_geomeans=compact_child_schur_geomeans,
                compact_csv_path=compact_csv_path,
                compact_json_path=compact_json_path,
                source_manifest_json=manifest_json,
                optimizer_payload=optimizer_payload,
                optimizer_csv_path=optimizer_csv_path,
                optimizer_json_path=optimizer_json_path,
                retrieved_append_geo_payload=retrieved_append_geo_payload,
                retrieved_append_geo_csv_path=retrieved_append_geo_csv_path,
                retrieved_append_geo_json_path=retrieved_append_geo_json_path,
                retrieved_append_geo_figure_paths=retrieved_append_geo_figure_paths,
                retrieved_append_geo_series_by_optimizer=retrieved_append_geo_series_by_optimizer,
                retrieved_append_geo_cost_rows_by_optimizer=retrieved_append_geo_cost_rows_by_optimizer,
            )
            if compile_pdf:
                if shutil.which("pdflatex") is None:
                    raise RuntimeError("pdflatex is required to compile this report")
                _compile_tex(tex_path)
            manifest["compact_child_schur_report"] = _write_compact_child_schur_artifacts(
                output_dir=output_dir,
                compact_stem=str(compact_stem),
                generated_utc=generated_utc,
                cost_rows_by_regime=cost_rows_by_regime,
                monotone_cost_rows_by_regime=monotone_cost_rows_by_regime,
                source_manifest_json=manifest_json,
                overlay_figure_paths=compact_overlay_figure_paths,
                overlay_series_by_regime=compact_overlay_series_by_regime,
                overlay_rows_by_regime=compact_overlay_rows_by_regime,
                compile_pdf=bool(compile_pdf),
            )
            manifest["pdf_exists"] = pdf_path.exists()
            manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return manifest

    current_rows = _load_current_rows(support_json)
    qiskit_rows = _load_qiskit_rows(qiskit_rows_json)
    child_records = _load_child_records(child_records_tsv)
    child_records.update(
        _load_snake_sfair_repair_records(snake_sfair_records_tsv, engine_key="native_forced")
    )
    monotone_records = _load_monotone_records(monotone_records_tsv)
    monotone_records.update(
        _load_snake_sfair_repair_records(snake_sfair_records_tsv, engine_key="legacy_monotone")
    )
    schur_overlay = _load_schur_overlay(schur_overlay_json)
    local_cost_rows = _load_local_cost_rows(local_costs_json)
    series_by_regime = _build_series(current_rows, qiskit_rows, child_records, schur_overlay)
    monotone_series_by_regime = _build_monotone_series(monotone_records)
    if reuse_existing_figures:
        figure_paths = _expected_figure_paths(figures_dir, "native_forced_")
        monotone_figure_paths = _expected_figure_paths(figures_dir, "monotone_nonforced_")
    else:
        figure_paths = _render_plots(series_by_regime, figures_dir, filename_prefix="native_forced_")
        monotone_figure_paths = _render_plots(
            monotone_series_by_regime,
            figures_dir,
            filename_prefix="monotone_nonforced_",
        )
    cost_rows_by_regime = {
        regime: _cost_rows_for_regime(regime, series, qiskit_rows, local_cost_rows, schur_overlay)
        for regime, series in series_by_regime.items()
    }
    monotone_cost_rows_by_regime = {
        regime: _cost_rows_for_monotone_regime(regime, series, local_cost_rows)
        for regime, series in monotone_series_by_regime.items()
    }
    compact_overlay_series_by_regime = _build_compact_overlay_series(
        series_by_regime=series_by_regime,
        monotone_series_by_regime=monotone_series_by_regime,
    )
    if reuse_existing_figures:
        compact_overlay_figure_paths = _expected_figure_paths(
            figures_dir,
            "compact_snake_child_schur_vs_geo_monotone_no_child_",
        )
    else:
        compact_overlay_figure_paths = _render_plots(
            compact_overlay_series_by_regime,
            figures_dir,
            filename_prefix="compact_snake_child_schur_vs_geo_monotone_no_child_",
            figsize=(5.2, 1.08),
            axis_fontsize=5.6,
            title_fontsize=6.6,
            tick_fontsize=5.0,
            legend_fontsize=4.2,
            tight_pad=0.12,
        )
    compact_overlay_rows_by_regime = _build_compact_overlay_rows(
        cost_rows_by_regime=cost_rows_by_regime,
        monotone_cost_rows_by_regime=monotone_cost_rows_by_regime,
    )
    tex_path = output_dir / f"{stem}.tex"
    pdf_path = output_dir / f"{stem}.pdf"
    manifest_json = output_dir / f"{stem}.provenance.json"
    compact_csv_path = output_dir / f"{compact_stem}.csv"
    compact_json_path = output_dir / f"{compact_stem}.json"
    optimizer_csv_path = output_dir / "paper_i_hh_child_fairness_optimizer_diagnostic_20260625.csv"
    optimizer_json_path = output_dir / "paper_i_hh_child_fairness_optimizer_diagnostic_20260625.json"
    retrieved_append_geo_csv_path = output_dir / "paper_i_hh_child_fairness_retrieved_optimizer_rows_20260626.csv"
    retrieved_append_geo_json_path = output_dir / "paper_i_hh_child_fairness_retrieved_optimizer_rows_20260626.json"
    compact_child_schur_rows = _build_compact_child_schur_rows(
        cost_rows_by_regime=cost_rows_by_regime,
        monotone_cost_rows_by_regime=monotone_cost_rows_by_regime,
    )
    compact_child_schur_geomeans = _compact_geomean_rows(compact_child_schur_rows)
    optimizer_payload: dict[str, Any] | None
    if reuse_existing_sidecars and optimizer_json_path.exists():
        existing_optimizer_payload = _read_json(optimizer_json_path)
        optimizer_payload = dict(existing_optimizer_payload) if isinstance(existing_optimizer_payload, Mapping) else None
    else:
        optimizer_payload = _write_optimizer_addendum_sidecars(
            csv_path=optimizer_csv_path,
            json_path=optimizer_json_path,
            generated_utc=datetime.now(timezone.utc).isoformat(),
            powell_summary_json=powell_partial_summary_json,
            powell_tuning_summary_json=powell_tuning_summary_json,
            powell_tuning_overlay_png=powell_tuning_overlay_png,
            optimizer_crosscheck_preflight_json=optimizer_crosscheck_preflight_json,
        )
    retrieved_append_geo_payload: dict[str, Any] | None
    if reuse_existing_sidecars and retrieved_append_geo_json_path.exists():
        existing_retrieved_payload = _read_json(retrieved_append_geo_json_path)
        retrieved_append_geo_payload = dict(existing_retrieved_payload) if isinstance(existing_retrieved_payload, Mapping) else None
    else:
        retrieved_append_geo_payload = _write_retrieved_append_geo_sidecars(
            csv_path=retrieved_append_geo_csv_path,
            json_path=retrieved_append_geo_json_path,
            generated_utc=datetime.now(timezone.utc).isoformat(),
            retrieved_dir=retrieved_append_geo_dir,
            local_optimizer_dir=local_optimizer_dir,
            powell_summary_json=powell_partial_summary_json,
        )
    retrieved_append_geo_series_by_regime = _build_retrieved_append_geo_series(retrieved_append_geo_payload)
    retrieved_append_geo_cost_rows_by_regime = _build_retrieved_append_geo_cost_rows(retrieved_append_geo_payload)
    retrieved_append_geo_series_by_optimizer = _retrieved_series_by_optimizer(retrieved_append_geo_series_by_regime)
    retrieved_append_geo_cost_rows_by_optimizer = _retrieved_cost_rows_by_optimizer(retrieved_append_geo_cost_rows_by_regime)
    retrieved_append_geo_figure_paths: dict[str, dict[str, str]] = {}
    for optimizer, opt_series_by_regime in retrieved_append_geo_series_by_optimizer.items():
        if reuse_existing_figures and reuse_existing_sidecars:
            retrieved_append_geo_figure_paths[optimizer] = _expected_figure_paths(
                figures_dir,
                f"retrieved_{optimizer}_optimizer_rows_",
            )
        elif reuse_existing_figures:
            retrieved_append_geo_figure_paths[optimizer] = _render_plots_pil(
                opt_series_by_regime,
                figures_dir,
                filename_prefix=f"retrieved_{optimizer}_optimizer_rows_",
            )
        else:
            retrieved_append_geo_figure_paths[optimizer] = _render_plots(
                opt_series_by_regime,
                figures_dir,
                filename_prefix=f"retrieved_{optimizer}_optimizer_rows_",
                figsize=(5.15, 0.95),
                axis_fontsize=5.6,
                title_fontsize=6.4,
                tick_fontsize=5.0,
                legend_fontsize=4.1,
                tight_pad=0.12,
            )
    manifest: dict[str, Any] = {
        "schema": "paper_i_hh_child_fairness_incremental_report_v5",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "figure_render_policy": (
            "reuse_existing_all_figures"
            if reuse_existing_figures and reuse_existing_sidecars
            else "reuse_existing_static_figures_plus_pil_page5_live_figures"
            if reuse_existing_figures
            else "matplotlib_render_all_figures"
        ),
        "sidecar_refresh_policy": "reuse_existing_sidecars" if reuse_existing_sidecars else "refresh_sidecars_from_sources",
        "shot_column_policy": {
            "S_unfair": (
                "Legacy or mixed shot/work proxy retained for diagnostic continuity. For SNAKE this may come from "
                "grouped/controller/event-ledger accounting and must not be compared directly with Append/Geo child rows."
            ),
            "S_fair": (
                "Common expanded-work shot proxy for fair comparison. Append/Geo use the existing expanded probe proxy. "
                "SNAKE runtime-child rows on pages 1--2 are shown as current-input diagnostic rows, with fair-S blocked "
                "until corrected global Pauli-child-pool evidence is available."
            ),
        },
        "scope": (
            "Five-page Paper-I HH Pauli-child fairness overlay. Page 1 uses native-forced SPSA current/follow-up rows; "
            "Page 2 uses monotone/non-forced SPSA child/no-child rows with Schur append-prune warm-start SNAKE; "
            "Page 3 gives compact SNAKE child+Schur divided-by-Geo ratio comparisons; "
            "Page 4 records a Powell/BFGS optimizer diagnostic addendum; "
            "Pages 5 and 6 record newly fetched and local Geo/Append/SNAKE optimizer-study cells, split into Powell and ROTOSOLVE views."
        ),
        "page_semantics": {
            "page_1": "native-forced SPSA child-fairness overlay; SNAKE runtime-child rows are rendered as current-input diagnostic rows, not corrected-global child-pool evidence",
            "page_2": "monotone/non-forced SPSA overlay; SNAKE runtime-child rows are rendered as current-input diagnostic rows, not corrected-global child-pool evidence",
            "page_3": "compact child+Schur ratios; strict corrected-global SNAKE child+Schur anchor remains blocked until global Pauli-child evidence is available",
            "page_4": "optimizer diagnostic addendum only",
            "page_5": "fetched/local Powell optimizer rows; SNAKE included when corrected global child-pool rows complete",
            "page_6": "fetched/local ROTOSOLVE optimizer rows; SNAKE included when corrected global child-pool rows complete",
        },
        "pool_exposure_policy": {
            "macro_rows": "full_meta_minus_hva macro-generator pool",
            "append_geo_child_rows": "Pauli-child expanded candidate pool",
            "snake_runtime_child_rows_pages_1_2": "old SNAKE phase-3 runtime child shortlisting; displayed as diagnostic/current-input evidence only",
            "snake_corrected_global_child_rows": "requires global_pauli_child_sets_v1 / pauli_child_sets_v1 candidate-pool expansion",
            "strict_ratio_anchor": "compact/global-corrected pages do not accept runtime-child diagnostic rows as corrected-global evidence",
        },
        "snake_child_pool_repair_status": {
            "status": f"pending:{SNAKE_CORRECTED_CHILD_POOL_BLOCKER}",
            "replacement_run_required_for_pages": [3, 5, 6],
            "accepted_modes": sorted(SNAKE_GLOBAL_CHILD_POOL_MODES),
            "diagnostic_runtime_child_mode_pages_1_2": PAULI_CHILD_MODE,
        },
        "child_split_policy": (
            "Append/Geo child rows use generic_adapt_runtime_split_mode=shortlist_pauli_children_v1, "
            "symmetry_policy=off, generic_adapt_stop_policy=fixed_horizon_no_target_v1, "
            "resource_pool_term_cap=0."
        ),
        "current_support_json": _rel(support_json),
        "current_support_json_sha256": _sha256(support_json),
        "current_qiskit_rows_json": _rel(qiskit_rows_json),
        "current_qiskit_rows_json_sha256": _sha256(qiskit_rows_json),
        "child_records_tsv": _rel(child_records_tsv) if child_records_tsv.exists() else str(child_records_tsv),
        "child_records_tsv_sha256": _sha256(child_records_tsv) if child_records_tsv.exists() else None,
        "monotone_records_tsv": _rel(monotone_records_tsv) if monotone_records_tsv.exists() else str(monotone_records_tsv),
        "monotone_records_tsv_sha256": _sha256(monotone_records_tsv) if monotone_records_tsv.exists() else None,
        "snake_sfair_records_tsv": _rel(snake_sfair_records_tsv) if snake_sfair_records_tsv.exists() else str(snake_sfair_records_tsv),
        "snake_sfair_records_tsv_sha256": _sha256(snake_sfair_records_tsv) if snake_sfair_records_tsv.exists() else None,
        "local_costs_json": _rel(local_costs_json) if local_costs_json.exists() else str(local_costs_json),
        "local_costs_json_sha256": _sha256(local_costs_json) if local_costs_json.exists() else None,
        "schur_overlay_json": _rel(schur_overlay_json) if schur_overlay_json.exists() else str(schur_overlay_json),
        "schur_overlay_json_sha256": _sha256(schur_overlay_json) if schur_overlay_json.exists() else None,
        "schur_overlay_import_policy": (
            "Page-1 imports the existing native-forced maxiter-200 depth-30 SNAKE+Pauli-child+Schur "
            "append-prune warm-start rows from the referenced overlay JSON. The imported strong-strong row follows "
            "the overlay's from-scratch 8-worker replacement."
        ),
        "monotone_page_policy": (
            "Optimization-result curves are primary. Qiskit compile-cost rows are filled from local sidecars "
            "when completed result JSONs have been fetched."
        ),
        "compact_child_schur_report_policy": (
            "Derived compact ratio view anchored only on corrected-global SNAKE child+Schur evidence. "
            "Runtime-child diagnostic rows from pages 1--2 are intentionally blocked as ratio anchors."
        ),
        "compact_overlay_policy": (
            "Compact child+Schur companion PDF includes corrected-global SNAKE child+Schur only when available; "
            "otherwise the strict SNAKE/Geo ratios stay blocked. Values below one favor SNAKE once ratios are complete."
        ),
        "optimizer_diagnostic_policy": (
            "Page 4 is an optimizer-study addendum only. Powell/BFGS rows do not replace active SPSA "
            "fair-comparison manuscript/report cells."
        ),
        "retrieved_append_geo_policy": (
            "Pages 5 and 6 consume the local CHTC retrieval archive plus the local alternator output root for Geo-ADAPT, append-only ADAPT, and corrected SNAKE rows when present. "
            "A numeric row requires stdout status completed and a terminal progress-ledger event; cell-manifest success "
            "alone is not enough for retrieved generic rows; local SNAKE rows may use cell-manifest success plus native result JSON/history. "
            "Powell and ROTOSOLVE plots/tables are rendered separately."
        ),
        "local_optimizer_dir": _rel(local_optimizer_dir) if local_optimizer_dir.exists() else str(local_optimizer_dir),
        "sidecars": {
            "main_provenance_json": _rel(manifest_json),
            "compact_csv": _rel(compact_csv_path),
            "compact_json": _rel(compact_json_path),
            "optimizer_csv": _rel(optimizer_csv_path),
            "optimizer_json": _rel(optimizer_json_path),
            "retrieved_optimizer_csv": _rel(retrieved_append_geo_csv_path),
            "retrieved_optimizer_json": _rel(retrieved_append_geo_json_path),
        },
        "tex": _rel(tex_path),
        "pdf": _rel(pdf_path),
        "manifest_json": _rel(manifest_json),
        "figures": figure_paths,
        "monotone_figures": monotone_figure_paths,
        "compact_overlay_figures": compact_overlay_figure_paths,
        "series": {
            regime: [
                {
                    "variant_key": s.variant_key,
                    "label": s.label,
                    "method": s.method,
                    "status": s.status,
                    "point_count": len(s.points),
                    "marker_k": s.marker_k,
                    "marker_y": s.marker_y,
                    "marker_policy": s.marker_policy,
                    "delta_e": s.delta_e,
                    "source_json": s.source_json,
                    "source_sha256": s.source_sha256,
                    "blocker": s.blocker,
                }
                for s in series
            ]
            for regime, series in series_by_regime.items()
        },
        "monotone_series": {
            regime: [
                {
                    "variant_key": s.variant_key,
                    "label": s.label,
                    "method": s.method,
                    "status": s.status,
                    "point_count": len(s.points),
                    "marker_k": s.marker_k,
                    "marker_y": s.marker_y,
                    "marker_policy": s.marker_policy,
                    "delta_e": s.delta_e,
                    "source_json": s.source_json,
                    "source_sha256": s.source_sha256,
                    "blocker": s.blocker,
                }
                for s in series
            ]
            for regime, series in monotone_series_by_regime.items()
        },
        "compact_overlay_series": {
            regime: [
                {
                    "variant_key": s.variant_key,
                    "label": s.label,
                    "method": s.method,
                    "status": s.status,
                    "point_count": len(s.points),
                    "marker_k": s.marker_k,
                    "marker_y": s.marker_y,
                    "marker_policy": s.marker_policy,
                    "delta_e": s.delta_e,
                    "source_json": s.source_json,
                    "source_sha256": s.source_sha256,
                    "blocker": s.blocker,
                }
                for s in series
            ]
            for regime, series in compact_overlay_series_by_regime.items()
        },
        "cost_rows": cost_rows_by_regime,
        "monotone_cost_rows": monotone_cost_rows_by_regime,
        "compact_overlay_rows": compact_overlay_rows_by_regime,
        "optimizer_diagnostic": optimizer_payload,
        "retrieved_append_geo": retrieved_append_geo_payload,
        "retrieved_append_geo_figures": retrieved_append_geo_figure_paths,
        "retrieved_append_geo_figures_by_optimizer": retrieved_append_geo_figure_paths,
        "retrieved_append_geo_series": {
            regime: [
                {
                    "variant_key": s.variant_key,
                    "label": s.label,
                    "method": s.method,
                    "status": s.status,
                    "point_count": len(s.points),
                    "marker_k": s.marker_k,
                    "marker_y": s.marker_y,
                    "marker_policy": s.marker_policy,
                    "delta_e": s.delta_e,
                    "source_json": s.source_json,
                    "source_sha256": s.source_sha256,
                    "blocker": s.blocker,
                }
                for s in series
            ]
            for regime, series in retrieved_append_geo_series_by_regime.items()
        },
        "retrieved_append_geo_series_by_optimizer": {
            optimizer: {
                regime: [
                    {
                        "variant_key": s.variant_key,
                        "label": s.label,
                        "method": s.method,
                        "status": s.status,
                        "point_count": len(s.points),
                        "marker_k": s.marker_k,
                        "marker_y": s.marker_y,
                        "marker_policy": s.marker_policy,
                        "delta_e": s.delta_e,
                        "source_json": s.source_json,
                        "source_sha256": s.source_sha256,
                        "blocker": s.blocker,
                    }
                    for s in series
                ]
                for regime, series in by_regime.items()
            }
            for optimizer, by_regime in retrieved_append_geo_series_by_optimizer.items()
        },
        "retrieved_append_geo_cost_rows": retrieved_append_geo_cost_rows_by_regime,
        "retrieved_append_geo_cost_rows_by_optimizer": retrieved_append_geo_cost_rows_by_optimizer,
    }
    _write_tex(
        tex_path=tex_path,
        manifest=manifest,
        figure_paths=figure_paths,
        series_by_regime=series_by_regime,
        cost_rows_by_regime=cost_rows_by_regime,
        monotone_figure_paths=monotone_figure_paths,
        monotone_series_by_regime=monotone_series_by_regime,
        monotone_cost_rows_by_regime=monotone_cost_rows_by_regime,
        compact_child_schur_rows=compact_child_schur_rows,
        compact_child_schur_geomeans=compact_child_schur_geomeans,
        compact_csv_path=compact_csv_path,
        compact_json_path=compact_json_path,
        source_manifest_json=manifest_json,
        optimizer_payload=optimizer_payload,
        optimizer_csv_path=optimizer_csv_path,
        optimizer_json_path=optimizer_json_path,
        retrieved_append_geo_payload=retrieved_append_geo_payload,
        retrieved_append_geo_csv_path=retrieved_append_geo_csv_path,
        retrieved_append_geo_json_path=retrieved_append_geo_json_path,
        retrieved_append_geo_figure_paths=retrieved_append_geo_figure_paths,
        retrieved_append_geo_series_by_optimizer=retrieved_append_geo_series_by_optimizer,
        retrieved_append_geo_cost_rows_by_optimizer=retrieved_append_geo_cost_rows_by_optimizer,
    )
    if compile_pdf:
        if shutil.which("pdflatex") is None:
            raise RuntimeError("pdflatex is required to compile this report")
        _compile_tex(tex_path)
    manifest["compact_child_schur_report"] = _write_compact_child_schur_artifacts(
        output_dir=output_dir,
        compact_stem=str(compact_stem),
        generated_utc=str(manifest["generated_utc"]),
        cost_rows_by_regime=cost_rows_by_regime,
        monotone_cost_rows_by_regime=monotone_cost_rows_by_regime,
        source_manifest_json=manifest_json,
        overlay_figure_paths=compact_overlay_figure_paths,
        overlay_series_by_regime=compact_overlay_series_by_regime,
        overlay_rows_by_regime=compact_overlay_rows_by_regime,
        compile_pdf=bool(compile_pdf),
    )
    manifest["pdf_exists"] = pdf_path.exists()
    manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-json", type=Path, default=DEFAULT_SUPPORT_JSON)
    parser.add_argument("--qiskit-rows-json", type=Path, default=DEFAULT_QISKIT_ROWS_JSON)
    parser.add_argument("--child-records-tsv", type=Path, default=DEFAULT_CHILD_RECORDS_TSV)
    parser.add_argument("--monotone-records-tsv", type=Path, default=DEFAULT_MONOTONE_RECORDS_TSV)
    parser.add_argument("--snake-sfair-records-tsv", type=Path, default=DEFAULT_SNAKE_SFAIR_RECORDS_TSV)
    parser.add_argument("--local-costs-json", type=Path, default=DEFAULT_LOCAL_COSTS_JSON)
    parser.add_argument("--schur-overlay-json", type=Path, default=DEFAULT_SCHUR_OVERLAY_JSON)
    parser.add_argument("--powell-partial-summary-json", type=Path, default=DEFAULT_POWELL_PARTIAL_SUMMARY_JSON)
    parser.add_argument("--powell-tuning-summary-json", type=Path, default=DEFAULT_POWELL_TUNING_SUMMARY_JSON)
    parser.add_argument("--powell-tuning-overlay-png", type=Path, default=DEFAULT_POWELL_TUNING_OVERLAY_PNG)
    parser.add_argument(
        "--optimizer-crosscheck-preflight-json",
        type=Path,
        default=DEFAULT_OPTIMIZER_CROSSCHECK_PREFLIGHT_JSON,
    )
    parser.add_argument("--retrieved-append-geo-dir", type=Path, default=DEFAULT_RETRIEVED_APPEND_GEO_DIR)
    parser.add_argument("--local-optimizer-dir", type=Path, default=DEFAULT_LOCAL_SNAKE_ROTOSOLVE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default=DEFAULT_STEM)
    parser.add_argument("--compact-stem", default=DEFAULT_COMPACT_STEM)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument(
        "--reuse-existing-figures",
        action="store_true",
        help="Reuse existing static figures and regenerate only live optimizer plots with the PIL fallback.",
    )
    parser.add_argument(
        "--reuse-existing-sidecars",
        action="store_true",
        help="Reuse existing optimizer/retrieval sidecars instead of rescanning raw output trees.",
    )
    parser.add_argument(
        "--fast-update",
        action="store_true",
        help="Fast layout/LaTeX refresh: reuse existing sidecars and figures; use only when evidence inputs did not change.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_report(
        support_json=_resolve(args.support_json),
        qiskit_rows_json=_resolve(args.qiskit_rows_json),
        child_records_tsv=_resolve(args.child_records_tsv),
        monotone_records_tsv=_resolve(args.monotone_records_tsv),
        snake_sfair_records_tsv=_resolve(args.snake_sfair_records_tsv),
        local_costs_json=_resolve(args.local_costs_json),
        schur_overlay_json=_resolve(args.schur_overlay_json),
        powell_partial_summary_json=_resolve(args.powell_partial_summary_json),
        powell_tuning_summary_json=_resolve(args.powell_tuning_summary_json),
        powell_tuning_overlay_png=_resolve(args.powell_tuning_overlay_png),
        optimizer_crosscheck_preflight_json=_resolve(args.optimizer_crosscheck_preflight_json),
        retrieved_append_geo_dir=_resolve(args.retrieved_append_geo_dir),
        local_optimizer_dir=_resolve(args.local_optimizer_dir),
        output_dir=_resolve(args.output_dir),
        stem=str(args.stem),
        compact_stem=str(args.compact_stem),
        compile_pdf=not bool(args.no_compile),
        reuse_existing_figures=bool(args.reuse_existing_figures or args.fast_update),
        reuse_existing_sidecars=bool(args.reuse_existing_sidecars or args.fast_update),
    )
    print(
        json.dumps(
            {k: manifest[k] for k in ("tex", "pdf", "manifest_json", "pdf_exists", "compact_child_schur_report")},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
