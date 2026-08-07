#!/usr/bin/env python3
"""Build the Paper-I HH full-meta singleton symmetry matrix report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BATCH_ID = "paper_i_hh_fullmeta_singleton_symmetry_matrix_20260629_v1"
DEFAULT_RECORDS_TSV = (
    REPO_ROOT
    / "chtc"
    / "phase3_optuna"
    / "input"
    / DEFAULT_BATCH_ID
    / "paper_i_hh_spsa_budget_ladder_records.tsv"
)
DEFAULT_MANIFEST_JSON = (
    REPO_ROOT
    / "chtc"
    / "phase3_optuna"
    / "input"
    / DEFAULT_BATCH_ID
    / "paper_i_hh_spsa_budget_ladder_manifest.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "pdf" / "paper_i_hh_fullmeta_singleton_symmetry_matrix_20260629"
DEFAULT_STEM = "paper_i_hh_fullmeta_singleton_symmetry_matrix_20260629"
DEFAULT_RESULT_ROOTS = (
    REPO_ROOT / "raw_outputs" / DEFAULT_BATCH_ID,
    REPO_ROOT
    / "raw_outputs"
    / "chtc_fetches"
    / "paper_i_hh_fullmeta_singleton_symmetry_matrix_20260629"
    / "raw_outputs"
    / DEFAULT_BATCH_ID,
)
LOCAL_FIXCHECK_OUTPUT_DIR = (
    REPO_ROOT / "output" / "pdf" / "paper_i_hh_fullmeta_singleton_local_fixcheck_20260629"
)
LOCAL_FIXCHECK_STEM = "paper_i_hh_fullmeta_singleton_local_fixcheck_20260629"
LOCAL_FIXCHECK_INPUTS = (
    {
        "optimizer": "POWELL",
        "records_tsv": REPO_ROOT
        / "chtc"
        / "phase3_optuna"
        / "input"
        / "paper_i_hh_fullmeta_singleton_symmetry_20260629_v4_fixed_generic_powell"
        / "paper_i_hh_spsa_budget_ladder_records.tsv",
        "manifest_json": REPO_ROOT
        / "chtc"
        / "phase3_optuna"
        / "input"
        / "paper_i_hh_fullmeta_singleton_symmetry_20260629_v4_fixed_generic_powell"
        / "paper_i_hh_spsa_budget_ladder_manifest.json",
        "local_roots": (
            REPO_ROOT
            / "raw_outputs"
            / "paper_i_hh_fullmeta_singleton_symmetry_weakweak_local_powell_fixcheck_20260629",
            REPO_ROOT
            / "raw_outputs"
            / "paper_i_hh_fullmeta_singleton_symmetry_local_parallel_fixcheck_20260630_powell",
            REPO_ROOT
            / "raw_outputs"
            / "paper_i_hh_fullmeta_singleton_symmetry_local_snake_reference_replay_20260630_powell",
            REPO_ROOT
            / "raw_outputs"
            / "paper_i_hh_fullmeta_singleton_symmetry_local_snake_phase0_replay_20260630_powell",
            REPO_ROOT
            / "raw_outputs"
            / "paper_i_hh_fullmeta_singleton_symmetry_local_unified_fixcheck_20260630_powell",
        ),
    },
    {
        "optimizer": "ROTOSOLVE",
        "records_tsv": REPO_ROOT
        / "chtc"
        / "phase3_optuna"
        / "input"
        / "paper_i_hh_fullmeta_singleton_symmetry_20260629_v4_fixed_generic_rotosolve"
        / "paper_i_hh_spsa_budget_ladder_records.tsv",
        "manifest_json": REPO_ROOT
        / "chtc"
        / "phase3_optuna"
        / "input"
        / "paper_i_hh_fullmeta_singleton_symmetry_20260629_v4_fixed_generic_rotosolve"
        / "paper_i_hh_spsa_budget_ladder_manifest.json",
        "local_roots": (
            REPO_ROOT
            / "raw_outputs"
            / "paper_i_hh_fullmeta_singleton_symmetry_weakweak_local_rotosolve_fixcheck_20260629",
            REPO_ROOT
            / "raw_outputs"
            / "paper_i_hh_fullmeta_singleton_symmetry_local_parallel_fixcheck_20260630_rotosolve",
            REPO_ROOT
            / "raw_outputs"
            / "paper_i_hh_fullmeta_singleton_symmetry_local_snake_reference_replay_20260630_rotosolve",
            REPO_ROOT
            / "raw_outputs"
            / "paper_i_hh_fullmeta_singleton_symmetry_local_snake_phase0_replay_20260630_rotosolve",
            REPO_ROOT
            / "raw_outputs"
            / "paper_i_hh_fullmeta_singleton_symmetry_local_unified_fixcheck_20260630_rotosolve",
        ),
    },
    {
        "optimizer": "SPSA",
        "records_tsv": REPO_ROOT
        / "chtc"
        / "phase3_optuna"
        / "input"
        / "paper_i_hh_fullmeta_singleton_symmetry_matrix_20260629_v3_fixedcode_snake_spsa"
        / "paper_i_hh_spsa_budget_ladder_records.tsv",
        "manifest_json": REPO_ROOT
        / "chtc"
        / "phase3_optuna"
        / "input"
        / "paper_i_hh_fullmeta_singleton_symmetry_matrix_20260629_v3_fixedcode_snake_spsa"
        / "paper_i_hh_spsa_budget_ladder_manifest.json",
        "local_roots": (
            REPO_ROOT
            / "raw_outputs"
            / "paper_i_hh_fullmeta_singleton_symmetry_local_snake_reference_replay_20260630_spsa",
            REPO_ROOT
            / "raw_outputs"
            / "paper_i_hh_fullmeta_singleton_symmetry_local_unified_fixcheck_20260630_spsa",
        ),
    },
)

REGIME_ORDER = (
    "weak-weak",
    "intermediate-weak",
    "strong-weak",
    "weak-strong",
    "intermediate-strong",
    "strong-strong",
)

PLOT_START_CONVENTION = "post_selection_refit_history_point"
PLOT_START_CONVENTION_NOTE = (
    "Trajectory plots start at the first recorded post-selection/post-refit "
    "ADAPT history point for every method. SNAKE delta_abs_prev pre-step "
    "values are not plotted unless all methods provide a safe matching "
    "pre-ADAPT point."
)
METHOD_ORDER = ("snake", "geo", "append")
METHOD_LABEL = {
    "snake": "SNAKE",
    "geo": "Geo-ADAPT",
    "append": "append-only ADAPT",
}
METHOD_SHORT_LABEL = {
    "snake": "SNAKE",
    "geo": "Geo",
    "append": "append",
}
METHOD_COLOR = {
    "snake": "#2B6CB0",
    "geo": "#B83280",
    "append": "#2F855A",
}
METHOD_MARKER = {
    "snake": "o",
    "geo": "s",
    "append": "^",
}
SNAKE_CANONICAL_EFFECTIVE_FLAGS = {
    "--phase1-lambda-theta": "0.001",
    "--phase2-rho": "0.5",
    "--phase2-w-shot": "0.05",
    "--phase3-backend-w-depth": "0.15",
    "--adapt-window-size": "50",
    "--adapt-window-topk": "50",
    "--phase3-geometry-window-size": "10",
    "--phase1-prune-fraction": "0.4",
    "--phase2-batch-near-degenerate-ratio": "0.98",
    "--phase3-batch-near-degenerate-ratio": "0.98",
    "--phase2-batch-rank-rel-tol": "0.25",
    "--phase3-batch-rank-rel-tol": "0.25",
    "--phase2-batch-additivity-tol": "0.25",
    "--phase3-batch-additivity-tol": "0.25",
    "--phase1-maturity-cap-min": "10",
    "--phase1-maturity-cap-max": "25",
    "--phase2-maturity-cap-min": "8",
    "--phase2-maturity-cap-max": "25",
    "--phase3-maturity-cap-min": "4",
    "--phase3-maturity-cap-max": "10",
    "--phase-maturity-shot-min": "1",
    "--phase-maturity-shot-max": "1",
    "--phase1-maturity-shot-cap": "1",
    "--phase2-maturity-shot-cap": "1",
    "--phase3-maturity-shot-cap": "1",
}
SNAKE_CANONICAL_ENABLED_FLAGS = ("--phase-live-hysteresis-disabled",)
SNAKE_CANONICAL_DISABLED_FLAGS = ("--phase-live-hysteresis-enabled",)
MATRIX_LABELS = (
    "A_native_staged_singleton_hard_guard",
    "A_native_staged_singleton_no_guard",
    "B_common_phase0_singleton_hard_guard",
    "B_common_phase0_singleton_no_guard",
    "C_macro_only",
)
EXTRA_MATRIX_LABELS = (
    "A_native_staged_singleton_true_no_guard",
)
ALL_MATRIX_LABELS = (*MATRIX_LABELS, *EXTRA_MATRIX_LABELS)
MATRIX_TITLE = {
    "A_native_staged_singleton_hard_guard": "A1: native/staged singleton, hard guard",
    "A_native_staged_singleton_no_guard": "A2: native/staged singleton, no guard",
    "A_native_staged_singleton_true_no_guard": "A3: native/staged singleton, true no guard",
    "B_common_phase0_singleton_hard_guard": "B1: common Phase-0 singleton, hard guard",
    "B_common_phase0_singleton_no_guard": "B2: common Phase-0 singleton, no guard",
    "C_macro_only": "C: macro-generator only",
}
MATRIX_NOTE = {
    "A_native_staged_singleton_hard_guard": (
        "SNAKE opens singleton Pauli children through the archival Phase-III split with hard-guard symmetry. "
        "Geo and append use the generic runtime singleton split with hard-guard symmetry."
    ),
    "A_native_staged_singleton_no_guard": (
        "SNAKE uses archival Phase-III singleton split with parent symmetry forwarding. "
        "Geo and append use generic runtime singleton split with no symmetry guard."
    ),
    "A_native_staged_singleton_true_no_guard": (
        "SNAKE uses archival Phase-III singleton split with child-set symmetry policy off. "
        "This is a SNAKE-only repair/add-on route unless matching comparator rows are explicitly generated."
    ),
    "B_common_phase0_singleton_hard_guard": (
        "All methods consume a shared Phase-0 singleton Pauli-child pool with hard-guard symmetry."
    ),
    "B_common_phase0_singleton_no_guard": (
        "All methods consume a shared/common Phase-0 singleton Pauli-child pool with shared_pauli_pool_symmetry_policy=off. "
        "Pending rows are configured no-guard until result sidecars prove symmetry_gate_enforced=false."
    ),
    "C_macro_only": "All methods run the full-meta macro-generator pool with no Pauli-child expansion.",
}
LOCAL_FIXCHECK_LABELS = (
    "local_powell_A_native_staged_singleton_hard_guard",
    "local_powell_A_native_staged_singleton_no_guard",
    "local_rotosolve_A_native_staged_singleton_hard_guard",
    "local_rotosolve_A_native_staged_singleton_no_guard",
    "local_powell_B_common_phase0_singleton_hard_guard",
    "local_powell_B_common_phase0_singleton_no_guard",
    "local_powell_C_macro_only",
    "local_rotosolve_B_common_phase0_singleton_hard_guard",
    "local_rotosolve_B_common_phase0_singleton_no_guard",
    "local_rotosolve_C_macro_only",
    "local_spsa_A_native_staged_singleton_hard_guard",
    "local_spsa_A_native_staged_singleton_no_guard",
    "local_spsa_B_common_phase0_singleton_hard_guard",
    "local_spsa_B_common_phase0_singleton_no_guard",
    "local_spsa_C_macro_only",
)
LOCAL_FIXCHECK_TITLE = {
    "local_powell_A_native_staged_singleton_hard_guard": "Local POWELL A1: hard guard",
    "local_powell_A_native_staged_singleton_no_guard": "Local POWELL A2: patched no guard",
    "local_rotosolve_A_native_staged_singleton_hard_guard": "Local ROTOSOLVE A1: hard guard",
    "local_rotosolve_A_native_staged_singleton_no_guard": "Local ROTOSOLVE A2: patched no guard",
    "local_powell_B_common_phase0_singleton_hard_guard": "Local POWELL B1: Phase-0 hard guard",
    "local_powell_B_common_phase0_singleton_no_guard": "Local POWELL B2: Phase-0 no guard",
    "local_powell_C_macro_only": "Local POWELL C: macro only",
    "local_rotosolve_B_common_phase0_singleton_hard_guard": "Local ROTOSOLVE B1: Phase-0 hard guard",
    "local_rotosolve_B_common_phase0_singleton_no_guard": "Local ROTOSOLVE B2: Phase-0 no guard",
    "local_rotosolve_C_macro_only": "Local ROTOSOLVE C: macro only",
    "local_spsa_A_native_staged_singleton_hard_guard": "Local SPSA A1: hard guard",
    "local_spsa_A_native_staged_singleton_no_guard": "Local SPSA A2: parent no guard",
    "local_spsa_B_common_phase0_singleton_hard_guard": "Local SPSA B1: Phase-0 hard guard",
    "local_spsa_B_common_phase0_singleton_no_guard": "Local SPSA B2: Phase-0 no guard",
    "local_spsa_C_macro_only": "Local SPSA C: macro only",
}
LOCAL_FIXCHECK_NOTE = {
    "local_powell_A_native_staged_singleton_hard_guard": (
        "Local fixed probe. Geo and append use hard-guard singleton splitting. "
        "SNAKE rows are local replays, not imported old references."
    ),
    "local_powell_A_native_staged_singleton_no_guard": (
        "Local fixed probe. Geo and append use patched true no-guard splitting "
        "(symmetry_gate_enforced=false). SNAKE uses parent-inherited Phase-III splitting, not true off."
    ),
    "local_rotosolve_A_native_staged_singleton_hard_guard": (
        "Local fixed probe after the ROTOSOLVE stencil repair. "
        "Geo and append use hard-guard singleton splitting. SNAKE rows are local replays."
    ),
    "local_rotosolve_A_native_staged_singleton_no_guard": (
        "Local fixed probe after the ROTOSOLVE stencil repair. Geo and append use patched true "
        "no-guard splitting. SNAKE uses parent-inherited Phase-III splitting, not true off."
    ),
    "local_powell_B_common_phase0_singleton_hard_guard": (
        "Local SNAKE Phase-0 shared singleton Pauli-child replay under POWELL. "
        "This is the strict common-exposure hard-guard control."
    ),
    "local_powell_B_common_phase0_singleton_no_guard": (
        "Local SNAKE Phase-0 shared singleton Pauli-child replay under POWELL with explicit shared_pauli_pool_symmetry_policy=off; executed rows must expose symmetry_gate_enforced=false."
    ),
    "local_powell_C_macro_only": (
        "Local macro-generator-only control. No Pauli-child expansion is active."
    ),
    "local_rotosolve_B_common_phase0_singleton_hard_guard": (
        "Local SNAKE Phase-0 shared singleton Pauli-child replay under ROTOSOLVE. "
        "This is the strict common-exposure hard-guard control."
    ),
    "local_rotosolve_B_common_phase0_singleton_no_guard": (
        "Local SNAKE Phase-0 shared singleton Pauli-child replay under ROTOSOLVE with explicit shared_pauli_pool_symmetry_policy=off; executed rows must expose symmetry_gate_enforced=false."
    ),
    "local_rotosolve_C_macro_only": (
        "Local macro-generator-only control after the ROTOSOLVE stencil repair. "
        "No Pauli-child expansion is active."
    ),
    "local_spsa_A_native_staged_singleton_hard_guard": (
        "Local SNAKE SPSA replay using the source-locked Paper-I HH SPSA schedule. "
        "Geo and append SPSA rows remain absent until their Paper-I schedules are source-locked."
    ),
    "local_spsa_A_native_staged_singleton_no_guard": (
        "Local SNAKE SPSA replay using the source-locked Paper-I HH SPSA schedule. "
        "SNAKE no-guard means parent-inherited Phase-III splitting, not true off."
    ),
    "local_spsa_B_common_phase0_singleton_hard_guard": (
        "Local SNAKE SPSA Phase-0 shared singleton Pauli-child replay using the source-locked Paper-I HH SPSA schedule."
    ),
    "local_spsa_B_common_phase0_singleton_no_guard": (
        "Local SNAKE SPSA Phase-0 shared singleton Pauli-child replay with explicit shared_pauli_pool_symmetry_policy=off; executed rows must expose symmetry_gate_enforced=false."
    ),
    "local_spsa_C_macro_only": (
        "Local SNAKE SPSA macro-generator-only replay using the source-locked Paper-I HH SPSA schedule."
    ),
}


@dataclass(frozen=True)
class LoadedRow:
    record_id: str
    matrix_label: str
    regime: str
    method: str
    optimizer: str | None
    pool_contract: str | None
    symmetry_policy: str | None
    evidence_kind: str
    status: str
    iteration: int | None
    depth: int | None
    abs_delta_e: float | None
    fidelity: float | None
    n2q: int | None
    d2q: int | None
    dc: int | None
    cost_source: str | None
    cost_status: str | None
    s_alg: float | None
    s_grad: float | None
    s_refit: float | None
    s_metric: float | None
    s_outer: float | None
    phase_ledger: str | None
    phase0_events: int | None
    phase0_candidates: int | None
    phase1_events: int | None
    phase1_candidates: int | None
    phase2_events: int | None
    phase2_candidates: int | None
    phase3_events: int | None
    phase3_candidates: int | None
    s_work_status: str | None
    s_work_source: str | None
    s_work_status_detail: str | None
    pool_size: int | None
    expanded_pool_size: int | None
    parent_pool_size: int | None
    child_pool_size: int | None
    fidelity_source: str | None
    source_json: str | None
    source_sha256: str | None
    source_dir: str | None
    settings_status: str | None
    settings_source: str | None
    requested_shared_pauli_pool_mode: str | None
    requested_shared_pauli_pool_symmetry_policy: str | None
    requested_shared_pauli_pool_max_subset_size: str | None
    observed_shared_pauli_pool_symmetry_policy: str | None
    observed_shared_pauli_pool_symmetry_gate_enforced: bool | None
    shared_pauli_pool_runtime_status: str | None
    trajectory_points: tuple[tuple[int, float], ...]
    note: str


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _path_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return token or "default"


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _float_or_none(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _int_or_none(value: Any) -> int | None:
    parsed = _float_or_none(value)
    if parsed is None:
        return None
    return int(round(parsed))


def _first_int(*values: Any) -> int | None:
    for value in values:
        parsed = _int_or_none(value)
        if parsed is not None:
            return parsed
    return None


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
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def _fmt_sci(value: Any) -> str:
    parsed = _float_or_none(value)
    if parsed is None:
        return "--"
    return f"{parsed:.2e}"


def _fmt_float(value: Any, digits: int = 6) -> str:
    parsed = _float_or_none(value)
    if parsed is None:
        return "--"
    return f"{parsed:.{digits}f}"


def _fmt_int(value: Any) -> str:
    parsed = _int_or_none(value)
    if parsed is None:
        return "--"
    return str(parsed)


def _fmt_s(value: Any) -> str:
    parsed = _float_or_none(value)
    if parsed is None:
        return "--"
    if abs(parsed - round(parsed)) < 1e-9:
        return f"{int(round(parsed)):,}"
    return f"{parsed:.3g}"


def _work_value(mapping: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _float_or_none(mapping.get(key))
        if value is not None:
            return value
    return None


def _fmt_compact_count(value: int | None) -> str:
    if value is None:
        return "--"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 10_000:
        return f"{value / 1_000:.0f}k"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}k"
    return str(value)


PHASE_KEYS = ("phase0", "phase1", "phase2", "phase3")
SNAKE_TERMINAL_WORK_SEMANTICS_VERSION = "snake_terminal_s_alg_winner_lineage_v1"
SNAKE_TERMINAL_BEAM_ROW_POLICY = "beam_terminal_winner_history_v1"
SNAKE_TERMINAL_BEAM_AGGREGATE_SCOPE = "all_expanded_scored_branches"
SNAKE_TERMINAL_WINNER_WORK_SCOPES = {"winner_lineage_terminal", "winner_lineage_only"}


def _snake_terminal_s_alg_display_semantics_status(sidecar: Mapping[str, Any]) -> tuple[str, str | None]:
    if sidecar.get("work_semantics_version") != SNAKE_TERMINAL_WORK_SEMANTICS_VERSION:
        return "stale:missing_or_old_work_semantics_version", "regenerate terminal sidecar"
    explicit = sidecar.get("S_alg_display_semantics_status")
    if isinstance(explicit, str) and explicit:
        detail = sidecar.get("S_alg_display_semantics_detail")
        return explicit, str(detail) if detail is not None else None
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


def _phase_counts_from_summary(summary: Mapping[str, Any] | None) -> dict[str, dict[str, int | None]]:
    counts = {phase: {"events": None, "candidates": None} for phase in PHASE_KEYS}
    if not isinstance(summary, Mapping):
        return counts
    by_phase = summary.get("per_phase") or summary.get("by_phase")
    if not isinstance(by_phase, Mapping):
        return counts
    for phase in PHASE_KEYS:
        payload = by_phase.get(phase)
        if not isinstance(payload, Mapping):
            continue
        counts[phase]["events"] = _int_or_none(payload.get("candidate_work_event_count"))
        counts[phase]["candidates"] = _int_or_none(
            payload.get("candidate_count_total") or payload.get("records_evaluated")
        )
    return counts


def _phase_ledger_from_counts(counts: Mapping[str, Mapping[str, int | None]]) -> str | None:
    entries: list[str] = []
    for phase, label in (("phase0", "p0"), ("phase1", "p1"), ("phase2", "p2"), ("phase3", "p3")):
        payload = counts.get(phase)
        if not isinstance(payload, Mapping):
            continue
        events = _int_or_none(payload.get("events"))
        candidates = _int_or_none(payload.get("candidates"))
        if events is None and candidates is None:
            continue
        entries.append(f"{label} {_fmt_compact_count(events)}/{_fmt_compact_count(candidates)}")
    return " ".join(entries) or None


def _phase_ledger_from_summary(summary: Mapping[str, Any] | None) -> str | None:
    return _phase_ledger_from_counts(_phase_counts_from_summary(summary))


def _phase_kwargs(counts: Mapping[str, Mapping[str, int | None]]) -> dict[str, int | None]:
    return {
        "phase0_events": _int_or_none(counts.get("phase0", {}).get("events")),
        "phase0_candidates": _int_or_none(counts.get("phase0", {}).get("candidates")),
        "phase1_events": _int_or_none(counts.get("phase1", {}).get("events")),
        "phase1_candidates": _int_or_none(counts.get("phase1", {}).get("candidates")),
        "phase2_events": _int_or_none(counts.get("phase2", {}).get("events")),
        "phase2_candidates": _int_or_none(counts.get("phase2", {}).get("candidates")),
        "phase3_events": _int_or_none(counts.get("phase3", {}).get("events")),
        "phase3_candidates": _int_or_none(counts.get("phase3", {}).get("candidates")),
    }


def _component_sum_matches(
    s_alg: float | None,
    s_grad: float | None,
    s_refit: float | None,
    s_metric: float | None,
    s_outer: float | None,
) -> bool:
    if s_alg is None or None in {s_grad, s_refit, s_metric, s_outer}:
        return False
    total = float(s_grad or 0.0) + float(s_refit or 0.0) + float(s_metric or 0.0) + float(s_outer or 0.0)
    return math.isclose(total, float(s_alg), rel_tol=1e-9, abs_tol=1e-6)


def _hamiltonian_work_sum(s_refit: float | None, s_outer: float | None) -> float | None:
    if s_refit is None or s_outer is None:
        return None
    return float(s_refit) + float(s_outer)


def _component_sum_detail(
    s_alg: float | None,
    s_grad: float | None,
    s_refit: float | None,
    s_metric: float | None,
    s_outer: float | None,
) -> str:
    if s_alg is None:
        return "missing_S_alg"
    missing = [
        name
        for name, value in (
            ("S_grad", s_grad),
            ("S_refit", s_refit),
            ("S_metric", s_metric),
            ("S_H_out", s_outer),
        )
        if value is None
    ]
    if missing:
        return "missing_components:" + ",".join(missing)
    total = float(s_grad or 0.0) + float(s_refit or 0.0) + float(s_metric or 0.0) + float(s_outer or 0.0)
    return f"component_sum={total:.12g};S_alg={float(s_alg):.12g}"


def _snake_s_work(work: Mapping[str, Any] | None, qiskit_cost: Mapping[str, Any]) -> dict[str, Any]:
    display_semantics_status, display_semantics_detail = _snake_terminal_s_alg_display_semantics_status(qiskit_cost)
    if display_semantics_status != "ok":
        return {
            "s_alg": None,
            "s_grad": None,
            "s_refit": None,
            "s_metric": None,
            "s_outer": None,
            "s_work_status": display_semantics_status,
            "s_work_source": "paper_i_terminal_qiskit_cost.json:S_alg_display_semantics",
            "s_work_status_detail": display_semantics_detail or display_semantics_status,
        }
    if (
        qiskit_cost.get("work_semantics_version") == SNAKE_TERMINAL_WORK_SEMANTICS_VERSION
        and str(qiskit_cost.get("S_alg_status") or "") == "ok"
    ):
        s_alg = _float_or_none(qiskit_cost.get("S_alg"))
        s_grad = _float_or_none(qiskit_cost.get("S_alg_N_grad_probe"))
        s_refit = _float_or_none(qiskit_cost.get("S_alg_N_H_refit_eval"))
        s_metric = _float_or_none(qiskit_cost.get("S_alg_N_metric_probe"))
        s_outer = _float_or_none(qiskit_cost.get("S_alg_N_H_outer_eval"))
        source = "paper_i_terminal_qiskit_cost.json:S_alg_N_components"
        if _component_sum_matches(s_alg, s_grad, s_refit, s_metric, s_outer):
            return {
                "s_alg": s_alg,
                "s_grad": s_grad,
                "s_refit": s_refit,
                "s_metric": s_metric,
                "s_outer": s_outer,
                "s_work_status": "ok",
                "s_work_source": source,
                "s_work_status_detail": "terminal sidecar components_sum_to_S_alg",
            }
        return {
            "s_alg": s_alg,
            "s_grad": None,
            "s_refit": None,
            "s_metric": None,
            "s_outer": None,
            "s_work_status": "mismatch:component_sum",
            "s_work_source": source,
            "s_work_status_detail": _component_sum_detail(s_alg, s_grad, s_refit, s_metric, s_outer),
        }
    if not isinstance(work, Mapping):
        return {
            "s_alg": _float_or_none(qiskit_cost.get("S_alg")),
            "s_grad": None,
            "s_refit": None,
            "s_metric": None,
            "s_outer": None,
            "s_work_status": "blocked:missing_snake_algorithmic_work",
            "s_work_source": None,
            "s_work_status_detail": "snake_algorithmic_work.json missing",
        }
    s_alg = _float_or_none(work.get("S_alg"))
    if s_alg is None:
        s_alg = _float_or_none(qiskit_cost.get("S_alg"))
    top_status = str(work.get("S_alg_status") or "")
    nested = work.get("algorithmic_measurement_work")
    nested_status = str(nested.get("status") or "") if isinstance(nested, Mapping) else ""
    component_counts = work.get("component_counts")
    source = "snake_algorithmic_work.json:component_counts"
    if not isinstance(component_counts, Mapping):
        component_counts = nested.get("components") if isinstance(nested, Mapping) else None
        source = "snake_algorithmic_work.json:algorithmic_measurement_work.components"
    if top_status != "ok" or nested_status != "ok":
        return {
            "s_alg": None,
            "s_grad": None,
            "s_refit": None,
            "s_metric": None,
            "s_outer": None,
            "s_work_status": "blocked:non_ok_status",
            "s_work_source": source if isinstance(component_counts, Mapping) else None,
            "s_work_status_detail": f"S_alg_status={top_status or 'missing'};nested_status={nested_status or 'missing'}",
        }
    if not isinstance(component_counts, Mapping):
        return {
            "s_alg": s_alg,
            "s_grad": None,
            "s_refit": None,
            "s_metric": None,
            "s_outer": None,
            "s_work_status": "blocked:missing_components",
            "s_work_source": None,
            "s_work_status_detail": "component_counts and nested components missing",
        }
    s_grad = _work_value(component_counts, "N_grad_probe", "S_alg_N_grad_probe")
    s_refit = _work_value(component_counts, "N_H_refit_eval", "S_alg_N_H_refit_eval")
    s_metric = _work_value(component_counts, "N_metric_probe", "S_alg_N_metric_probe")
    s_outer = _work_value(component_counts, "N_H_outer_eval", "S_alg_N_H_outer_eval")
    if not _component_sum_matches(s_alg, s_grad, s_refit, s_metric, s_outer):
        return {
            "s_alg": s_alg,
            "s_grad": None,
            "s_refit": None,
            "s_metric": None,
            "s_outer": None,
            "s_work_status": "mismatch:component_sum",
            "s_work_source": source,
            "s_work_status_detail": _component_sum_detail(s_alg, s_grad, s_refit, s_metric, s_outer),
        }
    return {
        "s_alg": s_alg,
        "s_grad": s_grad,
        "s_refit": s_refit,
        "s_metric": s_metric,
        "s_outer": s_outer,
        "s_work_status": "ok",
        "s_work_source": source,
        "s_work_status_detail": "components_sum_to_S_alg",
    }


def _generic_s_work(result: Mapping[str, Any]) -> dict[str, Any]:
    s_alg = _float_or_none(result.get("S_alg"))
    if s_alg is None:
        return {
            "s_alg": None,
            "s_grad": None,
            "s_refit": None,
            "s_metric": None,
            "s_outer": None,
            "s_work_status": "blocked:missing_S_alg",
            "s_work_source": None,
            "s_work_status_detail": "S_alg missing",
        }
    s_grad = _float_or_none(result.get("S_alg_N_grad_probe"))
    s_refit = _float_or_none(result.get("S_alg_N_H_refit_eval"))
    s_metric = _float_or_none(result.get("S_alg_N_metric_probe"))
    s_outer = _float_or_none(result.get("S_alg_N_H_outer_eval"))
    source = "generic_static_single.json:S_alg_N_components"
    if not _component_sum_matches(s_alg, s_grad, s_refit, s_metric, s_outer):
        return {
            "s_alg": s_alg,
            "s_grad": None,
            "s_refit": None,
            "s_metric": None,
            "s_outer": None,
            "s_work_status": "mismatch:component_sum",
            "s_work_source": source,
            "s_work_status_detail": _component_sum_detail(s_alg, s_grad, s_refit, s_metric, s_outer),
        }
    return {
        "s_alg": s_alg,
        "s_grad": s_grad,
        "s_refit": s_refit,
        "s_metric": s_metric,
        "s_outer": s_outer,
        "s_work_status": "ok",
        "s_work_source": source,
        "s_work_status_detail": "components_sum_to_S_alg",
    }


def _first_int_from_mapping(mapping: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = _int_or_none(mapping.get(key))
        if value is not None:
            return value
    return None


def _pool_size_fields(*payloads: Mapping[str, Any] | None) -> dict[str, int | None]:
    merged: list[Mapping[str, Any]] = [payload for payload in payloads if isinstance(payload, Mapping)]
    return {
        "pool_size": next(
            (
                _first_int_from_mapping(payload, "pool_size", "operator_pool_size", "macro_pool_size")
                for payload in merged
                if _first_int_from_mapping(payload, "pool_size", "operator_pool_size", "macro_pool_size") is not None
            ),
            None,
        ),
        "expanded_pool_size": next(
            (
                _first_int_from_mapping(payload, "expanded_pool_size", "expanded_pool_term_count")
                for payload in merged
                if _first_int_from_mapping(payload, "expanded_pool_size", "expanded_pool_term_count") is not None
            ),
            None,
        ),
        "parent_pool_size": next(
            (
                _first_int_from_mapping(payload, "parent_pool_size", "parent_count", "split_parent_count")
                for payload in merged
                if _first_int_from_mapping(payload, "parent_pool_size", "parent_count", "split_parent_count") is not None
            ),
            None,
        ),
        "child_pool_size": next(
            (
                _first_int_from_mapping(payload, "child_pool_size", "child_count", "child_set_count")
                for payload in merged
                if _first_int_from_mapping(payload, "child_pool_size", "child_count", "child_set_count") is not None
            ),
            None,
        ),
    }


def _read_records(records_tsv: Path) -> list[dict[str, str]]:
    with records_tsv.open(newline="", encoding="utf-8") as fh:
        return [{str(k): "" if v is None else str(v) for k, v in row.items()} for row in csv.DictReader(fh, delimiter="\t")]


def _record_dirs(record_id: str, result_roots: Sequence[Path]) -> Iterable[Path]:
    for root in result_roots:
        candidate = root / record_id
        if candidate.exists():
            yield candidate


def _record_suffix(record_id: str) -> str:
    return record_id.split("__", 1)[1] if "__" in record_id else record_id


def _record_dirs_by_suffix(record_id: str, result_roots: Sequence[Path]) -> Iterable[Path]:
    suffix = _record_suffix(record_id)
    if not suffix:
        return
    for root in result_roots:
        if not root.exists():
            continue
        for candidate in root.iterdir():
            if candidate.is_dir() and _record_suffix(candidate.name) == suffix:
                yield candidate


def _row_optimizer(row: Mapping[str, str]) -> str | None:
    value = str(row.get("optimizer") or row.get("adapt_optimizer_kind") or "").strip()
    return value.upper() if value else None


def _report_optimizer_from_rows(rows: Sequence[LoadedRow]) -> str | None:
    values = sorted({str(row.optimizer).upper() for row in rows if row.optimizer})
    if len(values) == 1:
        return values[0]
    if values:
        return "+".join(values)
    return None


def _row_pool_contract(row: Mapping[str, str]) -> str | None:
    value = str(row.get("pool_contract") or row.get("hh_adaptive_pool_profile") or "").strip()
    return value or None


def _row_symmetry_policy(row: Mapping[str, str]) -> str | None:
    value = str(row.get("symmetry_policy") or "").strip()
    if value:
        return value
    if row.get("method_key") == "snake":
        value = str(row.get("snake_phase3_runtime_split_child_set_symmetry_policy") or "").strip()
    else:
        value = str(row.get("generic_adapt_runtime_split_symmetry_policy") or "").strip()
    return value or None


def _row_evidence_kind(row: Mapping[str, str]) -> str:
    return str(row.get("evidence_kind") or "candidate").strip() or "candidate"


def _requested_shared_pauli_pool(row: Mapping[str, str]) -> dict[str, str | None]:
    return {
        "mode": str(row.get("shared_pauli_pool_mode") or "").strip() or None,
        "symmetry_policy": str(row.get("shared_pauli_pool_symmetry_policy") or "").strip() or None,
        "max_subset_size": str(row.get("shared_pauli_pool_max_subset_size") or "").strip() or None,
    }


def _shared_pauli_pool_contract_status(
    row: Mapping[str, str],
    contract: Mapping[str, Any] | None,
    *,
    result_present: bool,
) -> tuple[str | None, str | None, bool | None]:
    label = str(row.get("matrix_label") or "").strip()
    requested = _requested_shared_pauli_pool(row)
    requested_mode = requested.get("mode")
    requested_policy = requested.get("symmetry_policy")
    requested_cap = requested.get("max_subset_size")
    if label != "B_common_phase0_singleton_no_guard":
        if isinstance(contract, Mapping):
            observed_policy = str(contract.get("symmetry_policy") or "") or None
            gate = contract.get("symmetry_gate_enforced")
            return "observed:runtime_contract_present", observed_policy, bool(gate) if gate is not None else None
        return None, None, None
    if requested_mode != "shared_pauli_child_sets_v1" or requested_policy != "off" or requested_cap != "1":
        return (
            "mismatch:label_requires_shared_pool_off_cap1",
            None,
            None,
        )
    if not result_present:
        return "configured_pending:no_result", None, None
    if not isinstance(contract, Mapping):
        return "mismatch:runtime_contract_missing", None, None
    observed_policy = str(contract.get("symmetry_policy") or "") or None
    gate_raw = contract.get("symmetry_gate_enforced")
    gate = bool(gate_raw) if gate_raw is not None else None
    if observed_policy == "off" and gate is False:
        return "observed:true_no_guard", observed_policy, gate
    return f"mismatch:runtime_contract_policy:{observed_policy}:gate:{gate}", observed_policy, gate


def _status_with_shared_pool_validation(default_status: str, shared_contract_status: str | None) -> str:
    if shared_contract_status and shared_contract_status.startswith("mismatch:"):
        return "evidence-invalid"
    return default_status


def _source_record_has_resume_prefix(row: Mapping[str, str]) -> bool:
    if str(row.get("method_key") or "") != "snake":
        return False
    text = "\n".join(
        str(row.get(key) or "")
        for key in (
            "source_command_args_json",
            "source_command_sh",
            "snake_cli_overrides_json",
            "source_contract_note",
            "record_id",
        )
    )
    return "--adapt-resume-scaffold-json" in text or "adapt_ref_base_depth" in text


def _result_path_for_loaded_row(row: LoadedRow) -> Path | None:
    if not row.source_json:
        return None
    path = Path(row.source_json)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _loaded_snake_has_resume_prefix(row: LoadedRow) -> bool:
    if row.method != "snake":
        return False
    resume_tokens = ("resume_repair", "resume_scaffold", "resume_from", "_resume_")
    source_text = " ".join(value or "" for value in (row.source_json, row.source_dir, row.note))
    if any(token in source_text for token in resume_tokens):
        return True
    path = _result_path_for_loaded_row(row)
    if path is None or not path.exists():
        return False
    try:
        payload = _read_json(path)
    except Exception:
        return False
    adapt_vqe = payload.get("adapt_vqe") if isinstance(payload, Mapping) else None
    if not isinstance(adapt_vqe, Mapping):
        return False
    ref_depth = _int_or_none(adapt_vqe.get("adapt_ref_base_depth"))
    if ref_depth is not None and ref_depth > 0:
        return True
    history = adapt_vqe.get("history")
    if isinstance(history, Sequence) and not isinstance(history, (str, bytes)):
        for item in history:
            if not isinstance(item, Mapping):
                continue
            step_depth = _int_or_none(item.get("depth"))
            cumulative = _int_or_none(item.get("depth_cumulative"))
            if step_depth is not None and cumulative is not None and cumulative > step_depth:
                return True
            break
    return False


def _purged_record_from_source(row: Mapping[str, str], reason: str) -> dict[str, Any]:
    return {
        "record_id": str(row.get("record_id") or ""),
        "matrix_label": str(row.get("matrix_label") or ""),
        "regime": str(row.get("display_regime") or ""),
        "method": str(row.get("method_key") or ""),
        "optimizer": _row_optimizer(row),
        "status": "source-record-excluded",
        "source_json": str(row.get("source_json") or ""),
        "source_command_args_json": str(row.get("source_command_args_json") or ""),
        "reason": reason,
    }


def _purged_record_from_loaded(row: LoadedRow, reason: str) -> dict[str, Any]:
    return {
        "record_id": row.record_id,
        "matrix_label": row.matrix_label,
        "regime": row.regime,
        "method": row.method,
        "optimizer": row.optimizer,
        "status": row.status,
        "source_json": row.source_json,
        "source_dir": row.source_dir,
        "source_sha256": row.source_sha256,
        "reason": reason,
    }


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists() and path.is_file():
            return path
    return None


def _result_path_for_method(method_key: str, record_dir: Path) -> Path | None:
    if method_key == "snake":
        return _first_existing((record_dir / "json" / "result.json", record_dir / "result.json"))
    return _first_existing(
        (
            record_dir / "result" / "generic_static_single.json",
            record_dir / "result" / "result.json",
            record_dir / "generic_static_single.json",
        )
    )


def _posthoc_fidelity(record_dir: Path) -> dict[str, Any] | None:
    path = record_dir / "paper_i_posthoc_fidelity.json"
    if not path.exists():
        return None
    try:
        payload = _read_json(path)
    except Exception:
        return None
    if not isinstance(payload, Mapping):
        return None
    if str(payload.get("status") or "") != "computed":
        return None
    fidelity = _float_or_none(payload.get("fidelity"))
    if fidelity is None:
        infidelity = _float_or_none(payload.get("infidelity"))
        if infidelity is not None:
            fidelity = 1.0 - infidelity
    if fidelity is None:
        return None
    return {
        "fidelity": fidelity,
        "fidelity_source": str(
            payload.get("fidelity_source")
            or "posthoc_same_cutoff_dense_sector_exact_state_v1"
        ),
        "path": _rel(path),
    }


def _select_record_dir(row: Mapping[str, str], result_roots: Sequence[Path]) -> Path | None:
    record_dirs: list[Path] = []
    seen: set[Path] = set()
    for record_dir in (
        *_record_dirs(str(row["record_id"]), result_roots),
        *_record_dirs_by_suffix(str(row["record_id"]), result_roots),
    ):
        if record_dir not in seen:
            record_dirs.append(record_dir)
            seen.add(record_dir)
    if not record_dirs:
        return None
    method_key = str(row.get("method_key") or "")
    for record_dir in record_dirs:
        if _result_path_for_method(method_key, record_dir) is not None:
            return record_dir
    return record_dirs[0]


def _progress_points(path: Path) -> tuple[tuple[int, float], ...]:
    if not path.exists():
        return ()
    points: list[tuple[int, float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, Mapping):
            continue
        x = _first_int(payload.get("iteration"), payload.get("k"), payload.get("depth"))
        y = (
            _float_or_none(payload.get("abs_delta_e"))
            or _float_or_none(payload.get("abs_delta_e_after"))
            or _float_or_none(payload.get("same_cutoff_abs_delta_e"))
            or _float_or_none(payload.get("delta_E_abs"))
        )
        if x is not None and y is not None and y > 0:
            points.append((x, y))
    return tuple(points)


def _snake_points(adapt_vqe: Mapping[str, Any]) -> tuple[tuple[int, float], ...]:
    points: list[tuple[int, float]] = []
    history = adapt_vqe.get("history")
    if isinstance(history, Sequence) and not isinstance(history, (str, bytes)):
        for index, item in enumerate(history):
            if not isinstance(item, Mapping):
                continue
            # SNAKE history records have historically used a field named
            # "depth" for the controller step.  Do not use it here: report
            # x-axes and k columns are ADAPT iteration indices, not terminal
            # ansatz/circuit depths.
            x = _first_int(
                item.get("iteration"),
                item.get("adapt_iteration"),
                item.get("k_iteration"),
                item.get("step"),
                item.get("k"),
            )
            if x is None:
                x = index
            y = (
                _float_or_none(item.get("delta_abs_current"))
                or _float_or_none(item.get("abs_delta_e"))
                or _float_or_none(item.get("benchmark_target_abs_delta_e_current"))
            )
            if x is not None and y is not None and y > 0:
                points.append((x, y))
    final_iteration = _snake_iteration_count(adapt_vqe)
    final_error = _float_or_none(adapt_vqe.get("abs_delta_e") or adapt_vqe.get("benchmark_target_abs_delta_e_current"))
    if final_iteration is not None and final_error is not None and final_error > 0:
        if points and points[-1][0] == final_iteration:
            points[-1] = (final_iteration, final_error)
        else:
            points.append((final_iteration, final_error))
    return tuple(points)


def _snake_iteration_count(adapt_vqe: Mapping[str, Any]) -> int | None:
    explicit = _first_int(
        adapt_vqe.get("adapt_num_iterations"),
        adapt_vqe.get("adapt_iteration_count"),
        adapt_vqe.get("iteration_count"),
        adapt_vqe.get("nit"),
    )
    if explicit is not None:
        return explicit
    history = adapt_vqe.get("history")
    if isinstance(history, Sequence) and not isinstance(history, (str, bytes)):
        return len(history)
    return None


def _generic_points(result: Mapping[str, Any], progress_path: Path | None) -> tuple[tuple[int, float], ...]:
    points: list[tuple[int, float]] = []
    history = result.get("adapt_history")
    if isinstance(history, Sequence) and not isinstance(history, (str, bytes)):
        for index, item in enumerate(history):
            if not isinstance(item, Mapping):
                continue
            x = _first_int(item.get("iteration"), item.get("k"))
            if x is None:
                x = index
            y = (
                _float_or_none(item.get("abs_delta_e_same_cutoff_after"))
                or _float_or_none(item.get("abs_delta_e_after"))
                or _float_or_none(item.get("delta_E_abs_after"))
            )
            if x is not None and y is not None and y > 0:
                points.append((x, y))
    if not points and progress_path is not None:
        points.extend(_progress_points(progress_path))
    final_depth = _generic_iteration_count(result)
    final_error = _float_or_none(result.get("abs_delta_e") or result.get("delta_E_abs"))
    if final_depth is not None and final_error is not None and final_error > 0:
        if points and points[-1][0] == final_depth:
            points[-1] = (final_depth, final_error)
        else:
            points.append((final_depth, final_error))
    return tuple(points)


def _generic_iteration_count(result: Mapping[str, Any]) -> int | None:
    parsed = _first_int(result.get("adapt_num_iterations"), result.get("adapt_iteration_count"))
    if parsed is not None:
        return parsed
    history = result.get("adapt_history")
    if isinstance(history, Sequence) and not isinstance(history, (str, bytes)):
        return len(history)
    return _first_int(result.get("adapt_depth_reached"), result.get("depth"))


def _compile_snake_terminal_qiskit_cost(record_dir: Path, result_path: Path) -> dict[str, Any]:
    """Compile the terminal SNAKE ansatz with the shared Table-I Qiskit convention."""

    sidecar_path = record_dir / "paper_i_terminal_qiskit_cost.json"
    source_sha = _sha256(result_path)
    if sidecar_path.exists():
        try:
            cached = _read_json(sidecar_path)
        except Exception:
            cached = None
        if (
            isinstance(cached, Mapping)
            and cached.get("source_sha256") == source_sha
            and cached.get("work_semantics_version") == SNAKE_TERMINAL_WORK_SEMANTICS_VERSION
        ):
            return dict(cached)
    try:
        repo_text = str(REPO_ROOT)
        if repo_text not in sys.path:
            sys.path.insert(0, repo_text)
        from pipelines.reporting.build_paper_i_hh_child_fairness_pdf import _compile_local_snake_terminal_cost

        return dict(_compile_local_snake_terminal_cost(record_dir, result_path))
    except Exception as exc:  # pragma: no cover - report should fail closed on optional Qiskit path
        return {
            "schema": "paper_i_hh_fullmeta_matrix_snake_terminal_qiskit_cost_v1",
            "work_semantics_version": SNAKE_TERMINAL_WORK_SEMANTICS_VERSION,
            "status": f"blocked:{type(exc).__name__}",
            "compile_error": str(exc),
            "source_json": _rel(result_path),
            "source_sha256": _sha256(result_path),
            "S_alg_display_semantics_status": "blocked:terminal_cost_sidecar_unavailable",
            "S_alg_display_semantics_detail": str(exc),
        }


def _compiled_value_if_qiskit(cost: Mapping[str, Any], *keys: str) -> int | None:
    status = str(cost.get("status") or "").lower()
    source = str(cost.get("compiled_resource_source_kind") or cost.get("source_kind") or "").lower()
    if status not in {"done", "ok"}:
        return None
    if "qiskit" not in source or not bool(cost.get("compiled_resource_qiskit_validated")):
        return None
    for key in keys:
        parsed = _int_or_none(cost.get(key))
        if parsed is not None:
            return parsed
    return None


def _validated_generic_qiskit_cost(result: Mapping[str, Any]) -> tuple[bool, str]:
    source = str(result.get("compiled_resource_source_kind") or "")
    status = str(result.get("compiled_circuit_stats_status") or "")
    values = {
        "N2q": _int_or_none(result.get("compiled_count_2q_total")),
        "D2q": _int_or_none(result.get("compiled_depth_2q_total")),
        "Dc": _int_or_none(result.get("compiled_depth_total")),
    }
    if not bool(result.get("compiled_resource_qiskit_validated")):
        return False, "blocked:qiskit_validation_flag_false"
    if status.lower() != "ok":
        return False, f"blocked:qiskit_status_{status or 'missing'}"
    if "qiskit" not in source.lower():
        return False, "blocked:qiskit_source_kind_missing"
    if any(value is None for value in values.values()):
        return False, "blocked:qiskit_cost_columns_missing"
    if min(int(value) for value in values.values() if value is not None) < 0:
        return False, "blocked:qiskit_cost_negative"
    if int(values["Dc"]) < int(values["D2q"]):
        return False, "blocked:qiskit_depth_order_invalid"
    return True, "ok"


def _argv_flag_value(args: Sequence[str], flag: str) -> str | None:
    for index, item in enumerate(args):
        if item == flag:
            if index + 1 < len(args) and not str(args[index + 1]).startswith("--"):
                return str(args[index + 1])
            return "present"
        prefix = f"{flag}="
        if str(item).startswith(prefix):
            return str(item)[len(prefix) :]
    return None


def _snake_effective_settings_status(record_dir: Path) -> tuple[str | None, str | None]:
    manifest_path = record_dir / "cell_manifest.json"
    if not manifest_path.exists():
        return None, None
    try:
        manifest = _read_json(manifest_path)
    except Exception as exc:
        return f"blocked:unreadable_cell_manifest:{type(exc).__name__}", _rel(manifest_path)
    command = manifest.get("command") if isinstance(manifest, Mapping) else None
    if not isinstance(command, Sequence) or isinstance(command, (str, bytes)):
        return "blocked:missing_effective_command", _rel(manifest_path)
    args = [str(item) for item in command]
    mismatches: list[str] = []
    for flag, expected in SNAKE_CANONICAL_EFFECTIVE_FLAGS.items():
        actual = _argv_flag_value(args, flag)
        if actual != expected:
            mismatches.append(f"{flag}={actual or 'missing'} expected {expected}")
    for flag in SNAKE_CANONICAL_ENABLED_FLAGS:
        if _argv_flag_value(args, flag) != "present":
            mismatches.append(f"{flag}=missing")
    for flag in SNAKE_CANONICAL_DISABLED_FLAGS:
        if _argv_flag_value(args, flag) is not None:
            mismatches.append(f"{flag}=present")
    if mismatches:
        return "mismatch:" + "; ".join(mismatches[:8]), _rel(manifest_path)
    return "ok:effective_command_matches_canonical_overlay", _rel(manifest_path)


def _load_snake(row: Mapping[str, str], record_dir: Path) -> LoadedRow:
    result_path = _result_path_for_method("snake", record_dir)
    if result_path is None:
        note = "missing result.json"
        return _empty_loaded(row, status="pending", note=note, record_dir=record_dir)
    payload = _read_json(result_path)
    adapt_vqe = payload.get("adapt_vqe") if isinstance(payload, Mapping) else None
    if not isinstance(adapt_vqe, Mapping):
        return _empty_loaded(row, status="failed", note="missing adapt_vqe payload", record_dir=record_dir)
    qiskit_cost = _compile_snake_terminal_qiskit_cost(record_dir, result_path)
    work_path = _first_existing((record_dir / "snake_algorithmic_work.json",))
    work = _read_json(work_path) if work_path is not None else {}
    s_work = _snake_s_work(work if isinstance(work, Mapping) else None, qiskit_cost)
    controller_summary = adapt_vqe.get("controller_measurement_work_summary")
    phase_counts = _phase_counts_from_summary(controller_summary if isinstance(controller_summary, Mapping) else None)
    pool_sizes = _pool_size_fields(
        adapt_vqe,
        adapt_vqe.get("shared_pauli_pool_contract") if isinstance(adapt_vqe.get("shared_pauli_pool_contract"), Mapping) else None,
        work if isinstance(work, Mapping) else None,
    )
    cost_source = str(qiskit_cost.get("compiled_resource_source_kind") or "")
    cost_status = str(qiskit_cost.get("status") or "blocked:qiskit_cost_missing")
    fidelity = _float_or_none(adapt_vqe.get("exact_state_fidelity"))
    settings_status, settings_source = _snake_effective_settings_status(record_dir)
    requested_shared_pool = _requested_shared_pauli_pool(row)
    shared_contract = adapt_vqe.get("shared_pauli_pool_contract")
    shared_contract_status, observed_shared_policy, observed_shared_gate = _shared_pauli_pool_contract_status(
        row,
        shared_contract if isinstance(shared_contract, Mapping) else None,
        result_present=True,
    )
    note_bits = [f"terminal Qiskit cost {cost_status}"]
    if settings_status:
        note_bits.append(f"settings {settings_status}")
    if shared_contract_status:
        note_bits.append(f"shared-pool {shared_contract_status}")
    return LoadedRow(
        record_id=str(row["record_id"]),
        matrix_label=str(row["matrix_label"]),
        regime=str(row["display_regime"]),
        method="snake",
        optimizer=_row_optimizer(row),
        pool_contract=_row_pool_contract(row),
        symmetry_policy=_row_symmetry_policy(row),
        evidence_kind=_row_evidence_kind(row),
        status=_status_with_shared_pool_validation("done", shared_contract_status),
        iteration=_snake_iteration_count(adapt_vqe),
        depth=_int_or_none(adapt_vqe.get("ansatz_depth")),
        abs_delta_e=_float_or_none(adapt_vqe.get("abs_delta_e") or adapt_vqe.get("benchmark_target_abs_delta_e_current")),
        fidelity=fidelity,
        n2q=_compiled_value_if_qiskit(qiskit_cost, "compiled_count_2q_total", "N2q"),
        d2q=_compiled_value_if_qiskit(qiskit_cost, "compiled_depth_2q_total", "D2q"),
        dc=_compiled_value_if_qiskit(qiskit_cost, "compiled_depth_total", "Dc"),
        cost_source=cost_source or None,
        cost_status=cost_status,
        s_alg=s_work["s_alg"],
        s_grad=s_work["s_grad"],
        s_refit=s_work["s_refit"],
        s_metric=s_work["s_metric"],
        s_outer=s_work["s_outer"],
        phase_ledger=_phase_ledger_from_counts(phase_counts),
        **_phase_kwargs(phase_counts),
        s_work_status=s_work["s_work_status"],
        s_work_source=s_work["s_work_source"],
        s_work_status_detail=s_work["s_work_status_detail"],
        **pool_sizes,
        fidelity_source=str(adapt_vqe.get("exact_state_fidelity_source") or "adapt_vqe.exact_state_fidelity")
        if fidelity is not None
        else None,
        source_json=_rel(result_path),
        source_sha256=_sha256(result_path),
        source_dir=_rel(record_dir),
        settings_status=settings_status,
        settings_source=settings_source,
        requested_shared_pauli_pool_mode=requested_shared_pool.get("mode"),
        requested_shared_pauli_pool_symmetry_policy=requested_shared_pool.get("symmetry_policy"),
        requested_shared_pauli_pool_max_subset_size=requested_shared_pool.get("max_subset_size"),
        observed_shared_pauli_pool_symmetry_policy=observed_shared_policy,
        observed_shared_pauli_pool_symmetry_gate_enforced=observed_shared_gate,
        shared_pauli_pool_runtime_status=shared_contract_status,
        trajectory_points=_snake_points(adapt_vqe),
        note="; ".join(note_bits),
    )


def _load_generic(row: Mapping[str, str], record_dir: Path) -> LoadedRow:
    result_path = _result_path_for_method(str(row.get("method_key") or ""), record_dir)
    if result_path is None:
        return _empty_loaded(row, status="pending", note="missing generic result", record_dir=record_dir)
    payload = _read_json(result_path)
    result = payload.get("result") if isinstance(payload, Mapping) else None
    if not isinstance(result, Mapping):
        result = payload if isinstance(payload, Mapping) else {}
    progress_path = _first_existing((record_dir / "adapt_iteration_progress.jsonl",))
    fidelity = (
        _float_or_none(result.get("exact_state_fidelity"))
        or _float_or_none(result.get("fidelity_exact"))
        or _float_or_none(result.get("state_fidelity"))
    )
    infidelity = _float_or_none(result.get("infidelity_exact"))
    if fidelity is None and infidelity is not None:
        fidelity = 1.0 - infidelity
    if fidelity is not None:
        fidelity_source = "generic_static_result"
    elif result.get("infidelity_status"):
        fidelity_source = str(result.get("infidelity_status"))
    else:
        fidelity_source = None
    posthoc = _posthoc_fidelity(record_dir)
    if posthoc is not None:
        fidelity = _float_or_none(posthoc.get("fidelity"))
        fidelity_source = str(posthoc.get("fidelity_source") or "posthoc_fidelity")
    cost_source = str(result.get("compiled_resource_source_kind") or "")
    cost_ok, cost_status = _validated_generic_qiskit_cost(result)
    requested_shared_pool = _requested_shared_pauli_pool(row)
    shared_contract_raw = result.get("shared_pauli_pool_contract") or payload.get("shared_pauli_pool_contract")
    shared_contract_status, observed_shared_policy, observed_shared_gate = _shared_pauli_pool_contract_status(
        row,
        shared_contract_raw if isinstance(shared_contract_raw, Mapping) else None,
        result_present=True,
    )
    note_bits = [cost_status]
    if posthoc is not None:
        note_bits.append(f"fidelity sidecar {_rel(record_dir / 'paper_i_posthoc_fidelity.json')}")
    if shared_contract_status:
        note_bits.append(f"shared-pool {shared_contract_status}")
    s_work = _generic_s_work(result)
    controller_summary = (
        result.get("controller_measurement_work_summary")
        if isinstance(result.get("controller_measurement_work_summary"), Mapping)
        else None
    )
    phase_counts = _phase_counts_from_summary(controller_summary)
    pool_sizes = _pool_size_fields(
        result,
        payload if isinstance(payload, Mapping) else None,
        shared_contract_raw if isinstance(shared_contract_raw, Mapping) else None,
    )
    return LoadedRow(
        record_id=str(row["record_id"]),
        matrix_label=str(row["matrix_label"]),
        regime=str(row["display_regime"]),
        method=str(row["method_key"]),
        optimizer=_row_optimizer(row),
        pool_contract=_row_pool_contract(row),
        symmetry_policy=_row_symmetry_policy(row),
        evidence_kind=_row_evidence_kind(row),
        status=_status_with_shared_pool_validation("done", shared_contract_status),
        iteration=_generic_iteration_count(result),
        depth=_int_or_none(result.get("adapt_depth_reached") or result.get("depth")),
        abs_delta_e=_float_or_none(result.get("abs_delta_e") or result.get("delta_E_abs")),
        fidelity=fidelity,
        n2q=_int_or_none(result.get("compiled_count_2q_total")) if cost_ok else None,
        d2q=_int_or_none(result.get("compiled_depth_2q_total")) if cost_ok else None,
        dc=_int_or_none(result.get("compiled_depth_total")) if cost_ok else None,
        cost_source=cost_source or None,
        cost_status=cost_status,
        s_alg=s_work["s_alg"],
        s_grad=s_work["s_grad"],
        s_refit=s_work["s_refit"],
        s_metric=s_work["s_metric"],
        s_outer=s_work["s_outer"],
        phase_ledger=_phase_ledger_from_counts(phase_counts),
        **_phase_kwargs(phase_counts),
        s_work_status=s_work["s_work_status"],
        s_work_source=s_work["s_work_source"],
        s_work_status_detail=s_work["s_work_status_detail"],
        **pool_sizes,
        fidelity_source=fidelity_source,
        source_json=_rel(result_path),
        source_sha256=_sha256(result_path),
        source_dir=_rel(record_dir),
        settings_status=None,
        settings_source=None,
        requested_shared_pauli_pool_mode=requested_shared_pool.get("mode"),
        requested_shared_pauli_pool_symmetry_policy=requested_shared_pool.get("symmetry_policy"),
        requested_shared_pauli_pool_max_subset_size=requested_shared_pool.get("max_subset_size"),
        observed_shared_pauli_pool_symmetry_policy=observed_shared_policy,
        observed_shared_pauli_pool_symmetry_gate_enforced=observed_shared_gate,
        shared_pauli_pool_runtime_status=shared_contract_status,
        trajectory_points=_generic_points(result, progress_path),
        note="; ".join(note_bits),
    )


def _empty_loaded(
    row: Mapping[str, str],
    *,
    status: str,
    note: str,
    record_dir: Path | None = None,
) -> LoadedRow:
    requested_shared_pool = _requested_shared_pauli_pool(row)
    shared_contract_status, observed_shared_policy, observed_shared_gate = _shared_pauli_pool_contract_status(
        row,
        None,
        result_present=False,
    )
    note_out = note
    if shared_contract_status:
        note_out = f"{note}; shared-pool {shared_contract_status}"
    return LoadedRow(
        record_id=str(row["record_id"]),
        matrix_label=str(row["matrix_label"]),
        regime=str(row["display_regime"]),
        method=str(row["method_key"]),
        optimizer=_row_optimizer(row),
        pool_contract=_row_pool_contract(row),
        symmetry_policy=_row_symmetry_policy(row),
        evidence_kind=_row_evidence_kind(row),
        status=status,
        iteration=None,
        depth=None,
        abs_delta_e=None,
        fidelity=None,
        n2q=None,
        d2q=None,
        dc=None,
        cost_source=None,
        cost_status=None,
        s_alg=None,
        s_grad=None,
        s_refit=None,
        s_metric=None,
        s_outer=None,
        phase_ledger=None,
        phase0_events=None,
        phase0_candidates=None,
        phase1_events=None,
        phase1_candidates=None,
        phase2_events=None,
        phase2_candidates=None,
        phase3_events=None,
        phase3_candidates=None,
        s_work_status=None,
        s_work_source=None,
        s_work_status_detail=None,
        pool_size=None,
        expanded_pool_size=None,
        parent_pool_size=None,
        child_pool_size=None,
        fidelity_source=None,
        source_json=None,
        source_sha256=None,
        source_dir=None if record_dir is None else _rel(record_dir),
        settings_status=None,
        settings_source=None,
        requested_shared_pauli_pool_mode=requested_shared_pool.get("mode"),
        requested_shared_pauli_pool_symmetry_policy=requested_shared_pool.get("symmetry_policy"),
        requested_shared_pauli_pool_max_subset_size=requested_shared_pool.get("max_subset_size"),
        observed_shared_pauli_pool_symmetry_policy=observed_shared_policy,
        observed_shared_pauli_pool_symmetry_gate_enforced=observed_shared_gate,
        shared_pauli_pool_runtime_status=shared_contract_status,
        trajectory_points=(),
        note=note_out,
    )


def _load_row(row: Mapping[str, str], result_roots: Sequence[Path]) -> LoadedRow:
    if str(row.get("runnable") or "").lower() != "true":
        return _empty_loaded(row, status="blocked", note=str(row.get("blocked_reason") or row.get("blocker") or "blocked"))
    record_dir = _select_record_dir(row, result_roots)
    if record_dir is None:
        return _empty_loaded(row, status="pending", note="result not fetched")
    if row.get("method_key") == "snake":
        return _load_snake(row, record_dir)
    return _load_generic(row, record_dir)


def load_rows(
    records_tsv: Path,
    result_roots: Sequence[Path],
    *,
    label_order: Sequence[str] = MATRIX_LABELS,
) -> list[LoadedRow]:
    label_set = set(label_order)
    records = [record for record in _read_records(records_tsv) if str(record.get("matrix_label") or "") in label_set]
    rows = [_load_row(record, result_roots) for record in records]
    return sorted(
        rows,
        key=lambda item: (
            label_order.index(item.matrix_label) if item.matrix_label in label_order else 99,
            REGIME_ORDER.index(item.regime) if item.regime in REGIME_ORDER else 99,
            METHOD_ORDER.index(item.method) if item.method in METHOD_ORDER else 99,
        ),
    )


def load_rows_with_purge(
    records_tsv: Path,
    result_roots: Sequence[Path],
    *,
    label_order: Sequence[str] = MATRIX_LABELS,
) -> tuple[list[LoadedRow], list[dict[str, Any]]]:
    label_set = set(label_order)
    rows: list[LoadedRow] = []
    purged_rows: list[dict[str, Any]] = []
    for record in _read_records(records_tsv):
        if str(record.get("matrix_label") or "") not in label_set:
            continue
        if _source_record_has_resume_prefix(record):
            purged_rows.append(
                _purged_record_from_source(
                    record,
                    "snake_resume_prefix_source_record_violates_depth_zero_fair_contract",
                )
            )
            continue
        loaded = _load_row(record, result_roots)
        if _loaded_snake_has_resume_prefix(loaded):
            purged_rows.append(
                _purged_record_from_loaded(
                    loaded,
                    "snake_resume_prefix_result_violates_depth_zero_fair_contract",
                )
            )
            continue
        rows.append(loaded)
    return (
        sorted(
            rows,
            key=lambda item: (
                label_order.index(item.matrix_label) if item.matrix_label in label_order else 99,
                REGIME_ORDER.index(item.regime) if item.regime in REGIME_ORDER else 99,
                METHOD_ORDER.index(item.method) if item.method in METHOD_ORDER else 99,
            ),
        ),
        purged_rows,
    )


def purge_resume_prefix_rows(rows: Sequence[LoadedRow]) -> tuple[list[LoadedRow], list[dict[str, Any]]]:
    kept: list[LoadedRow] = []
    purged_rows: list[dict[str, Any]] = []
    for row in rows:
        if _loaded_snake_has_resume_prefix(row):
            purged_rows.append(
                _purged_record_from_loaded(
                    row,
                    "snake_resume_prefix_result_violates_depth_zero_fair_contract",
                )
            )
            continue
        kept.append(row)
    return kept, purged_rows


def _sort_loaded_rows(rows: Sequence[LoadedRow], label_order: Sequence[str]) -> list[LoadedRow]:
    return sorted(
        rows,
        key=lambda item: (
            label_order.index(item.matrix_label) if item.matrix_label in label_order else 99,
            REGIME_ORDER.index(item.regime) if item.regime in REGIME_ORDER else 99,
            METHOD_ORDER.index(item.method) if item.method in METHOD_ORDER else 99,
        ),
    )


def _loaded_row_key(row: LoadedRow) -> tuple[str, str, str]:
    return (row.matrix_label, row.regime, row.method)


def _purged_row_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("matrix_label") or ""), str(row.get("regime") or ""), str(row.get("method") or ""))


def _rows_by_key(rows: Sequence[LoadedRow]) -> dict[tuple[str, str, str], LoadedRow]:
    return {(row.matrix_label, row.regime, row.method): row for row in rows}


def load_depth_zero_repair_rows(
    records_tsv: Path,
    result_roots: Sequence[Path],
    *,
    label_order: Sequence[str] = MATRIX_LABELS,
) -> tuple[list[LoadedRow], list[dict[str, Any]]]:
    """Load depth-zero repair rows without purging on historical anchor text.

    Repair records intentionally retain source-row provenance in fields such as
    ``source_command_args_json``.  The active command is defined after
    ``snake_cli_overrides_json`` is applied, so source-text resume flags are not
    a sufficient reason to suppress these rows.  We still reject a fetched
    repair result if the loaded runtime artifact itself contains a resume
    prefix.
    """

    label_set = set(label_order)
    rows: list[LoadedRow] = []
    purged_rows: list[dict[str, Any]] = []
    for record in _read_records(records_tsv):
        if str(record.get("matrix_label") or "") not in label_set:
            continue
        loaded = _load_row(record, result_roots)
        if _loaded_snake_has_resume_prefix(loaded):
            purged_rows.append(
                _purged_record_from_loaded(
                    loaded,
                    "snake_depth_zero_repair_result_still_contains_resume_prefix",
                )
            )
            continue
        rows.append(
            replace(
                loaded,
                note="; ".join(bit for bit in (loaded.note, "depth-zero repair overlay") if bit),
            )
        )
    return _sort_loaded_rows(rows, label_order), purged_rows


def overlay_depth_zero_repair_rows(
    rows: Sequence[LoadedRow],
    purged_rows: Sequence[Mapping[str, Any]],
    repair_rows: Sequence[LoadedRow],
    *,
    label_order: Sequence[str] = MATRIX_LABELS,
) -> tuple[list[LoadedRow], list[dict[str, Any]]]:
    if not repair_rows:
        return _sort_loaded_rows(rows, label_order), [dict(item) for item in purged_rows]
    by_key = _rows_by_key(rows)
    repair_keys = {_loaded_row_key(row) for row in repair_rows}
    for row in repair_rows:
        by_key[_loaded_row_key(row)] = row
    filtered_purged = [dict(item) for item in purged_rows if _purged_row_key(item) not in repair_keys]
    return _sort_loaded_rows(tuple(by_key.values()), label_order), filtered_purged


def _plot_contract_regime(
    output_dir: Path,
    label: str,
    regime: str,
    rows: Sequence[LoadedRow],
    *,
    figure_namespace: str = "default",
) -> str | None:
    plotted_rows = [row for row in rows if row.trajectory_points]
    if not plotted_rows:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.15, 1.45), dpi=180)
    for row in plotted_rows:
        xs = [point[0] for point in row.trajectory_points]
        ys = [point[1] for point in row.trajectory_points]
        if not xs or not ys:
            continue
        ax.plot(
            xs,
            ys,
            label=METHOD_SHORT_LABEL[row.method],
            color=METHOD_COLOR[row.method],
            marker=METHOD_MARKER[row.method],
            linewidth=1.05,
            markersize=2.5,
            markevery=max(1, len(xs) // 5),
        )
    ax.set_yscale("log")
    ax.set_xlabel("iteration", fontsize=7)
    ax.set_ylabel("|Delta E|", fontsize=7)
    ax.set_title(regime, fontsize=8)
    ax.grid(True, which="both", linewidth=0.35, alpha=0.35)
    ax.tick_params(axis="both", labelsize=6)
    ax.legend(fontsize=5.4, frameon=False, loc="best")
    fig.tight_layout(pad=0.22)
    plot_path = (
        output_dir
        / "figures"
        / _path_token(figure_namespace)
        / f"{_path_token(figure_namespace)}__{label}__{regime.replace('-', '_')}.png"
    )
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path)
    plt.close(fig)
    return str(plot_path.relative_to(output_dir))


def build_plots(
    output_dir: Path,
    rows: Sequence[LoadedRow],
    *,
    label_order: Sequence[str] = MATRIX_LABELS,
    figure_namespace: str = "default",
) -> dict[str, dict[str, str]]:
    plots: dict[str, dict[str, str]] = {}
    for label in label_order:
        for regime in REGIME_ORDER:
            subset = [row for row in rows if row.matrix_label == label and row.regime == regime]
            rel = _plot_contract_regime(output_dir, label, regime, subset, figure_namespace=figure_namespace)
            if rel:
                plots.setdefault(label, {})[regime] = rel
    return plots


def _plot_box(image_rel: str | None, label: str, placeholder: str) -> str:
    if image_rel:
        return rf"\includegraphics[width=0.98\linewidth,height=1.08in,keepaspectratio]{{{_tex_escape(image_rel)}}}"
    return "\n".join(
        [
            r"\fbox{%",
            r"\begin{minipage}[c][1.03in][c]{0.96\linewidth}",
            r"\centering\scriptsize",
            _tex_escape(label) + r"\\",
            _tex_escape(placeholder),
            r"\end{minipage}%",
            r"}",
        ]
    )


def _cost_table(
    rows: Sequence[LoadedRow],
    *,
    expected_methods: Sequence[str] = METHOD_ORDER,
    purged_methods: Sequence[str] = (),
) -> str:
    by_method = {row.method: row for row in rows}
    purged_method_set = set(purged_methods)
    lines = [
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{llrrrrrrr}",
        r"\hline",
        r"Method & status & $k_{\rm iter}$ & $d_{\rm ans}$ & $|\Delta E|$ & $F$ & $N_{2q}$ & $D_{2q}$ & $D_c$ \\",
        r"\hline",
    ]
    if not expected_methods:
        lines.append(r"\multicolumn{9}{c}{No rows in this report batch} \\")
    for method in expected_methods:
        row = by_method.get(method)
        if row is None:
            status = "repair pending" if method in purged_method_set else "missing"
            values = [METHOD_SHORT_LABEL[method], status, "--", "--", "--", "--", "--", "--", "--"]
        elif row.status in {"done", "reference"}:
            values = [
                METHOD_SHORT_LABEL[method],
                row.status,
                _fmt_int(row.iteration),
                _fmt_int(row.depth),
                _fmt_sci(row.abs_delta_e),
                _fmt_float(row.fidelity),
                _fmt_int(row.n2q),
                _fmt_int(row.d2q),
                _fmt_int(row.dc),
            ]
        else:
            values = [
                METHOD_SHORT_LABEL[method],
                row.status,
                "--",
                "--",
                "--",
                "--",
                "--",
                "--",
                "--",
            ]
        lines.append(" & ".join(_tex_escape(value) for value in values) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}%", r"}"])
    return "\n".join(lines)


def _work_table(
    rows: Sequence[LoadedRow],
    *,
    expected_methods: Sequence[str] = METHOD_ORDER,
) -> str:
    by_method = {row.method: row for row in rows}
    lines = [
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\hline",
        r"Method & P0 & P1 & P2 & P3 & $S_{\rm grad}$ & $S_H$ & $S_{\rm metric}$ & $S_{\rm alg}$ \\",
        r"\hline",
    ]
    if not expected_methods:
        lines.append(r"\multicolumn{9}{c}{No rows in this report batch} \\")
    for method in expected_methods:
        row = by_method.get(method)
        if row is None or row.status not in {"done", "reference"}:
            values = [METHOD_SHORT_LABEL[method], "--", "--", "--", "--", "--", "--", "--", "--"]
        else:
            values = [
                METHOD_SHORT_LABEL[method],
                _fmt_s(row.phase0_candidates),
                _fmt_s(row.phase1_candidates),
                _fmt_s(row.phase2_candidates),
                _fmt_s(row.phase3_candidates),
                _fmt_s(row.s_grad),
                _fmt_s(_hamiltonian_work_sum(row.s_refit, row.s_outer)),
                _fmt_s(row.s_metric),
                _fmt_s(row.s_alg),
            ]
        lines.append(" & ".join(_tex_escape(value) for value in values) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}%", r"}"])
    return "\n".join(lines)


def _expected_methods_for_label(rows: Sequence[LoadedRow], label: str) -> tuple[str, ...]:
    return tuple(method for method in METHOD_ORDER if any(row.matrix_label == label and row.method == method for row in rows))


def _purged_methods_for_panel(
    purged_rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
    regime: str,
) -> tuple[str, ...]:
    methods = {
        str(item.get("method") or "")
        for item in purged_rows
        if isinstance(item, Mapping)
        and item.get("matrix_label") == label
        and item.get("regime") == regime
        and str(item.get("method") or "") in METHOD_ORDER
    }
    return tuple(method for method in METHOD_ORDER if method in methods)


def _matrix_title_with_optimizer(title: str, optimizer: str | None) -> str:
    if not optimizer:
        return title
    upper_title = title.upper()
    known_optimizers = ("POWELL", "ROTOSOLVE", "SPSA")
    if any(name in upper_title for name in known_optimizers):
        return title
    return f"{optimizer}: {title}"


def _contract_page(
    *,
    label: str,
    rows: Sequence[LoadedRow],
    plots: Mapping[str, Mapping[str, str]],
    matrix_title: Mapping[str, str] = MATRIX_TITLE,
    matrix_note: Mapping[str, str] = MATRIX_NOTE,
    optimizer: str | None = None,
    first_page_manifest: str | None = None,
    purged_rows: Sequence[Mapping[str, Any]] = (),
) -> list[str]:
    lines: list[str] = []
    if first_page_manifest:
        lines.extend([first_page_manifest, r"\vspace{0.05in}"])
    lines.extend(
        [
            rf"\textbf{{{_tex_escape(_matrix_title_with_optimizer(matrix_title.get(label, label), optimizer))}}}\\[-0.24em]",
            rf"{{\scriptsize {_tex_escape(matrix_note.get(label, ''))}\par}}",
            r"\vspace{0.04in}",
        ]
    )
    by_key = _rows_by_key(rows)
    for idx, regime in enumerate(REGIME_ORDER):
        if idx % 3 == 0:
            if idx:
                lines.append(r"\par\vspace{0.055in}")
            lines.append(r"\noindent")
        else:
            lines.append(r"\hfill")
        subset = [by_key[(label, regime, method)] for method in METHOD_ORDER if (label, regime, method) in by_key]
        purged_methods = _purged_methods_for_panel(purged_rows, label=label, regime=regime)
        purged_count = len(purged_methods)
        expected_methods = tuple(
            method
            for method in METHOD_ORDER
            if any(row.method == method for row in subset) or method in purged_methods
        )
        done = sum(1 for row in subset if row.status == "done")
        reference = sum(1 for row in subset if row.status == "reference")
        blocked = sum(1 for row in subset if row.status == "blocked")
        pending = sum(1 for row in subset if row.status == "pending")
        failed = sum(1 for row in subset if row.status == "failed")
        invalid = sum(1 for row in subset if row.status == "evidence-invalid")
        if not subset and purged_count:
            status = f"{purged_count} purged by depth-zero fair contract"
        elif not subset:
            status = "missing local evidence"
        else:
            status = f"{done} done, {reference} reference, {pending} pending, {blocked} blocked"
            if purged_count:
                status = f"{status}, {purged_count} purged"
        if failed:
            status = f"{status}, {failed} failed"
        if invalid:
            status = f"{status}, {invalid} invalid"
        if blocked == len(subset) and subset:
            placeholder = "blocked by contract"
        elif invalid:
            placeholder = "evidence invalid"
        elif failed:
            placeholder = "result failed or incomplete"
        elif purged_count and not subset:
            placeholder = "purged by depth-zero fair contract"
        else:
            placeholder = "pending result trajectory"
        image_rel = plots.get(label, {}).get(regime) if isinstance(plots.get(label), Mapping) else None
        lines.extend(
            [
                r"\begin{minipage}[t]{0.318\linewidth}",
                r"\centering",
                rf"\textbf{{\scriptsize {_tex_escape(regime)}}}\\[-0.15em]",
                _plot_box(image_rel, regime, placeholder),
                r"\vspace{0.02in}",
                rf"{{\tiny {_tex_escape(status)}\par}}",
                r"\vspace{0.02in}",
                _cost_table(subset, expected_methods=expected_methods, purged_methods=purged_methods),
                r"\vspace{0.015in}",
                _work_table(subset, expected_methods=expected_methods),
                r"\end{minipage}",
            ]
        )
    return lines


def _tex_comment_block(name: str, payload: Mapping[str, Any]) -> list[str]:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return [f"% BEGIN_{name}", f"% {blob}", f"% END_{name}"]


def _make_manifest(
    *,
    records_tsv: Path,
    manifest_json: Path,
    output_dir: Path,
    stem: str,
    result_roots: Sequence[Path],
    rows: Sequence[LoadedRow],
    plots: Mapping[str, Mapping[str, str]],
    label_order: Sequence[str] = MATRIX_LABELS,
    report_mode: str = "full_matrix",
    report_title: str | None = None,
    purged_rows: Sequence[Mapping[str, Any]] = (),
    repair_records_tsvs: Sequence[Path] = (),
    repair_result_roots: Sequence[Path] = (),
) -> dict[str, Any]:
    input_manifest = _read_json(manifest_json) if manifest_json.exists() else {}
    source_contract = input_manifest.get("source_contract") if isinstance(input_manifest, Mapping) else {}
    optimizer = _report_optimizer_from_rows(rows) or "ROTOSOLVE"
    optimizer_overlay_id = optimizer.lower().replace("+", "_")
    if isinstance(source_contract, Mapping):
        optimizer = _report_optimizer_from_rows(rows) or str(source_contract.get("optimizer") or optimizer)
        optimizer_overlay_id = str(source_contract.get("optimizer_overlay_id") or optimizer_overlay_id)
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        counts.setdefault(row.matrix_label, {}).setdefault(row.status, 0)
        counts[row.matrix_label][row.status] += 1
    sidecar_json = output_dir / f"{stem}.json"
    sidecar_csv = output_dir / f"{stem}.csv"
    sidecar_md = output_dir / f"{stem}.md"
    tex_path = output_dir / f"{stem}.tex"
    return {
        "schema": "paper_i_hh_fullmeta_singleton_symmetry_matrix_report_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "report_mode": report_mode,
        "evidence_status": "candidate_incomplete_or_live",
        "records_tsv": _rel(records_tsv),
        "records_tsv_sha256": _sha256(records_tsv),
        "input_manifest_json": _rel(manifest_json),
        "input_manifest_sha256": _sha256(manifest_json),
        "output_pdf": _rel(output_dir / f"{stem}.pdf"),
        "output_tex": _rel(tex_path),
        "output_json": _rel(sidecar_json),
        "output_csv": _rel(sidecar_csv),
        "output_md": _rel(sidecar_md),
        "result_roots": [_rel(path) for path in result_roots],
        "repair_records_tsvs": [_rel(path) for path in repair_records_tsvs],
        "repair_result_roots": [_rel(path) for path in repair_result_roots],
        "contract": {
            "pool_contract": "full_meta_unfiltered",
            "hva_policy": "included",
            "optimizer": optimizer,
            "optimizer_overlay_id": optimizer_overlay_id,
            "maxiter": 200,
            "depth_cap": 30,
            "child_subset_size": 1,
            "run_class": "candidate",
            "method_order": list(METHOD_ORDER),
            "regime_order": list(REGIME_ORDER),
            "matrix_labels": list(MATRIX_LABELS),
            "display_matrix_labels": list(label_order),
            "report_title": report_title or f"Paper-I HH Full-Meta Singleton Symmetry Matrix: {optimizer}",
            "blocked_policy": (
                "Unsupported rows remain blocked only when their label is included in this report. "
                "Old pre-implementation B2 placeholders should be omitted from active reports once "
                "dedicated true no-guard B2 rows exist."
            ),
            "depth_zero_fair_contract": (
                "All methods must start from a depth-zero ansatz/scaffold. SNAKE rows sourced from "
                "nonzero resume/prefix scaffolds are excluded from active plots, tables, and CSV rows."
            ),
            "plot_start_convention": PLOT_START_CONVENTION,
            "plot_start_convention_note": PLOT_START_CONVENTION_NOTE,
        },
        "source_contract": source_contract,
        "counts_by_matrix_label": counts,
        "purge_policy": "exclude_snake_resume_prefix_rows_from_active_depth_zero_fair_contract",
        "purged_rows": list(purged_rows),
        "plots": plots,
        "rows": [row.__dict__ for row in rows],
    }


def _manifest_strip(manifest: Mapping[str, Any]) -> str:
    contract = manifest.get("contract") if isinstance(manifest.get("contract"), Mapping) else {}
    total_done = sum(1 for row in manifest.get("rows", []) if isinstance(row, Mapping) and row.get("status") == "done")
    total_rows = len(manifest.get("rows", [])) if isinstance(manifest.get("rows"), Sequence) else 0
    return "\n".join(
        [
            r"{\scriptsize",
            r"\textbf{Parameter manifest.} ",
            rf"Generated {_tex_escape(manifest.get('generated_utc'))}. ",
            rf"Pool: {_tex_escape(contract.get('pool_contract'))}; HVA {_tex_escape(contract.get('hva_policy'))}; "
            rf"optimizer {_tex_escape(contract.get('optimizer'))}; maxiter {_tex_escape(contract.get('maxiter'))}; depth cap {_tex_escape(contract.get('depth_cap'))}; singleton cap {_tex_escape(contract.get('child_subset_size'))}. ",
            rf"Rows loaded: {_tex_escape(total_done)} done / {_tex_escape(total_rows)} total. ",
            r"Cost columns use Qiskit-compiled final ansatz circuits only; proxy costs are not displayed. ",
            r"The second table reports phase candidate exposure totals P0--P3 and validated algorithmic work as $S_{\rm grad}$, $S_H$, $S_{\rm metric}$, and $S_{\rm alg}=S_{\rm grad}+S_H+S_{\rm metric}$. Here $S_H$ counts Hamiltonian-energy evaluations, combining refit and outer objective calls; $S_{\rm grad}$ counts SNAKE Phase-0 gradient plus Phase-1 screened tangent work, or comparator gradient probes; $S_{\rm metric}$ counts Geo Gram or SNAKE Phase-2/3 geometry-curvature work. The refit/outer split and detailed phase event/candidate ledgers remain in CSV/JSON sidecars. ",
            r"Trajectory plots start at the first recorded post-selection/post-refit ADAPT history point for every method; SNAKE \texttt{delta\_abs\_prev} pre-step values are not plotted unless all methods provide a safe matching pre-ADAPT point. ",
            r"The plot x-axis and table $k_{\rm iter}$ are ADAPT iterations; $d_{\rm ans}$ is final logical scaffold/ansatz depth, not Qiskit circuit depth. ",
            r"Run labels containing \texttt{depth30} mean logical ansatz/scaffold depth cap 30, not fixed 30 outer-controller iterations. SNAKE may have $k_{\rm iter}<30$ because one controller step can retain/add multiple logical blocks or terminate at the depth cap. ",
            r"SNAKE terminal ansatz depth is never used as $k_{\rm iter}$. ",
            r"SNAKE rows with a nonzero resume/prefix scaffold are excluded from active plots, tables, and CSV rows because the fair contract requires all methods to start from depth zero; excluded row ids are recorded in the JSON sidecar. ",
            r"SNAKE canonical settings are validated from the effective launched \texttt{cell\_manifest.command}; historical \texttt{source\_command\_args\_json} may remain regime-dependent. ",
            r"Fidelity is shown when exact-state fidelity/infidelity is present in the source row or a validated posthoc fidelity sidecar exists; otherwise it remains blank with sidecar status. ",
            rf"Sidecars: \nolinkurl{{{_tex_escape(manifest.get('output_json'))}}}, \nolinkurl{{{_tex_escape(manifest.get('output_csv'))}}}. ",
            r"Evidence status: candidate, not promoted.",
            r"\par}",
        ]
    )


def write_tex(
    output_dir: Path,
    stem: str,
    manifest: Mapping[str, Any],
    rows: Sequence[LoadedRow],
    plots: Mapping[str, Mapping[str, str]],
    *,
    label_order: Sequence[str] = MATRIX_LABELS,
    matrix_title: Mapping[str, str] = MATRIX_TITLE,
    matrix_note: Mapping[str, str] = MATRIX_NOTE,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    tex_path = output_dir / f"{stem}.tex"
    comment = {
        "schema": "paper_i_hh_fullmeta_singleton_symmetry_matrix_tex_provenance_v1",
        "json": manifest.get("output_json"),
        "csv": manifest.get("output_csv"),
        "records_tsv": manifest.get("records_tsv"),
        "input_manifest_json": manifest.get("input_manifest_json"),
        "contract": manifest.get("contract"),
        "purge_policy": manifest.get("purge_policy"),
        "purged_rows": manifest.get("purged_rows"),
    }
    lines: list[str] = _tex_comment_block("MACHINE_READABLE_MATRIX_PROVENANCE", comment)
    lines.extend(
        [
            r"\documentclass[10pt]{article}",
            r"\usepackage[landscape,margin=0.25in]{geometry}",
            r"\usepackage{graphicx}",
            r"\usepackage{hyperref}",
            r"\usepackage{xcolor}",
            r"\hypersetup{colorlinks=true,linkcolor=black,urlcolor=blue}",
            r"\pagestyle{empty}",
            r"\setlength{\parindent}{0pt}",
            r"\setlength{\tabcolsep}{2pt}",
            r"\renewcommand{\arraystretch}{0.84}",
            r"\begin{document}",
            r"\sloppy",
            r"\textbf{\Large "
            + _tex_escape((manifest.get("contract") or {}).get("report_title", "Paper-I HH Full-Meta Singleton Symmetry Matrix"))
            + r"}\\[0.25em]",
        ]
    )
    for index, label in enumerate(label_order):
        if index:
            lines.append(r"\newpage")
        first_manifest = _manifest_strip(manifest) if index == 0 else None
        contract = manifest.get("contract") if isinstance(manifest.get("contract"), Mapping) else {}
        purged_rows = manifest.get("purged_rows") if isinstance(manifest.get("purged_rows"), Sequence) else ()
        lines.extend(
            _contract_page(
                label=label,
                rows=rows,
                plots=plots,
                matrix_title=matrix_title,
                matrix_note=matrix_note,
                optimizer=str(contract.get("optimizer") or "") or None,
                first_page_manifest=first_manifest,
                purged_rows=purged_rows,
            )
        )
    lines.extend([r"\end{document}", ""])
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    return tex_path


def write_csv(output_dir: Path, stem: str, rows: Sequence[LoadedRow]) -> Path:
    path = output_dir / f"{stem}.csv"
    fieldnames = (
        "matrix_label",
        "regime",
        "method",
        "optimizer",
        "pool_contract",
        "symmetry_policy",
        "evidence_kind",
        "status",
        "depth",
        "k_iteration",
        "abs_delta_e",
        "fidelity",
        "N2q",
        "D2q",
        "Dc",
        "cost_source",
        "cost_status",
        "S_alg",
        "S_grad",
        "S_refit",
        "S_metric",
        "S_outer",
        "phase_ledger",
        "phase0_events",
        "phase0_candidates",
        "phase1_events",
        "phase1_candidates",
        "phase2_events",
        "phase2_candidates",
        "phase3_events",
        "phase3_candidates",
        "s_work_status",
        "s_work_source",
        "s_work_status_detail",
        "pool_size",
        "expanded_pool_size",
        "parent_pool_size",
        "child_pool_size",
        "fidelity_source",
        "source_json",
        "source_sha256",
        "source_dir",
        "settings_status",
        "settings_source",
        "requested_shared_pauli_pool_mode",
        "requested_shared_pauli_pool_symmetry_policy",
        "requested_shared_pauli_pool_max_subset_size",
        "observed_shared_pauli_pool_symmetry_policy",
        "observed_shared_pauli_pool_symmetry_gate_enforced",
        "shared_pauli_pool_runtime_status",
        "note",
    )
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "matrix_label": row.matrix_label,
                    "regime": row.regime,
                    "method": row.method,
                    "optimizer": row.optimizer or "",
                    "pool_contract": row.pool_contract or "",
                    "symmetry_policy": row.symmetry_policy or "",
                    "evidence_kind": row.evidence_kind,
                    "status": row.status,
                    "depth": "" if row.depth is None else row.depth,
                    "k_iteration": "" if row.iteration is None else row.iteration,
                    "abs_delta_e": "" if row.abs_delta_e is None else row.abs_delta_e,
                    "fidelity": "" if row.fidelity is None else row.fidelity,
                    "N2q": "" if row.n2q is None else row.n2q,
                    "D2q": "" if row.d2q is None else row.d2q,
                    "Dc": "" if row.dc is None else row.dc,
                    "cost_source": row.cost_source or "",
                    "cost_status": row.cost_status or "",
                    "S_alg": "" if row.s_alg is None else row.s_alg,
                    "S_grad": "" if row.s_grad is None else row.s_grad,
                    "S_refit": "" if row.s_refit is None else row.s_refit,
                    "S_metric": "" if row.s_metric is None else row.s_metric,
                    "S_outer": "" if row.s_outer is None else row.s_outer,
                    "phase_ledger": row.phase_ledger or "",
                    "phase0_events": "" if row.phase0_events is None else row.phase0_events,
                    "phase0_candidates": "" if row.phase0_candidates is None else row.phase0_candidates,
                    "phase1_events": "" if row.phase1_events is None else row.phase1_events,
                    "phase1_candidates": "" if row.phase1_candidates is None else row.phase1_candidates,
                    "phase2_events": "" if row.phase2_events is None else row.phase2_events,
                    "phase2_candidates": "" if row.phase2_candidates is None else row.phase2_candidates,
                    "phase3_events": "" if row.phase3_events is None else row.phase3_events,
                    "phase3_candidates": "" if row.phase3_candidates is None else row.phase3_candidates,
                    "s_work_status": row.s_work_status or "",
                    "s_work_source": row.s_work_source or "",
                    "s_work_status_detail": row.s_work_status_detail or "",
                    "pool_size": "" if row.pool_size is None else row.pool_size,
                    "expanded_pool_size": "" if row.expanded_pool_size is None else row.expanded_pool_size,
                    "parent_pool_size": "" if row.parent_pool_size is None else row.parent_pool_size,
                    "child_pool_size": "" if row.child_pool_size is None else row.child_pool_size,
                    "fidelity_source": row.fidelity_source or "",
                    "source_json": row.source_json or "",
                    "source_sha256": row.source_sha256 or "",
                    "source_dir": row.source_dir or "",
                    "settings_status": row.settings_status or "",
                    "settings_source": row.settings_source or "",
                    "requested_shared_pauli_pool_mode": row.requested_shared_pauli_pool_mode or "",
                    "requested_shared_pauli_pool_symmetry_policy": row.requested_shared_pauli_pool_symmetry_policy or "",
                    "requested_shared_pauli_pool_max_subset_size": row.requested_shared_pauli_pool_max_subset_size or "",
                    "observed_shared_pauli_pool_symmetry_policy": row.observed_shared_pauli_pool_symmetry_policy or "",
                    "observed_shared_pauli_pool_symmetry_gate_enforced": "" if row.observed_shared_pauli_pool_symmetry_gate_enforced is None else str(bool(row.observed_shared_pauli_pool_symmetry_gate_enforced)).lower(),
                    "shared_pauli_pool_runtime_status": row.shared_pauli_pool_runtime_status or "",
                    "note": row.note,
                }
            )
    return path


def write_md(
    output_dir: Path,
    stem: str,
    manifest: Mapping[str, Any],
    rows: Sequence[LoadedRow],
    *,
    label_order: Sequence[str] = MATRIX_LABELS,
) -> Path:
    path = output_dir / f"{stem}.md"
    contract = manifest.get("contract") if isinstance(manifest.get("contract"), Mapping) else {}
    lines = [
        f"# {contract.get('report_title', 'Paper-I HH Full-Meta Singleton Symmetry Matrix')}",
        "",
        f"Generated: `{manifest['generated_utc']}`",
        "",
        "This is candidate evidence, not a promoted Paper-I table source.",
        "",
        "## Contract",
        "",
        "- Pool: `full_meta_unfiltered`.",
        "- HVA: included.",
        f"- Optimizer: `{contract.get('optimizer', '')}`, maxiter `200`, depth cap `30`.",
        "- Singleton child cap: `1`.",
        "- Unsupported rows remain visible as `blocked` only when their label is included in this report.",
        "- Old pre-implementation B2 blocked placeholders are omitted from active reports once dedicated true no-guard B2 rows exist.",
        "- SNAKE rows with nonzero resume/prefix scaffolds are excluded from active report rows because this fair contract requires depth-zero starts for all methods.",
        "- Trajectory plots use a common post-selection/post-refit first-history-point convention; SNAKE pre-step `delta_abs_prev` points are not plotted unless every method has a safe matching pre-ADAPT point.",
        "- B2 no-guard rows are `configured_pending` until a runtime shared-pool contract reports `symmetry_policy=off` and `symmetry_gate_enforced=false`; mismatches are flagged in `shared_pauli_pool_runtime_status`.",
        "- Cost columns are Qiskit-compiled final ansatz circuit costs only; SNAKE graph-span proxy fields are not displayed.",
        "- Fidelity is displayed when source rows expose exact-state fidelity/infidelity or when a validated posthoc fidelity sidecar is present.",
        "- For SNAKE rows, `source_command_args_json` is the historical anchor and may remain regime-dependent. Use `settings_status` and `settings_source`; those validate the effective launched `cell_manifest.command` after `snake_cli_overrides_json` is applied.",
        "",
        "## Status By Contract",
        "",
        "| Matrix label | done | pending | blocked | invalid | failed |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in label_order:
        subset = [row for row in rows if row.matrix_label == label]
        counts = {status: sum(1 for row in subset if row.status == status) for status in ("done", "pending", "blocked", "evidence-invalid", "failed")}
        lines.append(
            f"| `{label}` | {counts['done']} | {counts['pending']} | {counts['blocked']} | "
            f"{counts['evidence-invalid']} | {counts['failed']} |"
        )
    lines.extend(
        [
            "",
            "## Purged Rows",
            "",
        ]
    )
    purged_rows = manifest.get("purged_rows") if isinstance(manifest.get("purged_rows"), Sequence) else []
    if purged_rows:
        lines.extend(
            [
                "| Matrix label | Regime | Method | Optimizer | Reason |",
                "|---|---|---|---|---|",
            ]
        )
        for item in purged_rows:
            if not isinstance(item, Mapping):
                continue
            lines.append(
                f"| `{item.get('matrix_label', '')}` | `{item.get('regime', '')}` | "
                f"`{item.get('method', '')}` | `{item.get('optimizer', '')}` | "
                f"`{item.get('reason', '')}` |"
            )
    else:
        lines.append("No rows were purged by the depth-zero fair-contract filter.")
    lines.extend(
        [
            "",
            "## Sidecars",
            "",
            f"- PDF: `{manifest['output_pdf']}`",
            f"- TeX: `{manifest['output_tex']}`",
            f"- JSON: `{manifest['output_json']}`",
            f"- CSV: `{manifest['output_csv']}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def compile_tex(tex_path: Path) -> None:
    if shutil.which("pdflatex") is None:
        raise RuntimeError("pdflatex is required for this LaTeX-built report")
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        cwd=str(tex_path.parent),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )


def _local_fixcheck_label(optimizer: str, source_label: str) -> str:
    prefix = optimizer.lower()
    return f"local_{prefix}_{source_label}"


def _local_rows(
    *,
    optimizer: str,
    records_tsv: Path,
    local_roots: Sequence[Path],
) -> list[LoadedRow]:
    existing_roots = tuple(root for root in local_roots if root.exists())
    if not existing_roots:
        return []
    dirs_by_suffix: dict[str, Path] = {}
    for root in existing_roots:
        for path in root.iterdir():
            if path.is_dir():
                dirs_by_suffix[_record_suffix(path.name)] = path
    rows: list[LoadedRow] = []
    for record in _read_records(records_tsv):
        if record.get("method_key") not in {"snake", "geo", "append"}:
            continue
        matrix_label = str(record.get("matrix_label") or "")
        staged_labels = {
            "A_native_staged_singleton_hard_guard",
            "A_native_staged_singleton_no_guard",
        }
        phase0_labels = {
            "B_common_phase0_singleton_hard_guard",
            "B_common_phase0_singleton_no_guard",
        }
        macro_labels = {"C_macro_only"}
        if matrix_label not in staged_labels | phase0_labels | macro_labels:
            continue
        record_dir = dirs_by_suffix.get(_record_suffix(str(record.get("record_id") or "")))
        if matrix_label in staged_labels and record_dir is None:
            continue
        if matrix_label in phase0_labels and record.get("method_key") != "snake":
            continue
        local_record = dict(record)
        if record_dir is not None:
            local_record["record_id"] = record_dir.name
        local_record["matrix_label"] = _local_fixcheck_label(optimizer, matrix_label)
        local_record["optimizer"] = optimizer
        local_record["evidence_kind"] = "local"
        loaded = _load_row(local_record, existing_roots)
        note_bits = [loaded.note, "local fixed probe"]
        if matrix_label == "A_native_staged_singleton_no_guard":
            if record.get("method_key") == "snake":
                note_bits.append("SNAKE parent-inherited Phase-III split, not true off")
            else:
                note_bits.append("patched generic no-guard row; symmetry_gate_enforced=false expected")
        if matrix_label == "B_common_phase0_singleton_hard_guard":
            note_bits.append("SNAKE Phase-0 shared singleton hard-guard row")
        if matrix_label == "B_common_phase0_singleton_no_guard":
            note_bits.append("shared Phase-0 true no-guard requires observed symmetry_gate_enforced=false")
        if matrix_label == "C_macro_only":
            note_bits.append("macro-generator-only control; no Pauli-child expansion")
        rows.append(
            replace(
                loaded,
                optimizer=optimizer,
                evidence_kind="local",
                note="; ".join(bit for bit in note_bits if bit),
            )
        )
    return rows


def build_report(
    *,
    records_tsv: Path,
    manifest_json: Path,
    output_dir: Path,
    stem: str,
    result_roots: Sequence[Path],
    label_order: Sequence[str] = MATRIX_LABELS,
    matrix_title: Mapping[str, str] = MATRIX_TITLE,
    matrix_note: Mapping[str, str] = MATRIX_NOTE,
    report_mode: str = "full_matrix",
    report_title: str | None = None,
    repair_records_tsvs: Sequence[Path] = (),
    repair_result_roots: Sequence[Path] = (),
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, purged_rows = load_rows_with_purge(records_tsv, result_roots, label_order=label_order)
    repair_purged_rows: list[dict[str, Any]] = []
    repair_rows: list[LoadedRow] = []
    for repair_records_tsv in repair_records_tsvs:
        loaded_repair_rows, loaded_repair_purged = load_depth_zero_repair_rows(
            repair_records_tsv,
            repair_result_roots,
            label_order=label_order,
        )
        repair_rows.extend(loaded_repair_rows)
        repair_purged_rows.extend(loaded_repair_purged)
    rows, purged_rows = overlay_depth_zero_repair_rows(
        rows,
        purged_rows,
        repair_rows,
        label_order=label_order,
    )
    purged_rows.extend(repair_purged_rows)
    plots = build_plots(output_dir, rows, label_order=label_order, figure_namespace=stem)
    all_result_roots = tuple(result_roots) + tuple(repair_result_roots)
    manifest = _make_manifest(
        records_tsv=records_tsv,
        manifest_json=manifest_json,
        output_dir=output_dir,
        stem=stem,
        result_roots=all_result_roots,
        rows=rows,
        plots=plots,
        label_order=label_order,
        report_mode=report_mode,
        report_title=report_title,
        purged_rows=purged_rows,
        repair_records_tsvs=repair_records_tsvs,
        repair_result_roots=repair_result_roots,
    )
    tex_path = write_tex(
        output_dir,
        stem,
        manifest,
        rows,
        plots,
        label_order=label_order,
        matrix_title=matrix_title,
        matrix_note=matrix_note,
    )
    csv_path = write_csv(output_dir, stem, rows)
    json_path = output_dir / f"{stem}.json"
    md_path = write_md(output_dir, stem, manifest, rows, label_order=label_order)
    manifest["output_csv"] = _rel(csv_path)
    manifest["output_json"] = _rel(json_path)
    manifest["output_md"] = _rel(md_path)
    manifest["output_tex"] = _rel(tex_path)
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    compile_tex(tex_path)
    return manifest


def build_local_fixcheck_report(
    *,
    output_dir: Path = LOCAL_FIXCHECK_OUTPUT_DIR,
    stem: str = LOCAL_FIXCHECK_STEM,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[LoadedRow] = []
    local_result_roots: list[Path] = []
    input_records: list[dict[str, Any]] = []
    for item in LOCAL_FIXCHECK_INPUTS:
        optimizer = str(item["optimizer"])
        records_tsv = Path(item["records_tsv"])
        manifest_json = Path(item["manifest_json"])
        local_roots = tuple(Path(path) for path in item["local_roots"])
        local_result_roots.extend(local_roots)
        input_records.append(
            {
                "optimizer": optimizer,
                "records_tsv": _rel(records_tsv),
                "records_tsv_sha256": _sha256(records_tsv),
                "manifest_json": _rel(manifest_json),
                "manifest_json_sha256": _sha256(manifest_json),
                "local_roots": [_rel(path) for path in local_roots],
            }
        )
        rows.extend(_local_rows(optimizer=optimizer, records_tsv=records_tsv, local_roots=local_roots))
    rows, purged_rows = purge_resume_prefix_rows(rows)
    rows = _sort_loaded_rows(rows, LOCAL_FIXCHECK_LABELS)
    plots = build_plots(output_dir, rows, label_order=LOCAL_FIXCHECK_LABELS, figure_namespace=stem)
    first_input = LOCAL_FIXCHECK_INPUTS[0]
    manifest = _make_manifest(
        records_tsv=Path(first_input["records_tsv"]),
        manifest_json=Path(first_input["manifest_json"]),
        output_dir=output_dir,
        stem=stem,
        result_roots=tuple(local_result_roots),
        rows=rows,
        plots=plots,
        label_order=LOCAL_FIXCHECK_LABELS,
        report_mode="local_fixcheck",
        report_title="Paper-I HH Full-Meta Singleton Local Fixcheck",
        purged_rows=purged_rows,
    )
    manifest["evidence_status"] = "diagnostic_local_fixcheck"
    if isinstance(manifest.get("contract"), dict):
        manifest["contract"]["optimizer"] = "POWELL+ROTOSOLVE+SPSA"
        manifest["contract"]["optimizer_overlay_id"] = "local_fixcheck_mixed"
    manifest["local_fixcheck_inputs"] = input_records
    manifest["local_fixcheck_semantics"] = {
        "generic_off": "patched true no-guard singleton split; rows should expose symmetry_gate_enforced=false",
        "generic_hard_guard": "symmetry-gated singleton split",
        "snake_no_guard": "SNAKE parent-inherited Phase-III split, not true off for A2; B2 shared/common Phase-0 rows require observed shared_pauli_pool_symmetry_policy=off and symmetry_gate_enforced=false",
        "snake_reference_policy": "old SNAKE CSV references are disabled; SNAKE cells are local replays only",
        "scope": "completed local fixed probes; missing rows remain absent until local result JSONs exist",
    }
    tex_path = write_tex(
        output_dir,
        stem,
        manifest,
        rows,
        plots,
        label_order=LOCAL_FIXCHECK_LABELS,
        matrix_title=LOCAL_FIXCHECK_TITLE,
        matrix_note=LOCAL_FIXCHECK_NOTE,
    )
    csv_path = write_csv(output_dir, stem, rows)
    json_path = output_dir / f"{stem}.json"
    md_path = write_md(output_dir, stem, manifest, rows, label_order=LOCAL_FIXCHECK_LABELS)
    manifest["output_csv"] = _rel(csv_path)
    manifest["output_json"] = _rel(json_path)
    manifest["output_md"] = _rel(md_path)
    manifest["output_tex"] = _rel(tex_path)
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    compile_tex(tex_path)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-fixcheck",
        action="store_true",
        help="Build the local POWELL/ROTOSOLVE fixed-probe diagnostic report.",
    )
    parser.add_argument("--records-tsv", type=Path, default=DEFAULT_RECORDS_TSV)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default=DEFAULT_STEM)
    parser.add_argument(
        "--result-root",
        action="append",
        type=Path,
        help="Result root containing record_id directories. May be repeated.",
    )
    parser.add_argument(
        "--repair-records-tsv",
        action="append",
        type=Path,
        help=(
            "Depth-zero repair records TSV to overlay onto the active report. "
            "May be repeated. These rows are validated from fetched runtime artifacts "
            "instead of purged from historical source-command resume text."
        ),
    )
    parser.add_argument(
        "--repair-result-root",
        action="append",
        type=Path,
        help="Result root containing depth-zero repair record_id directories. May be repeated.",
    )
    parser.add_argument(
        "--matrix-label",
        action="append",
        choices=ALL_MATRIX_LABELS,
        help=(
            "Matrix label to include in the active report. May be repeated. "
            "Use this to omit superseded placeholder labels, such as old B2 blocked rows, "
            "when a dedicated repair/add-on report supplies the current evidence."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.local_fixcheck:
        output_dir = LOCAL_FIXCHECK_OUTPUT_DIR if args.output_dir == DEFAULT_OUTPUT_DIR else args.output_dir
        stem = LOCAL_FIXCHECK_STEM if str(args.stem) == DEFAULT_STEM else str(args.stem)
        manifest = build_local_fixcheck_report(output_dir=output_dir, stem=stem)
        print(json.dumps({key: manifest[key] for key in ("output_pdf", "output_json", "output_csv", "output_md")}, indent=2, sort_keys=True))
        return 0
    result_roots = tuple(args.result_root or DEFAULT_RESULT_ROOTS)
    repair_records_tsvs = tuple(args.repair_records_tsv or ())
    repair_result_roots = tuple(args.repair_result_root or ())
    label_order = tuple(args.matrix_label or MATRIX_LABELS)
    manifest = build_report(
        records_tsv=args.records_tsv,
        manifest_json=args.manifest_json,
        output_dir=args.output_dir,
        stem=str(args.stem),
        result_roots=result_roots,
        label_order=label_order,
        repair_records_tsvs=repair_records_tsvs,
        repair_result_roots=repair_result_roots,
    )
    print(json.dumps({key: manifest[key] for key in ("output_pdf", "output_json", "output_csv", "output_md")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
