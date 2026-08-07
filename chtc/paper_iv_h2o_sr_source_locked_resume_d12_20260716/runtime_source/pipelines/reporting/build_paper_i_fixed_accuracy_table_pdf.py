#!/usr/bin/env python3
"""Build the approved Paper-I fixed-accuracy table PDF.

This helper exists to prevent Paper-I current-candidate reports from drifting
back to class-summary or alternate table shapes. It uses the approved
per-Hamiltonian grid as the output contract, optionally fills cells from the
current calibrated summary JSON, and keeps any old candidate PDF path only as a
provenance pointer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SOURCE_PDF = REPO_ROOT / "output/pdf/paper_i_current_candidate_results_20260519.pdf"
DEFAULT_SUMMARY_JSON = (
    REPO_ROOT
    / "raw_outputs/chtc_paper_i_current_20260519/fixed_accuracy_summary/"
    / "table_i_fixed_accuracy_calibrated_summary.json"
)
DEFAULT_OUTPUT_PDF = REPO_ROOT / "output/pdf/paper_i_approved_fixed_accuracy_table_20260519.pdf"

METHODS = [
    "HEA VQE",
    "family-informed VQE",
    "append-only ADAPT",
    "TETRIS-ADAPT",
    "Pos-Geo-ADAPT",
    "Qubit/QEB-ADAPT-VQE",
    "SNAKE",
]

GROUPS = [
    (
        "Fermionic Hamiltonians",
        [
            ("Hubbard", {"hubbard"}),
            ("ionic Hubbard", {"ionic_hubbard", "ionic_hubbard_chain"}),
            ("$t$-$t'$ Hubbard", {"ttprime_hubbard", "t_tprime_hubbard", "tprime_hubbard"}),
            ("extended Hubbard", {"extended_hubbard"}),
            ("spinless $t$-$V$", {"spinless_tv", "spinless_t_v"}),
        ],
    ),
    (
        "Bosonic Hamiltonians",
        [
            ("Bose-Hubbard", {"bose_hubbard"}),
            ("harmonic/Kerr chain", {"harmonic_kerr_chain", "harmonic_kerr"}),
            ("spin-boson", {"spin_boson"}),
        ],
    ),
    (
        "Fermion-boson Hamiltonians",
        [
            ("Hubbard-Holstein", {"hh", "hubbard_holstein"}),
            ("molecular-vibronic H$_2$", {"molecular_vibronic_h2", "vibronic_h2", "molecular_vibronic"}),
        ],
    ),
]

CANONICAL_FAMILY_BY_LABEL = {
    "Hubbard": "hubbard",
    "ionic Hubbard": "ionic_hubbard",
    "$t$-$t'$ Hubbard": "ttprime_hubbard",
    "extended Hubbard": "extended_hubbard",
    "spinless $t$-$V$": "spinless_tv",
    "Bose-Hubbard": "bose_hubbard",
    "harmonic/Kerr chain": "harmonic_kerr_chain",
    "spin-boson": "spin_boson",
    "Hubbard-Holstein": "hh",
    "molecular-vibronic H$_2$": "molecular_vibronic_h2",
}

CUTOFF_STRENGTH_BY_LABEL = {
    "Hubbard": r"--; $2/8$",
    "ionic Hubbard": r"--; $2/8$",
    "$t$-$t'$ Hubbard": r"--; $2/8$",
    "extended Hubbard": r"--; $(2,0.5)/(8,1.5)$",
    "spinless $t$-$V$": r"--; $0.5/1.5$",
    "Bose-Hubbard": r"$(1,4),(1,4);\,2/6$",
    "harmonic/Kerr chain": r"$(4,7),(4,7);\,1/0.75$",
    "spin-boson": r"$(2,5),(6,9);\,0.25/1.0$",
    "Hubbard-Holstein": r"\(\begin{array}{@{}c@{}}(1,4),(2,5);\\(2,0.25)/(8,1.0)\end{array}\)",
    "molecular-vibronic H$_2$": r"\(\begin{array}{@{}c@{}}(1,4),(3,6);\\0.25/1.0\end{array}\)",
}

METHOD_ALIASES = {
    "TETRIS-ADAPT-VQE": "TETRIS-ADAPT",
    "TETRIS ADAPT": "TETRIS-ADAPT",
    "Pos-Geo-ADAPT-VQE": "Pos-Geo-ADAPT",
    "PosGeo ADAPT": "Pos-Geo-ADAPT",
    "QEB-ADAPT-VQE": "Qubit/QEB-ADAPT-VQE",
    "Qubit ADAPT": "Qubit/QEB-ADAPT-VQE",
}

PHONON_FAMILIES = {
    "bose_hubbard",
    "harmonic_kerr_chain",
    "spin_boson",
    "hh",
    "molecular_vibronic_h2",
}

DEFERRED_PLACEHOLDER_FAMILIES: set[str] = set()
DEFERRED_PLACEHOLDER_REASON = "declared_placeholder_case_not_locked_or_wired"

EXPECTED_CASE_ID_BASE = {
    "hubbard": "hubbard",
    "ionic_hubbard": "ionic_hubbard",
    "ttprime_hubbard": "ttprime_hubbard",
    "extended_hubbard": "extended_hubbard",
    "spinless_tv": "spinless_tv",
    "bose_hubbard": "bose_hubbard",
    "harmonic_kerr_chain": "harmonic_kerr_chain",
    "spin_boson": "spin_boson",
    "hh": "hh",
    "molecular_vibronic_h2": "molecular_vibronic_h2",
}

ALGORITHM_ID_BY_METHOD = {
    "HEA VQE": "static_hea_qiskit_vqe",
    "family-informed VQE": "static_family_informed_vqe",
    "append-only ADAPT": "static_full_meta_append_adapt_vqe",
    "TETRIS-ADAPT": "static_tetris_qubit_adapt_vqe",
    "Pos-Geo-ADAPT": "static_pos_geo_adapt_vqe",
    "Qubit/QEB-ADAPT-VQE": "static_qubit_qeb_adapt_vqe",
    "SNAKE": "static_family_native_adapt_phase3",
}


@dataclass(frozen=True)
class TableCell:
    delta_e: str = "--"
    infidelity: str = "--"
    n_2q: str = "--"
    d_2q: str = "--"
    d_circ: str = "--"
    s_alg: str = "--"

    def as_latex_cells(self) -> list[str]:
        return [self.delta_e, self.infidelity, self.n_2q, self.d_2q, self.d_circ, self.s_alg]

    @property
    def s_norm(self) -> str:
        """Backward-compatible name for the final work column.

        The table column now reports validated ``S_alg``.  The old attribute
        name remains so older audit consumers can read historical manifests.
        """
        return self.s_alg


@dataclass
class LoadDiagnostics:
    summary_json: str | None = None
    historical_pareto_json: str | None = None
    display_overlay_json: str | None = None
    summary_missing: bool = False
    skipped_rows: list[str] = field(default_factory=list)
    duplicate_keys: list[str] = field(default_factory=list)
    duplicate_resolutions: list[dict[str, Any]] = field(default_factory=list)
    threshold_mismatch_rows: list[str] = field(default_factory=list)
    raw_row_audits: list[dict[str, Any]] = field(default_factory=list)
    expected_cell_audits: list[dict[str, Any]] = field(default_factory=list)
    summary_metadata: dict[str, Any] = field(default_factory=dict)
    historical_pareto_metadata: dict[str, Any] = field(default_factory=dict)
    display_overlay_metadata: dict[str, Any] = field(default_factory=dict)
    special_findings: dict[str, Any] = field(default_factory=dict)
    missing_expected_count: int = 0
    invalid_target_count: int = 0


def latex_escape(text: str) -> str:
    """Escape text for LaTeX, while preserving already-math Hamiltonian labels."""
    if "$" in text:
        return text
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    out = text
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def normalize_method(method: str | None) -> str | None:
    if method is None:
        return None
    method = str(method).strip()
    return METHOD_ALIASES.get(method, method)


def normalize_family(row: dict[str, Any]) -> str | None:
    family = row.get("family") or row.get("hamiltonian") or row.get("hamiltonian_id")
    if family is None:
        case_id = str(row.get("case_id") or row.get("benchmark_id") or row.get("record_id") or "")
        alias_pairs: list[tuple[str, str]] = []
        for _, hams in GROUPS:
            for label, aliases in hams:
                canonical = CANONICAL_FAMILY_BY_LABEL.get(label, max(aliases, key=len))
                alias_pairs.extend((alias, canonical) for alias in aliases)
        for alias, canonical in sorted(alias_pairs, key=lambda item: len(item[0]), reverse=True):
            if alias in case_id:
                return canonical
        return None
    raw_family = str(family).strip()
    for _, hams in GROUPS:
        for label, aliases in hams:
            if raw_family in aliases:
                return CANONICAL_FAMILY_BY_LABEL.get(label, raw_family)
    return raw_family


def normalize_regime(row: dict[str, Any]) -> str | None:
    raw = row.get("regime") or row.get("physics_regime")
    if raw is not None:
        lowered = str(raw).lower()
        if "weak" in lowered:
            return "weak"
        if "strong" in lowered:
            return "strong"
    case_id = str(row.get("case_id") or row.get("benchmark_id") or row.get("record_id") or "").lower()
    if "weak" in case_id:
        return "weak"
    if "strong" in case_id:
        return "strong"
    return None


def finite_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def first_finite_number(*values: Any) -> float | None:
    """Return the first finite numeric value, preserving valid zeros."""
    for value in values:
        number = finite_number(value)
        if number is not None:
            return number
    return None


def _first_int_value(*values: Any) -> int | None:
    for value in values:
        number = finite_number(value)
        if number is None:
            continue
        if abs(number - round(number)) <= 1e-9:
            return int(round(number))
    return None


def _extract_tagged_int(text: str | None, tag: str) -> int | None:
    if not text:
        return None
    pattern = rf"(?:^|[_/-]){re.escape(tag)}(\d+)(?:$|[_/-])"
    match = re.search(pattern, str(text).lower())
    if match:
        return int(match.group(1))
    return None


def _expected_case_id(family: str | None, regime: str | None) -> str | None:
    if family is None or regime is None:
        return None
    base = EXPECTED_CASE_ID_BASE.get(str(family))
    if base is None:
        return None
    if family == "molecular_vibronic_h2":
        nph_tag = "nph3" if regime == "strong" else "nph1"
        return f"{base}_L2_{nph_tag}_clean_{regime}"
    if family in PHONON_FAMILIES:
        return f"{base}_L2_nph2_clean_{regime}"
    return f"{base}_L2_clean_{regime}"


def _case_id_from_audit(raw: Mapping[str, Any] | None, *, family: str | None, regime: str | None) -> str | None:
    if raw is not None:
        for key in ("case_id", "benchmark_id"):
            value = raw.get(key)
            if value:
                return str(value)
    return _expected_case_id(family, regime)


def _infer_n_ph_work(row: Mapping[str, Any] | None, *, family: str | None, case_id: str | None) -> int | None:
    if family not in PHONON_FAMILIES:
        return None
    if row is not None:
        value = _first_int_value(
            row.get("n_ph_work"),
            row.get("n_ph_algorithm"),
            row.get("n_ph_alg"),
            row.get("boson_cutoff"),
        )
        if value is not None:
            return value
        for text_key in ("case_id", "benchmark_id", "record_id", "suite_profile"):
            value = _extract_tagged_int(str(row.get(text_key) or ""), "nph")
            if value is not None:
                return value
    value = _extract_tagged_int(case_id, "nph")
    if value is not None:
        return value
    return 2


def _infer_n_ph_ref(
    row: Mapping[str, Any] | None,
    *,
    family: str | None,
    case_id: str | None,
    n_ph_work: int | None,
) -> int | None:
    if family not in PHONON_FAMILIES:
        return None
    if row is not None:
        value = _first_int_value(
            row.get("n_ph_ref"),
            row.get("n_ph_ed"),
            row.get("n_ph_eval"),
            row.get("exact_reference_n_ph_max"),
            row.get("exact_reference_boson_cutoff"),
        )
        if value is not None:
            return value
        for text_key in ("case_id", "benchmark_id", "record_id", "suite_profile"):
            text = str(row.get(text_key) or "")
            value = _extract_tagged_int(text, "ref")
            if value is not None:
                return value
    value = _extract_tagged_int(case_id, "ref")
    if value is not None:
        return value
    if n_ph_work == 4:
        return 5
    if family == "molecular_vibronic_h2" and n_ph_work == 3:
        return 6
    return 4


def _paper_i_ladder_stage(
    row: Mapping[str, Any] | None,
    *,
    family: str | None,
    n_ph_work: int | None,
    n_ph_ref: int | None,
) -> str:
    """Backward-compatible audit field for the now-locked cutoff contract.

    Older reports used this field to encode an escalation ladder.  The Paper-I
    reporting contract no longer discovers phonon cutoffs by ladder.  We retain
    the field name so older audit consumers do not break, but the value now
    records that the row is expected to use its locked working/reference cutoff
    pair.
    """
    if family not in PHONON_FAMILIES:
        return "not_applicable_nonphonon"
    if row is not None:
        for key in ("paper_i_cutoff_contract", "paper_i_locked_cutoff_stage"):
            value = row.get(key)
            if value:
                return str(value)
    if n_ph_work is not None and n_ph_ref is not None:
        return "locked_known_cutoff"
    return "locked_cutoff_metadata_missing"


def _next_ladder_stage(stage: str | None) -> str | None:
    # Escalation ladders are obsolete for Paper-I reporting; cutoffs are locked
    # per row/family/regime before table promotion.
    return None


def _next_stage_case_id(case_id: str | None, next_stage: str | None) -> str | None:
    return None


def _cutoff_missing_reason(*, family: str | None, n_ph_work: int | None, n_ph_ref: int | None) -> str | None:
    if family not in PHONON_FAMILIES:
        return None
    if n_ph_work is None or n_ph_ref is None:
        return "phonon_cutoff_missing_or_unlocked"
    return None


def format_number(value: Any) -> str:
    number = finite_number(value)
    if number is None:
        return "--"
    if abs(number) >= 1000:
        return f"{number:.0f}"
    if abs(number) >= 1:
        return f"{number:.3g}"
    if number == 0:
        return "0"
    return f"{number:.2e}"


def format_delta(row: dict[str, Any], threshold: float) -> str:
    status = str(row.get("threshold_status") or "").lower()
    raw_abs_error = first_finite_number(
        row.get("abs_delta_e"),
        row.get("primary_error"),
        row.get("delta_e_mean_included"),
    )
    if raw_abs_error is not None:
        return format_number(max(0.0, raw_abs_error - threshold))
    excess = first_finite_number(
        row.get("delta_e_excess"),
        row.get("delta_E_excess"),
        row.get("target_excess_delta_e"),
    )
    if excess is not None:
        return format_number(max(0.0, excess))
    if "not_reached" in status:
        return "NR"
    if "running" in status:
        return "running"
    if "failed" in status:
        return "failed"
    if "upper_bound" in status:
        return "UB"
    return "--"


def format_infidelity(row: dict[str, Any]) -> str:
    for key in (
        "infidelity",
        "infidelity_reference",
        "infidelity_4",
        "infidelity_same",
        "infidelity_exact",
        "one_minus_fidelity",
        "one_minus_F",
        "1-F",
    ):
        value = first_finite_number(row.get(key))
        if value is not None:
            return format_number(value)
    fidelity = first_finite_number(row.get("fidelity"), row.get("F"))
    if fidelity is not None:
        return format_number(max(0.0, 1.0 - fidelity))
    return "--"


ELIGIBLE_COST_STATUSES = {
    "ok_native_first_hit",
    "ok_terminal_only_method",
    "not_reached_final_ansatz",
}
SNAKE_PROVISIONAL_COST_STATUSES = {
    "running_current_best_reached",
    "snake_current_status_first_crossing_reached",
}
SNAKE_AUDITED_COST_SOURCE = "snake_audited_first_crossing_compiled_cost"
FORBIDDEN_RESOURCE_SOURCE_TOKENS = (
    "proxy",
    "deterministic",
    "terminal",
    "final",
    "tie",
    "objective_score",
    "live_overlay",
    "live_snake_overlay",
    "supplemental",
    "synthetic",
    "fabricated",
    "current_best",
)

RESOURCE_MAPPING_KEYS = (
    "paper_i_first_crossing",
    "qiskit_compiled_first_hit_cost",
    "paper_i_first_crossing_compiled_cost",
    "snake_first_crossing_compiled_cost",
    "resource_proxies_at_crossing",
    "objective_score_components",
    "measurement_proxy_validation",
    "adapt_vqe",
    "continuation",
    "controller_measurement_work_summary",
    "selected_backend",
    "row_updates",
)

COUNT_2Q_KEYS = (
    "compiled_count_2q_total",
    "compiled_two_qubit_count",
    "compiled_count_2q",
    "count_2q",
    "N_2q",
    "n_2q",
)

DEPTH_2Q_KEYS = (
    "compiled_depth_2q_total",
    "compiled_two_qubit_depth",
    "compiled_depth_2q",
    "depth_2q",
    "D_2q",
    "d_2q",
)

CIRCUIT_DEPTH_KEYS = (
    "compiled_depth_total",
    "compiled_depth",
    "circuit_depth",
    "D_circ",
    "d_circ",
)

S_ALG_KEYS = (
    "S_alg",
    "s_alg",
)
S_ALG_OK_STATUSES = {"", "ok", "valid", "validated"}

CURRENT_BEST_ENRICHMENT_ROOT_CANDIDATES = (
    REPO_ROOT / "raw_outputs/chtc_paper_i_clean_current_20260520/local_metric_enrichment_current_best_20260523",
    REPO_ROOT / "raw_outputs/chtc_h2_rerun1_fetch/local_metric_enrichment_current_best_20260523",
)

PRIOR_CURRENT_BEST_SUMMARY_CANDIDATES = (
    REPO_ROOT / "raw_outputs/chtc_paper_i_current_20260519/fixed_accuracy_summary/table_i_fixed_accuracy_calibrated_summary.json",
)

RELATED_ARTIFACT_ROOT_CANDIDATES = (
    REPO_ROOT / "raw_outputs/chtc_snake_v6_fetch/unpacked/raw_outputs",
    REPO_ROOT / "raw_outputs/chtc_h2_rerun1_fetch/remote_raw_outputs",
    REPO_ROOT / "raw_outputs/chtc_h2_rerun1_fetch/unpacked/raw_outputs",
    REPO_ROOT / "raw_outputs/chtc_paper_i_clean_current_20260520/remote_raw_outputs",
    REPO_ROOT / "raw_outputs/chtc_paper_i_current_20260519",
)


def _row_is_snake_for_cost(row: Mapping[str, Any]) -> bool:
    method = normalize_method(str(row.get("method") or ""))
    algorithm_id = str(row.get("algorithm_id") or "")
    return method == "SNAKE" or algorithm_id == ALGORITHM_ID_BY_METHOD["SNAKE"]


def _row_is_historical_pareto_display(row: Mapping[str, Any]) -> bool:
    return (
        _row_is_snake_for_cost(row)
        and str(row.get("historical_pareto_role") or "") == "display_representative"
    )


def _row_is_display_overlay(row: Mapping[str, Any]) -> bool:
    return str(row.get("display_overlay_role") or "") == "approved_table_cell"


def _resource_source_text(row: Mapping[str, Any]) -> str:
    fields = []
    allowed_source_kinds = {
        "qiskit_compiled_first_hit_ansatz_circuit",
        "qiskit_compiled_final_ansatz_circuit",
        "qiskit_compiled_terminal_only_fixed_ansatz",
        "qiskit_compiled_current_best_ansatz_circuit_diagnostic",
        "snake_qiskit_compiled_first_hit_ansatz_circuit",
    }
    reportable_kind = str(
        row.get("first_hit_cost_source_kind")
        or row.get("compiled_resource_source_kind")
        or row.get("sidecar_source_kind")
        or ""
    ) in allowed_source_kinds
    for key in (
        "first_hit_cost_source_kind",
        "compiled_resource_source_kind",
        "sidecar_source_kind",
        "cost_source",
        "source",
        "first_hit_semantics",
        "compiled_resource_validation_reason",
        "compiled_circuit_stats_status",
        "compiled_depth_2q_semantics",
        "depth_2q_semantics",
    ):
        value = row.get(key)
        if value is not None:
            if key in {"first_hit_cost_source_kind", "compiled_resource_source_kind", "sidecar_source_kind"} and value in allowed_source_kinds:
                continue
            if reportable_kind and key in {"source", "first_hit_semantics"}:
                fields.append(str(value).lower().replace("terminal", "").replace("final", ""))
                continue
            fields.append(str(value))
    return " ".join(fields).lower()


def _iter_resource_maps(mapping: Mapping[str, Any] | None) -> Sequence[Mapping[str, Any]]:
    """Return nested mappings that may carry compiled circuit or work fields."""
    if not isinstance(mapping, Mapping):
        return []
    out: list[Mapping[str, Any]] = []
    stack: list[Mapping[str, Any]] = [mapping]
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        ident = id(current)
        if ident in seen:
            continue
        seen.add(ident)
        out.append(current)
        for key in RESOURCE_MAPPING_KEYS:
            value = current.get(key)
            if isinstance(value, Mapping):
                stack.append(value)
            elif isinstance(value, list):
                stack.extend(item for item in value[:4] if isinstance(item, Mapping))
        rows = current.get("rows")
        if isinstance(rows, list):
            stack.extend(item for item in rows[:4] if isinstance(item, Mapping))
    return out


def _compiled_triplet_from_mapping(mapping: Mapping[str, Any]) -> tuple[float, float, float] | None:
    count_2q = first_finite_number(*(mapping.get(key) for key in COUNT_2Q_KEYS))
    depth_2q = first_finite_number(*(mapping.get(key) for key in DEPTH_2Q_KEYS))
    circuit_depth = first_finite_number(*(mapping.get(key) for key in CIRCUIT_DEPTH_KEYS))
    if count_2q is None or depth_2q is None or circuit_depth is None:
        return None
    if count_2q < 0.0 or depth_2q < 0.0 or circuit_depth < 0.0:
        return None
    if circuit_depth < depth_2q:
        return None
    return (count_2q, depth_2q, circuit_depth)


def _valid_s_alg_value(row: Mapping[str, Any]) -> float | None:
    for mapping in (row, *_iter_resource_maps(row)):
        value = first_finite_number(*(mapping.get(key) for key in S_ALG_KEYS))
        if value is not None:
            status = str(mapping.get("S_alg_status") or mapping.get("s_alg_status") or "").strip().lower()
            if status in S_ALG_OK_STATUSES:
                return value
            continue
        proxy = first_finite_number(mapping.get("measurement_work_proxy"))
        proxy_source = str(mapping.get("measurement_work_proxy_source") or "").strip()
        proxy_status = str(mapping.get("measurement_work_proxy_status") or "").strip().lower()
        if proxy is not None and proxy_source == "S_alg" and proxy_status in S_ALG_OK_STATUSES:
            return proxy
        algorithmic = mapping.get("algorithmic_measurement_work")
        if isinstance(algorithmic, Mapping):
            value = first_finite_number(algorithmic.get("S_alg"), algorithmic.get("s_alg"))
            status = str(
                algorithmic.get("S_alg_status")
                or algorithmic.get("s_alg_status")
                or algorithmic.get("status")
                or ""
            ).strip().lower()
            if value is not None and status in S_ALG_OK_STATUSES:
                return value
    return None


def _work_value_from_resource_maps(row: Mapping[str, Any]) -> float | None:
    return _valid_s_alg_value(row)


def _current_best_resource_values(row: Mapping[str, Any]) -> tuple[float, float, float, float | None] | None:
    for mapping in _iter_resource_maps(row):
        triplet = _compiled_triplet_from_mapping(mapping)
        if triplet is None:
            continue
        return (*triplet, _valid_s_alg_value(row))
    return None


def _has_full_current_best_resource_values(row: Mapping[str, Any]) -> bool:
    return _current_best_resource_values(row) is not None


def _resource_display_from_current_best(row: Mapping[str, Any], status: str) -> tuple[bool, str]:
    if "running_no_completed" in status:
        return False, "running_no_completed_trial_no_compiled_resource_display"
    if _current_best_resource_values(row) is None:
        return False, "required_qiskit_resource_missing:count_2q,depth_2q,circuit_depth"
    return True, "qiskit_compiled_current_best_or_first_hit_cost"


def _forbidden_resource_source_reason(row: Mapping[str, Any]) -> str | None:
    text = _resource_source_text(row)
    for token in FORBIDDEN_RESOURCE_SOURCE_TOKENS:
        if token in text:
            return f"forbidden_resource_source_{token}"
    return None


def _depth_ordering_display_reason(row: Mapping[str, Any]) -> str | None:
    depth_2q = first_finite_number(row.get("depth_2q"), row.get("D_2q"), row.get("d_2q"))
    circuit_depth = first_finite_number(row.get("circuit_depth"), row.get("D_circ"), row.get("d_circ"))
    if depth_2q is None or circuit_depth is None:
        return None
    source_text = _resource_source_text(row)
    if "count" in source_text and "depth" not in source_text:
        return "compiled_depth_2q_semantic_mismatch_count_not_depth"
    if float(circuit_depth) < float(depth_2q):
        return "compiled_depth_total_less_than_two_qubit_depth"
    return None


def _resource_display_policy(row: Mapping[str, Any], status: str | None = None) -> tuple[bool, str]:
    status = str(status if status is not None else row.get("threshold_status") or "").lower()
    if not status:
        return False, "missing_threshold_status"
    if _row_is_display_overlay(row):
        if row.get("resource_display_allowed") is True:
            return True, str(row.get("resource_display_reason") or "approved_pdf_recovered_qiskit_cost")
        return False, str(row.get("resource_display_reason") or "approved_pdf_recovered_status_only")
    current_best_allowed, current_best_reason = _resource_display_from_current_best(row, status)
    if current_best_allowed:
        return True, current_best_reason
    if "upper_bound" in status:
        return False, "terminal_upper_bound_not_promoted"
    if row.get("cost_included") is not True:
        return False, "cost_included_false"
    if status not in ELIGIBLE_COST_STATUSES:
        return False, f"threshold_status_not_resource_eligible:{status}"
    required_resources = {
        "count_2q": first_finite_number(row.get("count_2q"), row.get("N_2q"), row.get("n_2q")),
        "depth_2q": first_finite_number(row.get("depth_2q"), row.get("D_2q"), row.get("d_2q")),
        "circuit_depth": first_finite_number(row.get("circuit_depth"), row.get("D_circ"), row.get("d_circ")),
    }
    missing_resources = [name for name, value in required_resources.items() if value is None]
    if missing_resources:
        return False, "required_qiskit_resource_missing:" + ",".join(missing_resources)
    depth_reason = _depth_ordering_display_reason(row)
    if depth_reason is not None:
        return False, depth_reason
    for name, value in (
        ("count_2q", first_finite_number(row.get("count_2q"), row.get("N_2q"), row.get("n_2q"))),
        ("depth_2q", first_finite_number(row.get("depth_2q"), row.get("D_2q"), row.get("d_2q"))),
        ("circuit_depth", first_finite_number(row.get("circuit_depth"), row.get("D_circ"), row.get("d_circ"))),
        ("S_alg", _valid_s_alg_value(row)),
    ):
        if value is not None and float(value) < 0.0:
            return False, f"invalid_negative_resource_value:{name}"
    forbidden = _forbidden_resource_source_reason(row)
    if forbidden is not None:
        return False, forbidden
    if _row_is_snake_for_cost(row):
        if status != "ok_native_first_hit":
            return False, "snake_status_not_validated_first_hit"
        if str(row.get("cost_source") or "") != SNAKE_AUDITED_COST_SOURCE:
            return False, "snake_cost_source_not_audited_first_crossing"
        if str(row.get("sidecar_validation_status") or "") != "ok":
            return False, "snake_sidecar_validation_not_ok"
        if row.get("sidecar_hash_verified") is not True:
            return False, "snake_sidecar_hash_not_verified"
        return True, "validated_snake_qiskit_compiled_first_hit_sidecar"
    if row.get("resource_display_allowed") is not True:
        return False, "resource_display_allowed_not_true"
    if str(row.get("compiled_resource_validation_status") or "") != "ok":
        return False, "compiled_resource_validation_not_ok"
    return True, "validated_qiskit_compiled_first_hit_or_terminal_only_cost"


def _cost_status_allows_display(row: Mapping[str, Any], status: str) -> bool:
    allowed, _reason = _resource_display_policy(row, status)
    return allowed


DISPLAY_OVERLAY_CELL_KEYS = {
    "delta_e": ("display_delta_e", "delta_e_display", "delta_e"),
    "infidelity": ("display_infidelity", "infidelity_display", "one_minus_fidelity", "1-F"),
    "n_2q": ("display_n_2q", "n_2q_display", "n_2q", "N_2q", "count_2q"),
    "d_2q": ("display_d_2q", "d_2q_display", "d_2q", "D_2q", "depth_2q"),
    "d_circ": ("display_d_circ", "d_circ_display", "d_circ", "D_circ", "circuit_depth"),
    "s_alg": ("display_s_alg", "s_alg_display", "s_alg", "S_alg"),
}


def _display_overlay_value(row: Mapping[str, Any], logical_key: str) -> str:
    if logical_key == "s_alg":
        return format_number(_valid_s_alg_value(row))
    for key in DISPLAY_OVERLAY_CELL_KEYS[logical_key]:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return "--"


def _cell_from_display_overlay_row(row: Mapping[str, Any]) -> TableCell:
    return TableCell(
        delta_e=_display_overlay_value(row, "delta_e"),
        infidelity=_display_overlay_value(row, "infidelity"),
        n_2q=_display_overlay_value(row, "n_2q"),
        d_2q=_display_overlay_value(row, "d_2q"),
        d_circ=_display_overlay_value(row, "d_circ"),
        s_alg=_display_overlay_value(row, "s_alg"),
    )


def cost_cells_from_row(row: dict[str, Any]) -> tuple[str, str, str, str]:
    if _row_is_display_overlay(row):
        cell = _cell_from_display_overlay_row(row)
        return cell.n_2q, cell.d_2q, cell.d_circ, cell.s_alg
    status = str(row.get("threshold_status") or "").lower()
    current_best = _current_best_resource_values(row)
    if current_best is not None and "running_no_completed" not in status:
        count_2q, depth_2q, circuit_depth, work = current_best
        return (
            format_number(count_2q),
            format_number(depth_2q),
            format_number(circuit_depth),
            format_number(work),
        )
    if "upper_bound" in status:
        return ("UB", "UB", "UB", "UB")
    if row.get("cost_included") is not True:
        return ("--", "--", "--", "--")
    if not _cost_status_allows_display(row, status):
        return ("--", "--", "--", "--")
    return (
        format_number(first_finite_number(row.get("count_2q"), row.get("N_2q"), row.get("n_2q"))),
        format_number(first_finite_number(row.get("depth_2q"), row.get("D_2q"), row.get("d_2q"))),
        format_number(first_finite_number(row.get("circuit_depth"), row.get("D_circ"), row.get("d_circ"))),
        format_number(_valid_s_alg_value(row)),
    )


def cell_from_row(row: dict[str, Any], threshold: float) -> TableCell:
    if _row_is_display_overlay(row):
        return _cell_from_display_overlay_row(row)
    n_2q, d_2q, d_circ, s_alg = cost_cells_from_row(row)
    return TableCell(
        delta_e=format_delta(row, threshold),
        infidelity=format_infidelity(row),
        n_2q=n_2q,
        d_2q=d_2q,
        d_circ=d_circ,
        s_alg=s_alg,
    )


def _sha256_for_path(path_text: Any) -> str | None:
    if not path_text:
        return None
    try:
        path = Path(str(path_text))
    except TypeError:
        return None
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary_metadata(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    keys = (
        "schema",
        "records_path",
        "output_root",
        "enrichment_root",
        "thresholds",
        "target_profile",
        "threshold_policy",
        "expected_count",
        "payload_count",
        "missing_count",
        "live_snake_overlay",
        "local_overrides",
    )
    return {key: payload.get(key) for key in keys if key in payload}


def _expected_case_id(family: str | None, regime: str | None) -> str | None:
    base = EXPECTED_CASE_ID_BASE.get(family)
    if base is None:
        return None
    if family == "molecular_vibronic_h2":
        return f"{base}_L2_nph1_clean_{regime}"
    if family in PHONON_FAMILIES:
        return f"{base}_L2_nph2_clean_{regime}"
    return f"{base}_L2_clean_{regime}"


def _supplemental_remote_roots(summary_json: Path, metadata: Mapping[str, Any] | None = None) -> list[Path]:
    roots: list[Path] = []
    for ancestor in [summary_json.parent, *summary_json.parents]:
        candidate = ancestor / "remote_raw_outputs"
        if candidate.exists() and candidate.is_dir() and candidate not in roots:
            roots.append(candidate)
    if metadata is not None and metadata.get("output_root"):
        output_root = _resolve_artifact_path(metadata.get("output_root"), summary_json=summary_json)
        if output_root is not None:
            for ancestor in [output_root, *output_root.parents]:
                candidate = ancestor / "remote_raw_outputs"
                if candidate.exists() and candidate.is_dir() and candidate not in roots:
                    roots.append(candidate)
    return roots


def _result_payload_row(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    result = payload.get("result")
    if isinstance(result, Mapping):
        return result
    rows = payload.get("rows")
    if isinstance(rows, list) and rows and isinstance(rows[0], Mapping):
        return rows[0]
    return payload


def _threshold_lookup_keys(threshold: float) -> tuple[str, ...]:
    sci = f"{float(threshold):.0e}"
    compact = sci.replace("-0", "-").replace("-", "_")
    legacy = compact.replace("e_", "e")
    keys = (
        f"first_hit_{compact}",
        f"first_hit_{legacy}",
        f"first_hit_abs_delta_e_le_{compact}",
        f"first_hit_abs_delta_e_le_{legacy}",
        sci,
        sci.replace("-0", "-"),
        str(float(threshold)),
        str(threshold),
    )
    return tuple(dict.fromkeys(keys))


def _supplemental_threshold_row(
    *,
    algorithm_id: str,
    result_row: Mapping[str, Any],
    threshold: float,
) -> tuple[Mapping[str, Any], str]:
    if algorithm_id in {ALGORITHM_ID_BY_METHOD["HEA VQE"], ALGORITHM_ID_BY_METHOD["family-informed VQE"]}:
        return result_row, "terminal_row_fixed_method"
    for key in _threshold_lookup_keys(float(threshold)):
        direct = result_row.get(key)
        if isinstance(direct, Mapping):
            return direct, str(key)
    hits = result_row.get("benchmark_first_hits")
    if isinstance(hits, Mapping):
        for key in _threshold_lookup_keys(float(threshold)):
            candidate = hits.get(key)
            if isinstance(candidate, Mapping):
                return candidate, f"benchmark_first_hits[{key}]"
    return result_row, "terminal_row"


def _table_i_threshold_cost_from_row(
    *,
    algorithm_id: str,
    row: Mapping[str, Any],
    threshold: float,
    threshold_source: str,
) -> dict[str, Any]:
    """Local lightweight threshold classifier for report-only supplemental rows.

    The full enrichment module imports the static runner stack, which is too
    heavy for a PDF rebuild.  This helper intentionally extracts only the
    fields needed by this table renderer; canonical sidecar generation and
    validation remain in the exact-bench pipeline.
    """

    status = str(row.get("threshold_status") or "")
    abs_delta_e = first_finite_number(
        row.get("abs_delta_e"),
        row.get("primary_error"),
        row.get("delta_e_mean_included"),
        row.get("error"),
    )
    if not status:
        if abs_delta_e is None:
            status = "unknown"
        elif algorithm_id in {ALGORITHM_ID_BY_METHOD["HEA VQE"], ALGORITHM_ID_BY_METHOD["family-informed VQE"]}:
            status = "ok_terminal_only_method" if abs_delta_e <= float(threshold) else "not_reached"
        elif "first_hit" in threshold_source or "benchmark_first_hits" in threshold_source:
            status = "ok_native_first_hit" if abs_delta_e <= float(threshold) else "not_reached"
        elif abs_delta_e <= float(threshold):
            status = "terminal_upper_bound_missing_native_first_hit"
        else:
            status = "not_reached"

    triplet = _current_best_resource_values(row)
    has_triplet = triplet is not None
    source_kind = str(
        row.get("first_hit_cost_source_kind")
        or row.get("compiled_resource_source_kind")
        or row.get("source_kind")
        or ""
    )
    if "not_reached" in status and source_kind in {
        "qiskit_compiled_final_ansatz_circuit",
        "qiskit_compiled_terminal_only_fixed_ansatz",
    }:
        status = "not_reached_final_ansatz"
    promoted_resource = status in ELIGIBLE_COST_STATUSES
    diagnostic_resource = bool(
        has_triplet
        and not promoted_resource
        and "running_no_completed" not in status
        and "failed" not in status
        and "invalid" not in status
        and status != "unknown"
    )
    resource_display_allowed = bool(has_triplet and (promoted_resource or diagnostic_resource))
    if promoted_resource:
        compiled_status = "ok" if has_triplet else "missing"
        compiled_reason = None if has_triplet else "compiled_count_2q_or_depth_2q_or_depth_total_missing"
        cost_source = row.get("cost_source") or "qiskit_compiled_supplemental_row"
        method_cost_semantics = row.get("method_cost_semantics")
    elif diagnostic_resource:
        compiled_status = "ok_current_best_display"
        compiled_reason = "full_compiled_triplet_for_displayed_diagnostic_row"
        cost_source = row.get("cost_source") or "qiskit_compiled_diagnostic_row_cost"
        method_cost_semantics = row.get("method_cost_semantics") or "current_best_diagnostic_not_first_hit"
    else:
        compiled_status = "missing" if not has_triplet else "not_promoted"
        compiled_reason = (
            "compiled_count_2q_or_depth_2q_or_depth_total_missing"
            if not has_triplet
            else "supplemental_row_not_valid_first_hit_cost"
        )
        cost_source = row.get("cost_source") or "qiskit_compiled_supplemental_row"
        method_cost_semantics = row.get("method_cost_semantics")
    out = {
        "threshold_status": status,
        "abs_delta_e": abs_delta_e,
        "S_alg": _work_value_from_resource_maps(row),
        "cost_source": cost_source,
        "source": row.get("source") or "supplemental_remote_raw_output",
        "first_hit_semantics": row.get("first_hit_semantics"),
        "method_cost_semantics": method_cost_semantics,
        "resource_display_allowed": resource_display_allowed,
        "compiled_resource_validation_status": compiled_status,
        "compiled_resource_validation_reason": compiled_reason,
        "first_hit_cost_source_kind": (
            "qiskit_compiled_terminal_only_fixed_ansatz"
            if status == "ok_terminal_only_method"
            else (
                "qiskit_compiled_first_hit_ansatz_circuit"
                if status == "ok_native_first_hit"
                else (
                    source_kind
                    if status == "not_reached_final_ansatz" and source_kind
                    else "qiskit_compiled_current_best_ansatz_circuit_diagnostic"
                )
            )
        ),
        "source_resource_fields_present": has_triplet,
    }
    if triplet is not None:
        count_2q, depth_2q, circuit_depth, _work = triplet
        out.update(
            {
                "count_2q": count_2q,
                "depth_2q": depth_2q,
                "circuit_depth": circuit_depth,
                "compiled_count_2q_total": count_2q,
                "compiled_depth_2q_total": depth_2q,
                "compiled_depth_total": circuit_depth,
            }
        )
    for key in (
        "components",
        "paper_i_first_crossing",
        "sidecar_validation_status",
        "sidecar_validation_reason",
        "sidecar_hash_verified",
        "sidecar_source_kind",
        "snake_first_crossing_cost_sidecar_key",
        "snake_first_crossing_history_position_tau",
        "source_result_sha256",
    ):
        if key in row:
            out[key] = row.get(key)
    return out


def _snake_supplemental_cost(
    *,
    cost: Mapping[str, Any],
    payload: Mapping[str, Any],
    result_row: Mapping[str, Any],
    path: Path,
    case_id: str,
    threshold: float,
) -> dict[str, Any]:
    """Normalize fetched SNAKE artifacts for current-status reporting.

    This is deliberately a consumer, not a sidecar producer.  Supplemental
    artifacts may fill missing SNAKE status/error cells, but terminal compiled
    totals are never transformed into ``snake_first_crossing_compiled_cost_v1``.
    Numeric resources are preserved only when the artifact already contains a
    sidecar that the enrichment classifier validated.
    """

    out = dict(cost)
    if (
        str(out.get("threshold_status") or "") == "ok_native_first_hit"
        and out.get("resource_display_allowed") is True
        and str(out.get("sidecar_validation_status") or "") == "ok"
        and out.get("sidecar_hash_verified") is True
    ):
        return out
    payload_status = str(payload.get("status") or result_row.get("status") or "").lower()
    failure_reason = result_row.get("failure_reason") or payload.get("failure_reason")
    if payload_status == "failed" or failure_reason:
        return {
            **out,
            "threshold_status": "failed",
            "reason": str(failure_reason or payload_status or "failed"),
            "abs_delta_e": first_finite_number(result_row.get("abs_delta_e"), out.get("abs_delta_e")),
        }

    crossing = result_row.get("paper_i_first_crossing")
    if not isinstance(crossing, Mapping):
        crossing = None
    terminal_error = first_finite_number(result_row.get("abs_delta_e"), out.get("abs_delta_e"))
    crossing_error = first_finite_number(
        None if crossing is None else crossing.get("primary_error_at_crossing"),
    )
    crossed = bool(
        crossing is not None
        and (crossing.get("reached") is True or str(crossing.get("status") or "").lower() == "reached")
    )

    if crossed and crossing_error is not None and crossing_error <= float(threshold):
        return {
            **out,
            "threshold_status": "terminal_upper_bound_missing_native_first_hit",
            "abs_delta_e": float(crossing_error),
            "reason": "supplemental_snake_crossing_reached_without_valid_first_hit_compiled_sidecar",
            "source": "supplemental_remote_raw_output",
            "first_hit_semantics": "supplemental_snake_status_only_cost_sidecar_missing",
            "cost_source": None,
            "resource_display_allowed": False,
            "compiled_resource_validation_status": "missing",
            "compiled_resource_validation_reason": "supplemental_snake_valid_sidecar_missing",
            "sidecar_validation_status": out.get("sidecar_validation_status") or "missing",
            "sidecar_validation_reason": out.get("sidecar_validation_reason") or "supplemental_snake_valid_sidecar_missing",
            "sidecar_hash_verified": False,
            "paper_i_first_crossing": dict(crossing),
        }

    if terminal_error is not None:
        if terminal_error > float(threshold):
            return {
                **out,
                "threshold_status": "not_reached",
                "reason": "supplemental_snake_terminal_error_above_current_target",
                "abs_delta_e": float(terminal_error),
            }
        return {
            **out,
            "threshold_status": "terminal_upper_bound_missing_native_first_hit",
            "reason": "supplemental_snake_terminal_error_reached_without_current_target_first_crossing",
            "abs_delta_e": float(terminal_error),
            "resource_display_allowed": False,
            "compiled_resource_validation_status": "missing",
            "compiled_resource_validation_reason": "supplemental_snake_valid_sidecar_missing",
            "sidecar_validation_status": out.get("sidecar_validation_status") or "missing",
            "sidecar_validation_reason": out.get("sidecar_validation_reason") or "supplemental_snake_valid_sidecar_missing",
            "sidecar_hash_verified": False,
        }
    return out


def _supplemental_row_from_result_path(
    *,
    path: Path,
    spec: Mapping[str, str],
    case_id: str,
    algorithm_id: str,
    threshold: float,
) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, Mapping):
        return None
    result_row = _result_payload_row(payload)
    if not isinstance(result_row, Mapping):
        return None
    threshold_row, threshold_source = _supplemental_threshold_row(
        algorithm_id=algorithm_id,
        result_row=result_row,
        threshold=float(threshold),
    )
    record = {
        "record_id": f"static_table__{spec['canonical_family']}__{case_id}__{algorithm_id}",
        "family": spec["canonical_family"],
        "case_id": case_id,
        "algorithm_id": algorithm_id,
    }
    cost = _table_i_threshold_cost_from_row(
        algorithm_id=algorithm_id,
        row=threshold_row,
        threshold=float(threshold),
        threshold_source=str(threshold_source),
    )
    if algorithm_id == ALGORITHM_ID_BY_METHOD["SNAKE"]:
        cost = _snake_supplemental_cost(
            cost=cost,
            payload=payload,
            result_row=threshold_row,
            path=path,
            case_id=case_id,
            threshold=threshold,
        )
    threshold_status = str(cost.get("threshold_status") or "unknown")
    display_resources = bool(
        cost.get("resource_display_allowed") is True
        and _current_best_resource_values(cost) is not None
        and str(cost.get("compiled_resource_validation_status") or "")
        in {"ok", "ok_current_best_display", "approved_pdf_recovered"}
    )
    cost_included = (
        threshold_status in ELIGIBLE_COST_STATUSES
        and display_resources
        and str(cost.get("compiled_resource_validation_status") or "") == "ok"
    )
    return {
        "record_id": record["record_id"],
        "family": spec["canonical_family"],
        "case_id": case_id,
        "benchmark_id": case_id,
        "algorithm_id": algorithm_id,
        "method": spec["method"],
        "threshold": float(threshold),
        "threshold_source": f"supplemental_remote_raw_output:{path}:{threshold_source}",
        "threshold_status": threshold_status,
        "cost_included": bool(cost_included),
        "abs_delta_e": cost.get("abs_delta_e"),
        "S_alg": cost.get("S_alg") if display_resources else None,
        "N_metric": cost.get("N_metric") if display_resources else None,
        "metric_fraction": cost.get("metric_fraction") if display_resources else None,
        "count_2q": _first_number_or_none(cost.get("count_2q"), cost.get("compiled_count_2q_total")) if display_resources else None,
        "depth_2q": _first_number_or_none(cost.get("depth_2q"), cost.get("compiled_depth_2q_total")) if display_resources else None,
        "circuit_depth": _first_number_or_none(cost.get("circuit_depth"), cost.get("compiled_depth_total")) if display_resources else None,
        "payload_path": str(path),
        "source_payload_path": str(path),
        "cost_source": cost.get("cost_source"),
        "source": cost.get("source") or "supplemental_remote_raw_output",
        "first_hit_semantics": cost.get("first_hit_semantics"),
        "method_cost_semantics": cost.get("method_cost_semantics"),
        "resource_display_allowed": bool(display_resources),
        "compiled_resource_validation_status": cost.get("compiled_resource_validation_status"),
        "compiled_resource_validation_reason": cost.get("compiled_resource_validation_reason"),
        "first_hit_cost_source_kind": cost.get("first_hit_cost_source_kind"),
        "source_resource_fields_present": cost.get("source_resource_fields_present"),
        "sidecar_validation_status": cost.get("sidecar_validation_status"),
        "sidecar_validation_reason": cost.get("sidecar_validation_reason"),
        "sidecar_hash_verified": cost.get("sidecar_hash_verified"),
        "sidecar_source_kind": cost.get("sidecar_source_kind"),
        "snake_first_crossing_cost_sidecar_key": cost.get("snake_first_crossing_cost_sidecar_key"),
        "snake_first_crossing_history_position_tau": cost.get("snake_first_crossing_history_position_tau"),
        "source_result_sha256": cost.get("source_result_sha256"),
        "components": cost.get("components"),
    }


def _first_number_or_none(*values: Any) -> float | None:
    return first_finite_number(*values)


def _load_supplemental_result_rows(
    summary_json: Path,
    threshold: float,
    *,
    existing_keys: set[tuple[str, str, str]] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    roots = _supplemental_remote_roots(summary_json, metadata=metadata)
    if not roots:
        return []
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    existing_keys = existing_keys or set()
    for spec in _expected_cell_specs():
        if spec["canonical_family"] in DEFERRED_PLACEHOLDER_FAMILIES:
            continue
        algorithm_id = ALGORITHM_ID_BY_METHOD.get(spec["method"])
        case_id = _expected_case_id(family=spec["canonical_family"], regime=spec["regime"])
        if algorithm_id is None or case_id is None:
            continue
        record_id = f"static_table__{spec['canonical_family']}__{case_id}__{algorithm_id}"
        if record_id in seen:
            continue
        paths: list[Path] = []
        for root in roots:
            direct = root / record_id / "result/generic_static_single.json"
            if direct.exists():
                paths.append(direct)
            paths.extend(root.glob(f"*/{record_id}/result/generic_static_single.json"))
        if not paths:
            continue
        row = _supplemental_row_from_result_path(
            path=paths[0],
            spec=spec,
            case_id=case_id,
            algorithm_id=algorithm_id,
            threshold=threshold,
        )
        if row is not None:
            rows.append(row)
            seen.add(record_id)
    return rows


def _resolve_artifact_path(path_text: Any, *, summary_json: Path | None = None) -> Path | None:
    if not path_text:
        return None
    path = Path(str(path_text))
    if path.exists():
        return path
    candidates = [REPO_ROOT / path]
    if summary_json is not None:
        for ancestor in [summary_json.parent, *summary_json.parents]:
            candidates.append(ancestor / path)
            if ancestor.name == "raw_outputs":
                candidates.append(ancestor.parent / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _raw_outputs_tail(path: Path) -> Path | None:
    parts = path.parts
    if "raw_outputs" not in parts:
        return None
    index = parts.index("raw_outputs")
    if index + 1 >= len(parts):
        return None
    return Path(*parts[index + 1 :])


def _resolve_related_artifact_path(
    path_text: Any,
    *,
    base_path: Path | None = None,
    summary_json: Path | None = None,
) -> Path | None:
    if not path_text:
        return None
    path = Path(str(path_text))
    if path.exists():
        return path
    resolved = _resolve_artifact_path(path_text, summary_json=summary_json)
    if resolved is not None:
        return resolved
    candidates: list[Path] = []
    if base_path is not None:
        candidates.append(base_path.parent / path)
        candidates.extend(ancestor / path for ancestor in [base_path.parent, *base_path.parents])
    if path.is_absolute():
        text = str(path)
        for marker in ("/work/raw_outputs/", "/work/"):
            if marker in text:
                tail_text = text.split(marker, 1)[1]
                if marker == "/work/raw_outputs/":
                    tail = Path(tail_text)
                    candidates.extend(root / tail for root in RELATED_ARTIFACT_ROOT_CANDIDATES)
                else:
                    candidates.append(REPO_ROOT / tail_text)
    tail = _raw_outputs_tail(path)
    if tail is not None:
        if base_path is not None:
            for ancestor in [base_path.parent, *base_path.parents]:
                if ancestor.name == "remote_raw_outputs":
                    candidates.append(ancestor / tail)
                if ancestor.name == "unpacked":
                    candidates.append(ancestor / "raw_outputs" / tail)
                if ancestor.name.startswith("chtc_"):
                    candidates.append(ancestor / tail)
        candidates.extend(root / tail for root in RELATED_ARTIFACT_ROOT_CANDIDATES)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_json_mapping(path: Path | None) -> Mapping[str, Any] | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, Mapping) else None


def _apply_current_best_resources(
    row: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    source_label: str,
    work_fallback: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    out = dict(row)
    values = _current_best_resource_values(source)
    if values is None:
        return out
    count_2q, depth_2q, circuit_depth, work = values
    if work is None and isinstance(work_fallback, Mapping):
        work = _work_value_from_resource_maps(work_fallback)
    out.update(
        {
            "count_2q": count_2q,
            "depth_2q": depth_2q,
            "circuit_depth": circuit_depth,
            "compiled_count_2q_total": count_2q,
            "compiled_depth_2q_total": depth_2q,
            "compiled_depth_total": circuit_depth,
            "resource_display_allowed": True,
            "resource_display_reason": f"qiskit_compiled_current_best:{source_label}",
            "compiled_resource_validation_status": "ok_current_best_display",
            "compiled_resource_validation_reason": f"full_compiled_triplet_from_{source_label}",
            "current_best_resource_source": source_label,
        }
    )
    if work is not None:
        if first_finite_number(out.get("S_alg")) is None:
            out["S_alg"] = work
    return out


def _candidate_enrichment_roots(summary_json: Path | None, metadata: Mapping[str, Any]) -> list[Path]:
    roots: list[Path] = []
    raw_values = [metadata.get("enrichment_root")]
    output_root = metadata.get("output_root")
    if output_root:
        output_path = Path(str(output_root))
        raw_values.append(output_path.parent / "local_metric_enrichment_current_best_20260523")
    for value in [*raw_values, *CURRENT_BEST_ENRICHMENT_ROOT_CANDIDATES]:
        if not value:
            continue
        resolved = _resolve_artifact_path(value, summary_json=summary_json)
        if resolved is None:
            candidate = Path(str(value))
            resolved = candidate if candidate.exists() else REPO_ROOT / candidate
        if resolved.exists() and resolved.is_dir() and resolved not in roots:
            roots.append(resolved)
    return roots


def _augment_row_from_enrichment(row: Mapping[str, Any], enrichment_roots: Sequence[Path]) -> dict[str, Any]:
    out = dict(row)
    record_id = str(out.get("record_id") or "").strip()
    if not record_id:
        return out
    for root in enrichment_roots:
        path = root / record_id / "result/generic_static_metric_enrichment.json"
        payload = _load_json_mapping(path)
        if not isinstance(payload, Mapping):
            continue
        updates = payload.get("row_updates")
        if not isinstance(updates, Mapping):
            continue
        candidate = out
        for update_key, row_keys in (
            ("compiled_count_2q_total", ("compiled_count_2q_total", "count_2q")),
            ("count_2q", ("compiled_count_2q_total", "count_2q")),
            ("compiled_depth_2q_total", ("compiled_depth_2q_total", "depth_2q")),
            ("depth_2q", ("compiled_depth_2q_total", "depth_2q")),
            ("compiled_depth_total", ("compiled_depth_total", "circuit_depth")),
            ("circuit_depth", ("compiled_depth_total", "circuit_depth")),
        ):
            value = first_finite_number(updates.get(update_key))
            if value is None:
                continue
            for row_key in row_keys:
                if first_finite_number(candidate.get(row_key)) is None:
                    candidate[row_key] = value
        if not _has_full_current_best_resource_values(candidate):
            candidate = _apply_current_best_resources(
                candidate,
                updates,
                source_label="posthoc_qiskit_metric_enrichment",
                work_fallback=updates,
        )
        work = _work_value_from_resource_maps(updates)
        if work is not None:
            if first_finite_number(candidate.get("S_alg")) is None:
                candidate["S_alg"] = work
        if first_finite_number(
            candidate.get("infidelity"),
            candidate.get("infidelity_reference"),
            candidate.get("infidelity_4"),
            candidate.get("infidelity_same"),
        ) is None:
            infidelity = first_finite_number(
                updates.get("infidelity_reference"),
                updates.get("infidelity_4"),
                updates.get("infidelity_same"),
                updates.get("infidelity"),
            )
            if infidelity is not None:
                candidate["infidelity"] = infidelity
                if updates.get("infidelity_reference") is not None or updates.get("infidelity_4") is not None:
                    candidate["infidelity_reference"] = infidelity
                else:
                    candidate["infidelity_same"] = infidelity
        if _has_full_current_best_resource_values(candidate) or first_finite_number(candidate.get("infidelity")) is not None:
            candidate.setdefault("metric_enrichment_path", str(path))
            return candidate
    return out


def _augment_row_from_payload_resources(
    row: Mapping[str, Any],
    *,
    summary_json: Path | None,
) -> dict[str, Any]:
    out = dict(row)
    payload_path_text = out.get("payload_path") or out.get("result_path") or out.get("source_payload_path")
    payload_path = _resolve_related_artifact_path(payload_path_text, summary_json=summary_json)
    payload = _load_json_mapping(payload_path)
    if not isinstance(payload, Mapping):
        return out
    result_row = _result_payload_row(payload)
    if isinstance(result_row, Mapping):
        combined_result = dict(result_row)
        for key, value in out.items():
            if value is not None:
                combined_result[key] = value
        out = _apply_current_best_resources(
            out,
            combined_result,
            source_label="payload_result",
            work_fallback=result_row,
        )
        compile_json = result_row.get("compile_json") or payload.get("compile_json")
        compile_path = _resolve_related_artifact_path(
            compile_json,
            base_path=payload_path,
            summary_json=summary_json,
        )
        compile_payload = _load_json_mapping(compile_path)
        if isinstance(compile_payload, Mapping):
            before = out
            out = _apply_current_best_resources(
                out,
                compile_payload,
                source_label="payload_compile_scout",
                work_fallback=result_row,
            )
            if out is not before:
                out.setdefault("compile_scout_path", str(compile_path))
    if payload_path is not None:
        out.setdefault("source_payload_path", str(payload_path))
        out.setdefault("payload_path", str(payload_path))
    return out


def _should_load_prior_current_best_rows(summary_json: Path | None, metadata: Mapping[str, Any]) -> bool:
    text = " ".join(
        str(value or "")
        for value in (
            summary_json,
            metadata.get("records_path"),
            metadata.get("output_root"),
            metadata.get("live_snake_overlay"),
        )
    )
    return "paper_i_clean_current_status_tau2e4" in text or "chtc_paper_i_clean_current_20260520" in text


def _load_prior_current_best_rows(
    summary_json: Path | None,
    threshold: float,
    metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not _should_load_prior_current_best_rows(summary_json, metadata):
        return []
    rows: list[dict[str, Any]] = []
    if summary_json is not None:
        current = summary_json.resolve()
    else:
        current = None
    for path in PRIOR_CURRENT_BEST_SUMMARY_CANDIDATES:
        if not path.exists() or (current is not None and path.resolve() == current):
            continue
        payload = _load_json_mapping(path)
        raw_rows = payload.get("row_results") if isinstance(payload, Mapping) else None
        if not isinstance(raw_rows, list):
            continue
        for item in raw_rows:
            if not isinstance(item, Mapping):
                continue
            row_threshold = finite_number(item.get("threshold"))
            if row_threshold is not None and abs(row_threshold - float(threshold)) > 1e-12:
                continue
            out = dict(item)
            out.setdefault("threshold", float(threshold))
            out.setdefault("prior_current_best_source_path", str(path))
            out.setdefault("source", out.get("source") or "prior_current_best_summary_fallback")
            rows.append(out)
    return rows


def _snake_overlay_key(value: Any) -> str:
    return str(value or "").strip().lower()


def _load_live_snake_overlay_by_case(source_path: Any, *, summary_json: Path | None) -> tuple[dict[str, Mapping[str, Any]], Path | None, str | None]:
    path = _resolve_artifact_path(source_path, summary_json=summary_json)
    if path is None or not path.exists():
        return {}, path, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}, path, _sha256_for_path(path)
    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, Mapping):
        raw_entries = payload.get("rows") or payload.get("row_results") or payload.get("records") or []
        entries = raw_entries if isinstance(raw_entries, list) else []
    else:
        entries = []
    by_case: dict[str, Mapping[str, Any]] = {}
    for item in entries:
        if not isinstance(item, Mapping):
            continue
        direct_crossing = item.get("paper_i_first_crossing") if isinstance(item.get("paper_i_first_crossing"), Mapping) else {}
        objective = item.get("objective_score_components") if isinstance(item.get("objective_score_components"), Mapping) else {}
        objective_crossing = (
            objective.get("paper_i_first_crossing")
            if isinstance(objective.get("paper_i_first_crossing"), Mapping)
            else {}
        )
        for key in (
            item.get("benchmark_id"),
            item.get("case_id"),
            direct_crossing.get("benchmark_id") if isinstance(direct_crossing, Mapping) else None,
            objective_crossing.get("benchmark_id") if isinstance(objective_crossing, Mapping) else None,
            str(item.get("record_id") or "").removeprefix("live_snake_current_best__"),
            str(item.get("record_id") or "").removeprefix("live_snake_running_no_completed_trial__"),
        ):
            norm = _snake_overlay_key(key)
            if norm:
                by_case[norm] = item
    return by_case, path, _sha256_for_path(path)


def _nested_mapping(mapping: Mapping[str, Any] | None, *keys: str) -> Mapping[str, Any] | None:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current if isinstance(current, Mapping) else None


def _augment_snake_row_from_live_overlay(
    row: Mapping[str, Any],
    *,
    overlay_by_case: Mapping[str, Mapping[str, Any]],
    overlay_path: Path | None,
    overlay_sha256: str | None,
) -> dict[str, Any]:
    out = dict(row)
    if _lane_for_row(out, normalize_method(out.get("method"))) != "snake":
        return out
    if _row_is_historical_pareto_display(out) or _row_is_display_overlay(out):
        return out
    case_keys = [
        _snake_overlay_key(out.get("benchmark_id")),
        _snake_overlay_key(out.get("case_id")),
        _snake_overlay_key(str(out.get("record_id") or "").removeprefix("live_snake_current_best__")),
        _snake_overlay_key(str(out.get("record_id") or "").removeprefix("live_snake_running_no_completed_trial__")),
    ]
    overlay = next((overlay_by_case[key] for key in case_keys if key and key in overlay_by_case), None)
    if overlay is None:
        return out
    crossing = overlay.get("paper_i_first_crossing")
    if not isinstance(crossing, Mapping):
        crossing = _nested_mapping(overlay, "objective_score_components", "paper_i_first_crossing")
    if isinstance(crossing, Mapping) and not isinstance(out.get("paper_i_first_crossing"), Mapping):
        out["paper_i_first_crossing"] = dict(crossing)
        out.setdefault("history_position_tau", crossing.get("history_position_tau"))
        out.setdefault("first_crossing_reached", crossing.get("reached"))
        out.setdefault("first_crossing_status", crossing.get("status"))
        crossing_error = first_finite_number(
            crossing.get("primary_error_at_crossing"),
            crossing.get("abs_delta_e_at_crossing"),
            crossing.get("abs_error_at_crossing"),
        )
        if crossing_error is not None:
            out["abs_delta_e"] = float(crossing_error)
        if crossing.get("reached") is True and "running" in str(out.get("threshold_status") or "").lower():
            out["threshold_status"] = "running_current_best_reached"
    # Live overlays contain terminal tie-breaker resources in
    # objective_score_components; those are status diagnostics only and must
    # never be promoted into first-hit cost sidecars by the PDF builder.
    has_valid_existing_sidecar = (
        out.get("cost_included") is True
        and str(out.get("cost_source") or "") == SNAKE_AUDITED_COST_SOURCE
        and str(out.get("sidecar_validation_status") or "") == "ok"
        and out.get("sidecar_hash_verified") is True
    )
    if not has_valid_existing_sidecar:
        out["cost_included"] = False
        out["resource_display_allowed"] = False
        out.setdefault("compiled_resource_validation_status", "missing")
        out.setdefault("compiled_resource_validation_reason", "live_snake_overlay_status_only_no_valid_sidecar")
        out.setdefault("sidecar_validation_status", "missing")
        out.setdefault("sidecar_validation_reason", "live_snake_overlay_status_only_no_valid_sidecar")
        out["sidecar_hash_verified"] = False
    out.setdefault("method_cost_semantics", "snake_first_hit_sidecar_required")
    return out


def _expected_cell_specs() -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    for group_name, hamiltonians in GROUPS:
        for hamiltonian_label, aliases in hamiltonians:
            canonical = CANONICAL_FAMILY_BY_LABEL.get(hamiltonian_label, sorted(aliases)[0])
            for method in METHODS:
                for regime in ("weak", "strong"):
                    specs.append(
                        {
                            "group": group_name,
                            "hamiltonian_label": hamiltonian_label,
                            "canonical_family": canonical,
                            "method": method,
                            "regime": regime,
                            "lane": "snake" if method == "SNAKE" else "comparator",
                        }
                    )
    return specs


def _lane_for_row(row: Mapping[str, Any], method: str | None) -> str:
    algorithm_id = str(row.get("algorithm_id") or "")
    if method == "SNAKE" or algorithm_id == "static_family_native_adapt_phase3":
        return "snake"
    return "comparator"


def _row_priority(row: Mapping[str, Any], *, threshold_matches: bool) -> tuple[int, ...]:
    status = str(row.get("threshold_status") or "").lower()
    has_resources = _has_full_current_best_resource_values(row)
    has_work = _work_value_from_resource_maps(row) is not None
    if not threshold_matches:
        status_score = 0
    elif _row_is_display_overlay(row):
        if row.get("display_overlay_force") is True:
            status_score = 99
        elif _overlay_qiskit_resources_are_numeric(row):
            status_score = 100
        elif status in ELIGIBLE_COST_STATUSES:
            status_score = 58
        elif "not_reached" in status:
            status_score = 38
        elif "running" in status:
            status_score = 33
        elif "failed" in status or "invalid" in status:
            status_score = 18
        else:
            status_score = 8
    elif _row_is_historical_pareto_display(row) and has_resources:
        status_score = 90
    elif status in ELIGIBLE_COST_STATUSES and has_resources:
        status_score = 75
    elif status in ELIGIBLE_COST_STATUSES:
        status_score = 65
    elif "upper_bound" in status and has_resources:
        status_score = 68
    elif "not_reached" in status and has_resources:
        status_score = 62
    elif "running" in status and has_resources:
        status_score = 60
    elif "running" in status and first_finite_number(row.get("abs_delta_e"), row.get("primary_error"), row.get("delta_e_excess")) is not None:
        status_score = 55
    elif "not_reached" in status:
        status_score = 40
    elif "running" in status:
        # A running placeholder with no completed usable trial is less informative
        # than a fetched terminal not-reached/failed artifact for the same cell.
        status_score = 35
    elif "upper_bound" in status:
        status_score = 30
    elif "failed" in status or "invalid" in status:
        status_score = 20
    else:
        status_score = 10
    return (
        1 if threshold_matches else 0,
        status_score,
        1 if has_work else 0,
        1 if row.get("cost_included") is True else 0,
    )


def _cell_resources_displayed(cell: TableCell) -> bool:
    return any(value != "--" for value in (cell.n_2q, cell.d_2q, cell.d_circ, cell.s_alg))


def _cell_numeric_resources_displayed(cell: TableCell) -> bool:
    return any(
        finite_number(value) is not None
        for value in (cell.n_2q, cell.d_2q, cell.d_circ, cell.s_alg)
    )


def _cell_status_from_audit(audit: Mapping[str, Any] | None, cell: TableCell | None) -> str:
    if audit is None:
        return "missing"
    if not bool(audit.get("threshold_matches_requested")):
        return "invalid_target"
    status = str(audit.get("threshold_status") or "").lower()
    if "running" in status:
        return "running"
    if "not_reached" in status:
        return "not_reached"
    if "failed" in status:
        return "failed"
    if "upper_bound" in status:
        return "terminal_upper_bound"
    if "invalid" in status or "mismatch" in status:
        return "invalid_target"
    if audit.get("cost_included") is not True and not (cell and _cell_numeric_resources_displayed(cell)):
        return "cost_excluded"
    return "displayed"


def _cell_state_from_audit(audit: Mapping[str, Any] | None, cell: TableCell | None, *, method: str) -> str:
    if audit is None:
        return "missing-source"
    if not bool(audit.get("threshold_matches_requested")):
        return "invalid-target"
    threshold_status = str(audit.get("threshold_status") or "").lower()
    numeric_resources = bool(cell and _cell_numeric_resources_displayed(cell))
    resources_allowed = bool(audit.get("resource_display_allowed"))
    complete_trials = _first_int_value(audit.get("complete_trial_count"))
    if method == "SNAKE":
        if "running_no_completed" in threshold_status or (("running" in threshold_status) and complete_trials == 0):
            return "running-no-completed-trial"
        if "running" in threshold_status:
            reached = audit.get("first_crossing_reached") is True or "reached" in threshold_status
            if reached and numeric_resources and resources_allowed:
                return "running-with-current-best-hit-cost"
            if reached:
                return "running-current-best-hit-missing-first-hit-cost"
            return "running-current-best-not-reached"
    if "failed" in threshold_status:
        return "failed"
    if "not_reached" in threshold_status:
        return "not-reached-with-final-ansatz-cost" if numeric_resources and resources_allowed else "not-reached"
    if "upper_bound" in threshold_status:
        return "hit-with-terminal-upper-bound"
    if "invalid" in threshold_status or "mismatch" in threshold_status:
        return "invalid-target"
    if threshold_status == "ok_terminal_only_method":
        return "hit-with-terminal-only-method-cost" if numeric_resources and resources_allowed else "cost-excluded"
    if threshold_status == "ok_native_first_hit":
        return "hit-with-first-hit-cost" if numeric_resources and resources_allowed else "cost-excluded"
    return "cost-excluded"


def _audit_cutoff_fields(
    raw: Mapping[str, Any] | None,
    *,
    family: str | None,
    regime: str | None,
) -> dict[str, Any]:
    case_id = _case_id_from_audit(raw, family=family, regime=regime)
    n_ph_work = _infer_n_ph_work(raw, family=family, case_id=case_id)
    n_ph_ref = _infer_n_ph_ref(raw, family=family, case_id=case_id, n_ph_work=n_ph_work)
    stage = _paper_i_ladder_stage(raw, family=family, n_ph_work=n_ph_work, n_ph_ref=n_ph_ref)
    return {
        "case_id": case_id,
        "n_ph_work": n_ph_work,
        "n_ph_ref": n_ph_ref,
        "paper_i_ladder_stage": stage,
        "paper_i_cutoff_contract": "locked_known_cutoff" if family in PHONON_FAMILIES else "not_applicable_nonphonon",
        "cutoff_missing_reason": _cutoff_missing_reason(family=family, n_ph_work=n_ph_work, n_ph_ref=n_ph_ref),
    }


def _eligibility_fields(
    *,
    status: str,
    family: str | None,
    lane: str,
    stage: str | None,
    case_id: str | None,
) -> dict[str, Any]:
    eligible_for_rerun = status in {"missing", "invalid_target", "not_reached", "failed", "terminal_upper_bound", "cost_excluded"}
    if status == "deferred_placeholder":
        eligible_for_rerun = False
    eligible_for_escalation = False
    next_stage = None
    if status == "running":
        reason = "running_not_terminal"
    elif status == "missing":
        reason = "source_missing_rerun_candidate"
    elif status == "deferred_placeholder":
        reason = DEFERRED_PLACEHOLDER_REASON
    elif status == "invalid_target":
        reason = "invalid_target_rerun_candidate"
    elif status == "not_reached":
        if family in PHONON_FAMILIES:
            reason = "completed_not_reached_locked_cutoff_repair_or_settings_rerun"
        else:
            reason = "completed_not_reached_nonphonon_rerun_candidate"
    elif status == "failed":
        reason = "completed_failed_rerun_candidate"
    elif status == "terminal_upper_bound":
        reason = "terminal_upper_bound_rerun_candidate"
    elif status == "cost_excluded":
        reason = "cost_excluded_rerun_candidate"
    else:
        reason = "clean_target_evidence_available"
    return {
        "eligible_for_rerun": bool(eligible_for_rerun),
        "eligible_for_escalation": bool(eligible_for_escalation),
        "eligibility_reason": reason,
        "next_stage": next_stage if eligible_for_escalation else None,
        "next_stage_case_id": _next_stage_case_id(case_id, next_stage) if eligible_for_escalation else None,
        "escalation_reason": reason if eligible_for_escalation else None,
    }


def _snake_audit_fields(raw: Mapping[str, Any] | None, *, status: str, method: str) -> dict[str, Any]:
    if method != "SNAKE":
        return {}
    threshold_status = None if raw is None else raw.get("threshold_status")
    complete_trials = None if raw is None else raw.get("complete_trial_count")
    running_trials = None if raw is None else raw.get("running_trial_count")
    trial_count = None if raw is None else raw.get("trial_count")
    if raw is None:
        current_state = "missing_source"
        terminal_state = "missing_source"
        running_state = "missing_source"
        not_reached_state = "unknown_missing_source"
    else:
        current_state = str(threshold_status or status)
        if status == "running" or "running" in current_state:
            terminal_state = "not_terminal_running"
            if _first_int_value(complete_trials) == 0:
                running_state = "running_no_completed_trial"
            else:
                running_state = "running_with_completed_trials"
        elif status == "not_reached" or "not_reached" in current_state:
            terminal_state = "terminal_not_reached"
            running_state = "not_running"
        elif status == "displayed":
            terminal_state = "terminal_or_valid_hit"
            running_state = "not_running"
        else:
            terminal_state = status
            running_state = "not_running"
        if "not_reached" in current_state:
            not_reached_state = "current_best_not_reached" if "running" in current_state else "terminal_not_reached"
        elif raw.get("first_crossing_reached") is True or "reached" in current_state:
            not_reached_state = "reached"
        else:
            not_reached_state = "no_completed_trial" if _first_int_value(complete_trials) == 0 else "not_applicable"
    cost_source = None if raw is None else raw.get("cost_source")
    sidecar_present = bool(
        raw is not None
        and str(raw.get("threshold_status") or "").lower() == "ok_native_first_hit"
        and cost_source == SNAKE_AUDITED_COST_SOURCE
        and raw.get("cost_included") is True
        and str(raw.get("sidecar_validation_status") or "") == "ok"
        and raw.get("sidecar_hash_verified") is True
    )
    sidecar_source_text = "" if raw is None else _resource_source_text(raw)
    synthetic_sidecar_observed = bool(
        raw is not None
        and (
            isinstance(raw.get("paper_i_first_crossing_compiled_cost"), Mapping)
            or isinstance(raw.get("snake_first_crossing_compiled_cost"), Mapping)
            or raw.get("cost_source") == SNAKE_AUDITED_COST_SOURCE
        )
        and any(token in sidecar_source_text for token in ("live_overlay", "live_snake_overlay", "supplemental", "synthetic", "objective_score", "tie"))
    )
    return {
        "snake_current_state": current_state,
        "snake_terminal_state": terminal_state,
        "snake_running_state": running_state,
        "snake_not_reached_state": not_reached_state,
        "snake_complete_trial_count": complete_trials,
        "snake_running_trial_count": running_trials,
        "snake_trial_count": trial_count,
        "snake_best_trial_number": None if raw is None else raw.get("best_trial_number"),
        "snake_source_condor_job": None if raw is None else raw.get("source_condor_job"),
        "snake_first_crossing_reached": None if raw is None else raw.get("first_crossing_reached"),
        "snake_first_crossing_status": None if raw is None else raw.get("first_crossing_status"),
        "history_position_tau": None if raw is None else raw.get("history_position_tau"),
        "audited_first_crossing_compiled_cost_sidecar_present": sidecar_present,
        "audited_first_crossing_compiled_cost_sidecar_status": (
            "ok"
            if sidecar_present
            else str((raw or {}).get("sidecar_validation_status") or "absent")
        ),
        "sidecar_validation_status": None if raw is None else raw.get("sidecar_validation_status"),
        "sidecar_validation_reason": None if raw is None else raw.get("sidecar_validation_reason"),
        "sidecar_hash_verified": False if raw is None else bool(raw.get("sidecar_hash_verified")),
        "sidecar_source_kind": None if raw is None else raw.get("sidecar_source_kind"),
        "synthetic_snake_sidecar_observed": synthetic_sidecar_observed,
    }


def _build_expected_cell_audits(
    *,
    selected: Mapping[tuple[str, str, str], tuple[TableCell, dict[str, Any]]],
) -> list[dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    for spec in _expected_cell_specs():
        key = (spec["canonical_family"], spec["method"], spec["regime"])
        hit = selected.get(key)
        cell = hit[0] if hit is not None else None
        raw = hit[1] if hit is not None else None
        status = _cell_status_from_audit(raw, cell)
        missing_reason = None
        if status == "missing" and spec["canonical_family"] in DEFERRED_PLACEHOLDER_FAMILIES:
            status = "deferred_placeholder"
            missing_reason = DEFERRED_PLACEHOLDER_REASON
        elif status == "missing":
            missing_reason = "no_source_row_for_expected_key"
        elif status == "invalid_target":
            missing_reason = raw.get("skip_reason") if raw else "invalid_or_mismatched_source_row"
        cutoff_fields = _audit_cutoff_fields(raw, family=spec["canonical_family"], regime=spec["regime"])
        eligibility = _eligibility_fields(
            status=status,
            family=spec["canonical_family"],
            lane=spec["lane"],
            stage=cutoff_fields["paper_i_ladder_stage"],
            case_id=cutoff_fields["case_id"],
        )
        source_payload_path = None if raw is None else raw.get("source_payload_path")
        source_payload_missing_reason = None if raw is None else raw.get("source_payload_missing_reason")
        if raw is None and missing_reason is not None:
            source_payload_missing_reason = missing_reason
        raw_threshold_status = "" if raw is None else str(raw.get("threshold_status") or "")
        if raw is None:
            resource_allowed, resource_reason = False, "missing_source"
        else:
            resource_allowed = bool(raw.get("resource_display_allowed") is True)
            resource_reason = str(raw.get("resource_display_reason") or "")
            if not resource_reason:
                _resource_allowed_check, resource_reason = _resource_display_policy(raw, raw_threshold_status)
        cell_state = _cell_state_from_audit(raw, cell, method=spec["method"])
        if raw is None and spec["canonical_family"] in DEFERRED_PLACEHOLDER_FAMILIES:
            cell_state = "deferred-placeholder"
        elif raw is None:
            cell_state = "missing-source"
        audits.append(
            {
                **spec,
                **cutoff_fields,
                **eligibility,
                **_snake_audit_fields(raw, status=status, method=spec["method"]),
                "expected_key": list(key),
                "status": status,
                "cell_state": cell_state,
                "threshold_status": "missing" if raw is None else raw.get("threshold_status"),
                "cost_included": False if raw is None else bool(raw.get("cost_included")),
                "cost_source": None if raw is None else raw.get("cost_source"),
                "source": None if raw is None else raw.get("source"),
                "method_cost_semantics": None if raw is None else raw.get("method_cost_semantics"),
                "first_hit_cost_source_kind": None if raw is None else raw.get("first_hit_cost_source_kind"),
                "compiled_resource_validation_status": None if raw is None else raw.get("compiled_resource_validation_status"),
                "compiled_resource_validation_reason": None if raw is None else raw.get("compiled_resource_validation_reason"),
                "sidecar_validation_status": None if raw is None else raw.get("sidecar_validation_status"),
                "sidecar_validation_reason": None if raw is None else raw.get("sidecar_validation_reason"),
                "sidecar_hash_verified": False if raw is None else bool(raw.get("sidecar_hash_verified")),
                "sidecar_source_kind": None if raw is None else raw.get("sidecar_source_kind"),
                "source_resource_fields_present": None if raw is None else raw.get("source_resource_fields_present"),
                "historical_pareto_role": None if raw is None else raw.get("historical_pareto_role"),
                "historical_pareto_selection_rule": None if raw is None else raw.get("historical_pareto_selection_rule"),
                "historical_pareto_front_size": None if raw is None else raw.get("historical_pareto_front_size"),
                "historical_pareto_candidate_count": None if raw is None else raw.get("historical_pareto_candidate_count"),
                "historical_pareto_source_path": None if raw is None else raw.get("historical_pareto_source_path"),
                "display_overlay_role": None if raw is None else raw.get("display_overlay_role"),
                "display_overlay_source_path": None if raw is None else raw.get("display_overlay_source_path"),
                "display_overlay_source_label": None if raw is None else raw.get("display_overlay_source_label"),
                "resource_display_allowed": resource_allowed,
                "resource_display_reason": resource_reason,
                "display_delta_e": None if cell is None else cell.delta_e,
                "display_infidelity": None if cell is None else cell.infidelity,
                "display_n_2q": None if cell is None else cell.n_2q,
                "display_d_2q": None if cell is None else cell.d_2q,
                "display_d_circ": None if cell is None else cell.d_circ,
                "display_s_alg": None if cell is None else cell.s_alg,
                "resources_displayed": False if cell is None else _cell_resources_displayed(cell),
                "resource_numeric_displayed": False if cell is None else _cell_numeric_resources_displayed(cell),
                "source_row_index": None if raw is None else raw.get("row_index"),
                "source_record_id": None if raw is None else raw.get("record_id"),
                "source_payload_path": source_payload_path,
                "source_payload_path_kind": None if raw is None else raw.get("source_payload_path_kind"),
                "source_payload_missing_reason": source_payload_missing_reason,
                "payload_path": source_payload_path,
                "payload_sha256": None if raw is None else raw.get("source_payload_sha256"),
                "missing_reason": missing_reason,
                "eligible_for_display": status not in {"missing", "invalid_target", "deferred_placeholder"},
            }
        )
    return audits


def _harmonic_kerr_pos_geo_finding(
    expected_cell_audits: Sequence[Mapping[str, Any]],
    raw_row_audits: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    key = ["harmonic_kerr_chain", "Pos-Geo-ADAPT", "weak"]
    cell = next((item for item in expected_cell_audits if item.get("expected_key") == key), None)
    near_misses = [
        item
        for item in raw_row_audits
        if item.get("normalized_key") == key and not bool(item.get("display_candidate"))
    ]
    valid = bool(
        cell
        and cell.get("status") == "displayed"
        and cell.get("threshold_status") == "ok_native_first_hit"
        and cell.get("cost_included") is True
    )
    return {
        "expected_key": key,
        "status": None if cell is None else cell.get("status"),
        "valid_clean_ok_native_first_hit": valid,
        "source_row_index": None if cell is None else cell.get("source_row_index"),
        "payload_path": None if cell is None else cell.get("payload_path"),
        "missing_reason": None if cell is None else cell.get("missing_reason"),
        "near_miss_count": len(near_misses),
        "near_misses": near_misses[:8],
    }


def _snake_cost_findings(raw_row_audits: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    snake_rows = [
        item
        for item in raw_row_audits
        if item.get("lane") == "snake" and item.get("status") != "deferred_placeholder"
    ]
    valid = [
        item
        for item in snake_rows
        if str(item.get("threshold_status") or "").lower() == "ok_native_first_hit"
        and item.get("cost_source") == SNAKE_AUDITED_COST_SOURCE
        and item.get("cost_included") is True
        and item.get("sidecar_validation_status") == "ok"
        and item.get("sidecar_hash_verified") is True
    ]
    return {
        "snake_row_count": len(snake_rows),
        "audited_first_crossing_compiled_cost_count": len(valid),
        "audited_first_crossing_compiled_cost_denominator": len(snake_rows),
        "audited_first_crossing_compiled_cost_summary": f"{len(valid)}/{len(snake_rows)}",
        "any_audited_first_crossing_compiled_cost": bool(valid),
        "status_counts": {
            status: sum(1 for item in snake_rows if item.get("threshold_status") == status)
            for status in sorted({str(item.get("threshold_status")) for item in snake_rows})
        },
        "running_state_counts": {
            status: sum(1 for item in snake_rows if item.get("snake_running_state") == status)
            for status in sorted({str(item.get("snake_running_state")) for item in snake_rows})
        },
        "terminal_state_counts": {
            status: sum(1 for item in snake_rows if item.get("snake_terminal_state") == status)
            for status in sorted({str(item.get("snake_terminal_state")) for item in snake_rows})
        },
        "not_reached_state_counts": {
            status: sum(1 for item in snake_rows if item.get("snake_not_reached_state") == status)
            for status in sorted({str(item.get("snake_not_reached_state")) for item in snake_rows})
        },
        "sidecar_status_counts": {
            status: sum(1 for item in snake_rows if item.get("audited_first_crossing_compiled_cost_sidecar_status") == status)
            for status in sorted({str(item.get("audited_first_crossing_compiled_cost_sidecar_status")) for item in snake_rows})
        },
    }


def _load_historical_pareto_rows(
    historical_pareto_json: Path | None,
    *,
    threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if historical_pareto_json is None:
        return [], {}
    metadata: dict[str, Any] = {
        "path": str(historical_pareto_json),
        "requested_threshold": float(threshold),
    }
    if not historical_pareto_json.exists():
        metadata["missing"] = True
        return [], metadata
    payload = json.loads(historical_pareto_json.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        metadata["invalid_payload"] = "not_mapping"
        return [], metadata
    rows: list[Mapping[str, Any]] = []
    if isinstance(payload.get("row_results"), list):
        rows.extend(item for item in payload["row_results"] if isinstance(item, Mapping))
    else:
        groups = payload.get("groups")
        if isinstance(groups, list):
            for group in groups:
                if not isinstance(group, Mapping):
                    continue
                display = group.get("display_representative")
                if isinstance(display, Mapping):
                    rows.append(display)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        row_threshold = finite_number(row.get("threshold") or row.get("tau_phys"))
        if row_threshold is not None and abs(row_threshold - float(threshold)) > 1e-12:
            rejected.append({"row_index": index, "reason": "threshold_mismatch", "threshold": row_threshold})
            continue
        out = dict(row)
        out.setdefault("threshold", float(threshold))
        out.setdefault("threshold_status", "ok_native_first_hit")
        out.setdefault("method", "SNAKE")
        out.setdefault("algorithm_id", ALGORITHM_ID_BY_METHOD["SNAKE"])
        out.setdefault("cost_included", True)
        out.setdefault("historical_pareto_role", "display_representative")
        out.setdefault("historical_pareto_source_path", str(historical_pareto_json))
        out.setdefault("historical_pareto_selection_rule", payload.get("display_representative_rule"))
        out.setdefault("historical_pareto_payload_schema", payload.get("schema"))
        accepted.append(out)
    metadata.update(
        {
            "schema": payload.get("schema"),
            "selection_policy": payload.get("selection_policy"),
            "display_representative_rule": payload.get("display_representative_rule"),
            "source_threshold": payload.get("threshold"),
            "accepted_display_row_count": len(accepted),
            "rejected_display_row_count": len(rejected),
            "rejected_display_rows": rejected[:32],
            "accepted_candidate_count": payload.get("accepted_candidate_count"),
            "rejection_counts": payload.get("rejection_counts"),
        }
    )
    return accepted, metadata


def _overlay_status_from_cell(row: Mapping[str, Any], method: str | None) -> str:
    explicit = row.get("threshold_status")
    if explicit:
        return str(explicit)
    delta = _display_overlay_value(row, "delta_e").strip().lower()
    if delta in {"nr", "not reached", "not_reached"}:
        return "not_reached"
    if delta in {"running", "failed", "ub", "n/a", "--"}:
        return {"ub": "terminal_upper_bound_missing_native_first_hit"}.get(delta, delta)
    number = finite_number(delta)
    if number is None:
        return "unknown"
    if number == 0.0:
        return "ok_terminal_only_method" if method in {"HEA VQE", "family-informed VQE"} else "ok_native_first_hit"
    return "not_reached"


def _overlay_qiskit_resources_are_numeric(row: Mapping[str, Any]) -> bool:
    return all(
        finite_number(_display_overlay_value(row, key)) is not None
        for key in ("n_2q", "d_2q", "d_circ")
    )


def _load_display_overlay_rows(
    display_overlay_json: Path | None,
    *,
    threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if display_overlay_json is None:
        return [], {}
    metadata: dict[str, Any] = {
        "path": str(display_overlay_json),
        "requested_threshold": float(threshold),
    }
    if not display_overlay_json.exists():
        metadata["missing"] = True
        return [], metadata
    payload = json.loads(display_overlay_json.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        metadata["invalid_payload"] = "not_mapping"
        return [], metadata
    raw_rows = payload.get("row_results") or payload.get("rows") or payload.get("cells") or []
    rows = raw_rows if isinstance(raw_rows, list) else []
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    source_label = str(payload.get("source_label") or "approved_pdf_recovered_cell_overlay")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            rejected.append({"row_index": index, "reason": "not_mapping"})
            continue
        row_threshold = finite_number(row.get("threshold") or row.get("tau_phys"))
        if row_threshold is not None and abs(row_threshold - float(threshold)) > 1e-12:
            rejected.append({"row_index": index, "reason": "threshold_mismatch", "threshold": row_threshold})
            continue
        method = normalize_method(row.get("method"))
        qiskit_resources_numeric = _overlay_qiskit_resources_are_numeric(row)
        s_alg_numeric = finite_number(_display_overlay_value(row, "s_alg")) is not None
        out = dict(row)
        out["method"] = method or row.get("method")
        if method in ALGORITHM_ID_BY_METHOD:
            out.setdefault("algorithm_id", ALGORITHM_ID_BY_METHOD[method])
        out.setdefault("threshold", float(threshold))
        out.setdefault("threshold_status", _overlay_status_from_cell(out, method))
        out.setdefault("display_overlay_role", "approved_table_cell")
        out.setdefault("display_overlay_source_path", str(display_overlay_json))
        out.setdefault("display_overlay_source_label", source_label)
        out.setdefault("source", "approved_pdf_recovered_cell_overlay")
        out.setdefault("threshold_source", f"approved_pdf_recovered_cell_overlay:{display_overlay_json}")
        out.setdefault("resource_display_allowed", qiskit_resources_numeric)
        resource_display_reason = "approved_pdf_recovered_status_only"
        if qiskit_resources_numeric:
            resource_display_reason = (
                "approved_pdf_recovered_qiskit_cost_with_validated_s_alg"
                if s_alg_numeric
                else "approved_pdf_recovered_qiskit_cost_s_alg_missing"
            )
        out.setdefault(
            "resource_display_reason",
            resource_display_reason,
        )
        out.setdefault("cost_included", qiskit_resources_numeric)
        out.setdefault("compiled_resource_validation_status", "approved_pdf_recovered")
        out.setdefault("compiled_resource_validation_reason", "recovered_from_approved_table_text")
        out.setdefault("first_hit_cost_source_kind", "qiskit_compiled_first_hit_ansatz_circuit")
        out.setdefault("method_cost_semantics", "approved_table_display_cell")
        accepted.append(out)
    metadata.update(
        {
            "schema": payload.get("schema"),
            "source_label": source_label,
            "source_note": payload.get("source_note"),
            "source_threshold": payload.get("threshold"),
            "accepted_display_row_count": len(accepted),
            "rejected_display_row_count": len(rejected),
            "rejected_display_rows": rejected[:32],
        }
    )
    return accepted, metadata


def load_rows_with_diagnostics(
    summary_json: Path | None,
    threshold: float,
    historical_pareto_json: Path | None = None,
    display_overlay_json: Path | None = None,
) -> tuple[dict[tuple[str, str, str], TableCell], LoadDiagnostics]:
    diagnostics = LoadDiagnostics(
        summary_json=None if summary_json is None else str(summary_json),
        historical_pareto_json=None if historical_pareto_json is None else str(historical_pareto_json),
        display_overlay_json=None if display_overlay_json is None else str(display_overlay_json),
    )
    historical_rows, historical_metadata = _load_historical_pareto_rows(
        historical_pareto_json,
        threshold=float(threshold),
    )
    diagnostics.historical_pareto_metadata = historical_metadata
    display_overlay_rows, display_overlay_metadata = _load_display_overlay_rows(
        display_overlay_json,
        threshold=float(threshold),
    )
    diagnostics.display_overlay_metadata = display_overlay_metadata
    if summary_json is None or not summary_json.exists():
        diagnostics.summary_missing = True
        payload: Any = {"row_results": []}
        live_snake_overlay_source = None
        live_snake_overlay_by_case: dict[str, Mapping[str, Any]] = {}
        live_snake_overlay_resolved_path = None
        live_snake_overlay_sha256 = None
    else:
        payload = json.loads(summary_json.read_text())
        diagnostics.summary_metadata = _summary_metadata(payload)
        live_snake_overlay = diagnostics.summary_metadata.get("live_snake_overlay")
        live_snake_overlay_source = None
        if isinstance(live_snake_overlay, Mapping):
            live_snake_overlay_source = live_snake_overlay.get("source_path")
        live_snake_overlay_by_case, live_snake_overlay_resolved_path, live_snake_overlay_sha256 = _load_live_snake_overlay_by_case(
            live_snake_overlay_source,
            summary_json=summary_json,
        )
    rows_raw = payload if isinstance(payload, list) else payload.get("row_results", [])
    rows = rows_raw if isinstance(rows_raw, list) else []
    enrichment_roots = _candidate_enrichment_roots(summary_json, diagnostics.summary_metadata)
    prior_current_best_rows = _load_prior_current_best_rows(summary_json, threshold, diagnostics.summary_metadata)
    existing_keys = {
        (family, method, regime)
        for row in rows
        if isinstance(row, dict)
        and (finite_number(row.get("threshold")) is not None and abs(float(finite_number(row.get("threshold"))) - float(threshold)) <= 1e-12)
        for family, method, regime in [
            (normalize_family(row), normalize_method(row.get("method")), normalize_regime(row))
        ]
        if family is not None and method is not None and regime is not None
    }
    supplemental_rows = (
        []
        if summary_json is None or not summary_json.exists()
        else _load_supplemental_result_rows(
            summary_json,
            threshold,
            existing_keys=existing_keys,
            metadata=diagnostics.summary_metadata,
        )
    )
    rows = [*prior_current_best_rows, *rows, *supplemental_rows, *historical_rows, *display_overlay_rows]
    selected: dict[tuple[str, str, str], tuple[TableCell, dict[str, Any], tuple[int, ...]]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            diagnostics.skipped_rows.append(f"row[{index}]: non-dict")
            diagnostics.raw_row_audits.append({"row_index": index, "skip_reason": "non_dict", "display_candidate": False})
            continue
        row = _augment_snake_row_from_live_overlay(
            row,
            overlay_by_case=live_snake_overlay_by_case,
            overlay_path=live_snake_overlay_resolved_path,
            overlay_sha256=live_snake_overlay_sha256,
        )
        row = _augment_row_from_enrichment(row, enrichment_roots)
        row = _augment_row_from_payload_resources(row, summary_json=summary_json)
        row_threshold = finite_number(row.get("threshold"))
        threshold_matches = row_threshold is not None and abs(row_threshold - threshold) <= 1e-12
        family = normalize_family(row)
        method = normalize_method(row.get("method"))
        regime = normalize_regime(row)
        key = (family, method, regime) if family is not None and method is not None and regime is not None else None
        lane = _lane_for_row(row, method)
        payload_path = row.get("payload_path") or row.get("result_path") or row.get("source_payload_path")
        payload_path_kind = "row_payload" if payload_path else None
        if payload_path is None and lane == "snake" and live_snake_overlay_source:
            payload_path = str(live_snake_overlay_resolved_path or live_snake_overlay_source)
            payload_path_kind = "live_snake_overlay_summary"
        source_payload_missing_reason = None if payload_path else "source_payload_path_absent_in_summary_row"
        case_id = _case_id_from_audit(row, family=family, regime=regime)
        n_ph_work = _infer_n_ph_work(row, family=family, case_id=case_id)
        n_ph_ref = _infer_n_ph_ref(row, family=family, case_id=case_id, n_ph_work=n_ph_work)
        stage = _paper_i_ladder_stage(row, family=family, n_ph_work=n_ph_work, n_ph_ref=n_ph_ref)
        audit: dict[str, Any] = {
            "row_index": index,
            "record_id": row.get("record_id"),
            "case_id": case_id,
            "benchmark_id": row.get("benchmark_id"),
            "family_raw": row.get("family") or row.get("hamiltonian") or row.get("hamiltonian_id"),
            "family_normalized": family,
            "method_raw": row.get("method"),
            "method_normalized": method,
            "regime": regime,
            "algorithm_id": row.get("algorithm_id"),
            "lane": lane,
            "threshold": row_threshold,
            "threshold_source": row.get("threshold_source"),
            "threshold_matches_requested": threshold_matches,
            "threshold_status": row.get("threshold_status"),
            "cost_included": bool(row.get("cost_included")),
            "cost_source": row.get("cost_source"),
            "source": row.get("source"),
            "method_cost_semantics": row.get("method_cost_semantics"),
            "resource_display_allowed": _resource_display_policy(row, row.get("threshold_status"))[0],
            "resource_display_reason": _resource_display_policy(row, row.get("threshold_status"))[1],
            "compiled_resource_validation_status": row.get("compiled_resource_validation_status"),
            "compiled_resource_validation_reason": row.get("compiled_resource_validation_reason"),
            "first_hit_cost_source_kind": row.get("first_hit_cost_source_kind"),
            "source_resource_fields_present": row.get("source_resource_fields_present"),
            "sidecar_validation_status": row.get("sidecar_validation_status"),
            "sidecar_validation_reason": row.get("sidecar_validation_reason"),
            "sidecar_hash_verified": bool(row.get("sidecar_hash_verified")),
            "sidecar_source_kind": row.get("sidecar_source_kind"),
            "complete_trial_count": row.get("complete_trial_count"),
            "running_trial_count": row.get("running_trial_count"),
            "trial_count": row.get("trial_count"),
            "best_trial_number": row.get("best_trial_number"),
            "source_condor_job": row.get("source_condor_job"),
            "first_crossing_reached": row.get("first_crossing_reached"),
            "first_crossing_status": row.get("first_crossing_status"),
            "history_position_tau": row.get("history_position_tau"),
            "historical_pareto_role": row.get("historical_pareto_role"),
            "historical_pareto_selection_rule": row.get("historical_pareto_selection_rule"),
            "historical_pareto_front_size": row.get("historical_pareto_front_size"),
            "historical_pareto_candidate_count": row.get("historical_pareto_candidate_count"),
            "historical_pareto_source_path": row.get("historical_pareto_source_path"),
            "display_overlay_role": row.get("display_overlay_role"),
            "display_overlay_source_path": row.get("display_overlay_source_path"),
            "display_overlay_source_label": row.get("display_overlay_source_label"),
            "n_ph_work": n_ph_work,
            "n_ph_ref": n_ph_ref,
            "paper_i_ladder_stage": stage,
            "paper_i_cutoff_contract": "locked_known_cutoff" if family in PHONON_FAMILIES else "not_applicable_nonphonon",
            "cutoff_missing_reason": _cutoff_missing_reason(family=family, n_ph_work=n_ph_work, n_ph_ref=n_ph_ref),
            "source_payload_path": None if payload_path is None else str(payload_path),
            "source_payload_path_kind": payload_path_kind,
            "source_payload_missing_reason": source_payload_missing_reason,
            "source_payload_sha256": _sha256_for_path(payload_path),
            "payload_path": None if payload_path is None else str(payload_path),
            "payload_sha256": _sha256_for_path(payload_path),
            "normalized_key": None if key is None else list(key),
            "display_candidate": False,
            "display_priority": None,
            "skip_reason": None,
            **_snake_audit_fields(row, status="raw", method=str(method or "")),
        }
        diagnostics.raw_row_audits.append(audit)
        if row_threshold is None:
            audit["skip_reason"] = "missing_threshold"
            diagnostics.threshold_mismatch_rows.append(f"row[{index}]: missing threshold")
            continue
        if not threshold_matches:
            audit["skip_reason"] = "threshold_mismatch"
            diagnostics.threshold_mismatch_rows.append(f"row[{index}]: threshold={row_threshold}")
            continue
        if key is None:
            audit["skip_reason"] = "missing_normalized_key"
            diagnostics.skipped_rows.append(
                f"row[{index}]: record_id={row.get('record_id')!r}, family={family!r}, method={method!r}, regime={regime!r}"
            )
            continue
        priority = _row_priority(row, threshold_matches=True)
        audit["display_candidate"] = True
        audit["display_priority"] = list(priority)
        cell = cell_from_row(row, threshold)
        existing = selected.get(key)
        if existing is not None:
            diagnostics.duplicate_keys.append(f"{key[0]}|{key[1]}|{key[2]}")
            chosen = "new" if priority >= existing[2] else "existing"
            diagnostics.duplicate_resolutions.append(
                {
                    "key": list(key),
                    "existing_row_index": existing[1].get("row_index"),
                    "new_row_index": index,
                    "chosen": chosen,
                    "existing_priority": list(existing[2]),
                    "new_priority": list(priority),
                }
            )
            if priority < existing[2]:
                continue
        selected[key] = (cell, audit, priority)
    out = {key: value[0] for key, value in selected.items()}
    selected_audits = {key: (value[0], value[1]) for key, value in selected.items()}
    diagnostics.expected_cell_audits = _build_expected_cell_audits(selected=selected_audits)
    diagnostics.missing_expected_count = sum(1 for item in diagnostics.expected_cell_audits if item.get("status") == "missing")
    diagnostics.invalid_target_count = sum(1 for item in diagnostics.expected_cell_audits if item.get("status") == "invalid_target")
    diagnostics.special_findings = {
        "harmonic_kerr_weak_pos_geo": _harmonic_kerr_pos_geo_finding(
            diagnostics.expected_cell_audits,
            diagnostics.raw_row_audits,
        ),
        "snake_audited_first_crossing_costs": _snake_cost_findings(diagnostics.expected_cell_audits),
    }
    return out, diagnostics


def load_rows(
    summary_json: Path,
    threshold: float,
    historical_pareto_json: Path | None = None,
    display_overlay_json: Path | None = None,
) -> dict[tuple[str, str, str], TableCell]:
    rows, _ = load_rows_with_diagnostics(
        summary_json,
        threshold,
        historical_pareto_json=historical_pareto_json,
        display_overlay_json=display_overlay_json,
    )
    return rows


def find_cell(
    rows: dict[tuple[str, str, str], TableCell],
    aliases: set[str],
    method: str,
    regime: str,
) -> TableCell:
    for alias in aliases:
        hit = rows.get((alias, method, regime))
        if hit is not None:
            return hit
    return TableCell()


def render_row(hamiltonian: str, aliases: set[str], method: str, rows: dict[tuple[str, str, str], TableCell]) -> str:
    weak = find_cell(rows, aliases, method, "weak").as_latex_cells()
    strong = find_cell(rows, aliases, method, "strong").as_latex_cells()
    cutoff_strength = CUTOFF_STRENGTH_BY_LABEL.get(hamiltonian, "--")
    cells = [latex_escape(hamiltonian), cutoff_strength, latex_escape(method), *weak, *strong]
    return " & ".join(cells) + r"\\"


def render_diagnostics(diagnostics: LoadDiagnostics | None) -> str:
    if diagnostics is None:
        return ""
    lines: list[str] = []
    if diagnostics.summary_missing:
        lines.append(r"\item Source summary JSON was missing; all cells are placeholders.")
    if diagnostics.skipped_rows:
        examples = "; ".join(latex_escape(item) for item in diagnostics.skipped_rows[:8])
        suffix = "" if len(diagnostics.skipped_rows) <= 8 else f"; ... {len(diagnostics.skipped_rows) - 8} more"
        lines.append(rf"\item Skipped source rows: {len(diagnostics.skipped_rows)} ({examples}{suffix}).")
    if diagnostics.threshold_mismatch_rows:
        examples = "; ".join(latex_escape(item) for item in diagnostics.threshold_mismatch_rows[:8])
        suffix = "" if len(diagnostics.threshold_mismatch_rows) <= 8 else f"; ... {len(diagnostics.threshold_mismatch_rows) - 8} more"
        lines.append(rf"\item Clean-threshold gate skipped rows: {len(diagnostics.threshold_mismatch_rows)} ({examples}{suffix}).")
    if diagnostics.historical_pareto_metadata:
        count = diagnostics.historical_pareto_metadata.get("accepted_display_row_count", 0)
        path = diagnostics.historical_pareto_metadata.get("path", diagnostics.historical_pareto_json)
        path_label = Path(str(path)).name if path else "none"
        lines.append(
            rf"\item SNAKE historical Route-A/SPSA Pareto overlay supplied {count} display representative row(s) from {latex_escape(path_label)}."
        )
    if diagnostics.display_overlay_metadata:
        count = diagnostics.display_overlay_metadata.get("accepted_display_row_count", 0)
        path = diagnostics.display_overlay_metadata.get("path", diagnostics.display_overlay_json)
        path_label = Path(str(path)).name if path else "none"
        label = diagnostics.display_overlay_metadata.get("source_label", "approved-pdf recovery")
        lines.append(
            rf"\item Approved-table display overlay supplied {count} recovered current-evidence cell(s) from {latex_escape(path_label)} ({latex_escape(str(label))})."
        )
        note = str(diagnostics.display_overlay_metadata.get("source_note") or "").strip()
        if note:
            lines.append(rf"\item Approved-table display overlay note: {latex_escape(note)}")
    if diagnostics.duplicate_keys:
        examples = "; ".join(latex_escape(item) for item in diagnostics.duplicate_keys[:8])
        suffix = "" if len(diagnostics.duplicate_keys) <= 8 else f"; ... {len(diagnostics.duplicate_keys) - 8} more"
        lines.append(rf"\item Duplicate table keys resolved by deterministic source priority: {len(diagnostics.duplicate_keys)} ({examples}{suffix}).")
    if diagnostics.expected_cell_audits:
        deferred_placeholder_count = sum(
            1 for item in diagnostics.expected_cell_audits if item.get("status") == "deferred_placeholder"
        )
        lines.append(
            rf"\item Expected-cell audit: {diagnostics.missing_expected_count} actionable missing cells, {diagnostics.invalid_target_count} invalid-target cells, and {deferred_placeholder_count} deferred placeholder cells in the clean table skeleton."
        )
    hk = diagnostics.special_findings.get("harmonic_kerr_weak_pos_geo", {})
    if hk:
        lines.append(
            rf"\item Harmonic/Kerr weak Pos-Geo source-map: status={latex_escape(str(hk.get('status')))}, valid clean native first hit={str(bool(hk.get('valid_clean_ok_native_first_hit'))).lower()}, near misses={hk.get('near_miss_count', 0)}."
        )
    snake = diagnostics.special_findings.get("snake_audited_first_crossing_costs", {})
    if snake:
        lines.append(
            rf"\item SNAKE audited first-crossing compiled-cost sidecars: {snake.get('audited_first_crossing_compiled_cost_count', 0)} valid of {snake.get('snake_row_count', 0)} SNAKE rows."
        )
    if not lines:
        lines.append(r"\item Source rows loaded without skipped-row, duplicate-key, or expected-cell diagnostics.")
    return "\n".join(
        [
            r"\vspace{6pt}",
            r"\noindent\textbf{Source/status diagnostics.}",
            r"\begin{itemize}\footnotesize",
            *lines,
            r"\item Resource columns report the available Qiskit-compiled/current-best ansatz-circuit cost for the displayed evidence row; \(\delta E\) carries the hit/miss status. Audited first-crossing sidecar counts remain provenance diagnostics, not a display gate.",
            r"\end{itemize}",
        ]
    )


def _cost_semantics_audit(diagnostics: LoadDiagnostics | None) -> dict[str, Any]:
    expected = [] if diagnostics is None else list(diagnostics.expected_cell_audits)
    raw_rows = [] if diagnostics is None else list(diagnostics.raw_row_audits)
    numeric = [item for item in expected if item.get("resource_numeric_displayed") is True]
    forbidden_numeric = [
        item
        for item in numeric
        if item.get("resource_display_allowed") is not True
    ]

    def _is_diagnostic_resource_display(item: Mapping[str, Any]) -> bool:
        return (
            item.get("resource_display_allowed") is True
            and item.get("cost_included") is not True
            and item.get("cost_source") != SNAKE_AUDITED_COST_SOURCE
        )

    synthetic_numeric = [
        item
        for item in numeric
        if item.get("lane") == "snake"
        and not _is_diagnostic_resource_display(item)
        and any(
            token in _resource_source_text(item)
            for token in ("live_overlay", "live_snake_overlay", "supplemental", "synthetic", "objective_score", "tie")
        )
    ]
    forbidden_observed = [
        item
        for item in [*expected, *raw_rows]
        if _forbidden_resource_source_reason(item) is not None
    ]
    synthetic_observed = [
        item
        for item in [*expected, *raw_rows]
        if item.get("lane") == "snake"
        and (
            item.get("synthetic_snake_sidecar_observed") is True
            or (
                item.get("cost_source") == SNAKE_AUDITED_COST_SOURCE
                and not _is_diagnostic_resource_display(item)
                and any(
                    token in _resource_source_text(item)
                    for token in ("live_overlay", "live_snake_overlay", "supplemental", "synthetic", "objective_score", "tie")
                )
            )
        )
    ]
    statusless_cost_included = [
        item
        for item in raw_rows
        if item.get("cost_included") is True and not str(item.get("threshold_status") or "")
    ]
    adaptive_terminal_numeric = [
        item
        for item in numeric
        if item.get("lane") != "snake"
        and (
            "upper_bound" in str(item.get("threshold_status") or "").lower()
            or (
                "terminal" in str(item.get("first_hit_cost_source_kind") or item.get("source") or "").lower()
                and item.get("method_cost_semantics") != "terminal_only_fixed_ansatz"
                and str(item.get("first_hit_cost_source_kind") or "") != "qiskit_compiled_terminal_only_fixed_ansatz"
                and str(item.get("first_hit_cost_source_kind") or "") != "qiskit_compiled_final_ansatz_circuit"
            )
        )
    ]
    running_no_completed_numeric = [
        item
        for item in numeric
        if item.get("cell_state") == "running-no-completed-trial"
    ]
    invalid_depth_ordering = [
        item
        for item in [*expected, *raw_rows]
        if str(item.get("resource_display_reason") or item.get("compiled_resource_validation_reason") or "")
        in {"compiled_depth_total_less_than_two_qubit_depth", "compiled_depth_2q_semantic_mismatch_count_not_depth"}
    ]
    numeric_invalid_depth_ordering = [
        item
        for item in numeric
        if str(item.get("resource_display_reason") or item.get("compiled_resource_validation_reason") or "")
        in {"compiled_depth_total_less_than_two_qubit_depth", "compiled_depth_2q_semantic_mismatch_count_not_depth"}
    ]
    pass_fail_counts = {
        "numeric_resource_without_display_permission_count": len(forbidden_numeric),
        "running_no_completed_trial_numeric_resource_count": len(running_no_completed_numeric),
        "invalid_depth_ordering_numeric_resource_count": len(numeric_invalid_depth_ordering),
    }
    return {
        "schema": "paper_i_fixed_accuracy_cost_semantics_audit_v1",
        "numeric_resource_row_count": len(numeric),
        "forbidden_numeric_resource_source_count": len(forbidden_numeric),
        "forbidden_resource_source_count": len(forbidden_observed),
        "synthetic_snake_sidecar_count": len(synthetic_numeric),
        "synthetic_snake_sidecar_observed_count": len(synthetic_observed),
        "statusless_cost_included_count": len(statusless_cost_included),
        "adaptive_terminal_fallback_numeric_resource_count": len(adaptive_terminal_numeric),
        "running_no_completed_trial_numeric_resource_count": len(running_no_completed_numeric),
        "invalid_depth_ordering_count": len(invalid_depth_ordering),
        "invalid_depth_ordering_numeric_resource_count": len(numeric_invalid_depth_ordering),
        "numeric_resource_semantics_pass": all(value == 0 for value in pass_fail_counts.values()),
        "numeric_resource_rows": [item.get("expected_key") for item in numeric],
        "forbidden_numeric_resource_rows": [item.get("expected_key") for item in forbidden_numeric],
    }


def build_audit_manifest(
    *,
    source_pdf: Path | None,
    summary_json: Path | None,
    threshold: float,
    diagnostics: LoadDiagnostics | None,
) -> dict[str, Any]:
    metadata = diagnostics.summary_metadata if diagnostics is not None else {}
    return {
        "schema": "paper_i_fixed_accuracy_table_audit_v1",
        "source_summary_json": None if summary_json is None else str(summary_json),
        "historical_pareto_json": None if diagnostics is None else diagnostics.historical_pareto_json,
        "display_overlay_json": None if diagnostics is None else diagnostics.display_overlay_json,
        "source_candidate_pdf": None if source_pdf is None else str(source_pdf),
        "target_profile": metadata.get("target_profile") or "paper_i_phys_v1",
        "threshold": float(threshold),
        "threshold_label": format_number(threshold),
        "summary_metadata": metadata,
        "historical_pareto_metadata": {} if diagnostics is None else diagnostics.historical_pareto_metadata,
        "display_overlay_metadata": {} if diagnostics is None else diagnostics.display_overlay_metadata,
        "source_roots": {
            "comparator": metadata.get("output_root"),
            "snake": metadata.get("output_root"),
            "snake_live_overlay": metadata.get("live_snake_overlay", {}).get("source_path")
            if isinstance(metadata.get("live_snake_overlay"), Mapping)
            else None,
            "enrichment": metadata.get("enrichment_root"),
            "records": metadata.get("records_path"),
        },
        "raw_row_audits": [] if diagnostics is None else diagnostics.raw_row_audits,
        "expected_cell_audits": [] if diagnostics is None else diagnostics.expected_cell_audits,
        "special_findings": {} if diagnostics is None else diagnostics.special_findings,
        "cost_semantics": _cost_semantics_audit(diagnostics),
        "diagnostics": {
            "summary_missing": False if diagnostics is None else diagnostics.summary_missing,
            "skipped_rows": [] if diagnostics is None else diagnostics.skipped_rows,
            "duplicate_keys": [] if diagnostics is None else diagnostics.duplicate_keys,
            "duplicate_resolutions": [] if diagnostics is None else diagnostics.duplicate_resolutions,
            "threshold_mismatch_rows": [] if diagnostics is None else diagnostics.threshold_mismatch_rows,
            "missing_expected_count": 0 if diagnostics is None else diagnostics.missing_expected_count,
            "invalid_target_count": 0 if diagnostics is None else diagnostics.invalid_target_count,
            "deferred_placeholder_count": 0
            if diagnostics is None
            else sum(1 for item in diagnostics.expected_cell_audits if item.get("status") == "deferred_placeholder"),
        },
    }


def write_audit_manifest(
    path: Path,
    *,
    source_pdf: Path | None,
    summary_json: Path | None,
    threshold: float,
    diagnostics: LoadDiagnostics | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            build_audit_manifest(
                source_pdf=source_pdf,
                summary_json=summary_json,
                threshold=threshold,
                diagnostics=diagnostics,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def render_tex(
    source_pdf: Path | None,
    summary_json: Path | None,
    threshold: float,
    rows: dict[tuple[str, str, str], TableCell],
    n_ph_algorithm: str,
    n_ph_ed: str,
    diagnostics: LoadDiagnostics | None = None,
) -> str:
    source_comment = str(source_pdf) if source_pdf else "none"
    summary_comment = str(summary_json) if summary_json else "none"
    body: list[str] = []
    for group_name, hamiltonians in GROUPS:
        body.append(r"\midrule")
        body.append(rf"\multicolumn{{15}}{{l}}{{\textbf{{{group_name}}}}}\\")
        for hamiltonian, aliases in hamiltonians:
            for method in METHODS:
                body.append(render_row(hamiltonian, aliases, method, rows))
            body.append(r"\addlinespace")
    if body and body[-1] == r"\addlinespace":
        body.pop()

    return rf"""\documentclass{{article}}
\usepackage[margin=0.35in,landscape]{{geometry}}
\usepackage{{booktabs,longtable,array,caption,xcolor}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern}}
\usepackage{{microtype}}
\pagestyle{{plain}}
\captionsetup{{font=small,skip=4pt}}
% Generated by pipelines/reporting/build_paper_i_fixed_accuracy_table_pdf.py
% Source candidate PDF: {source_comment}
% Source summary JSON: {summary_comment}
% Physical threshold: {threshold}
\begin{{document}}
\begin{{center}}
{{\Large\bfseries Paper-I approved fixed-accuracy table}}\par\vspace{{2pt}}
{{\small Per-Hamiltonian candidate-results grid generated in the approved Paper-I report shape.}}\par\vspace{{6pt}}
\end{{center}}

\noindent\textbf{{Cutoff/strength column:}}
For phonon rows, each entry reports
$(n_{{\rm ph}}^{{\rm work}},n_{{\rm ph}}^{{\rm ED}})_w,(n_{{\rm ph}}^{{\rm work}},n_{{\rm ph}}^{{\rm ED}})_s;\lambda_{{w}}/\lambda_{{s}}$.
Here $\lambda$ is Hamiltonian-specific: $U/t$ for Hubbard-type rows, $(U/t,V/t)$ for extended Hubbard,
$V/t$ for spinless $t$-$V$, $U_b/J$ for Bose-Hubbard, $\omega_0/t$ for harmonic/Kerr with the Kerr
coefficient fixed, $g/\omega_0$ for spin-boson, $(U/t,g/t)$ for Hubbard-Holstein, and $g_{{\rm ep}}$
for molecular-vibronic H$_2$.\par\vspace{{6pt}}

\noindent\textbf{{$\delta E$ reference:}}
For phonon-containing rows, $(n_{{\rm ph}}^{{\rm work}},n_{{\rm ph}}^{{\rm ED}})$ is the row-wise pair shown in the
cutoff/strength column; nonphonon rows have no phonon cutoff.
$\tau_{{\rm phys}}={format_number(threshold)}$,
$\Delta E_{{\rm primary}}=E_{{\rm alg}}(n_{{\rm ph}}^{{\rm algorithm}})-E_{{\rm ref}}(n_{{\rm ph}}^{{\rm ED}})$,
and $\delta E=\max(0,|\Delta E_{{\rm primary}}|-\tau_{{\rm phys}})$.\par\vspace{{6pt}}

\scriptsize
\setlength{{\tabcolsep}}{{2.0pt}}
\renewcommand{{\arraystretch}}{{1.08}}
\begin{{longtable}}{{p{{0.105\textwidth}}p{{0.105\textwidth}}p{{0.145\textwidth}}rrrrrr|rrrrrr}}
\caption{{Paper-I fixed-accuracy current-evidence table at $\tau_{{\rm phys}}={format_number(threshold)}$. Rows are grouped by Hamiltonian family, but every data row lists the individual Hamiltonian rather than a class-level aggregate. The cutoff/strength column reports weak/strong ordered cutoff pairs followed by $\lambda_w/\lambda_s$, with $\lambda$ defined row-wise in the preceding note. Each Hamiltonian repeats the full method set. Weak and strong regime blocks retain the approved quality/resource columns: target-excess energy $\delta E$, infidelity diagnostic, two-qubit count, two-qubit depth, circuit depth, and validated algorithmic estimator/probe work. \(N_{{2q}}\), \(D_{{2q}}\), and \(D_{{\rm circ}}\) are Qiskit-compiled ansatz-circuit metrics for the displayed current-best evidence row when available; \(\delta E=0\) identifies target hits, while nonzero \(\delta E\), NR, running, or failed statuses identify misses/incomplete evidence. \(S_{{\rm alg}}\) is the event-ledger algorithmic estimator/probe work count from the same evidence row when available; legacy normalized-work fields, raw shot totals, and grouped proxy fields are not promoted into this column.}}\label{{tab:paper_i_approved_fixed_accuracy}}\\
\toprule
Hamiltonian & cutoff/strength & Method
& \multicolumn{{6}}{{c}}{{weak regime}}
& \multicolumn{{6}}{{c}}{{strong regime}}\\
\cmidrule(lr){{4-9}}\cmidrule(lr){{10-15}}
&
&
& $\delta E$ & $1-F$ & $N_{{2q}}$ & $D_{{2q}}$ & $D_{{\rm circ}}$ & $S_{{\rm alg}}$
& $\delta E$ & $1-F$ & $N_{{2q}}$ & $D_{{2q}}$ & $D_{{\rm circ}}$ & $S_{{\rm alg}}$\\
\midrule
\endfirsthead
\toprule
Hamiltonian & cutoff/strength & Method
& \multicolumn{{6}}{{c}}{{weak regime}}
& \multicolumn{{6}}{{c}}{{strong regime}}\\
\cmidrule(lr){{4-9}}\cmidrule(lr){{10-15}}
&
&
& $\delta E$ & $1-F$ & $N_{{2q}}$ & $D_{{2q}}$ & $D_{{\rm circ}}$ & $S_{{\rm alg}}$
& $\delta E$ & $1-F$ & $N_{{2q}}$ & $D_{{2q}}$ & $D_{{\rm circ}}$ & $S_{{\rm alg}}$\\
\midrule
\endhead
\midrule
\multicolumn{{15}}{{r}}{{continued on next page}}\\
\endfoot
\bottomrule
\endlastfoot
{chr(10).join(body)}
\end{{longtable}}
{render_diagnostics(diagnostics)}
\end{{document}}
"""


def compile_latex(tex_path: Path, output_pdf: Path) -> None:
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("pdflatex") is not None:
        cmd = [
            "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-output-directory",
            str(tex_path.parent),
            str(tex_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        built_pdf = tex_path.with_suffix(".pdf")
    elif shutil.which("tectonic") is not None:
        cmd = [
            "tectonic",
            "--keep-logs",
            "--outdir",
            str(tex_path.parent),
            str(tex_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        built_pdf = tex_path.with_suffix(".pdf")
    else:
        raise RuntimeError("No LaTeX engine available; install pdflatex or tectonic")
    if built_pdf.resolve() != output_pdf.resolve():
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(built_pdf, output_pdf)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pdf", type=Path, default=DEFAULT_SOURCE_PDF)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--threshold", type=float, default=2e-4)
    parser.add_argument("--n-ph-algorithm", default="placeholder")
    parser.add_argument("--n-ph-ed", default="placeholder")
    parser.add_argument("--output-pdf", type=Path, default=DEFAULT_OUTPUT_PDF)
    parser.add_argument("--output-tex", type=Path, default=None)
    parser.add_argument(
        "--historical-pareto-json",
        type=Path,
        default=None,
        help="Optional Route-A SPSA SNAKE historical Pareto ledger; display representatives override weaker live SNAKE cost rows.",
    )
    parser.add_argument(
        "--display-overlay-json",
        type=Path,
        default=None,
        help="Optional approved-table cell overlay; recovered display cells override newer diagnostic rows.",
    )
    parser.add_argument("--audit-json", type=Path, default=None, help="Optional machine-readable source-map/audit JSON to write.")
    parser.add_argument("--no-compile", action="store_true", help="Write TeX but do not run pdflatex.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_pdf = args.source_pdf.resolve() if args.source_pdf else None
    summary_json = args.summary_json.resolve() if args.summary_json else None
    output_pdf = args.output_pdf.resolve()
    output_tex = args.output_tex.resolve() if args.output_tex else output_pdf.with_suffix(".tex")
    historical_pareto_json = args.historical_pareto_json.resolve() if args.historical_pareto_json else None
    display_overlay_json = args.display_overlay_json.resolve() if args.display_overlay_json else None
    audit_json = args.audit_json.resolve() if args.audit_json else None

    rows, diagnostics = load_rows_with_diagnostics(
        summary_json,
        args.threshold,
        historical_pareto_json=historical_pareto_json,
        display_overlay_json=display_overlay_json,
    )
    tex = render_tex(
        source_pdf,
        summary_json,
        args.threshold,
        rows,
        str(args.n_ph_algorithm),
        str(args.n_ph_ed),
        diagnostics,
    )
    output_tex.parent.mkdir(parents=True, exist_ok=True)
    output_tex.write_text(tex)
    if audit_json is not None:
        write_audit_manifest(
            audit_json,
            source_pdf=source_pdf,
            summary_json=summary_json,
            threshold=args.threshold,
            diagnostics=diagnostics,
        )

    if not args.no_compile:
        compile_latex(output_tex, output_pdf)

    print(f"Wrote TeX: {output_tex}")
    if not args.no_compile:
        print(f"Wrote PDF: {output_pdf}")
    if source_pdf:
        print(f"Source PDF: {source_pdf}")
    if historical_pareto_json:
        print(f"Historical Pareto JSON: {historical_pareto_json}")
    if display_overlay_json:
        print(f"Display overlay JSON: {display_overlay_json}")
    if summary_json:
        print(f"Source summary JSON: {summary_json}")
    if audit_json is not None:
        print(f"Wrote audit JSON: {audit_json}")


if __name__ == "__main__":
    main()
