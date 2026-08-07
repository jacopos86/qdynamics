#!/usr/bin/env python3
"""Build the corrected Paper-I HH parent-comparator page-13 support PDF.

The report is intentionally standalone: it does not edit ``Paper_I.tex`` and
does not launch scientific runs.  It consumes the corrected append-only ADAPT
and Geo-ADAPT result JSONs, retains the frozen visible SNAKE evidence, compiles
the corrected comparator plateau prefixes with the coefficient-aware Qiskit
route, and emits a manifest-first LaTeX PDF plus JSON/CSV sidecars.
"""

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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCHEMA = "paper_i_hh_corrected_parent_comparator_page13_report_v1"
STEM = "paper_i_hh_corrected_parent_comparators_page13_20260710"
QISKIT_COMPILE_CONVENTION = "table_i_basis_gate_transpile_v1"
GROUPED_EXACT_SYNTHESIS_ID = "commuting_pauli_or_active_support_unitary_exact_v1"
DEFAULT_WEAK_WEAK_ROOT = REPO_ROOT / (
    "raw_outputs/paper_i_hh_weak_weak_parent_only_comparator_fix_"
    "powell200_depth30_local_20260710_v1"
)
DEFAULT_CORRECTED_ROOT = REPO_ROOT / (
    "raw_outputs/paper_i_hh_six_regime_corrected_parent_comparators_"
    "powell200_depth30_local_20260710_v1"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output/pdf/paper_i_hh_corrected_parent_comparators_20260710"
SNAKE_PLOT_PROVENANCE = REPO_ROOT / (
    "MATH/paper_details/figures/paper_i_physical_lane_snake_duplicate_20260708/"
    "paper_i_physical_lane_snake_duplicate_20260708_append_parent_only_provenance.json"
)
PAPER_I_PROVENANCE = REPO_ROOT / (
    "MATH/paper_facing/paper_I_static_scaffold/provenance/"
    "Paper_I_provenance.json"
)
PAPER_I_TEX = REPO_ROOT / "MATH/paper_details/Paper_I.tex"
SNAKE_S_SHADOW = REPO_ROOT / (
    "output/pdf/paper_i_hh_s_accounting_shadow_20260709/"
    "paper_i_hh_s_accounting_shadow_20260709.json"
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
    "weak-weak": "Weak--weak",
    "intermediate-weak": "Intermediate--weak",
    "strong-weak": "Strong--weak",
    "weak-strong": "Weak--strong",
    "intermediate-strong": "Intermediate--strong",
    "strong-strong": "Strong--strong",
}
REGIME_PHYSICS = {
    "weak-weak": {"U_over_t": 0.25, "lambda": 0.25, "g_ep": 0.3535533905932738, "n_ph_work": 2, "n_ph_ref": 5, "same_cutoff_exact": -0.9183531194992409, "external_exact": -0.9183814647368327},
    "intermediate-weak": {"U_over_t": 1.25, "lambda": 0.25, "g_ep": 0.3535533905932738, "n_ph_work": 2, "n_ph_ref": 5, "same_cutoff_exact": -0.4949956391087026, "external_exact": -0.49500550257876347},
    "strong-weak": {"U_over_t": 8.0, "lambda": 0.25, "g_ep": 0.3535533905932738, "n_ph_work": 2, "n_ph_ref": 5, "same_cutoff_exact": 0.5264587007998435, "external_exact": 0.5264586847322538},
    "weak-strong": {"U_over_t": 0.25, "lambda": 1.25, "g_ep": 0.7905694150420949, "n_ph_work": 4, "n_ph_ref": 7, "same_cutoff_exact": -1.13857920035935, "external_exact": -1.138720638074999},
    "intermediate-strong": {"U_over_t": 1.25, "lambda": 1.25, "g_ep": 0.7905694150420949, "n_ph_work": 4, "n_ph_ref": 7, "same_cutoff_exact": -0.623910404831391, "external_exact": -0.6239396137518906},
    "strong-strong": {"U_over_t": 8.0, "lambda": 1.25, "g_ep": 0.7905694150420949, "n_ph_work": 4, "n_ph_ref": 7, "same_cutoff_exact": 0.5205762777107074, "external_exact": 0.5205762765682556},
}
METHOD_ORDER = ("snake", "geo", "append")
METHOD_DISPLAY = {"snake": "SNAKE", "geo": "Geo", "append": "Append"}
METHOD_ALGORITHM = {
    "geo": "static_geo_adapt_vqe",
    "append": "static_full_meta_append_adapt_vqe",
}
METHOD_STYLE = {
    "snake": {"color": "#E45756", "marker": "*"},
    "geo": {"color": "#54A24B", "marker": "^"},
    "append": {"color": "#4C78A8", "marker": "o"},
}
PLATEAU_REL_TOL = 0.10


@dataclass(frozen=True)
class CurvePoint:
    k: int
    error: float


@dataclass
class DisplayRow:
    regime: str
    method: str
    k_pl: int
    history_position: int | None
    logical_depth: int
    abs_delta_e: float
    n2q: int
    d2q: int
    dc: int
    s_alg: int
    s_components: dict[str, int]
    curve: list[CurvePoint]
    source_json: str
    source_sha256: str
    cost_source: str
    cost_metadata: dict[str, Any]
    validation: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "regime": self.regime,
            "method": self.method,
            "method_display": METHOD_DISPLAY[self.method],
            "k_pl": self.k_pl,
            "history_position": self.history_position,
            "logical_depth": self.logical_depth,
            "abs_delta_e": self.abs_delta_e,
            "N2q": self.n2q,
            "D2q": self.d2q,
            "Dc": self.dc,
            "S_alg": self.s_alg,
            "S_components": dict(self.s_components),
            "trajectory_points": [point.__dict__ for point in self.curve],
            "source_json": self.source_json,
            "source_sha256": self.source_sha256,
            "cost_source": self.cost_source,
            "cost_metadata": self.cost_metadata,
            "validation": self.validation,
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_source_path(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def corrected_result_path(
    regime: str,
    method: str,
    *,
    weak_weak_root: Path,
    corrected_root: Path,
) -> Path:
    root = weak_weak_root if regime == "weak-weak" else corrected_root / regime
    return root / method / "result/generic_static_single.json"


def _positive_error(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"Non-finite energy error: {value!r}")
    return max(float(value), 1.0e-16)


def generic_curve(payload: Mapping[str, Any]) -> list[CurvePoint]:
    result = payload["result"]
    history = list(result.get("adapt_history") or [])
    if not history:
        raise ValueError("Corrected comparator has no adapt_history")
    exact = float(result["same_cutoff_exact_gs_energy"])
    initial = abs(float(history[0]["energy_before"]) - exact)
    points = [CurvePoint(0, _positive_error(initial))]
    for index, row in enumerate(history):
        error = row.get("abs_delta_e_same_cutoff_after")
        if error is None:
            error = abs(float(row["energy_after"]) - exact)
        history_position = int(row.get("history_position", index))
        if history_position != index:
            raise ValueError(f"Non-contiguous history position {history_position} at index {index}")
        points.append(CurvePoint(index + 1, _positive_error(float(error))))
    return points


def first_plateau_history_row(
    payload: Mapping[str, Any],
    *,
    rel_tol: float = PLATEAU_REL_TOL,
    max_iterations: int | None = None,
) -> tuple[int, int, int, float]:
    """Return (history position, plotted k, logical depth, error)."""

    result = payload["result"]
    history = list(result.get("adapt_history") or [])
    if not history:
        raise ValueError("Cannot select plateau from empty history")
    if max_iterations is not None:
        horizon = int(max_iterations)
        if horizon <= 0:
            raise ValueError("max_iterations must be positive when provided")
        history = history[:horizon]
        if len(history) != horizon:
            raise ValueError(
                f"Requested plateau horizon {horizon} exceeds history length {len(history)}"
            )
    errors: list[float] = []
    for row in history:
        value = row.get("abs_delta_e_same_cutoff_after")
        if value is None:
            value = abs(float(row["energy_after"]) - float(result["same_cutoff_exact_gs_energy"]))
        errors.append(float(value))
    threshold = (1.0 + float(rel_tol)) * min(errors)
    for position, (row, error) in enumerate(zip(history, errors)):
        if error <= threshold:
            return position, position + 1, int(row["depth_after"]), float(error)
    raise AssertionError("Plateau selector did not select a history row")


def prefix_query_ledger(history: Sequence[Mapping[str, Any]], history_position: int) -> dict[str, int]:
    rows = list(history[: int(history_position) + 1])
    if len(rows) != int(history_position) + 1:
        raise IndexError(history_position)

    def total(key: str) -> int:
        return sum(int(row.get(key) or 0) for row in rows)

    components = {
        "N_H_outer": total("outer_hamiltonian_eval_count"),
        "N_H_refit": total("optimizer_nfev"),
        "N_grad_selector": total("selector_gradient_probe_count"),
        "N_grad_qngd": total("qngd_gradient_operator_probe_count_total"),
        "N_metric_selector": total("selector_metric_probe_count"),
        "N_metric_qngd": total("qngd_metric_operator_probe_count_total"),
        "N_other_quantum": total("N_other_quantum"),
    }
    components["S_alg"] = sum(components.values())
    return components


def reference_state_from_runtime_seed(result_path: Path, payload: Mapping[str, Any]) -> tuple[Any, Path]:
    import numpy as np

    raw = payload.get("runtime_seed_json")
    sibling_seed = result_path.parent / "runtime_seed.json"
    seed_path = Path(str(raw)) if raw else sibling_seed
    if not seed_path.is_absolute():
        seed_path = REPO_ROOT / seed_path
    if not seed_path.is_file() and sibling_seed.is_file():
        # Condor result payloads retain their remote repo-relative seed path.
        # A scoped fetch relocates the complete record directory, so the
        # sibling seed is the hashable local authority when that remote path is
        # absent from the active checkout.
        seed_path = sibling_seed
    seed = read_json(seed_path)
    state = seed.get("ansatz_input_state") or {}
    nq = int(state["nq_total"])
    vector = np.zeros(2**nq, dtype=complex)
    amplitudes = state.get("amplitudes_qn_to_q0") or {}
    for bitstring, amplitude in amplitudes.items():
        if len(str(bitstring)) != nq:
            raise ValueError(f"Reference bitstring width mismatch in {seed_path}")
        vector[int(str(bitstring), 2)] = complex(float(amplitude.get("re", 0.0)), float(amplitude.get("im", 0.0)))
    norm = float(np.linalg.norm(vector))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-10):
        raise ValueError(f"Reference state norm is {norm} in {seed_path}")
    return vector, seed_path


def reconstruct_generic_prefix_candidates(
    payload: Mapping[str, Any],
    *,
    history_position: int,
) -> tuple[list[Any], dict[str, Any]]:
    from pipelines.exact_bench.generic_static_adapt_variants import (
        _candidate_pauli_terms_payload,
        _expand_pool_with_shared_pauli_children,
        _get_config,
        _resolve_context_from_spec,
        build_full_meta_candidate_pool,
    )

    spec = payload["spec"]
    context = _resolve_context_from_spec(SimpleNamespace(base_pipeline_args=tuple(spec["base_pipeline_args"])))
    parent_pool = tuple(build_full_meta_candidate_pool(context, max_terms=None))
    result = payload["result"]
    pool, shared_meta = _expand_pool_with_shared_pauli_children(
        pool=parent_pool,
        context=context,
        config=_get_config(str(payload["algorithm_id"])),
        mode=str(result.get("shared_pauli_pool_mode") or "off"),
        symmetry_policy=str(result.get("shared_pauli_pool_symmetry_policy") or "off"),
        max_subset_size=int(result.get("shared_pauli_pool_max_subset_size") or 3),
        max_terms=None,
    )
    pool = tuple(pool)
    by_label = {str(candidate.label): candidate for candidate in pool}
    if len(by_label) != len(pool):
        raise ValueError("Full-meta pool contains duplicate labels")
    expected_labels = list(result.get("pool_labels") or [])
    if expected_labels and expected_labels != [str(candidate.label) for candidate in pool]:
        raise ValueError("Rebuilt full-meta pool label order differs from run payload")
    rebuilt_modes = {
        str(candidate.label): str(candidate.execution_mode or "termwise_product")
        for candidate in pool
    }
    rebuilt_paulis = {
        str(candidate.label): list(candidate.pauli_labels_exyz)
        for candidate in pool
    }
    rebuilt_supports = {
        str(candidate.label): list(candidate.support)
        for candidate in pool
    }
    if dict(result.get("pool_execution_modes") or {}) != rebuilt_modes:
        raise ValueError("Rebuilt full-meta pool execution modes differ from run payload")
    if dict(result.get("pool_pauli_labels_exyz") or {}) != rebuilt_paulis:
        raise ValueError("Rebuilt full-meta pool Pauli labels differ from run payload")
    if dict(result.get("pool_qubit_supports") or {}) != rebuilt_supports:
        raise ValueError("Rebuilt full-meta pool supports differ from run payload")
    if str(shared_meta.get("ordered_label_hash")) != str(result.get("shared_pauli_pool_ordered_label_hash")):
        raise ValueError("Rebuilt full-meta ordered-label hash differs from run payload")
    if str(shared_meta.get("ordered_pool_hash")) != str(result.get("shared_pauli_pool_ordered_pool_hash")):
        raise ValueError("Rebuilt coefficient-bearing pool hash differs from run payload")
    selected: list[Any] = []
    skipped_rows = 0
    appended_rows = 0
    history = list(result["adapt_history"])
    for row_index, row in enumerate(history[: int(history_position) + 1]):
        labels = list(row.get("selected_batch_labels") or [])
        modes = list(row.get("selected_batch_execution_modes") or [])
        skipped = bool(row.get("geo_immediate_repeat_skipped"))
        appended_count = int(row.get("appended_operator_count") or 0)
        if skipped:
            skipped_rows += 1
            if labels or appended_count:
                raise ValueError(f"Geo repeat-skip row {row_index} also appends a generator")
            continue
        if appended_count != len(labels):
            raise ValueError(f"Append-count mismatch at history row {row_index}")
        if not labels:
            continue
        candidates = []
        for label_index, label in enumerate(labels):
            if str(label) not in by_label:
                raise KeyError(f"Selected label missing from rebuilt pool: {label}")
            candidate = by_label[str(label)]
            if modes and str(modes[label_index]) != str(candidate.execution_mode):
                raise ValueError(f"Execution-mode mismatch for {label}")
            candidates.append(candidate)
        insertion = row.get("selected_insertion_position")
        if len(candidates) == 1 and insertion is not None:
            position = int(insertion)
            if not 0 <= position <= len(selected):
                raise IndexError(f"Invalid insertion position {position} at row {row_index}")
            selected.insert(position, candidates[0])
        else:
            selected.extend(candidates)
        appended_rows += 1
    expected_depth = int(history[int(history_position)]["depth_after"])
    if len(selected) != expected_depth:
        raise ValueError(f"Reconstructed depth {len(selected)} != history depth {expected_depth}")
    prefix_semantics = [
        {
            "label": str(candidate.label),
            "execution_mode": str(candidate.execution_mode or "termwise_product"),
            "pauli_labels_exyz": list(candidate.pauli_labels_exyz),
            "pauli_terms": _candidate_pauli_terms_payload(candidate),
        }
        for candidate in selected
    ]
    prefix_semantics_sha256 = hashlib.sha256(
        json.dumps(prefix_semantics, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return selected, {
        "rebuilt_pool_size": len(pool),
        "rebuilt_pool_ordered_label_hash": str(shared_meta.get("ordered_label_hash")),
        "rebuilt_pool_ordered_pool_hash": str(shared_meta.get("ordered_pool_hash")),
        "selected_generator_count": len(selected),
        "prefix_generator_semantics_sha256": prefix_semantics_sha256,
        "prefix_generator_semantics": prefix_semantics,
        "repeat_skip_rows_through_prefix": skipped_rows,
        "append_rows_through_prefix": appended_rows,
    }


def validate_corrected_payload(
    payload: Mapping[str, Any],
    *,
    regime: str,
    method: str,
    expected_iterations: int | None = None,
) -> dict[str, Any]:
    result = payload["result"]
    expected_algorithm = METHOD_ALGORITHM[method]
    physics = REGIME_PHYSICS[regime]
    base_args = list((payload.get("spec") or {}).get("base_pipeline_args") or [])

    def arg_value(flag: str) -> str | None:
        try:
            return str(base_args[base_args.index(flag) + 1])
        except (ValueError, IndexError):
            return None

    history = list(result.get("adapt_history") or [])
    observed_iterations = int(result.get("adapt_num_iterations") or 0)
    expected_horizon = int(
        expected_iterations
        if expected_iterations is not None
        else (result.get("adapt_max_iterations") or observed_iterations)
    )
    if expected_horizon <= 0:
        raise ValueError(f"Invalid expected iteration horizon for {regime}/{method}: {expected_horizon}")
    pool_size = int(result.get("pool_term_count") or 0)
    terminal_components = {
        "N_H_outer": int(result.get("N_H_outer_eval") or 0),
        "N_H_refit": int(result.get("N_H_refit_eval") or 0),
        "N_grad": int(result.get("N_grad") or 0),
        "N_metric": int(result.get("N_metric") or 0),
        "N_other": int(result.get("N_other_quantum") or 0),
    }
    history_ledger = prefix_query_ledger(history, len(history) - 1) if history else {}

    checks = {
        "status_success": str(payload.get("status")) in {"completed", "success", "ok"}
        and str(result.get("status")) in {"completed", "success", "ok"},
        "algorithm_id": str(payload.get("algorithm_id")) == expected_algorithm,
        "powell": str(result.get("adapt_optimizer_kind") or result.get("optimizer_kind")).upper() == "POWELL",
        "powell_maxiter_200": int(result.get("optimizer_maxiter") or 0) == 200,
        "fixed_horizon_scans": observed_iterations == expected_horizon,
        "history_has_expected_rows": len(history) == expected_horizon,
        "full_meta": str(result.get("base_pool_name") or result.get("pool_name")) == "full_meta",
        "full_meta_unfiltered": str(result.get("hh_adaptive_pool_profile")) == "full_meta_unfiltered",
        "hva_included": not bool(result.get("hh_full_meta_minus_hva_active")),
        "parent_pool_size": int(result.get("pool_term_count") or 0)
        == (97 if int(physics["n_ph_work"]) == 2 else 103),
        "runtime_split_disabled": not bool(result.get("generic_adapt_runtime_split_enabled")),
        "shared_pauli_pool_disabled": not bool(result.get("shared_pauli_pool_enabled")),
        "same_cutoff_primary": str(result.get("primary_energy_metric")) == "same_cutoff_abs_delta_e",
        "terminal_diagnostics_excluded": not bool(result.get("adapt_terminal_diagnostic_queries_in_S_alg")),
        "seed_42": int(result.get("seed") or -1) == 42,
        "physics_L2": int(arg_value("--L") or -1) == 2,
        "physics_u": math.isclose(float(arg_value("--u") or "nan"), float(physics["U_over_t"]), rel_tol=0.0, abs_tol=1.0e-12),
        "physics_g_ep": math.isclose(float(arg_value("--g-ep") or "nan"), float(physics["g_ep"]), rel_tol=0.0, abs_tol=1.0e-12),
        "physics_n_ph": int(arg_value("--n-ph-max") or -1) == int(physics["n_ph_work"]),
        "physics_no_drive": math.isclose(float(arg_value("--dv") or "nan"), 0.0, rel_tol=0.0, abs_tol=1.0e-15),
        "same_cutoff_exact_lock": math.isclose(float(result.get("same_cutoff_exact_gs_energy")), float(physics["same_cutoff_exact"]), rel_tol=0.0, abs_tol=1.0e-12),
        "external_exact_lock": math.isclose(float(result.get("exact_reference_energy")), float(physics["external_exact"]), rel_tol=0.0, abs_tol=1.0e-12),
        "external_cutoff_lock": int(result.get("exact_reference_n_ph_max") or -1) == int(physics["n_ph_ref"]),
        "terminal_s_component_sum": sum(terminal_components.values()) == int(result.get("S_alg") or -1),
        "history_outer_matches_terminal": history_ledger.get("N_H_outer") == terminal_components["N_H_outer"],
        "history_refit_matches_terminal": history_ledger.get("N_H_refit") == terminal_components["N_H_refit"],
        "history_gradient_matches_terminal": (
            history_ledger.get("N_grad_selector", 0) + history_ledger.get("N_grad_qngd", 0)
            == terminal_components["N_grad"]
        ),
        "history_metric_matches_terminal": (
            history_ledger.get("N_metric_selector", 0) + history_ledger.get("N_metric_qngd", 0)
            == terminal_components["N_metric"]
        ),
        "history_s_matches_terminal": history_ledger.get("S_alg") == int(result.get("S_alg") or -1),
        "each_row_charges_one_outer_h": all(
            int(row.get("outer_hamiltonian_eval_count") or 0) == 1 for row in history
        ),
        "each_row_charges_full_pool_gradient": all(
            int(row.get("selector_gradient_probe_count") or 0) == pool_size for row in history
        ),
        "full_pool_gradient_charge": terminal_components["N_grad"] == expected_horizon * pool_size,
        "one_outer_h_per_scan": terminal_components["N_H_outer"] == expected_horizon,
    }
    if method == "geo":
        skip_count = sum(bool(row.get("geo_immediate_repeat_skipped")) for row in history)
        checks.update(
            geo_immediate_repeat_blocked=bool(result.get("geo_immediate_repeat_blocked")),
            geo_post_score_pre_append=(
                str(result.get("geo_immediate_repeat_policy_stage"))
                == "post_full_pool_selection_skip_append"
            ),
            geo_metric_full_pair_charge=(
                terminal_components["N_metric"]
                == expected_horizon * pool_size * (pool_size + 1) // 2
            ),
            geo_each_row_charges_full_metric=(
                all(
                    int(row.get("selector_metric_probe_count") or 0)
                    == pool_size * (pool_size + 1) // 2
                    for row in history
                )
            ),
            geo_skip_depth_identity=(
                int(result.get("adapt_depth_reached") or -1) + skip_count
                == expected_horizon
            ),
            geo_skip_rows_append_nothing=all(
                (not bool(row.get("geo_immediate_repeat_skipped")))
                or (
                    int(row.get("appended_operator_count") or 0) == 0
                    and not list(row.get("selected_batch_labels") or [])
                )
                for row in history
            ),
        )
    else:
        checks.update(
            append_only=bool(result.get("adapt_append_only")),
            selection_with_replacement=bool(result.get("adapt_selection_with_replacement")),
            append_metric_zero=terminal_components["N_metric"] == 0,
            append_each_row_metric_zero=all(
                int(row.get("selector_metric_probe_count") or 0) == 0 for row in history
            ),
            append_depth_matches_horizon=(
                int(result.get("adapt_depth_reached") or -1) == expected_horizon
            ),
            append_one_generator_per_scan=all(
                int(row.get("appended_operator_count") or 0) == 1 for row in history
            ),
        )
    if not all(checks.values()):
        failed = [key for key, value in checks.items() if not value]
        raise ValueError(f"Corrected payload validation failed for {regime}/{method}: {failed}")
    return checks


def build_corrected_row(
    regime: str,
    method: str,
    result_path: Path,
    *,
    expected_iterations: int | None = None,
    plateau_horizon: int | None = None,
    grouped_exact_max_active_qubits: int = 5,
) -> DisplayRow:
    from pipelines.exact_bench.generic_static_adapt_variants import _ansatz_terms_from_candidates
    from pipelines.exact_bench.table_i_qiskit_resource_compile import (
        TABLE_I_GROUPED_EXACT_SYNTHESIS_ID,
        TABLE_I_QISKIT_COMPILE_CONVENTION,
        TableIQiskitCompileConfig,
        compile_table_i_ansatz_terms,
    )

    if TABLE_I_QISKIT_COMPILE_CONVENTION != QISKIT_COMPILE_CONVENTION:
        raise ValueError("Qiskit compile convention drifted from the report contract")
    if TABLE_I_GROUPED_EXACT_SYNTHESIS_ID != GROUPED_EXACT_SYNTHESIS_ID:
        raise ValueError("Grouped-exact synthesis ID drifted from the report contract")

    payload = read_json(result_path)
    checks = validate_corrected_payload(
        payload,
        regime=regime,
        method=method,
        expected_iterations=expected_iterations,
    )
    history_position, k_pl, logical_depth, error = first_plateau_history_row(
        payload,
        max_iterations=plateau_horizon,
    )
    curve = generic_curve(payload)
    if not math.isclose(curve[k_pl].error, error, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(f"Marker/table error mismatch for {regime}/{method}")
    ledger = prefix_query_ledger(payload["result"]["adapt_history"], history_position)
    selected, reconstruction = reconstruct_generic_prefix_candidates(payload, history_position=history_position)
    reference_state, runtime_seed_path = reference_state_from_runtime_seed(result_path, payload)
    qiskit = compile_table_i_ansatz_terms(
        ops=_ansatz_terms_from_candidates(selected),
        num_qubits=int(payload["result"]["num_qubits"]),
        reference_state=reference_state,
        source_kind="qiskit_corrected_parent_comparator_plateau_prefix",
        config=TableIQiskitCompileConfig(
            grouped_exact_max_active_qubits=int(grouped_exact_max_active_qubits)
        ),
    )
    required_costs = ("compiled_count_2q_total", "compiled_depth_2q_total", "compiled_depth_total")
    if not bool(qiskit.get("compiled_resource_qiskit_validated")) or any(qiskit.get(key) is None for key in required_costs):
        raise ValueError(f"Qiskit prefix compile is not validated for {regime}/{method}")
    qiskit.update(
        prefix_history_position=history_position,
        prefix_k=k_pl,
        prefix_logical_depth=logical_depth,
        reconstruction=reconstruction,
        runtime_seed_json=rel(runtime_seed_path),
        runtime_seed_sha256=sha256(runtime_seed_path),
    )
    return DisplayRow(
        regime=regime,
        method=method,
        k_pl=k_pl,
        history_position=history_position,
        logical_depth=logical_depth,
        abs_delta_e=error,
        n2q=int(qiskit["compiled_count_2q_total"]),
        d2q=int(qiskit["compiled_depth_2q_total"]),
        dc=int(qiskit["compiled_depth_total"]),
        s_alg=int(ledger["S_alg"]),
        s_components=ledger,
        curve=curve,
        source_json=rel(result_path),
        source_sha256=sha256(result_path),
        cost_source="coefficient_aware_qiskit_plateau_prefix",
        cost_metadata=qiskit,
        validation=checks,
    )


def _snake_curve_from_source(curve_source: Path, table_source: Path) -> list[CurvePoint]:
    curve_payload = read_json(curve_source)
    table_payload = read_json(table_source)
    table_history = list((table_payload.get("adapt_vqe") or {}).get("history") or [])
    if not table_history:
        raise ValueError(f"SNAKE table source has no history: {table_source}")
    initial = _positive_error(float(table_history[0]["delta_abs_prev"]))
    if "points" in curve_payload:
        points = [CurvePoint(int(row["k"]), _positive_error(float(row["abs_delta_e"]))) for row in curve_payload["points"]]
        points.sort(key=lambda point: point.k)
        if not points or points[0].k != 0:
            raise ValueError(f"Stitched SNAKE curve does not begin at k=0: {curve_source}")
        points[0] = CurvePoint(0, initial)
        return points
    history = list((curve_payload.get("adapt_vqe") or {}).get("history") or [])
    if not history:
        raise ValueError(f"SNAKE curve source has no history: {curve_source}")
    return [CurvePoint(0, initial)] + [
        CurvePoint(index + 1, _positive_error(float(row["delta_abs_current"])))
        for index, row in enumerate(history)
    ]


def active_page13_snake_cells() -> dict[str, dict[str, int | float]]:
    """Read the six currently visible SNAKE mini-table rows from Paper_I.tex."""

    source = PAPER_I_TEX.read_text(encoding="utf-8")
    label_index = source.index(r"\label{fig:hh_main_results_composite}")
    start_index = source.rfind(r"\onecolumngrid", 0, label_index)
    if start_index < 0:
        raise ValueError("Could not isolate the active page-13 composite in Paper_I.tex")
    block = source[start_index:label_index]
    pattern = re.compile(
        r"^SNAKE\s*&\s*(\d+)\s*&\s*([0-9.eE+\-]+)\s*&\s*([0-9,]+)\s*&\s*"
        r"([0-9,]+)\s*&\s*([0-9,]+)\s*&\s*([0-9,]+)\s*\\\\\s*$",
        re.MULTILINE,
    )
    matches = list(pattern.finditer(block))
    if len(matches) != len(REGIME_ORDER):
        raise ValueError(f"Expected six active SNAKE table rows, found {len(matches)}")
    cells: dict[str, dict[str, int | float]] = {}
    for regime, match in zip(REGIME_ORDER, matches, strict=True):
        cells[regime] = {
            "k_pl": int(match.group(1)),
            "abs_delta_e": float(match.group(2)),
            "N2q": int(match.group(3).replace(",", "")),
            "D2q": int(match.group(4).replace(",", "")),
            "Dc": int(match.group(5).replace(",", "")),
            "S_alg": int(match.group(6).replace(",", "")),
        }
    return cells


def build_snake_rows() -> list[DisplayRow]:
    plot_provenance = read_json(SNAKE_PLOT_PROVENANCE)
    paper_provenance = read_json(PAPER_I_PROVENANCE)
    s_shadow = read_json(SNAKE_S_SHADOW)
    comparison_path = resolve_source_path(paper_provenance["comparison_json"])
    if sha256(comparison_path) != str(paper_provenance["comparison_json_sha256"]):
        raise ValueError("Frozen SNAKE Qiskit comparison-sidecar hash mismatch")
    comparison = read_json(comparison_path)
    plot_rows = {str(row["regime"]): row for row in plot_provenance["plots"]}
    cells = paper_provenance["changed_snake_cells"]
    shadow_rows = {
        str(row["regime"]): row
        for row in s_shadow["rows"]
        if str(row.get("method")) == "SNAKE"
    }
    active_cells = active_page13_snake_cells()
    rows: list[DisplayRow] = []
    for regime in REGIME_ORDER:
        cell = cells[regime]
        table_source = resolve_source_path(cell["source_json"])
        if sha256(table_source) != str(cell["source_json_sha256"]):
            raise ValueError(f"Frozen SNAKE table-source hash mismatch for {regime}")
        method_prov = next(row for row in plot_rows[regime]["methods"] if row["role_key"] == "snake")
        curve_source = resolve_source_path(method_prov["source_json"])
        if sha256(curve_source) != str(method_prov["source_sha256"]):
            raise ValueError(f"Frozen SNAKE curve-source hash mismatch for {regime}")
        curve = _snake_curve_from_source(curve_source, table_source)
        curve_map = {point.k: point.error for point in curve}
        k_pl = int(cell["k_pl"])
        error = float(cell["abs_delta_e"])
        if k_pl not in curve_map or not math.isclose(curve_map[k_pl], error, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"Frozen SNAKE marker/table mismatch for {regime}")
        shadow = shadow_rows[regime]
        shadow_source = resolve_source_path(shadow["source_json"])
        if (
            shadow_source.resolve() != table_source.resolve()
            or str(shadow.get("source_sha256")) != sha256(table_source)
            or str(shadow.get("mechanism_status")) != "ok_phase2_window_formula_v1"
            or int(shadow["k_pl"]) != k_pl
            or not bool(shadow.get("mechanism_formula_components_sum_to_s"))
        ):
            raise ValueError(f"SNAKE S-shadow mismatch for {regime}")
        comparison_row = next(
            (
                row
                for row in comparison["rows"]
                if str(row.get("regime")) == regime
                and str(row.get("source_json_sha256")) == sha256(table_source)
            ),
            None,
        )
        if comparison_row is None:
            raise ValueError(f"SNAKE Qiskit comparison row missing for {regime}")
        if not (
            str(comparison_row.get("qiskit_cost_status")) == "ok"
            and str(comparison_row.get("qiskit_compile_convention")) == QISKIT_COMPILE_CONVENTION
            and str(comparison_row.get("reference_state_status")) == "statevector_from_ansatz_input_state"
            and int(comparison_row.get("k_pl")) == k_pl
            and int(comparison_row.get("n2q")) == int(cell["N2q"])
            and int(comparison_row.get("d2q")) == int(cell["D2q"])
            and int(comparison_row.get("dc")) == int(cell["Dc"])
        ):
            raise ValueError(f"SNAKE Qiskit comparison row is not validated for {regime}")
        active = active_cells[regime]
        active_matches_locked_cell = (
            int(active["k_pl"]) == k_pl
            and math.isclose(float(active["abs_delta_e"]), error, rel_tol=5.0e-4, abs_tol=5.0e-12)
            and int(active["N2q"]) == int(cell["N2q"])
            and int(active["D2q"]) == int(cell["D2q"])
            and int(active["Dc"]) == int(cell["Dc"])
        )
        if not active_matches_locked_cell:
            raise ValueError(f"Active Paper-I SNAKE cell drifted from its locked evidence for {regime}")
        if int(active["S_alg"]) != int(shadow["mechanism_formula_s"]):
            raise ValueError(f"Active Paper-I SNAKE S does not match the corrected mechanism ledger for {regime}")
        s_components = {
            "N_grad": int(shadow["mechanism_formula_grad"]),
            "N_metric": int(shadow["mechanism_formula_metric"]),
            "N_H_refit": int(shadow["mechanism_formula_h_refit"]),
            "S_alg": int(shadow["mechanism_formula_s"]),
        }
        rows.append(
            DisplayRow(
                regime=regime,
                method="snake",
                k_pl=k_pl,
                history_position=k_pl - 1,
                logical_depth=k_pl,
                abs_delta_e=error,
                n2q=int(cell["N2q"]),
                d2q=int(cell["D2q"]),
                dc=int(cell["Dc"]),
                s_alg=int(shadow["mechanism_formula_s"]),
                s_components=s_components,
                curve=curve,
                source_json=rel(table_source),
                source_sha256=sha256(table_source),
                cost_source="frozen_visible_paper_i_qiskit_prefix",
                cost_metadata={
                    "curve_source_json": rel(curve_source),
                    "curve_source_sha256": sha256(curve_source),
                    "table_source_json": rel(table_source),
                    "table_source_sha256": sha256(table_source),
                    "qiskit_comparison_json": rel(comparison_path),
                    "qiskit_comparison_json_sha256": sha256(comparison_path),
                    "qiskit_compile_convention": str(comparison_row["qiskit_compile_convention"]),
                    "qiskit_cost_status": str(comparison_row["qiskit_cost_status"]),
                    "reference_state_status": str(comparison_row["reference_state_status"]),
                    "source_git_head": comparison_row.get("local_git_head"),
                    "marker_policy": "marker_at_visible_table_prefix",
                    "initial_point_policy": (
                        "report_normalization_replaces_source_k0_with_table_source_first_delta_abs_prev; "
                        "all_source_points_at_k_ge_1_are_retained"
                    ),
                    "s_source_json": rel(SNAKE_S_SHADOW),
                    "s_source_sha256": sha256(SNAKE_S_SHADOW),
                    "s_shadow_row_status": str(shadow.get("row_status")),
                    "s_shadow_mechanism_status": str(shadow.get("mechanism_status")),
                    "active_paper_i_tex": rel(PAPER_I_TEX),
                    "active_paper_i_tex_sha256": sha256(PAPER_I_TEX),
                    "active_visible_s": int(active["S_alg"]),
                },
                validation={
                    "frozen_source_hashes_match": True,
                    "marker_table_same_prefix": True,
                    "active_visible_cell_matches_locked_evidence": True,
                    "active_visible_s_matches_corrected_mechanism_ledger": True,
                },
            )
        )
    return rows


def plot_regime(regime: str, rows: Sequence[DisplayRow], *, figure_dir: Path, stem: str) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.15, 1.82))
    regime_rows = {row.method: row for row in rows if row.regime == regime}
    for method in METHOD_ORDER:
        row = regime_rows[method]
        style = METHOD_STYLE[method]
        xs = [point.k for point in row.curve]
        ys = [point.error for point in row.curve]
        ax.plot(xs, ys, color=style["color"], linestyle="-", linewidth=1.45, alpha=0.96)
        ax.scatter(
            [row.k_pl],
            [row.abs_delta_e],
            color=style["color"],
            marker=style["marker"],
            s=48 if method == "snake" else 27,
            edgecolor="black",
            linewidth=0.35,
            zorder=4,
        )
    ax.set_yscale("log")
    ax.set_xlim(left=0)
    ax.set_xlabel("ADAPT outer iteration $k$", fontsize=7.5)
    ax.set_ylabel(r"$|\Delta E|$", fontsize=7.5)
    ax.set_title(REGIME_DISPLAY[regime], fontsize=8.5)
    ax.tick_params(axis="both", labelsize=6.5)
    ax.grid(True, which="major", alpha=0.24, linewidth=0.45)
    handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_STYLE[method]["color"],
            linestyle="-",
            marker=METHOD_STYLE[method]["marker"],
            markersize=6 if method == "snake" else 4.5,
            label=f"{METHOD_DISPLAY[method]} ($k_{{\\rm pl}}={regime_rows[method].k_pl}$)",
        )
        for method in METHOD_ORDER
    ]
    ax.legend(handles=handles, loc="best", fontsize=4.8, frameon=False, handlelength=1.6)
    fig.tight_layout(pad=0.42)
    safe = regime.replace("-", "_")
    png = figure_dir / f"{stem}__{safe}.png"
    pdf = figure_dir / f"{stem}__{safe}.pdf"
    fig.savefig(png, dpi=260)
    fig.savefig(pdf)
    plt.close(fig)
    return {
        "regime": regime,
        "png": rel(png),
        "png_sha256": sha256(png),
        "pdf": rel(pdf),
        "pdf_sha256": sha256(pdf),
    }


def format_error(value: float) -> str:
    return f"{float(value):.3e}"


def latex_escape(value: str) -> str:
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
    return "".join(replacements.get(char, char) for char in str(value))


def latex_graphics_path(path: Path) -> str:
    raw = path.as_posix()
    if any(character in raw for character in ("%", "{", "}")):
        raise ValueError(f"TeX-unsafe graphics path: {raw}")
    return r"\detokenize{" + raw + "}"


def _table_tex(regime_rows: Mapping[str, DisplayRow]) -> str:
    lines = [
        r"\begin{tabular*}{\linewidth}{@{}l@{\extracolsep{\fill}}rrrrrr@{}}",
        r"\toprule",
        r"Method & $k_{\rm pl}$ & $|\Delta E|$ & $N_{2q}$ & $D_{2q}$ & $D_c$ & $S$\\",
        r"\midrule",
    ]
    for method in METHOD_ORDER:
        row = regime_rows[method]
        lines.append(
            f"{METHOD_DISPLAY[method]} & {row.k_pl} & {format_error(row.abs_delta_e)} & "
            f"{row.n2q:,} & {row.d2q:,} & {row.dc:,} & {row.s_alg:,} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular*}"])
    return "\n".join(lines)


def write_tex(
    path: Path,
    *,
    rows: Sequence[DisplayRow],
    figures: Sequence[Mapping[str, Any]],
    report_json: Path,
    report_csv: Path,
    generated_utc: str,
) -> None:
    by_regime = {regime: {row.method: row for row in rows if row.regime == regime} for regime in REGIME_ORDER}
    figure_by_regime = {str(row["regime"]): row for row in figures}
    source_comment = json.dumps(
        {
            "schema": SCHEMA,
            "report_json": rel(report_json),
            "report_csv": rel(report_csv),
            "run_class": "candidate",
            "manuscript_edited": False,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    manifest_rows = []
    for regime in REGIME_ORDER:
        append_row = by_regime[regime]["append"]
        geo_row = by_regime[regime]["geo"]
        physics = REGIME_PHYSICS[regime]
        manifest_rows.append(
            f"{latex_escape(REGIME_DISPLAY[regime])} & {physics['U_over_t']:g} & {physics['lambda']:g} & "
            f"{physics['g_ep']:.6f} & {physics['n_ph_work']}/{physics['n_ph_ref']} & "
            f"{physics['same_cutoff_exact']:.9f} & "
            f"{physics['external_exact']:.9f} & {append_row.source_sha256[:12]} & "
            f"{geo_row.source_sha256[:12]} \\\\"
        )
    panels: list[str] = []
    for index, regime in enumerate(REGIME_ORDER):
        figure = resolve_source_path(str(figure_by_regime[regime]["pdf"]))
        panels.extend(
            [
                r"\begin{minipage}[t]{0.322\textwidth}",
                r"\centering",
                f"\\includegraphics[width=\\linewidth]{{{latex_graphics_path(figure)}}}",
                r"\par\vspace{0.2ex}",
                _table_tex(by_regime[regime]),
                r"\end{minipage}",
            ]
        )
        if index in {0, 1, 3, 4}:
            panels.append(r"\hfill")
        elif index == 2:
            panels.append(r"\par\vspace{1.2ex}")
    tex = rf"""\documentclass[10pt]{{article}}
\usepackage[letterpaper,margin=0.38in]{{geometry}}
\usepackage{{booktabs,graphicx,caption,microtype,xcolor}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern}}
\pagestyle{{empty}}
\setlength{{\parindent}}{{0pt}}
% BEGIN_MACHINE_READABLE_CORRECTED_PARENT_COMPARATOR_REPORT
% {source_comment}
% END_MACHINE_READABLE_CORRECTED_PARENT_COMPARATOR_REPORT
\begin{{document}}
\begin{{center}}
{{\Large\bfseries Paper-I Hubbard--Holstein corrected parent-comparator reruns}}\\[0.4ex]
{{\small Manifest and page-13-style convergence/cost composite}}
\end{{center}}
\small
\begin{{tabular}}{{@{{}}p{{0.24\textwidth}}p{{0.71\textwidth}}@{{}}}}
\toprule
Field & Locked value\\
\midrule
Generated UTC & {latex_escape(generated_utc)}\\
Run class & candidate; no manuscript source was edited\\
Visible target & Paper-I page 13, six Hubbard--Holstein convergence panels and prefix-cost mini-tables\\
Corrected methods & Append-only ADAPT-VQE and the corrected Geo-ADAPT comparator; both use parent macro generators only\\
Pool & full-meta, HVA included, no runtime Pauli-child split, no shared Pauli pool\\
Model & $L=2$ open-boundary Hubbard--Holstein, $t=\omega_0=1$, blocked ordering, binary bosons, $N_\uparrow=N_\downarrow=1$, no drive\\
Reference state & resolved benchmark occupation basis state with one spin-up fermion, one spin-down fermion, and zero phonons; exact amplitudes and hashes are stored per corrected row\\
Outer policy & 30 fixed selector scans; exact/reference energies are reporting-only\\
Inner optimizer & Powell for every corrected comparator; maximum 200 iterations; seed 42\\
Geo repeat policy & score the full pool, then block an immediate repeat before append; skipped rows retain query work\\
Geo route scope & projected Fubini--Study natural-gradient selector; configured comparator deviations are the problem-local full-meta pool and Powell inner refit\\
Primary error & same-working-cutoff $|E_k-E_0|$; every curve begins at the reference-state error at $k=0$\\
Plateau policy & first completed history row within 10\% of that corrected trajectory's best error\\
Estimator-query cost & $S=N_{{H,\mathrm{{outer}}}}+N_{{H,\mathrm{{refit}}}}+N_{{\mathrm{{grad}}}}+N_{{\mathrm{{metric}}}}+N_{{\mathrm{{other}}}}$, accumulated through the same displayed prefix\\
Shot interpretation & $S$ counts logical estimator/probe events; it is not a physical hardware-shot allocation\\
Qiskit route & {latex_escape(QISKIT_COMPILE_CONVENTION)}; optimization level 0, transpiler seed 7, reference state included\\
Grouped-exact route & {latex_escape(GROUPED_EXACT_SYNTHESIS_ID)}; commuting sums use exact Pauli rotations and noncommuting blocks use exact active-support unitaries\\
SNAKE evidence & Frozen visible Paper-I trajectory points for $k\geq1$ and Qiskit cells; $k=0$ is normalized to the stored reference-state error, and stitched markers are placed at the visible table prefix\\
Machine sidecars & JSON and CSV beside this PDF; the JSON records full paths, hashes, trajectories, prefix ledgers, and synthesis metadata\\
\bottomrule
\end{{tabular}}

\vspace{{1.0ex}}
{{\bfseries Corrected source fingerprints}}\\[-0.3ex]
{{\footnotesize SHA-256 prefixes are shown here; full hashes and paths are in the JSON sidecar.}}
\begin{{center}}
\resizebox{{0.98\linewidth}}{{!}}{{%
\begin{{tabular}}{{lrrrcrrcc}}
\toprule
Regime & $U/t$ & $\lambda$ & $g_{{\rm ep}}$ & $M/M_{{\rm ref}}$ & $E_0(M)$ & $E_0(M_{{\rm ref}})$ & Append JSON & Geo JSON\\
\midrule
{chr(10).join(manifest_rows)}
\bottomrule
\end{{tabular}}
}}
\end{{center}}

\vfill
{{\footnotesize The table values on the next page are prefix-aligned: the plotted marker, energy error, compiled circuit, and estimator-query ledger all refer to one history row.}}
\clearpage
\begin{{center}}
\scriptsize
\setlength{{\tabcolsep}}{{0pt}}
\renewcommand{{\arraystretch}}{{0.92}}
{chr(10).join(panels)}
\captionof{{figure}}{{Hubbard--Holstein error-versus-iteration trajectories and Qiskit plateau-prefix costs after the corrected parent-generator Append-ADAPT and Geo-ADAPT reruns. The top row uses $M=2$ and the bottom row uses $M=4$. All curves are solid and carry one marker at the reported prefix $k_{{\rm pl}}$. Mini-tables report the same-prefix energy error, compiled two-qubit count $N_{{2q}}$, two-qubit depth $D_{{2q}}$, compiled depth $D_c$, and logical estimator-query count $S$.}}
\end{{center}}
\end{{document}}
"""
    path.write_text(tex, encoding="utf-8")


def write_csv_rows(path: Path, rows: Sequence[DisplayRow]) -> None:
    fields = [
        "regime",
        "method",
        "k_pl",
        "history_position",
        "logical_depth",
        "abs_delta_e",
        "N2q",
        "D2q",
        "Dc",
        "S_alg",
        "N_H_outer",
        "N_H_refit",
        "N_grad_selector",
        "N_grad_qngd",
        "N_metric_selector",
        "N_metric_qngd",
        "N_other_quantum",
        "source_json",
        "source_sha256",
        "cost_source",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "regime": row.regime,
                    "method": METHOD_DISPLAY[row.method],
                    "k_pl": row.k_pl,
                    "history_position": "" if row.history_position is None else row.history_position,
                    "logical_depth": row.logical_depth,
                    "abs_delta_e": f"{row.abs_delta_e:.17g}",
                    "N2q": row.n2q,
                    "D2q": row.d2q,
                    "Dc": row.dc,
                    "S_alg": row.s_alg,
                    "N_H_outer": row.s_components.get("N_H_outer", 0),
                    "N_H_refit": row.s_components.get("N_H_refit", 0),
                    "N_grad_selector": row.s_components.get("N_grad_selector", row.s_components.get("N_grad", 0)),
                    "N_grad_qngd": row.s_components.get("N_grad_qngd", 0),
                    "N_metric_selector": row.s_components.get("N_metric_selector", row.s_components.get("N_metric", 0)),
                    "N_metric_qngd": row.s_components.get("N_metric_qngd", 0),
                    "N_other_quantum": row.s_components.get("N_other_quantum", 0),
                    "source_json": row.source_json,
                    "source_sha256": row.source_sha256,
                    "cost_source": row.cost_source,
                }
            )


def compile_latex(tex_path: Path) -> Path:
    executable = shutil.which("latexmk")
    if executable:
        command = [executable, "-pdf", "-interaction=nonstopmode", "-halt-on-error", tex_path.name]
    else:
        executable = shutil.which("tectonic")
        if not executable:
            raise RuntimeError("Neither latexmk nor tectonic is available")
        command = [executable, "--keep-logs", "--outdir", str(tex_path.parent), tex_path.name]
    completed = subprocess.run(command, cwd=tex_path.parent, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"LaTeX build failed:\n{completed.stdout}\n{completed.stderr}")
    pdf_path = tex_path.with_suffix(".pdf")
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)
    return pdf_path


def validate_rows(rows: Sequence[DisplayRow]) -> dict[str, Any]:
    keys = [(row.regime, row.method) for row in rows]
    expected = [(regime, method) for regime in REGIME_ORDER for method in METHOD_ORDER]
    checks = {
        "exactly_18_rows": len(rows) == 18,
        "all_regime_method_pairs": sorted(keys) == sorted(expected),
        "all_costs_nonnegative": all(min(row.n2q, row.d2q, row.dc, row.s_alg) >= 0 for row in rows),
        "all_markers_on_curves": all(
            any(point.k == row.k_pl and math.isclose(point.error, row.abs_delta_e, rel_tol=0.0, abs_tol=1.0e-12) for point in row.curve)
            for row in rows
        ),
        "all_curves_begin_at_zero": all(row.curve and row.curve[0].k == 0 for row in rows),
        "all_s_component_sums_match": all(sum(value for key, value in row.s_components.items() if key != "S_alg") == row.s_alg for row in rows),
    }
    if not all(checks.values()):
        failed = [key for key, value in checks.items() if not value]
        raise ValueError(f"Report row validation failed: {failed}")
    return checks


def build(
    *,
    weak_weak_root: Path,
    corrected_root: Path,
    output_dir: Path,
    stem: str = STEM,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_utc = utc_now()
    visible_source_map = corrected_root / "visible_source_map.json"
    batch_manifest = corrected_root / "batch_manifest.json"
    resolver_traces = sorted((corrected_root / "source_locks").glob("*.json"))
    if not visible_source_map.is_file() or not batch_manifest.is_file() or len(resolver_traces) != 12:
        raise ValueError("Corrected six-regime source map, batch manifest, or 12 resolver locks are missing")
    rows = build_snake_rows()
    for regime in REGIME_ORDER:
        for method in ("geo", "append"):
            result_path = corrected_result_path(
                regime,
                method,
                weak_weak_root=weak_weak_root,
                corrected_root=corrected_root,
            )
            rows.append(build_corrected_row(regime, method, result_path))
    rows.sort(key=lambda row: (REGIME_ORDER.index(row.regime), METHOD_ORDER.index(row.method)))
    row_validation = validate_rows(rows)
    figure_dir = output_dir / "figures"
    figures = [plot_regime(regime, rows, figure_dir=figure_dir, stem=stem) for regime in REGIME_ORDER]
    report_json = output_dir / f"{stem}.json"
    report_csv = output_dir / f"{stem}.csv"
    report_tex = output_dir / f"{stem}.tex"
    write_csv_rows(report_csv, rows)
    write_tex(
        report_tex,
        rows=rows,
        figures=figures,
        report_json=report_json,
        report_csv=report_csv,
        generated_utc=generated_utc,
    )
    report_pdf = compile_latex(report_tex)
    payload = {
        "schema": SCHEMA,
        "generated_utc": generated_utc,
        "run_class": "candidate",
        "manuscript_edited": False,
        "target": {"pdf": "Paper_I.pdf", "page": 13, "figure_label": "fig:hh_main_results_composite"},
        "contract": {
            "corrected_methods": ["static_full_meta_append_adapt_vqe", "static_geo_adapt_vqe"],
            "optimizer": "POWELL",
            "optimizer_maxiter": 200,
            "selector_scans": 30,
            "pool": "full_meta",
            "parent_macro_generators_only": True,
            "hva_included": True,
            "runtime_pauli_child_split": False,
            "geo_immediate_repeat": "post_score_pre_append_block",
            "geo_selector": "full_pool_projected_fubini_study_natural_gradient",
            "geo_configured_deviations": [
                "problem_local_full_meta_pool_instead_of_excitation_pool",
                "powell_inner_optimizer_instead_of_fixed_step_qngd",
            ],
            "primary_error": "abs_delta_e_same_cutoff",
            "plateau_policy": "first_history_row_with_error_le_1p10_times_trajectory_minimum",
            "qiskit_compile_convention": QISKIT_COMPILE_CONVENTION,
            "grouped_exact_synthesis_id": GROUPED_EXACT_SYNTHESIS_ID,
            "S_formula": "N_H_outer + N_H_refit + N_grad + N_metric + N_other_quantum",
            "regime_physics": REGIME_PHYSICS,
        },
        "source_locks": {
            "weak_weak_root": rel(weak_weak_root),
            "corrected_root": rel(corrected_root),
            "visible_source_map": rel(visible_source_map),
            "visible_source_map_sha256": sha256(visible_source_map),
            "batch_manifest": rel(batch_manifest),
            "batch_manifest_sha256": sha256(batch_manifest),
            "resolver_traces": [
                {"path": rel(path), "sha256": sha256(path)} for path in resolver_traces
            ],
            "snake_plot_provenance": rel(SNAKE_PLOT_PROVENANCE),
            "snake_plot_provenance_sha256": sha256(SNAKE_PLOT_PROVENANCE),
            "paper_i_provenance": rel(PAPER_I_PROVENANCE),
            "paper_i_provenance_sha256": sha256(PAPER_I_PROVENANCE),
            "snake_qiskit_comparison_json": str(read_json(PAPER_I_PROVENANCE)["comparison_json"]),
            "snake_qiskit_comparison_json_sha256": str(
                read_json(PAPER_I_PROVENANCE)["comparison_json_sha256"]
            ),
            "active_paper_i_tex": rel(PAPER_I_TEX),
            "active_paper_i_tex_sha256": sha256(PAPER_I_TEX),
            "snake_s_shadow": rel(SNAKE_S_SHADOW),
            "snake_s_shadow_sha256": sha256(SNAKE_S_SHADOW),
        },
        "rows": [row.as_dict() for row in rows],
        "figures": figures,
        "validation": row_validation,
        "artifacts": {
            "pdf": rel(report_pdf),
            "pdf_sha256": sha256(report_pdf),
            "tex": rel(report_tex),
            "tex_sha256": sha256(report_tex),
            "csv": rel(report_csv),
            "csv_sha256": sha256(report_csv),
            "json": rel(report_json),
        },
    }
    write_json(report_json, payload)
    # A JSON document cannot contain a stable hash of itself.  Return the
    # final file hash to the caller without rewriting it into the document.
    payload["artifacts"]["json_sha256"] = sha256(report_json)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weak-weak-root", type=Path, default=DEFAULT_WEAK_WEAK_ROOT)
    parser.add_argument("--corrected-root", type=Path, default=DEFAULT_CORRECTED_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default=STEM)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build(
        weak_weak_root=args.weak_weak_root.resolve(),
        corrected_root=args.corrected_root.resolve(),
        output_dir=args.output_dir.resolve(),
        stem=str(args.stem),
    )
    print(json.dumps({"status": "ok", **payload["artifacts"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
