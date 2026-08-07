#!/usr/bin/env python3
"""Build L=3 HH intermediate-weak physical-lane support artifacts.

This script consumes one completed local Paper-I Hubbard--Holstein static ADAPT
result JSON and emits a standalone support bundle: same-cutoff error trajectory,
error-vs-iteration plot, Qiskit plateau resource table, provenance JSON, and a
LaTeX-built support PDF.  It intentionally does not edit manuscript sources or
promote artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.exact_bench.snake_table_i_measurement_work import (  # noqa: E402
    snake_algorithmic_work_from_payload,
)
from pipelines.exact_bench.table_i_qiskit_resource_compile import (  # noqa: E402
    TABLE_I_QISKIT_COMPILE_CONVENTION,
    TableICompileUnavailable,
    compile_table_i_pauli_label_groups,
)


DEFAULT_RESULT_JSON = (
    REPO_ROOT
    / "raw_outputs/paper_i_hh_l3_intermediate_weak_physical_lanes_raw42_21_frac0p4375_x3_fullreopt_powell200_nobatch_20260709_v1"
    / "intermediate_weak/json/result.json"
)
DEFAULT_OUT_DIR = REPO_ROOT / "output/pdf/paper_i_hh_l3_intermediate_weak_physical_lanes_support_20260709"
STEM = "paper_i_hh_l3_intermediate_weak_physical_lanes_support_20260709"
PLOT_STEM = "paper_i_hh_l3_intermediate_weak_error_vs_iteration_20260709"
PLATEAU_REL_TOL = 0.10
SCHEMA_VERSION = "paper_i_hh_l3_intermediate_weak_support_artifacts_v1"


@dataclass(frozen=True)
class TrajectoryPoint:
    history_position: int
    k_iter: int
    abs_delta_e: float
    error_source_field: str
    energy: float | None
    best_available_gain: float | None
    selected_op: str


@dataclass(frozen=True)
class PlateauSelection:
    history_position: int
    k_plateau: int
    abs_delta_e: float
    source_rule: str
    best_error: float
    threshold: float
    rel_tol: float
    canonical_plateau: bool


@dataclass
class CostRow:
    row_label: str
    source_kind: str
    status: str
    k_plateau: int | None
    history_position: int | None
    delta_E: float | None
    N2q: int | None
    D2q: int | None
    D_circ: int | None
    S: float | None
    S_alg_status: str
    qiskit_cost_status: str
    qiskit_compile_convention: str
    qiskit_version: str | None
    logical_operator_count: int | None
    runtime_rotation_count: int | None
    num_qubits: int | None
    note: str


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except Exception:
        return str(path)


def _git_value(args: Sequence[str]) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return ""
    return proc.stdout.strip()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _int_or_none(value: Any) -> int | None:
    try:
        out = int(value)
    except Exception:
        return None
    return out


def _adapt(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    adapt = payload.get("adapt_vqe")
    if isinstance(adapt, Mapping):
        return adapt
    return payload


def _history(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = _adapt(payload).get("history")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        return [row for row in raw if isinstance(row, Mapping)]
    return []


def _history_error(row: Mapping[str, Any], exact_energy: float | None = None) -> tuple[float | None, str]:
    for key in (
        "delta_abs_current",
        "abs_delta_e_same_cutoff_after",
        "abs_delta_e_after",
        "benchmark_target_abs_delta_e_current",
        "exact_final_state_benchmark_target_abs_delta_e_current",
        "exact_abs_delta_e_from_final_state",
        "abs_delta_e",
    ):
        value = _float_or_none(row.get(key))
        if value is not None and value >= 0.0:
            return value, key
    if exact_energy is not None:
        for key in ("energy_after_opt", "energy_after", "energy"):
            energy = _float_or_none(row.get(key))
            if energy is not None:
                return abs(float(energy) - float(exact_energy)), f"computed_abs_{key}_minus_exact_gs_energy"
    return None, "missing_same_cutoff_error"


def _display_k(row: Mapping[str, Any], fallback: int) -> tuple[int, str]:
    for key in ("depth_cumulative", "ansatz_depth", "depth", "k", "iteration"):
        parsed = _int_or_none(row.get(key))
        if parsed is not None and parsed > 0:
            return int(parsed), key
    return int(fallback), "history_position_fallback"


def _extract_trajectory(payload: Mapping[str, Any]) -> tuple[list[TrajectoryPoint], dict[str, Any]]:
    adapt = _adapt(payload)
    exact = _float_or_none(adapt.get("exact_gs_energy"))
    points: list[TrajectoryPoint] = []
    k_sources: list[str] = []
    error_sources: list[str] = []
    for idx, row in enumerate(_history(payload), start=1):
        err, err_key = _history_error(row, exact_energy=exact)
        if err is None:
            continue
        k_iter, k_key = _display_k(row, idx)
        k_sources.append(k_key)
        error_sources.append(err_key)
        points.append(
            TrajectoryPoint(
                history_position=int(idx),
                k_iter=int(k_iter),
                abs_delta_e=float(err),
                error_source_field=str(err_key),
                energy=_float_or_none(row.get("energy_after_opt") or row.get("energy_after") or row.get("energy")),
                best_available_gain=_float_or_none(row.get("best_available_gain")),
                selected_op=str(row.get("selected_op") or row.get("selected_operator") or ""),
            )
        )
    meta = {
        "status": "ok" if points else "missing_trajectory_points",
        "point_count": len(points),
        "k_source_fields": sorted(set(k_sources)),
        "error_source_fields": sorted(set(error_sources)),
        "same_cutoff_policy": "same_cutoff_fields_only_with_energy_minus_exact_gs_fallback",
    }
    return points, meta


def _select_plateau(points: Sequence[TrajectoryPoint], override_k: int | None = None) -> PlateauSelection:
    clean = [pt for pt in points if pt.abs_delta_e >= 0.0 and math.isfinite(pt.abs_delta_e)]
    if not clean:
        raise RuntimeError("No positive finite trajectory points available for plateau selection")
    best = min(pt.abs_delta_e for pt in clean)
    threshold = best * (1.0 + PLATEAU_REL_TOL)
    if override_k is not None:
        for pt in clean:
            if int(pt.k_iter) == int(override_k):
                return PlateauSelection(
                    history_position=int(pt.history_position),
                    k_plateau=int(pt.k_iter),
                    abs_delta_e=float(pt.abs_delta_e),
                    source_rule="cli_override_k_plateau",
                    best_error=float(best),
                    threshold=float(threshold),
                    rel_tol=PLATEAU_REL_TOL,
                    canonical_plateau=False,
                )
        raise RuntimeError(f"--plateau-k={override_k} did not match any trajectory k_iter")
    for pt in clean:
        if pt.abs_delta_e <= threshold:
            return PlateauSelection(
                history_position=int(pt.history_position),
                k_plateau=int(pt.k_iter),
                abs_delta_e=float(pt.abs_delta_e),
                source_rule="first_prefix_with_error_within_10pct_of_best_support_rule",
                best_error=float(best),
                threshold=float(threshold),
                rel_tol=PLATEAU_REL_TOL,
                canonical_plateau=False,
            )
    terminal = clean[-1]
    return PlateauSelection(
        history_position=int(terminal.history_position),
        k_plateau=int(terminal.k_iter),
        abs_delta_e=float(terminal.abs_delta_e),
        source_rule="terminal_fallback_no_point_within_threshold",
        best_error=float(best),
        threshold=float(threshold),
        rel_tol=PLATEAU_REL_TOL,
        canonical_plateau=False,
    )


def _complex_from_json(value: Any) -> complex | None:
    if isinstance(value, Mapping):
        real = _float_or_none(value.get("re", value.get("real")))
        imag = _float_or_none(value.get("im", value.get("imag")))
        if real is None and imag is None:
            return None
        return complex(0.0 if real is None else real, 0.0 if imag is None else imag)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) and len(value) >= 2:
        real = _float_or_none(value[0])
        imag = _float_or_none(value[1])
        if real is not None or imag is not None:
            return complex(0.0 if real is None else real, 0.0 if imag is None else imag)
    scalar = _float_or_none(value)
    return None if scalar is None else complex(float(scalar), 0.0)


def _statevector_from_state_payload(state: Mapping[str, Any] | None) -> tuple[np.ndarray | None, str, int | None]:
    if not isinstance(state, Mapping):
        return None, "missing_state_payload", None
    nq = _int_or_none(state.get("nq_total") or state.get("num_qubits") or state.get("nq"))
    amps = state.get("amplitudes_qn_to_q0")
    if nq is None or nq <= 0 or not isinstance(amps, Mapping):
        return None, "missing_statevector_amplitudes", nq
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
        return None, "empty_statevector_amplitudes", nq
    return vec, "statevector_from_ansatz_input_state", int(nq)


def _runtime_terms_group(block: Mapping[str, Any]) -> list[str]:
    terms = block.get("runtime_terms_exyz")
    if not isinstance(terms, Sequence) or isinstance(terms, (str, bytes, bytearray)):
        return []
    group: list[str] = []
    for term in terms:
        if isinstance(term, Mapping):
            pauli = str(term.get("pauli_exyz") or term.get("pauli") or "").strip().lower()
            if pauli:
                group.append(pauli)
        elif isinstance(term, str):
            text = term.strip().lower()
            if text:
                group.append(text)
    return group


def _prefix_pauli_groups(payload: Mapping[str, Any], history_position: int) -> tuple[list[list[str]] | None, dict[str, Any]]:
    adapt = _adapt(payload)
    history = _history(payload)
    parameterization = adapt.get("parameterization")
    blocks_raw = parameterization.get("blocks") if isinstance(parameterization, Mapping) else None
    blocks = [block for block in blocks_raw if isinstance(block, Mapping)] if isinstance(blocks_raw, Sequence) and not isinstance(blocks_raw, (str, bytes, bytearray)) else []
    if not blocks:
        return None, {"status": "missing_parameterization_blocks"}
    hp = int(history_position)
    if hp <= 0 or hp > len(blocks):
        return None, {"status": "history_position_out_of_parameterization_range", "history_position": hp, "block_count": len(blocks)}
    selected_labels = [str(row.get("selected_op") or "") for row in history[:hp]]
    block_labels = [str(block.get("candidate_label") or "") for block in blocks[:hp]]
    label_alignment_ok = selected_labels == block_labels
    operator_labels_raw = adapt.get("operators")
    operator_labels = [str(item) for item in operator_labels_raw] if isinstance(operator_labels_raw, Sequence) and not isinstance(operator_labels_raw, (str, bytes, bytearray)) else []
    terminal_operator_alignment_ok = operator_labels[: len(block_labels)] == block_labels if operator_labels else None
    if not label_alignment_ok:
        return None, {
            "status": "selected_history_prefix_mismatch_parameterization_prefix",
            "history_position": hp,
            "first_mismatches": [
                {"history_position": idx + 1, "selected_label": selected_labels[idx], "block_label": block_labels[idx]}
                for idx in range(min(len(selected_labels), len(block_labels)))
                if selected_labels[idx] != block_labels[idx]
            ][:10],
        }
    if terminal_operator_alignment_ok is False:
        return None, {
            "status": "terminal_operator_prefix_mismatch_parameterization_prefix",
            "history_position": hp,
        }
    groups: list[list[str]] = []
    bad_blocks: list[int] = []
    for idx, block in enumerate(blocks[:hp], start=1):
        group = _runtime_terms_group(block)
        if not group:
            bad_blocks.append(idx)
        groups.append(group)
    if bad_blocks:
        return None, {
            "status": "bad_or_missing_runtime_terms",
            "bad_history_positions": bad_blocks[:10],
            "bad_count": len(bad_blocks),
        }
    prune_rows = _prune_acceptance_rows(payload)
    prefix_prune_rows = [row for row in prune_rows if int(row.get("history_position", 0)) <= hp]
    active_label_set = {label for label in block_labels if label}
    selected_positions_by_label: dict[str, list[int]] = {}
    for pos, label in enumerate(block_labels, start=1):
        if label:
            selected_positions_by_label.setdefault(label, []).append(pos)
    overlapping_prunes: list[dict[str, Any]] = []
    later_reselected_prune_labels: list[dict[str, Any]] = []
    for row in prefix_prune_rows:
        row_hp = int(row.get("history_position", 0))
        labels = set(str(label) for label in row.get("accepted_decision_labels", []) if str(label))
        selected_label = str(row.get("selected_label") or "")
        if selected_label:
            labels.add(selected_label)
        prior_or_current_overlap = sorted(
            label
            for label in labels
            if any(pos <= row_hp for pos in selected_positions_by_label.get(label, []))
        )
        future_reselected = sorted(
            label
            for label in labels
            if label in active_label_set and label not in prior_or_current_overlap
        )
        if prior_or_current_overlap:
            item = dict(row)
            item["active_prefix_prior_or_current_overlap_labels"] = prior_or_current_overlap
            overlapping_prunes.append(item)
        if future_reselected:
            item = dict(row)
            item["future_reselected_labels"] = future_reselected
            item["future_reselected_positions"] = {
                label: [pos for pos in selected_positions_by_label.get(label, []) if pos > row_hp]
                for label in future_reselected
            }
            later_reselected_prune_labels.append(item)
    if overlapping_prunes:
        return None, {
            "status": "accepted_prune_targets_existing_selected_prefix_instance_replay_required",
            "history_position": hp,
            "overlapping_prune_rows": overlapping_prunes[:10],
            "overlapping_prune_count": len(overlapping_prunes),
        }
    return groups, {
        "status": "ok",
        "source": "adapt_vqe.parameterization.blocks_prefix_terminal_aligned_with_history",
        "history_position": hp,
        "logical_operator_count": len(groups),
        "selected_history_prefix_matches_parameterization_prefix": True,
        "terminal_operator_prefix_matches_parameterization_prefix": terminal_operator_alignment_ok,
        "accepted_prune_count_before_or_at_prefix": len(prefix_prune_rows),
        "accepted_prune_rows_before_or_at_prefix": prefix_prune_rows,
        "accepted_prune_existing_selected_instance_overlap_count": 0,
        "accepted_prune_later_reselected_rows": later_reselected_prune_labels,
        "accepted_prune_later_reselected_count": len(later_reselected_prune_labels),
        "prune_handling_note": "Qiskit prefix uses final parameterization block prefix after verified selected-history and terminal-operator alignment. Accepted prune labels before the plateau either do not appear in the compiled prefix or were reselected later than the prune row; no accepted prune targets an already-selected compiled-prefix instance.",
    }


def _num_qubits_from_groups(groups: Sequence[Sequence[str]], payload: Mapping[str, Any], fallback_nq: int | None = None) -> int | None:
    for group in groups:
        for label in group:
            text = str(label)
            if text:
                return len(text)
    if fallback_nq is not None and fallback_nq > 0:
        return int(fallback_nq)
    adapt = _adapt(payload)
    for source in (adapt, payload.get("hamiltonian") if isinstance(payload.get("hamiltonian"), Mapping) else None):
        if not isinstance(source, Mapping):
            continue
        for key in ("num_qubits", "nq", "nq_total", "total_qubits"):
            parsed = _int_or_none(source.get(key))
            if parsed is not None and parsed > 0:
                return int(parsed)
    return None


def _prune_acceptance_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(_history(payload), start=1):
        block = row.get("post_admission_prune")
        if not isinstance(block, Mapping):
            continue
        accepted = _int_or_none(block.get("accepted_count")) or 0
        if accepted <= 0:
            continue
        decisions = block.get("decisions")
        accepted_labels: list[str] = []
        accepted_indices: list[int] = []
        if isinstance(decisions, Sequence) and not isinstance(decisions, (str, bytes, bytearray)):
            for decision in decisions:
                if not isinstance(decision, Mapping) or not bool(decision.get("accepted")):
                    continue
                label = str(decision.get("label") or "")
                if label:
                    accepted_labels.append(label)
                parsed_idx = _int_or_none(decision.get("index"))
                if parsed_idx is not None:
                    accepted_indices.append(int(parsed_idx))
        rows.append(
            {
                "history_position": int(idx),
                "accepted_count": int(accepted),
                "selected_index": _int_or_none(block.get("selected_index")),
                "selected_label": str(block.get("selected_label") or ""),
                "accepted_decision_indices": accepted_indices,
                "accepted_decision_labels": accepted_labels,
                "prune_mode": str(block.get("prune_mode") or ""),
                "deletion_authority": str(block.get("deletion_authority") or ""),
            }
        )
    return rows


def _compile_qiskit_costs(payload: Mapping[str, Any], selection: PlateauSelection, result_json: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    groups, prefix_meta = _prefix_pauli_groups(payload, selection.history_position)
    if groups is None:
        return {
            "compiled_circuit_stats_status": prefix_meta.get("status", "prefix_groups_unavailable"),
            "compiled_count_2q_total": None,
            "compiled_depth_2q_total": None,
            "compiled_depth_total": None,
            "compile_convention": TABLE_I_QISKIT_COMPILE_CONVENTION,
            "qiskit_version": None,
            "logical_operator_count": None,
            "runtime_rotation_count": None,
            "num_qubits": None,
        }, prefix_meta
    ref_state, ref_status, ref_nq = _statevector_from_state_payload(payload.get("ansatz_input_state") if isinstance(payload, Mapping) else None)
    if ref_state is None:
        meta = dict(prefix_meta)
        meta.update(reference_state_status=ref_status, reference_state_included=False)
        return {
            "compiled_circuit_stats_status": "reference_state_required_for_qiskit_compile",
            "compiled_count_2q_total": None,
            "compiled_depth_2q_total": None,
            "compiled_depth_total": None,
            "compile_convention": TABLE_I_QISKIT_COMPILE_CONVENTION,
            "qiskit_version": None,
            "logical_operator_count": len(groups),
            "runtime_rotation_count": None,
            "num_qubits": ref_nq,
        }, meta
    nq = _num_qubits_from_groups(groups, payload, fallback_nq=ref_nq)
    if nq is None:
        out = {
            "compiled_circuit_stats_status": "missing_num_qubits",
            "compiled_count_2q_total": None,
            "compiled_depth_2q_total": None,
            "compiled_depth_total": None,
            "compile_convention": TABLE_I_QISKIT_COMPILE_CONVENTION,
            "qiskit_version": None,
            "logical_operator_count": len(groups),
            "runtime_rotation_count": None,
            "num_qubits": None,
        }
        meta = dict(prefix_meta)
        meta.update(reference_state_status=ref_status)
        return out, meta
    try:
        costs = compile_table_i_pauli_label_groups(
            pauli_label_groups=groups,
            num_qubits=int(nq),
            reference_state=ref_state,
            source_kind=f"l3_intermediate_weak_plateau_prefix_history_position_{selection.history_position}:{_rel(result_json)}",
        )
        costs = dict(costs)
        required = ("compiled_count_2q_total", "compiled_depth_2q_total", "compiled_depth_total")
        missing_required = [key for key in required if _int_or_none(costs.get(key)) is None]
        if missing_required and costs.get("compiled_circuit_stats_status") == "ok":
            costs["compiled_circuit_stats_status"] = "missing_required_qiskit_cost_fields"
            costs["missing_required_qiskit_cost_fields"] = missing_required
    except TableICompileUnavailable as exc:
        costs = {
            "compiled_circuit_stats_status": exc.status,
            "compiled_count_2q_total": None,
            "compiled_depth_2q_total": None,
            "compiled_depth_total": None,
            "compile_convention": TABLE_I_QISKIT_COMPILE_CONVENTION,
            "qiskit_version": None,
            "logical_operator_count": len(groups),
            "runtime_rotation_count": None,
            "num_qubits": int(nq),
            "error": exc.reason,
        }
    except Exception as exc:  # defensive: Qiskit environments vary locally
        costs = {
            "compiled_circuit_stats_status": f"qiskit_compile_failed:{type(exc).__name__}",
            "compiled_count_2q_total": None,
            "compiled_depth_2q_total": None,
            "compiled_depth_total": None,
            "compile_convention": TABLE_I_QISKIT_COMPILE_CONVENTION,
            "qiskit_version": None,
            "logical_operator_count": len(groups),
            "runtime_rotation_count": None,
            "num_qubits": int(nq),
            "error": str(exc),
        }
    meta = dict(prefix_meta)
    meta.update(
        reference_state_status=ref_status,
        reference_state_included=bool(ref_state is not None),
        num_qubits=int(nq),
        pauli_group_count=len(groups),
        pauli_term_count=sum(len(group) for group in groups),
    )
    return costs, meta


def _compute_s_alg(payload: Mapping[str, Any], selection: PlateauSelection, result_json: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        components, meta = snake_algorithmic_work_from_payload(
            payload,
            scope="display_prefix",
            history_position=int(selection.history_position),
            source_label=_rel(result_json),
        )
        return dict(components), dict(meta)
    except Exception as exc:
        return {
            "S_alg": None,
            "S_alg_status": f"failed:{type(exc).__name__}",
            "error": str(exc),
        }, {
            "status": f"failed:{type(exc).__name__}",
            "error": str(exc),
            "scope": "display_prefix",
            "history_position": int(selection.history_position),
        }


def _cost_row(
    selection: PlateauSelection,
    qiskit_costs: Mapping[str, Any],
    s_components: Mapping[str, Any],
) -> CostRow:
    s_value = _float_or_none(s_components.get("S_alg"))
    s_status = str(s_components.get("S_alg_status") or ("ok" if s_value is not None else "missing"))
    if s_status == "ok" and s_value is None:
        s_status = "missing_S_alg_value"
    n2q = _int_or_none(qiskit_costs.get("compiled_count_2q_total"))
    d2q = _int_or_none(qiskit_costs.get("compiled_depth_2q_total"))
    dcirc = _int_or_none(qiskit_costs.get("compiled_depth_total"))
    q_status = str(qiskit_costs.get("compiled_circuit_stats_status") or "missing")
    if q_status == "ok" and any(value is None for value in (n2q, d2q, dcirc)):
        q_status = "missing_required_qiskit_cost_fields"
    return CostRow(
        row_label="HH L=3 intermediate-weak physical-lane SNAKE plateau",
        source_kind="completed_local_result_json_support_artifact",
        status="ok" if q_status == "ok" and s_status == "ok" else "partial",
        k_plateau=int(selection.k_plateau),
        history_position=int(selection.history_position),
        delta_E=float(selection.abs_delta_e),
        N2q=n2q,
        D2q=d2q,
        D_circ=dcirc,
        S=s_value,
        S_alg_status=s_status,
        qiskit_cost_status=q_status,
        qiskit_compile_convention=str(qiskit_costs.get("compile_convention") or TABLE_I_QISKIT_COMPILE_CONVENTION),
        qiskit_version=None if qiskit_costs.get("qiskit_version") is None else str(qiskit_costs.get("qiskit_version")),
        logical_operator_count=_int_or_none(qiskit_costs.get("logical_operator_count")),
        runtime_rotation_count=_int_or_none(qiskit_costs.get("runtime_rotation_count")),
        num_qubits=_int_or_none(qiskit_costs.get("num_qubits")),
        note="Plateau is an inferred support-artifact prefix, not a manuscript-selected canonical plateau.",
    )


def _fmt_err(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{float(value):.3e}"


def _fmt_int(value: int | float | None) -> str:
    if value is None:
        return "--"
    try:
        return str(int(round(float(value))))
    except Exception:
        return "--"


def _fmt_float(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{float(value):.6g}"


def _tex_escape(value: Any) -> str:
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


def _tex_path(path: Path | str) -> str:
    text = str(path)
    delim = "|" if "|" not in text else "!"
    return rf"\path{delim}{text}{delim}"


def _write_trajectory_csv(points: Sequence[TrajectoryPoint], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "history_position",
                "k_iter",
                "abs_delta_e",
                "error_source_field",
                "energy",
                "best_available_gain",
                "selected_op",
            ],
        )
        writer.writeheader()
        for point in points:
            writer.writerow(asdict(point))


def _write_cost_csv(row: CostRow, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(row).keys()))
        writer.writeheader()
        writer.writerow(asdict(row))


def _write_error_plot(points: Sequence[TrajectoryPoint], selection: PlateauSelection, out_dir: Path) -> tuple[Path, Path]:
    x = [pt.k_iter for pt in points]
    y = [pt.abs_delta_e for pt in points]
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    ax.plot(x, y, color="#C44E52", linewidth=1.55, marker="o", markersize=3.0, label="L=3 intermediate-weak")
    ax.scatter(
        [selection.k_plateau],
        [selection.abs_delta_e],
        marker="*",
        s=95,
        color="#111111",
        edgecolor="white",
        linewidth=0.45,
        zorder=5,
        label=r"$k_{\rm plateau}$",
    )
    ax.axhline(selection.threshold, color="#777777", linewidth=0.75, linestyle="--", alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("ADAPT iteration $k$")
    ax.set_ylabel(r"$|E_k-E_{\rm exact}|$")
    ax.set_title("HH L=3 intermediate-weak")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
    ax.grid(True, which="both", axis="y", linestyle=":", linewidth=0.55, alpha=0.55)
    ax.grid(True, which="major", axis="x", linestyle=":", linewidth=0.4, alpha=0.35)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.tight_layout(pad=0.8)
    png = out_dir / f"{PLOT_STEM}.png"
    pdf = out_dir / f"{PLOT_STEM}.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def _latex_document(
    *,
    result_json: Path,
    result_sha: str | None,
    payload: Mapping[str, Any],
    selection: PlateauSelection,
    cost_row: CostRow,
    qiskit_meta: Mapping[str, Any],
    s_meta: Mapping[str, Any],
    plot_pdf: Path,
    trajectory_csv: Path,
    cost_csv: Path,
    provenance_json: Path,
) -> str:
    settings = payload.get("settings") if isinstance(payload.get("settings"), Mapping) else {}
    adapt = _adapt(payload)
    ham = payload.get("hamiltonian") if isinstance(payload.get("hamiltonian"), Mapping) else {}
    manifest_rows = [
        ("Schema", SCHEMA_VERSION),
        ("Generated UTC", datetime.now(timezone.utc).isoformat()),
        ("Source result JSON", "intermediate_weak/json/result.json; full path in provenance JSON"),
        ("Source SHA-256", result_sha or ""),
        ("Problem", settings.get("problem") or ham.get("problem") or "hh"),
        ("L", settings.get("L") or ham.get("L")),
        ("U/t", settings.get("u") or ham.get("u")),
        ("g", settings.get("g_ep") or ham.get("g_ep")),
        ("n_ph_max", settings.get("n_ph_max") or ham.get("n_ph_max")),
        ("ADAPT pool", settings.get("adapt_pool")),
        ("Refit policy", settings.get("adapt_reopt_policy")),
        ("Batching", settings.get("phase2_enable_batching")),
        ("Physical lanes", settings.get("physical_operator_lanes") or settings.get("physical_lane_shortlist_enabled")),
        ("Phase-I raw cap", settings.get("phase1_shortlist_size")),
        ("Phase-II raw cap", settings.get("phase2_shortlist_size")),
        ("Phase-II raw fraction", settings.get("phase2_shortlist_fraction")),
        ("Aggressiveness", settings.get("physical_lane_shortlist_aggressiveness")),
        ("Terminal depth", adapt.get("ansatz_depth")),
        ("Terminal same-cutoff error", adapt.get("abs_delta_e")),
        ("Manuscript edits", "false"),
        ("Remote actions", "false"),
    ]
    manifest_table = "\n".join(
        rf"{_tex_escape(label)} & {_tex_escape('' if value is None else value)} \\" for label, value in manifest_rows
    )
    cost_table = rf"""
\begin{{tabular}}{{lrrrrrr}}
\toprule
Row & $k_{{\rm plateau}}$ & $\Delta E$ & $N_{{2q}}$ & $D_{{2q}}$ & $D_{{\rm circ}}$ & $S_{{\rm alg}}$ \\
\midrule
{_tex_escape(cost_row.row_label)} & {_fmt_int(cost_row.k_plateau)} & {_fmt_err(cost_row.delta_E)} & {_fmt_int(cost_row.N2q)} & {_fmt_int(cost_row.D2q)} & {_fmt_int(cost_row.D_circ)} & {_fmt_int(cost_row.S)} \\
\bottomrule
\end{{tabular}}
""".strip()
    qiskit_notes = [
        ("Qiskit status", cost_row.qiskit_cost_status),
        ("Qiskit convention", cost_row.qiskit_compile_convention),
        ("Qiskit version", cost_row.qiskit_version or ""),
        ("Compiled circuit scope", qiskit_meta.get("compiled_circuit_scope") or qiskit_meta.get("source") or ""),
        ("Reference state", qiskit_meta.get("reference_state_status") or ""),
        ("$S_{\\rm alg}$ status", cost_row.S_alg_status),
        ("$S_{\\rm alg}$ scope", s_meta.get("scope") or "display_prefix"),
        ("Plateau rule", selection.source_rule),
        ("Best error", _fmt_err(selection.best_error)),
        ("10\\% threshold", _fmt_err(selection.threshold)),
    ]
    note_table = "\n".join(
        rf"{label} & {_tex_escape('' if value is None else value)} \\" for label, value in qiskit_notes
    )
    rel_plot = plot_pdf.name
    return rf"""\documentclass[10pt]{{article}}
\usepackage[margin=0.7in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{hyperref}}
\usepackage{{url}}
\usepackage[T1]{{fontenc}}
\hypersetup{{colorlinks=true,linkcolor=black,urlcolor=blue}}
\begin{{document}}
\section*{{Parameter manifest}}
\small
\begin{{tabular}}{{@{{}}p{{0.28\linewidth}}p{{0.66\linewidth}}@{{}}}}
\toprule
Field & Value \\
\midrule
{manifest_table}
\bottomrule
\end{{tabular}}

\section*{{Error versus ADAPT iteration}}
\begin{{center}}
\includegraphics[width=0.92\linewidth]{{{rel_plot}}}
\end{{center}}

\section*{{Qiskit plateau cost table}}
\small
{cost_table}

\medskip
\begin{{tabular}}{{@{{}}p{{0.32\linewidth}}p{{0.62\linewidth}}@{{}}}}
\toprule
Field & Value \\
\midrule
{note_table}
\bottomrule
\end{{tabular}}

\section*{{Artifact paths}}
\small
\begin{{tabular}}{{@{{}}p{{0.25\linewidth}}p{{0.69\linewidth}}@{{}}}}
\toprule
Artifact & Path \\
\midrule
Trajectory CSV & {_tex_path(trajectory_csv)} \\
Cost CSV & {_tex_path(cost_csv)} \\
Provenance JSON & {_tex_path(provenance_json)} \\
Plot PDF & {_tex_path(plot_pdf)} \\
\bottomrule
\end{{tabular}}

\section*{{Scope note}}
This is a standalone support artifact generated from a completed local result JSON.  The plateau rule is inferred for this support bundle and is not a manuscript-selected canonical plateau.  No manuscript source, manuscript PDF, CHTC job, or remote-runner job was modified.
\end{{document}}
"""


def _write_latex_report(
    *,
    tex_path: Path,
    result_json: Path,
    result_sha: str | None,
    payload: Mapping[str, Any],
    selection: PlateauSelection,
    cost_row: CostRow,
    qiskit_meta: Mapping[str, Any],
    s_meta: Mapping[str, Any],
    plot_pdf: Path,
    trajectory_csv: Path,
    cost_csv: Path,
    provenance_json: Path,
) -> None:
    tex = _latex_document(
        result_json=result_json,
        result_sha=result_sha,
        payload=payload,
        selection=selection,
        cost_row=cost_row,
        qiskit_meta=qiskit_meta,
        s_meta=s_meta,
        plot_pdf=plot_pdf,
        trajectory_csv=trajectory_csv,
        cost_csv=cost_csv,
        provenance_json=provenance_json,
    )
    tex_path.write_text(tex, encoding="utf-8")


def _build_latex(tex_path: Path) -> dict[str, Any]:
    cwd = tex_path.parent
    commands: list[list[str]] = []
    if shutil.which("latexmk"):
        commands.append(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", tex_path.name])
    if shutil.which("pdflatex"):
        commands.append(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name])
        commands.append(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name])
    if shutil.which("tectonic"):
        commands.append(["tectonic", tex_path.name])
    if not commands:
        return {"status": "latex_unavailable", "commands": []}
    logs: list[dict[str, Any]] = []
    if commands[0][0] == "latexmk":
        selected = [commands[0]]
    elif commands[0][0] == "pdflatex":
        selected = commands[:2]
    else:
        selected = [commands[0]]
    for cmd in selected:
        proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        logs.append({"command": cmd, "returncode": proc.returncode, "stdout_tail": proc.stdout[-4000:], "stderr_tail": proc.stderr[-4000:]})
        if proc.returncode != 0:
            return {"status": "failed", "commands": logs}
    pdf = tex_path.with_suffix(".pdf")
    return {"status": "ok" if pdf.exists() else "missing_pdf_after_latex", "commands": logs, "pdf": str(pdf)}


def _artifact_hashes(paths: Sequence[Path]) -> dict[str, str | None]:
    return {path.name: _sha256(path) for path in paths}


def build(result_json: Path, out_dir: Path, plateau_k: int | None = None) -> dict[str, Any]:
    result_json = result_json.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = _read_json(result_json)
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"Result JSON did not parse as an object: {result_json}")
    result_sha = _sha256(result_json)
    points, trajectory_meta = _extract_trajectory(payload)
    selection = _select_plateau(points, override_k=plateau_k)
    qiskit_costs, qiskit_meta = _compile_qiskit_costs(payload, selection, result_json)
    s_components, s_meta = _compute_s_alg(payload, selection, result_json)
    row = _cost_row(selection, qiskit_costs, s_components)

    trajectory_csv = out_dir / f"{STEM}_trajectory.csv"
    cost_csv = out_dir / f"{STEM}_qiskit_costs_at_plateau.csv"
    provenance_json = out_dir / f"{STEM}_provenance.json"
    tex_path = out_dir / f"{STEM}.tex"

    _write_trajectory_csv(points, trajectory_csv)
    _write_cost_csv(row, cost_csv)
    plot_png, plot_pdf = _write_error_plot(points, selection, out_dir)
    _write_latex_report(
        tex_path=tex_path,
        result_json=result_json,
        result_sha=result_sha,
        payload=payload,
        selection=selection,
        cost_row=row,
        qiskit_meta=qiskit_meta,
        s_meta=s_meta,
        plot_pdf=plot_pdf,
        trajectory_csv=trajectory_csv,
        cost_csv=cost_csv,
        provenance_json=provenance_json,
    )
    latex_build = _build_latex(tex_path)
    support_pdf = tex_path.with_suffix(".pdf")

    artifact_paths = [trajectory_csv, cost_csv, provenance_json, tex_path, plot_png, plot_pdf]
    if support_pdf.exists():
        artifact_paths.append(support_pdf)
    artifact_path_map = {
        "trajectory_csv": str(trajectory_csv),
        "plateau_cost_csv": str(cost_csv),
        "provenance_json": str(provenance_json),
        "support_tex": str(tex_path),
        "plot_png": str(plot_png),
        "plot_pdf": str(plot_pdf),
        "support_pdf": str(support_pdf) if support_pdf.exists() else None,
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "remote_actions": False,
        "manuscript_edits": False,
        "source_kind": "completed_local_result_json",
        "source_result_json": str(result_json),
        "source_result_json_rel": _rel(result_json),
        "source_result_sha256": result_sha,
        "repo_root": str(REPO_ROOT),
        "git_branch": _git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_head": _git_value(["rev-parse", "HEAD"]),
        "trajectory": trajectory_meta,
        "plateau_selection": asdict(selection),
        "qiskit_costs": dict(qiskit_costs),
        "qiskit_prefix_metadata": dict(qiskit_meta),
        "s_alg_components": dict(s_components),
        "s_alg_metadata": dict(s_meta),
        "cost_row": asdict(row),
        "settings_subset": {
            key: (payload.get("settings") or {}).get(key)
            for key in (
                "L",
                "problem",
                "u",
                "g_ep",
                "n_ph_max",
                "adapt_pool",
                "adapt_reopt_policy",
                "adapt_full_refit_every",
                "adapt_final_full_refit",
                "phase1_shortlist_size",
                "phase2_shortlist_size",
                "phase2_shortlist_fraction",
                "physical_lane_shortlist_aggressiveness",
                "phase2_enable_batching",
                "phase3_runtime_split_mode",
            )
            if isinstance(payload.get("settings"), Mapping)
        },
        "artifact_paths": artifact_path_map,
        "latex_build": latex_build,
    }
    # Do not embed a self-hash for the provenance JSON; it changes when written.
    hashable_artifacts = [path for path in artifact_paths if path != provenance_json]
    manifest["artifact_sha256"] = _artifact_hashes(hashable_artifacts)
    provenance_json.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-json", type=Path, default=DEFAULT_RESULT_JSON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--plateau-k", type=int, default=None, help="Optional explicit displayed plateau iteration.")
    args = parser.parse_args(argv)
    manifest = build(args.result_json, args.out_dir, plateau_k=args.plateau_k)
    print(json.dumps({
        "status": "ok",
        "out_dir": manifest["artifact_paths"],
        "cost_row": manifest["cost_row"],
        "latex_build_status": manifest["latex_build"].get("status"),
    }, indent=2, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
