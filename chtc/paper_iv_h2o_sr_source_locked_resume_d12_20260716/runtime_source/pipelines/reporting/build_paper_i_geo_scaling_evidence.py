#!/usr/bin/env python3
"""Build the deterministic Paper-I Geo-ADAPT scaling evidence appendix.

The builder is deliberately results-only.  It consumes the ordered 34-case
scaling slice of the corrected Geo-ADAPT inventory, reconstructs each selected
structural prefix from the fetched runtime seed plus adaptive history, compiles
coefficient-bearing prefix circuits when the local Qiskit route supports them,
and emits one one-column convergence plot and one immediately following table
per case.  It never edits ``Paper_I.tex`` or an existing figure asset.
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
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCHEMA = "paper_i_geo_scaling_evidence_v1"
PREFIX_SCHEMA = "paper_i_geo_scaling_structural_prefix_v1"
STEM = "paper_i_geo_scaling_evidence_20260711"
DEFAULT_INVENTORY = REPO_ROOT / (
    "output/pdf/paper_i_geo_evidence_inventory_20260711/"
    "paper_i_geo_evidence_inventory_20260711.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / f"output/pdf/{STEM}"
PLATEAU_REL_TOL = 0.10
GEO_COLOR = "#54A24B"
PLOT_ERROR_FLOOR = 1.0e-16

# This tuple is the run contract.  It is intentionally explicit: the builder
# selects these ordered pairs and never derives a Cartesian product.
EXPECTED_SCALING_CASE_IDS: tuple[str, ...] = (
    "hh_L3_nph2_scaling_weak_weak",
    "hh_L3_nph2_scaling_intermediate_weak",
    "hh_L3_nph2_scaling_strong_weak",
    "hh_L3_nph2_scaling_weak_strong",
    "hh_L3_nph2_scaling_intermediate_strong",
    "hh_L3_nph2_scaling_strong_strong",
    "hh_L4_nph1_scaling_weak_weak",
    "hh_L4_nph1_scaling_intermediate_weak",
    "hh_L4_nph1_scaling_strong_weak",
    "hh_L4_nph1_scaling_weak_strong",
    "hh_L4_nph1_scaling_intermediate_strong",
    "hh_L4_nph1_scaling_strong_strong",
    "hubbard_L2_scaling_weak",
    "hubbard_L2_scaling_strong",
    "hubbard_L3_scaling_weak",
    "hubbard_L3_scaling_strong",
    "hubbard_L4_scaling_weak",
    "hubbard_L4_scaling_strong",
    "spin_boson_L2_nph4_scaling_weak",
    "spin_boson_L2_nph4_scaling_strong",
    "spin_boson_L3_nph3_scaling_weak",
    "spin_boson_L3_nph3_scaling_strong",
    "spin_boson_L3_nph2_scaling_weak",
    "spin_boson_L3_nph2_scaling_strong",
    "spin_boson_L4_nph1_scaling_weak",
    "spin_boson_L4_nph1_scaling_strong",
    "bose_hubbard_L2_nph3_scaling_weak",
    "bose_hubbard_L2_nph3_scaling_strong",
    "bose_hubbard_L3_nph3_scaling_weak",
    "bose_hubbard_L3_nph3_scaling_strong",
    "bose_hubbard_L3_nph2_scaling_weak",
    "bose_hubbard_L3_nph2_scaling_strong",
    "bose_hubbard_L4_nph1_scaling_weak",
    "bose_hubbard_L4_nph1_scaling_strong",
)


@dataclass(frozen=True)
class CurvePoint:
    k: int
    error_raw: float
    error_plotted: float


@dataclass(frozen=True)
class PlateauSelection:
    history_position: int
    k_pl: int
    logical_depth: int
    error_raw: float
    error_plotted: float
    best_observed_error: float
    threshold: float


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


def resolve_path(raw: str | Path) -> Path:
    path = Path(str(raw))
    return path if path.is_absolute() else REPO_ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object at {path}")
    return payload


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def ordered_scaling_inventory_rows(inventory: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_rows = inventory.get("rows")
    if not isinstance(raw_rows, list):
        raise ValueError("Inventory has no rows list")
    rows = [
        dict(row)
        for row in raw_rows
        if isinstance(row, Mapping)
        and str(row.get("paper_placement")) == "appendix_scaling_results"
    ]
    observed = tuple(str(row.get("case_id")) for row in rows)
    if observed != EXPECTED_SCALING_CASE_IDS:
        raise ValueError(
            "Scaling inventory is not the exact ordered 34-case contract; "
            f"expected={EXPECTED_SCALING_CASE_IDS!r}, observed={observed!r}"
        )
    if len(set(observed)) != len(observed):
        raise ValueError("Scaling inventory contains duplicate case ids")
    return rows


def _finite_float(value: Any, *, label: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{label} is not finite: {value!r}")
    return out


def trajectory_points(result: Mapping[str, Any]) -> list[CurvePoint]:
    history = result.get("adapt_history")
    if not isinstance(history, list) or not history:
        raise ValueError("Geo result has no adaptive history")
    exact = _finite_float(result.get("same_cutoff_exact_gs_energy"), label="same-cutoff exact energy")
    initial = abs(_finite_float(history[0].get("energy_before"), label="initial energy") - exact)
    points = [CurvePoint(0, initial, max(initial, PLOT_ERROR_FLOOR))]
    for index, raw in enumerate(history):
        if not isinstance(raw, Mapping):
            raise TypeError(f"History row {index} is not an object")
        history_position = int(raw.get("history_position", index))
        if history_position != index:
            raise ValueError(
                f"Non-contiguous history position {history_position} at row {index}"
            )
        error = raw.get("abs_delta_e_same_cutoff_after")
        if error is None:
            error = abs(_finite_float(raw.get("energy_after"), label="history energy") - exact)
        value = abs(_finite_float(error, label="same-cutoff error"))
        points.append(CurvePoint(index + 1, value, max(value, PLOT_ERROR_FLOOR)))
    return points


def select_first_plateau(
    result: Mapping[str, Any],
    *,
    horizon: int,
    rel_tol: float = PLATEAU_REL_TOL,
) -> PlateauSelection:
    history = result.get("adapt_history")
    if not isinstance(history, list) or not history:
        raise ValueError("Cannot select a plateau from empty history")
    if int(horizon) <= 0 or len(history) != int(horizon):
        raise ValueError(
            f"Completed horizon/history mismatch: horizon={horizon}, history={len(history)}"
        )
    exact = _finite_float(result.get("same_cutoff_exact_gs_energy"), label="same-cutoff exact energy")
    errors: list[float] = []
    for raw in history:
        if not isinstance(raw, Mapping):
            raise TypeError("Adaptive history contains a non-object row")
        value = raw.get("abs_delta_e_same_cutoff_after")
        if value is None:
            value = abs(_finite_float(raw.get("energy_after"), label="history energy") - exact)
        errors.append(abs(_finite_float(value, label="same-cutoff error")))
    best = min(errors)
    threshold = (1.0 + float(rel_tol)) * best
    position = next(index for index, value in enumerate(errors) if value <= threshold)
    selected = history[position]
    raw_error = errors[position]
    return PlateauSelection(
        history_position=position,
        k_pl=position + 1,
        logical_depth=int(selected.get("depth_after") or 0),
        error_raw=raw_error,
        error_plotted=max(raw_error, PLOT_ERROR_FLOOR),
        best_observed_error=best,
        threshold=threshold,
    )


def prefix_query_ledger(
    history: Sequence[Mapping[str, Any]], history_position: int
) -> dict[str, int]:
    stop = int(history_position) + 1
    rows = list(history[:stop])
    if len(rows) != stop:
        raise IndexError(history_position)

    def total(*keys: str) -> int:
        return sum(
            sum(int(row.get(key) or 0) for key in keys)
            for row in rows
        )

    components = {
        "N_H_outer_eval": total("outer_hamiltonian_eval_count"),
        "N_H_refit_eval": total("optimizer_nfev"),
        "N_grad_probe": total(
            "selector_gradient_probe_count", "qngd_gradient_operator_probe_count_total"
        ),
        "N_metric_probe": total(
            "selector_metric_probe_count", "qngd_metric_operator_probe_count_total"
        ),
        "N_other_quantum": total("N_other_quantum"),
    }
    components["S"] = sum(components.values())
    return components


def resolve_runtime_seed_path(result_path: Path, payload: Mapping[str, Any]) -> Path:
    sibling = result_path.parent / "runtime_seed.json"
    candidates = [sibling]
    raw = payload.get("runtime_seed_json")
    if isinstance(raw, str) and raw.strip():
        candidates.extend((Path(raw), REPO_ROOT / raw, result_path.parent / Path(raw).name))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No fetched runtime_seed.json beside {result_path}")


def reconstruct_structural_prefix(
    *,
    seed: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    history_position: int,
) -> dict[str, Any]:
    adapt = seed.get("adapt_vqe")
    if not isinstance(adapt, Mapping):
        raise ValueError("runtime seed has no adapt_vqe object")
    arrays = {
        "operators": adapt.get("operators"),
        "execution_modes": adapt.get("selected_operator_execution_modes"),
        "pauli_terms": adapt.get("selected_operator_pauli_terms"),
        "supports": adapt.get("selected_operator_supports"),
        "theta": adapt.get("optimal_point"),
    }
    if any(not isinstance(value, list) for value in arrays.values()):
        missing = [key for key, value in arrays.items() if not isinstance(value, list)]
        raise ValueError(f"runtime seed is missing selected-generator arrays: {missing}")
    lengths = {key: len(value) for key, value in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"runtime-seed selected-generator length mismatch: {lengths}")

    selected_labels: list[str] = []
    prefix_labels: list[str] | None = None
    skip_count = 0
    for index, row in enumerate(history):
        labels = [str(item) for item in (row.get("selected_batch_labels") or [])]
        appended = int(row.get("appended_operator_count") or 0)
        skipped = bool(row.get("geo_immediate_repeat_skipped"))
        if skipped:
            skip_count += 1
            if labels or appended:
                raise ValueError(f"Immediate-repeat skip row {index} also appends a generator")
        else:
            if appended != len(labels):
                raise ValueError(
                    f"History row {index} appends {appended} generators but lists {len(labels)} labels"
                )
            insertion = row.get("selected_insertion_position")
            if insertion is not None and len(labels) == 1:
                position = int(insertion)
                if not 0 <= position <= len(selected_labels):
                    raise IndexError(f"Invalid insertion position {position} at history row {index}")
                selected_labels.insert(position, labels[0])
            else:
                selected_labels.extend(labels)
        if index == int(history_position):
            prefix_labels = list(selected_labels)

    if prefix_labels is None:
        raise IndexError(history_position)
    terminal_labels = [str(value) for value in arrays["operators"]]
    if selected_labels != terminal_labels:
        raise ValueError("Full history selected labels do not match runtime-seed operator order")
    depth = len(prefix_labels)
    if prefix_labels != terminal_labels[:depth]:
        raise ValueError("Selected history prefix is not the runtime-seed structural prefix")
    history_depth = int(history[int(history_position)].get("depth_after") or 0)
    if depth != history_depth:
        raise ValueError(
            f"Selected structural depth {depth} differs from history depth {history_depth}"
        )

    semantics: list[dict[str, Any]] = []
    for index in range(depth):
        raw_terms = arrays["pauli_terms"][index]
        if not isinstance(raw_terms, list) or not raw_terms:
            raise ValueError(f"Selected generator {index} has no coefficient-bearing Pauli terms")
        terms: list[dict[str, Any]] = []
        for term in raw_terms:
            if not isinstance(term, Mapping):
                raise TypeError(f"Selected generator {index} has a non-object Pauli term")
            terms.append(
                {
                    "pauli_exyz": str(term.get("pauli_exyz") or "").lower(),
                    "coeff_re": _finite_float(term.get("coeff_re", 0.0), label="Pauli coeff_re"),
                    "coeff_im": _finite_float(term.get("coeff_im", 0.0), label="Pauli coeff_im"),
                }
            )
        semantics.append(
            {
                "index": index,
                "label": prefix_labels[index],
                "execution_mode": str(arrays["execution_modes"][index] or "termwise_product"),
                "support": list(arrays["supports"][index] or []),
                "pauli_terms": terms,
            }
        )
    digest = hashlib.sha256(
        json.dumps(semantics, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    later_rows = history[int(history_position) + 1 :]
    no_later_append = all(int(row.get("appended_operator_count") or 0) == 0 for row in later_rows)
    terminal_parameters_match_structure = depth == len(terminal_labels) and no_later_append
    parameter_status = (
        "terminal_runtime_seed_parameters_available"
        if terminal_parameters_match_structure
        else "blocked_prefix_optimized_parameters_not_serialized"
    )
    return {
        "selected_generator_count": depth,
        "selected_labels": prefix_labels,
        "selected_generator_semantics": semantics,
        "selected_generator_semantics_sha256": digest,
        "repeat_skip_count_full_horizon": skip_count,
        "terminal_generator_count": len(terminal_labels),
        "terminal_parameters_match_selected_structure": terminal_parameters_match_structure,
        "selected_prefix_parameter_status": parameter_status,
        "selected_prefix_theta": (
            [float(value) for value in arrays["theta"]]
            if terminal_parameters_match_structure
            else None
        ),
    }


def _statevector_from_seed(seed: Mapping[str, Any]) -> Any:
    import numpy as np

    raw = seed.get("ansatz_input_state")
    if not isinstance(raw, Mapping):
        raise ValueError("runtime seed has no ansatz_input_state")
    nq = int(raw.get("nq_total") or 0)
    if nq <= 0:
        raise ValueError("runtime seed has invalid ansatz_input_state.nq_total")
    amplitudes = raw.get("amplitudes_qn_to_q0")
    if not isinstance(amplitudes, Mapping) or not amplitudes:
        raise ValueError("runtime seed reference state has no amplitudes")
    vector = np.zeros(1 << nq, dtype=complex)
    for bitstring, coefficient in amplitudes.items():
        text = str(bitstring)
        if len(text) != nq or set(text) - {"0", "1"}:
            raise ValueError(f"Invalid reference-state bitstring {text!r}")
        if not isinstance(coefficient, Mapping):
            raise TypeError(f"Invalid reference-state coefficient for {text}")
        vector[int(text, 2)] = complex(
            float(coefficient.get("re", 0.0)), float(coefficient.get("im", 0.0))
        )
    norm = float(np.linalg.norm(vector))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-10):
        raise ValueError(f"Reference-state norm is {norm}")
    return vector


def compile_prefix_qiskit(
    *,
    seed: Mapping[str, Any],
    reconstruction: Mapping[str, Any],
    grouped_exact_max_active_qubits: int,
    source_kind: str = "qiskit_coefficient_aware_geo_selected_prefix",
) -> dict[str, Any]:
    try:
        from pipelines.exact_bench.table_i_qiskit_resource_compile import (
            TABLE_I_QISKIT_COMPILE_CONVENTION,
            TableICompileUnavailable,
            TableIQiskitCompileConfig,
            compile_table_i_ansatz_terms,
        )
        from src.quantum.pauli_polynomial_class import PauliPolynomial
        from src.quantum.qubitization_module import PauliTerm
        from src.quantum.vqe_latex_python_pairs import AnsatzTerm

        state = _statevector_from_seed(seed)
        nq = int(seed["ansatz_input_state"]["nq_total"])
        ops: list[Any] = []
        for row in reconstruction["selected_generator_semantics"]:
            polynomial = PauliPolynomial(
                "JW",
                [
                    PauliTerm(
                        nq,
                        ps=str(term["pauli_exyz"]),
                        pc=complex(float(term["coeff_re"]), float(term["coeff_im"])),
                    )
                    for term in row["pauli_terms"]
                ],
            )
            ops.append(
                AnsatzTerm(
                    label=str(row["label"]),
                    polynomial=polynomial,
                    execution_mode=str(row["execution_mode"]),
                )
            )
        compiled = compile_table_i_ansatz_terms(
            ops=ops,
            num_qubits=nq,
            reference_state=state,
            source_kind=str(source_kind),
            config=TableIQiskitCompileConfig(
                grouped_exact_max_active_qubits=int(grouped_exact_max_active_qubits)
            ),
        )
        if not bool(compiled.get("compiled_resource_qiskit_validated")):
            raise ValueError("Qiskit prefix compile did not validate")
        return {
            "status": "ok",
            "blocked_reason": None,
            "compile_convention": TABLE_I_QISKIT_COMPILE_CONVENTION,
            "source_kind": compiled.get("compiled_resource_source_kind")
            or compiled.get("compiled_cost_source_kind")
            or str(source_kind),
            "N1q": compiled.get("compiled_count_1q_total"),
            "N2q": compiled.get("compiled_count_2q_total"),
            "D2q": compiled.get("compiled_depth_2q_total"),
            "Dcirc": compiled.get("compiled_depth_total"),
            "runtime_rotation_count": compiled.get("runtime_rotation_count"),
            "generator_coefficients_sha256": compiled.get("generator_coefficients_sha256"),
            "grouped_exact_synthesis_id": compiled.get("grouped_exact_synthesis_id"),
            "optimization_level": compiled.get("qiskit_transpile_optimization_level"),
            "transpile_seed": compiled.get("qiskit_transpile_seed"),
            "basis_gates": compiled.get("compiled_basis_gates"),
            "qiskit_version": compiled.get("qiskit_version"),
            "operator_synthesis": compiled.get("operator_synthesis"),
            "grouped_exact_max_active_qubits": int(grouped_exact_max_active_qubits),
        }
    except Exception as exc:
        status = getattr(exc, "status", "prefix_compile_exception")
        reason = getattr(exc, "reason", str(exc))
        return {
            "status": f"blocked:{status}",
            "blocked_reason": str(reason),
            "compile_convention": "table_i_basis_gate_transpile_v1",
            "source_kind": str(source_kind),
            "N1q": None,
            "N2q": None,
            "D2q": None,
            "Dcirc": None,
            "runtime_rotation_count": None,
            "generator_coefficients_sha256": reconstruction.get(
                "selected_generator_semantics_sha256"
            ),
            "grouped_exact_synthesis_id": None,
            "optimization_level": 0,
            "transpile_seed": 7,
            "basis_gates": None,
            "qiskit_version": None,
            "operator_synthesis": None,
            "grouped_exact_max_active_qubits": int(grouped_exact_max_active_qubits),
        }


def _terminal_query_identity(result: Mapping[str, Any]) -> dict[str, Any]:
    history = result.get("adapt_history")
    if not isinstance(history, list) or not history:
        return {"status": "blocked_missing_history"}
    ledger = prefix_query_ledger(history, len(history) - 1)
    source = int(result.get("S_alg") or -1)
    component_source = {
        "N_H_outer_eval": int(result.get("N_H_outer_eval") or 0),
        "N_H_refit_eval": int(result.get("N_H_refit_eval") or 0),
        "N_grad_probe": int(result.get("N_grad") or 0),
        "N_metric_probe": int(result.get("N_metric") or 0),
        "N_other_quantum": int(result.get("N_other_quantum") or 0),
    }
    checks = {
        key: int(ledger[key]) == int(component_source[key]) for key in component_source
    }
    checks["S"] = int(ledger["S"]) == source
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "history_terminal_ledger": ledger,
        "source_terminal_S_alg": source,
        "source_terminal_components": component_source,
    }


def _case_title(row: Mapping[str, Any]) -> str:
    family = str(row.get("family"))
    family_name = {
        "hh": "Hubbard--Holstein",
        "hubbard": "Hubbard",
        "spin_boson": "spin--boson",
        "bose_hubbard": "Bose--Hubbard",
    }.get(family, family)
    pieces = [family_name, f"L={int(row.get('L') or 0)}"]
    cutoff = row.get("cutoff") if isinstance(row.get("cutoff"), Mapping) else {}
    if cutoff.get("n_ph_work") is not None:
        pieces.append(f"phonon cutoff={int(cutoff['n_ph_work'])}")
    regime = str(row.get("display_regime") or "").replace("-", "--")
    if regime:
        pieces.append(regime)
    return ", ".join(pieces)


def _strict_replay_status(
    inventory_row: Mapping[str, Any], reconstruction: Mapping[str, Any]
) -> dict[str, Any]:
    source_replay = (
        inventory_row.get("trajectory_and_replay")
        if isinstance(inventory_row.get("trajectory_and_replay"), Mapping)
        else {}
    )
    if not bool(reconstruction.get("terminal_parameters_match_selected_structure")):
        status = "blocked_prefix_optimized_parameters_not_serialized"
        missing = ["selected_prefix_optimal_point"]
    else:
        status = "not_validated_terminal_runtime_seed_available"
        missing = ["strict_loader_validation"]
    return {
        "status": status,
        "strict_loader_validation": "not_run",
        "source_result_strict_replay_ready": bool(source_replay.get("strict_replay_ready")),
        "source_result_strict_replay_missing_fields": list(
            source_replay.get("strict_replay_missing_fields") or []
        ),
        "selected_prefix_missing_fields": missing,
        "structural_prefix_reconstruction": "pass",
        "runtime_seed_envelope": "complete",
    }


def analyze_case(
    inventory_row: Mapping[str, Any],
    *,
    order_index: int,
    prefix_dir: Path,
    compile_qiskit: bool,
    grouped_exact_max_active_qubits: int,
) -> dict[str, Any]:
    result_path = resolve_path(str(inventory_row["artifacts"]["result_json"]))
    if sha256(result_path) != str(inventory_row["artifacts"]["result_sha256"]):
        raise ValueError(f"Result hash mismatch for {result_path}")
    payload = read_json(result_path)
    result = payload.get("result")
    if not isinstance(result, Mapping):
        raise ValueError(f"Result payload has no result object: {result_path}")
    horizon = int(inventory_row["completion"]["outer_iteration_cap"])
    history = result.get("adapt_history")
    if not isinstance(history, list) or len(history) != horizon:
        raise ValueError(f"Incomplete history for {inventory_row['case_id']}")
    if str(inventory_row.get("validation", {}).get("status")) != "pass":
        raise ValueError(f"Inventory validation did not pass for {inventory_row['case_id']}")

    points = trajectory_points(result)
    plateau = select_first_plateau(result, horizon=horizon)
    if not math.isclose(
        points[plateau.k_pl].error_raw,
        plateau.error_raw,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise ValueError(f"Marker/curve mismatch for {inventory_row['case_id']}")
    ledger = prefix_query_ledger(history, plateau.history_position)
    terminal_identity = _terminal_query_identity(result)
    if terminal_identity["status"] != "pass":
        raise ValueError(f"Terminal query-accounting identity failed for {inventory_row['case_id']}")

    seed_path = resolve_runtime_seed_path(result_path, payload)
    seed = read_json(seed_path)
    reconstruction = reconstruct_structural_prefix(
        seed=seed,
        history=history,
        history_position=plateau.history_position,
    )
    strict_replay = _strict_replay_status(inventory_row, reconstruction)
    qiskit = (
        compile_prefix_qiskit(
            seed=seed,
            reconstruction=reconstruction,
            grouped_exact_max_active_qubits=grouped_exact_max_active_qubits,
        )
        if compile_qiskit
        else {
            "status": "blocked:qiskit_compile_disabled",
            "blocked_reason": "Builder invoked with --skip-qiskit",
            "compile_convention": "table_i_basis_gate_transpile_v1",
            "source_kind": "qiskit_coefficient_aware_geo_selected_prefix",
            "N1q": None,
            "N2q": None,
            "D2q": None,
            "Dcirc": None,
            "runtime_rotation_count": None,
            "generator_coefficients_sha256": reconstruction[
                "selected_generator_semantics_sha256"
            ],
            "grouped_exact_synthesis_id": None,
            "optimization_level": 0,
            "transpile_seed": 7,
            "basis_gates": None,
            "qiskit_version": None,
            "operator_synthesis": None,
        }
    )

    case_id = str(inventory_row["case_id"])
    prefix_path = prefix_dir / f"{case_id}__geo_k{plateau.k_pl:02d}_structural_prefix.json"
    prefix_payload = {
        "schema": PREFIX_SCHEMA,
        "case_id": case_id,
        "method": "Geo-ADAPT",
        "algorithm_id": "static_geo_adapt_vqe",
        "selected_prefix": {
            "history_position": plateau.history_position,
            "k_pl": plateau.k_pl,
            "logical_depth": plateau.logical_depth,
            "same_cutoff_abs_delta_e": plateau.error_raw,
            "energy_after": history[plateau.history_position].get("energy_after"),
        },
        "reconstruction": reconstruction,
        "strict_replay": strict_replay,
        "source": {
            "result_json": rel(result_path),
            "result_sha256": sha256(result_path),
            "runtime_seed_json": rel(seed_path),
            "runtime_seed_sha256": sha256(seed_path),
        },
        "settings": seed.get("settings"),
        "ansatz_input_state": seed.get("ansatz_input_state"),
        "query_ledger": ledger,
        "qiskit_prefix_cost": qiskit,
    }
    write_json(prefix_path, prefix_payload)

    cutoff = (
        dict(inventory_row.get("cutoff") or {})
        if isinstance(inventory_row.get("cutoff"), Mapping)
        else {}
    )
    method = dict(inventory_row.get("method") or {})
    return {
        "schema": f"{SCHEMA}_row_v1",
        "order_index": int(order_index),
        "case_id": case_id,
        "case_title_tex": _case_title(inventory_row),
        "family": str(inventory_row.get("family")),
        "L": int(inventory_row.get("L") or 0),
        "display_regime": str(inventory_row.get("display_regime") or ""),
        "cutoff_pair": {
            "n_ph_work": cutoff.get("n_ph_work"),
            "n_ph_ref": cutoff.get("n_ph_ref"),
            "reference_role": cutoff.get("reference_role"),
        },
        "method": {
            "label": "Geo-ADAPT",
            "algorithm_id": "static_geo_adapt_vqe",
            "optimizer": str(method.get("optimizer") or "powell").lower(),
            "optimizer_maxiter": int(method.get("optimizer_maxiter") or 0),
            "immediate_repeat_allowed": False,
            "immediate_repeat_policy": str(method.get("immediate_repeat_policy")),
            "selection_with_replacement": bool(method.get("selection_with_replacement")),
            "pool": str(method.get("pool")),
            "pool_policy": "full_meta_parent_generators_only",
            "parent_generator_policy": str(method.get("parent_generator_policy")),
            "generic_runtime_split_mode": str(method.get("generic_runtime_split_mode")),
            "shared_pauli_pool_mode": str(method.get("shared_pauli_pool_mode")),
        },
        "completed_horizon": horizon,
        "point_count": len(points),
        "trajectory_points": [point.__dict__ for point in points],
        "plateau_policy": {
            "id": "first_prefix_within_10_percent_of_best_observed_error_v1",
            "relative_tolerance": PLATEAU_REL_TOL,
            "best_observed_error": plateau.best_observed_error,
            "threshold": plateau.threshold,
            "selection_domain": f"completed_history_rows_1_through_{horizon}",
        },
        "marker": {
            "method": "Geo-ADAPT",
            "shape": "triangle",
            "count_on_curve": 1,
            "k": plateau.k_pl,
            "error_raw": plateau.error_raw,
            "error_plotted": plateau.error_plotted,
            "logical_depth": plateau.logical_depth,
        },
        "query_ledger": {
            **ledger,
            "definition": (
                "logical scalar estimator queries through selected history row; "
                "same-state reuse and symmetric metric entries counted once"
            ),
            "terminal_identity": terminal_identity,
        },
        "prefix_reconstruction": {
            "status": "pass",
            "source": "runtime_seed_plus_adapt_history",
            "sidecar_json": rel(prefix_path),
            "sidecar_sha256": sha256(prefix_path),
            **{key: value for key, value in reconstruction.items() if key != "selected_generator_semantics"},
        },
        "strict_replay": strict_replay,
        "qiskit_prefix_cost": qiskit,
        "sources": {
            "inventory_result_json": str(inventory_row["artifacts"]["result_json"]),
            "result_json": rel(result_path),
            "result_sha256": sha256(result_path),
            "runtime_seed_json": rel(seed_path),
            "runtime_seed_sha256": sha256(seed_path),
            "progress_jsonl": str(inventory_row["artifacts"].get("progress_jsonl")),
            "progress_sha256": str(inventory_row["artifacts"].get("progress_sha256")),
            "cell_manifest": str(inventory_row["artifacts"].get("cell_manifest")),
            "cell_manifest_sha256": str(inventory_row["artifacts"].get("cell_manifest_sha256")),
        },
    }


def plot_case(row: Mapping[str, Any], *, plot_dir: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    plot_dir.mkdir(parents=True, exist_ok=True)
    case_id = str(row["case_id"])
    pdf_path = plot_dir / f"{case_id}__geo_error_vs_iteration.pdf"
    png_path = plot_dir / f"{case_id}__geo_error_vs_iteration.png"
    points = list(row["trajectory_points"])
    x = [int(point["k"]) for point in points]
    y = [float(point["error_plotted"]) for point in points]
    marker = row["marker"]

    fig, ax = plt.subplots(figsize=(3.35, 2.55), constrained_layout=True)
    ax.plot(x, y, color=GEO_COLOR, linewidth=1.65, linestyle="-")
    ax.scatter(
        [int(marker["k"])],
        [float(marker["error_plotted"])],
        marker="^",
        s=54,
        color=GEO_COLOR,
        edgecolors="black",
        linewidths=0.45,
        zorder=4,
    )
    ax.set_yscale("log")
    ax.set_xlim(0, int(row["completed_horizon"]))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True, min_n_ticks=4))
    ax.set_xlabel("ADAPT iteration")
    ax.set_ylabel(r"Same-cutoff $|\Delta E|$")
    ax.grid(True, which="major", alpha=0.22, linewidth=0.5)
    legend_handle = Line2D(
        [0],
        [0],
        color=GEO_COLOR,
        linewidth=1.65,
        linestyle="-",
        marker="^",
        markersize=6,
        markeredgecolor="black",
        markeredgewidth=0.45,
        label=r"Geo-ADAPT ($\triangle$: $k_{\rm pl}$)",
    )
    ax.legend(handles=[legend_handle], loc="best", frameon=False, fontsize=7.5)
    fig.savefig(
        pdf_path,
        metadata={
            "Title": case_id,
            "Creator": Path(__file__).name,
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(
        png_path,
        dpi=240,
        metadata={"Software": Path(__file__).name},
    )
    plt.close(fig)
    return {
        "pdf": rel(pdf_path),
        "pdf_sha256": sha256(pdf_path),
        "png": rel(png_path),
        "png_sha256": sha256(png_path),
        "layout": "one_column_single_axes",
        "x_axis": "integer_adapt_iteration",
        "y_axis": "log_same_cutoff_abs_delta_e",
        "target_line": False,
        "curve_line_style": "solid",
        "curve_repeated_markers": False,
        "marker_count_on_curve": 1,
        "marker_shape": "triangle",
        "marker_k": int(marker["k"]),
        "marker_error_raw": float(marker["error_raw"]),
        "marker_error_plotted": float(marker["error_plotted"]),
        "zero_error_plot_floor": PLOT_ERROR_FLOOR,
    }


def latex_escape(value: Any) -> str:
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
    return "".join(replacements.get(char, char) for char in text)


def format_error_tex(value: float) -> str:
    number = float(value)
    if number == 0.0:
        return "0"
    exponent = int(math.floor(math.log10(abs(number))))
    if -3 <= exponent <= 2:
        return f"{number:.6g}"
    mantissa = number / (10.0**exponent)
    return rf"{mantissa:.3g}\times 10^{{{exponent}}}"


def format_integer_tex(value: int) -> str:
    return f"{int(value):,}".replace(",", "{,}")


def _cost_cell(qiskit: Mapping[str, Any], key: str) -> str:
    value = qiskit.get(key)
    return "--" if value is None else format_integer_tex(int(value))


def _reader_facing_replay_note(status: str) -> str:
    if status == "blocked_prefix_optimized_parameters_not_serialized":
        return (
            "The selected nonterminal prefix does not serialize its optimized "
            "parameters for an independent strict energy replay."
        )
    if status == "not_validated_terminal_runtime_seed_available":
        return (
            "A terminal runtime seed is available, but strict-loader replay was not run."
        )
    return "Strict-replay status is recorded in the machine-readable provenance."


def write_appendix_fragment(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "% Auto-generated Paper-I Geo-ADAPT scaling appendix fragment.",
        "% Requires graphicx, booktabs, and float. Define \\GeoScalingAssetRoot",
        "% relative to the LaTeX build directory before inputting this file.",
        r"\providecommand{\GeoScalingAssetRoot}{.}",
        "",
    ]
    for row in rows:
        case_id = str(row["case_id"])
        label_id = case_id.replace("_", "-")
        plot_rel = Path(str(row["plot"]["pdf"]))
        try:
            plot_from_output = plot_rel.relative_to(path.parent.relative_to(REPO_ROOT))
        except ValueError:
            plot_from_output = Path("plots") / plot_rel.name
        qiskit = row["qiskit_prefix_cost"]
        strict_status = str(row["strict_replay"]["status"])
        cost_status = str(qiskit["status"])
        cutoff = row["cutoff_pair"]
        if cutoff.get("n_ph_work") is None:
            cutoff_note = "phonon cutoff: not applicable"
        else:
            ref = "--" if cutoff.get("n_ph_ref") is None else str(int(cutoff["n_ph_ref"]))
            cutoff_note = (
                f"cutoff pair (working, reference)=({int(cutoff['n_ph_work'])}, {ref})"
            )
        lines.extend(
            [
                r"\begin{figure}[H]",
                r"  \centering",
                rf"  \includegraphics[width=0.98\columnwidth]{{\GeoScalingAssetRoot/{plot_from_output.as_posix()}}}",
                (
                    r"  \caption{Geo-ADAPT same-cutoff energy error versus ADAPT iteration for "
                    + str(row["case_title_tex"])
                    + r". The triangle marks the first prefix within 10\% of the best error observed over the completed horizon.}"
                ),
                rf"  \label{{fig:paper-i-geo-scaling-{label_id}}}",
                r"\end{figure}",
                r"\begin{table}[H]",
                r"  \centering",
                r"  \small",
                r"  \begin{tabular}{lrrrrrr}",
                r"    \toprule",
                r"    Method & $k_{\rm pl}$ & $|\Delta E|$ & $N_{2q}$ & $D_{2q}$ & $D_{\rm circ}$ & $S$ \\",
                r"    \midrule",
                (
                    r"    Geo-ADAPT & "
                    + str(int(row["marker"]["k"]))
                    + " & $"
                    + format_error_tex(float(row["marker"]["error_raw"]))
                    + "$ & "
                    + _cost_cell(qiskit, "N2q")
                    + " & "
                    + _cost_cell(qiskit, "D2q")
                    + " & "
                    + _cost_cell(qiskit, "Dcirc")
                    + " & "
                    + format_integer_tex(int(row["query_ledger"]["S"]))
                    + r" \\"
                ),
                r"    \bottomrule",
                r"  \end{tabular}",
                (
                    r"  \caption{Selected-prefix Geo-ADAPT evidence for "
                    + str(row["case_title_tex"])
                    + ". "
                    + latex_escape(cutoff_note)
                    + ". "
                    + (
                        "The coefficient-aware Qiskit prefix compilation succeeded. "
                        if cost_status == "ok"
                        else "The Qiskit prefix-compilation blocker is recorded in the provenance. "
                    )
                    + _reader_facing_replay_note(strict_status)
                    + "}"
                ),
                rf"  \label{{tab:paper-i-geo-scaling-{label_id}}}",
                r"\end{table}",
                r"\clearpage",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "order_index",
        "case_id",
        "family",
        "L",
        "display_regime",
        "n_ph_work",
        "n_ph_ref",
        "optimizer",
        "optimizer_maxiter",
        "immediate_repeat_allowed",
        "immediate_repeat_policy",
        "pool_policy",
        "completed_horizon",
        "point_count",
        "k_pl",
        "logical_depth",
        "marker_error_raw",
        "best_observed_error",
        "plateau_threshold",
        "S",
        "N_H_outer_eval",
        "N_H_refit_eval",
        "N_grad_probe",
        "N_metric_probe",
        "N_other_quantum",
        "qiskit_status",
        "N2q",
        "D2q",
        "Dcirc",
        "strict_replay_status",
        "source_json",
        "source_sha256",
        "runtime_seed_json",
        "runtime_seed_sha256",
        "prefix_sidecar_json",
        "plot_pdf",
        "plot_png",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            ledger = row["query_ledger"]
            marker = row["marker"]
            qiskit = row["qiskit_prefix_cost"]
            cutoff = row["cutoff_pair"]
            method = row["method"]
            writer.writerow(
                {
                    "order_index": row["order_index"],
                    "case_id": row["case_id"],
                    "family": row["family"],
                    "L": row["L"],
                    "display_regime": row["display_regime"],
                    "n_ph_work": cutoff.get("n_ph_work"),
                    "n_ph_ref": cutoff.get("n_ph_ref"),
                    "optimizer": method["optimizer"],
                    "optimizer_maxiter": method["optimizer_maxiter"],
                    "immediate_repeat_allowed": method["immediate_repeat_allowed"],
                    "immediate_repeat_policy": method["immediate_repeat_policy"],
                    "pool_policy": method["pool_policy"],
                    "completed_horizon": row["completed_horizon"],
                    "point_count": row["point_count"],
                    "k_pl": marker["k"],
                    "logical_depth": marker["logical_depth"],
                    "marker_error_raw": marker["error_raw"],
                    "best_observed_error": row["plateau_policy"]["best_observed_error"],
                    "plateau_threshold": row["plateau_policy"]["threshold"],
                    "S": ledger["S"],
                    "N_H_outer_eval": ledger["N_H_outer_eval"],
                    "N_H_refit_eval": ledger["N_H_refit_eval"],
                    "N_grad_probe": ledger["N_grad_probe"],
                    "N_metric_probe": ledger["N_metric_probe"],
                    "N_other_quantum": ledger["N_other_quantum"],
                    "qiskit_status": qiskit["status"],
                    "N2q": qiskit.get("N2q"),
                    "D2q": qiskit.get("D2q"),
                    "Dcirc": qiskit.get("Dcirc"),
                    "strict_replay_status": row["strict_replay"]["status"],
                    "source_json": row["sources"]["result_json"],
                    "source_sha256": row["sources"]["result_sha256"],
                    "runtime_seed_json": row["sources"]["runtime_seed_json"],
                    "runtime_seed_sha256": row["sources"]["runtime_seed_sha256"],
                    "prefix_sidecar_json": row["prefix_reconstruction"]["sidecar_json"],
                    "plot_pdf": row["plot"]["pdf"],
                    "plot_png": row["plot"]["png"],
                }
            )


def write_report_tex(
    path: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    inventory_path: Path,
    inventory_sha256: str,
    provenance_json: Path,
    provenance_csv: Path,
    manifest_json: Path,
    appendix_fragment: Path,
) -> None:
    qiskit_counts = Counter(str(row["qiskit_prefix_cost"]["status"]) for row in rows)
    strict_counts = Counter(str(row["strict_replay"]["status"]) for row in rows)
    qiskit_ok = int(qiskit_counts.get("ok", 0))
    qiskit_blocked = len(rows) - qiskit_ok
    strict_summary = "; ".join(f"{key}: {value}" for key, value in sorted(strict_counts.items()))
    tex = rf"""% BEGIN_MACHINE_READABLE_GEO_SCALING_EVIDENCE
% schema={SCHEMA}
% inventory_json={rel(inventory_path)}
% inventory_sha256={inventory_sha256}
% provenance_json={rel(provenance_json)}
% provenance_csv={rel(provenance_csv)}
% manifest_json={rel(manifest_json)}
% appendix_fragment={rel(appendix_fragment)}
% END_MACHINE_READABLE_GEO_SCALING_EVIDENCE
\documentclass[10pt]{{article}}
\ifdefined\pdfinfoomitdate\pdfinfoomitdate=1\fi
\ifdefined\pdftrailerid\pdftrailerid{{}}\fi
\ifdefined\pdfsuppressptexinfo\pdfsuppressptexinfo=15\fi
\usepackage[margin=0.72in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{float}}
\usepackage[hidelinks]{{hyperref}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{5pt}}
\newcommand{{\GeoScalingAssetRoot}}{{.}}
\begin{{document}}
\section*{{Parameter manifest}}
\begin{{tabular}}{{@{{}}p{{0.24\textwidth}}p{{0.72\textwidth}}@{{}}}}
\textbf{{Scope}} & Exactly 34 ordered Geo-ADAPT scaling cases from the fetched corrected inventory; no Cartesian expansion. \\
\textbf{{Method}} & Geo-ADAPT (\texttt{{static\_geo\_adapt\_vqe}}), full-meta parent generators only. \\
\textbf{{Replacement rule}} & Selection with replacement except that an immediate repeat is disabled. \\
\textbf{{Inner optimizer}} & Powell, maximum 200 inner iterations. \\
\textbf{{Outer horizons}} & Source-completed fixed horizons of 20, 30, or 50 iterations, case dependent. \\
\textbf{{Plateau rule}} & First accepted history prefix with same-cutoff error at most $1.10$ times the best observed error over the completed horizon. \\
\textbf{{Error axis}} & Same-cutoff $|E_{{\rm Geo}}-E_{{\rm ED}}|$; logarithmic scale; zero values are plotted at $10^{{-16}}$ and retained as zero in tables/provenance. \\
\textbf{{Plot policy}} & One one-column solid Geo curve per case, integer iteration ticks, exactly one triangle at $k_{{\rm pl}}$, and no target line. \\
\textbf{{Query currency}} & $S=N_{{H,\mathrm{{outer}}}}+N_{{H,\mathrm{{refit}}}}+N_{{\mathrm{{grad}}}}+N_{{\mathrm{{metric}}}}+N_{{\mathrm{{other}}}}$ through the selected history row. \\
\textbf{{Qiskit prefix costs}} & Coefficient-aware generator synthesis; optimization level 0; transpiler seed 7; {qiskit_ok} compiled, {qiskit_blocked} explicitly blocked. \\
\textbf{{Strict replay}} & {latex_escape(strict_summary)}. Structural prefix reconstruction is separate from strict energy replay. \\
\textbf{{Source inventory}} & \url{{{latex_escape(rel(inventory_path))}}}, SHA-256 \texttt{{{inventory_sha256}}}. \\
\textbf{{Machine-readable outputs}} & \url{{{latex_escape(rel(provenance_json))}}}; \url{{{latex_escape(rel(provenance_csv))}}}; \url{{{latex_escape(rel(manifest_json))}}}. \\
\textbf{{Manuscript mutation}} & None. This is a standalone support report and appendix-ready fragment. \\
\end{{tabular}}

\vfill
\textbf{{Ordered-pair contract.}} The 34 rows follow the inventory order verbatim: 12 Hubbard--Holstein scaling rows, 6 Hubbard rows, 8 spin--boson rows, and 8 Bose--Hubbard rows. The working/reference cutoff fields are carried case by case from the inventory.
\clearpage
\section*{{Appendix-ready Geo-ADAPT scaling evidence}}
\input{{{appendix_fragment.name}}}
\end{{document}}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tex, encoding="utf-8")


def compile_latex(tex_path: Path) -> dict[str, Any]:
    latexmk = shutil.which("latexmk")
    tectonic = shutil.which("tectonic")
    if latexmk:
        command = [
            latexmk,
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            tex_path.name,
        ]
        builder = "latexmk"
    elif tectonic:
        command = [tectonic, "--keep-logs", "--reruns", "2", tex_path.name]
        builder = "tectonic"
    else:
        raise RuntimeError("Neither latexmk nor tectonic is available")
    env = dict(os.environ)
    env.update(SOURCE_DATE_EPOCH="0", FORCE_SOURCE_DATE="1", TZ="UTC")
    completed = subprocess.run(
        command,
        cwd=tex_path.parent,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{builder} failed with exit {completed.returncode}\n"
            f"STDOUT:\n{completed.stdout[-8000:]}\nSTDERR:\n{completed.stderr[-8000:]}"
        )
    pdf_path = tex_path.with_suffix(".pdf")
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)
    return {
        "builder": builder,
        "command": command,
        "returncode": completed.returncode,
        "pdf": rel(pdf_path),
        "pdf_sha256": sha256(pdf_path),
        "log": rel(tex_path.with_suffix(".log")) if tex_path.with_suffix(".log").is_file() else None,
    }


def build_report(
    *,
    inventory_path: Path = DEFAULT_INVENTORY,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    compile_qiskit: bool = True,
    grouped_exact_max_active_qubits: int = 5,
) -> dict[str, Any]:
    inventory_path = inventory_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    prefix_dir = output_dir / "prefixes"
    inventory = read_json(inventory_path)
    inventory_rows = ordered_scaling_inventory_rows(inventory)

    rows: list[dict[str, Any]] = []
    for index, inventory_row in enumerate(inventory_rows):
        row = analyze_case(
            inventory_row,
            order_index=index,
            prefix_dir=prefix_dir,
            compile_qiskit=compile_qiskit,
            grouped_exact_max_active_qubits=grouped_exact_max_active_qubits,
        )
        row["plot"] = plot_case(row, plot_dir=plot_dir)
        rows.append(row)

    provenance_json = output_dir / f"{STEM}_provenance.json"
    provenance_csv = output_dir / f"{STEM}_provenance.csv"
    manifest_json = output_dir / f"{STEM}_manifest.json"
    appendix_fragment = output_dir / f"{STEM}_appendix_fragment.tex"
    report_tex = output_dir / f"{STEM}_support_report.tex"
    report_pdf = report_tex.with_suffix(".pdf")

    write_csv(provenance_csv, rows)
    write_appendix_fragment(appendix_fragment, rows)
    write_report_tex(
        report_tex,
        rows=rows,
        inventory_path=inventory_path,
        inventory_sha256=sha256(inventory_path),
        provenance_json=provenance_json,
        provenance_csv=provenance_csv,
        manifest_json=manifest_json,
        appendix_fragment=appendix_fragment,
    )
    latex = compile_latex(report_tex)

    qiskit_counts = Counter(str(row["qiskit_prefix_cost"]["status"]) for row in rows)
    strict_counts = Counter(str(row["strict_replay"]["status"]) for row in rows)
    provenance = {
        "schema": SCHEMA,
        "source_inventory_generated_utc": inventory.get("generated_utc"),
        "scope": (
            "Exactly 34 ordered Geo-ADAPT scaling cases; Hubbard-Holstein L=2 main-body rows excluded"
        ),
        "case_order_contract": list(EXPECTED_SCALING_CASE_IDS),
        "selection_policy": "ordered_inventory_rows_only_no_cartesian_expansion",
        "plateau_policy": {
            "id": "first_prefix_within_10_percent_of_best_observed_error_v1",
            "relative_tolerance": PLATEAU_REL_TOL,
        },
        "plot_policy": {
            "one_plot_per_case": True,
            "layout": "one_column_single_axes",
            "method": "Geo-ADAPT",
            "color": GEO_COLOR,
            "line_style": "solid",
            "curve_repeated_markers": False,
            "one_triangle_at_k_pl": True,
            "integer_x_ticks": True,
            "log_y": True,
            "target_line": False,
            "zero_error_plot_floor": PLOT_ERROR_FLOOR,
        },
        "method_contract": {
            "algorithm_id": "static_geo_adapt_vqe",
            "optimizer": "powell",
            "optimizer_maxiter": 200,
            "immediate_repeat_allowed": False,
            "immediate_repeat_policy": "with_replacement_except_immediate_repeat",
            "pool_policy": "full_meta_parent_generators_only",
        },
        "source": {
            "inventory_json": rel(inventory_path),
            "inventory_sha256": sha256(inventory_path),
        },
        "summary": {
            "row_count": len(rows),
            "family_counts": dict(sorted(Counter(str(row["family"]) for row in rows).items())),
            "qiskit_status_counts": dict(sorted(qiskit_counts.items())),
            "strict_replay_status_counts": dict(sorted(strict_counts.items())),
            "all_prefix_reconstructions_pass": all(
                row["prefix_reconstruction"]["status"] == "pass" for row in rows
            ),
            "all_terminal_query_identities_pass": all(
                row["query_ledger"]["terminal_identity"]["status"] == "pass"
                for row in rows
            ),
            "all_plot_marker_counts_one": all(
                int(row["plot"]["marker_count_on_curve"]) == 1 for row in rows
            ),
        },
        "artifacts": {
            "provenance_csv": rel(provenance_csv),
            "provenance_csv_sha256": sha256(provenance_csv),
            "appendix_fragment_tex": rel(appendix_fragment),
            "appendix_fragment_sha256": sha256(appendix_fragment),
            "support_report_tex": rel(report_tex),
            "support_report_tex_sha256": sha256(report_tex),
            "support_report_pdf": rel(report_pdf),
            "support_report_pdf_sha256": sha256(report_pdf),
            "plot_directory": rel(plot_dir),
            "prefix_directory": rel(prefix_dir),
            "latex_build": latex,
        },
        "rows": rows,
    }
    write_json(provenance_json, provenance)
    manifest = {
        "schema": f"{SCHEMA}_manifest_v1",
        "source_inventory_generated_utc": inventory.get("generated_utc"),
        "parameter_manifest": {
            "case_count": len(rows),
            "case_order": list(EXPECTED_SCALING_CASE_IDS),
            "method": "Geo-ADAPT",
            "algorithm_id": "static_geo_adapt_vqe",
            "optimizer": "powell",
            "optimizer_maxiter": 200,
            "immediate_repeat_allowed": False,
            "pool_policy": "full_meta_parent_generators_only",
            "plateau_relative_tolerance": PLATEAU_REL_TOL,
            "qiskit_grouped_exact_max_active_qubits": int(grouped_exact_max_active_qubits),
        },
        "source": provenance["source"],
        "summary": provenance["summary"],
        "artifacts": {
            **provenance["artifacts"],
            "provenance_json": rel(provenance_json),
            "provenance_json_sha256": sha256(provenance_json),
        },
    }
    write_json(manifest_json, manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-qiskit", action="store_true")
    parser.add_argument("--grouped-exact-max-active-qubits", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_report(
        inventory_path=args.inventory,
        output_dir=args.output_dir,
        compile_qiskit=not args.skip_qiskit,
        grouped_exact_max_active_qubits=args.grouped_exact_max_active_qubits,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
