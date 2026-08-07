#!/usr/bin/env python3
"""Prepare AVQDS comparator artifacts for the AP-McLachlan Results PDF."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.hardcoded.adapt_circuit_execution import build_ansatz_circuit
from pipelines.qiskit_backend_tools import (
    compile_circuit_for_backend,
    compiled_gate_stats,
    rank_compile_rows,
    resolve_backend_targets,
    safe_circuit_depth,
)
from pipelines.scaffold.runtime_loader import load_scaffold_runtime_input
from pipelines.time_dynamics.benchmarks.legacy_native import (
    _build_layout_for_terms,
    _compiled_executor_for_terms,
    _copy_theta_by_layout_blocks,
    _native_hamiltonian_flow,
    _normalize_state,
    _prepare_scaffold_state,
    _runtime_variational_bundle,
    _state_diagnostic_row,
)
from pipelines.time_dynamics.benchmarks.avqds_tetris import (
    TetrisPoolAtom,
    _term_for_atom,
    initial_avqds_tetris_variational_bundle,
)
from pipelines.time_dynamics.tables.dynamics_benchmark_contract import (
    DynamicsBenchmarkCase,
    json_safe,
)


DEFAULT_BACKEND = "FakeMarrakesh"
DEFAULT_PREFERRED_FAKE_BACKENDS = ("FakeMarrakesh", "FakeNighthawk", "FakeFez")
SEED_EXACT_OVERLAY_TOLERANCE = 1.0e-5
AVQDS_SCHEMA_CONFIG = {
    "generic_avqds_benchmark_v1": {
        "family": "AVQDS",
        "steps_key": "avqds_steps",
        "residual_key": "rhs_residual_ratio",
    },
    "generic_avqds_t_benchmark_v1": {
        "family": "PF-target adaptive tangent",
        "steps_key": "avqds_t_steps",
        "residual_key": "target_tangent_residual_ratio",
    },
    "generic_avqds_tetris_benchmark_v1": {
        "family": "AVQDS(T)",
        "steps_key": "avqds_tetris_steps",
        "residual_key": "rhs_residual_ratio",
    },
}


@dataclass(frozen=True)
class TerminalReconstruction:
    case: DynamicsBenchmarkCase
    runtime_input: Any
    flow: Any
    terms: tuple[Any, ...]
    layout: Any
    theta_runtime: np.ndarray
    psi_ref: np.ndarray
    state: np.ndarray
    parity: Mapping[str, Any]
    drive_aligned_ansatz: Mapping[str, Any]
    diagnostic_redundancy_stress: Mapping[str, Any]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON payload must be an object: {path}")
    return dict(payload)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(json_safe(dict(payload)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _max_abs_difference(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    try:
        a = np.asarray(left, dtype=float).reshape(-1)
        b = np.asarray(right, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if (
        a.shape != b.shape
        or a.size == 0
        or not np.all(np.isfinite(a))
        or not np.all(np.isfinite(b))
    ):
        return None
    return float(np.max(np.abs(a - b)))


def _schema_config(payload: Mapping[str, Any]) -> dict[str, str]:
    schema = str(payload.get("schema_version"))
    config = AVQDS_SCHEMA_CONFIG.get(schema)
    if config is None:
        raise ValueError(
            f"Expected one of {tuple(AVQDS_SCHEMA_CONFIG)}, got {schema!r}."
        )
    return dict(config)


def _layout_payload(layout: Any) -> dict[str, Any]:
    return {
        "mode": str(layout.mode),
        "term_order": str(layout.term_order),
        "ignore_identity": bool(layout.ignore_identity),
        "coefficient_tolerance": float(layout.coefficient_tolerance),
        "logical_parameter_count": int(layout.logical_parameter_count),
        "runtime_parameter_count": int(layout.runtime_parameter_count),
        "blocks": [
            {
                "candidate_label": str(block.candidate_label),
                "logical_index": int(block.logical_index),
                "runtime_start": int(block.runtime_start),
                "runtime_count": int(block.runtime_count),
                "execution_mode": str(block.execution_mode),
                "runtime_terms_exyz": [
                    {
                        "pauli_exyz": str(term.pauli_exyz),
                        "coeff_real": float(term.coeff_real),
                        "nq": int(term.nq),
                    }
                    for term in block.terms
                ],
            }
            for block in layout.blocks
        ],
    }


def reconstruct_terminal_avqds(
    payload: Mapping[str, Any],
) -> TerminalReconstruction:
    schema_config = _schema_config(payload)
    case_raw = payload.get("case")
    if not isinstance(case_raw, Mapping):
        raise ValueError("AVQDS payload is missing its benchmark case.")
    case = DynamicsBenchmarkCase.from_mapping(case_raw)
    runtime_input = load_scaffold_runtime_input(
        case.artifact_json,
        loader_mode=case.loader_mode,
        generator_family=case.generator_family,
        fallback_family=case.fallback_family,
    )
    flow = _native_hamiltonian_flow(case, runtime_input)
    is_tetris = str(payload.get("schema_version")) == "generic_avqds_tetris_benchmark_v1"
    if is_tetris:
        (
            terms,
            layout,
            theta,
            psi_ref,
            executor,
            drive_aligned,
            redundancy_stress,
        ) = initial_avqds_tetris_variational_bundle(
            case=case,
            runtime_input=runtime_input,
            flow=flow,
        )
    else:
        terms, layout, theta, psi_ref, executor, drive_aligned = (
            _runtime_variational_bundle(
                runtime_input,
                hamiltonian=flow.hamiltonian,
                drive_aligned_ansatz=bool(flow.drive_enabled),
            )
        )
        redundancy_stress = {
            "enabled": False,
            "applied": False,
            "reason": "not_avqds_tetris",
        }
    current_terms = tuple(terms)
    theta_current = np.asarray(theta, dtype=float).reshape(-1)
    candidate_pool = tuple(runtime_input.candidate_pool_terms)
    events_raw = payload.get("append_events", ())
    steps_raw = payload.get(schema_config["steps_key"], ())
    if not isinstance(events_raw, Sequence) or isinstance(events_raw, (str, bytes)):
        raise ValueError("AVQDS append_events must be a sequence.")
    if not isinstance(steps_raw, Sequence) or isinstance(steps_raw, (str, bytes)):
        raise ValueError(
            f"AVQDS step ledger {schema_config['steps_key']} must be a sequence."
        )
    events_by_interval: dict[int, list[dict[str, Any]]] = {}
    for event in events_raw:
        if isinstance(event, Mapping):
            events_by_interval.setdefault(int(event["interval_index"]), []).append(dict(event))

    for expected_index, step_raw in enumerate(steps_raw):
        if not isinstance(step_raw, Mapping):
            raise ValueError(f"AVQDS step {expected_index} is not an object.")
        step = dict(step_raw)
        interval_index = int(step.get("interval_index", expected_index))
        if interval_index != expected_index:
            raise ValueError(
                f"AVQDS steps are not contiguous at {expected_index}: {interval_index}."
            )
        for event in events_by_interval.get(interval_index, ()):
            if is_tetris:
                global_layer_index = int(event["global_layer_index"])
                pauli_terms = tuple(str(value) for value in event.get("pauli_terms", ()))
                if int(event.get("count", len(pauli_terms))) != len(pauli_terms):
                    raise ValueError(
                        f"TETRIS layer {global_layer_index} count does not match Pauli terms."
                    )
                appended_terms = []
                for local_index, pauli_exyz in enumerate(pauli_terms):
                    atom = TetrisPoolAtom(
                        pool_index=int(local_index),
                        pauli_exyz=str(pauli_exyz),
                        qubit_support=tuple(
                            index
                            for index, letter in enumerate(str(pauli_exyz).lower())
                            if letter not in {"e", "i"}
                        ),
                        source_labels=("terminal_reconstruction",),
                        nq=len(str(pauli_exyz)),
                    )
                    appended_terms.append(
                        _term_for_atom(
                            atom,
                            label=(
                                f"avqds_tetris_layer_{global_layer_index}::"
                                f"{local_index}::{pauli_exyz}"
                            ),
                        )
                    )
                current_terms = current_terms + tuple(appended_terms)
            else:
                candidate_index = int(event["candidate_pool_index"])
                if candidate_index < 0 or candidate_index >= len(candidate_pool):
                    raise IndexError(
                        f"AVQDS candidate index {candidate_index} is outside pool size "
                        f"{len(candidate_pool)}."
                    )
                current_terms = current_terms + (candidate_pool[candidate_index],)
            new_layout = _build_layout_for_terms(current_terms, reference_layout=layout)
            theta_current = _copy_theta_by_layout_blocks(
                old_theta=theta_current,
                old_layout=layout,
                new_layout=new_layout,
            )
            layout = new_layout
            executor = _compiled_executor_for_terms(current_terms, layout)
        theta_dot = np.asarray(step.get("theta_dot", ()), dtype=float).reshape(-1)
        if int(theta_dot.size) != int(theta_current.size):
            raise ValueError(
                f"AVQDS theta_dot length mismatch at interval {interval_index}: "
                f"{theta_dot.size} vs {theta_current.size}."
            )
        theta_current = theta_current + float(step["dt"]) * theta_dot
        declared_count = int(step.get("parameter_count", theta_current.size))
        if declared_count != int(theta_current.size):
            raise ValueError(
                f"AVQDS parameter count mismatch at interval {interval_index}: "
                f"{declared_count} vs {theta_current.size}."
            )

    state = _normalize_state(_prepare_scaffold_state(executor, psi_ref, theta_current))
    trajectory = payload.get("trajectory", ())
    if not isinstance(trajectory, Sequence) or not trajectory:
        raise ValueError("AVQDS payload has no terminal trajectory row.")
    terminal_saved = trajectory[-1]
    if not isinstance(terminal_saved, Mapping):
        raise ValueError("AVQDS terminal trajectory row is invalid.")
    terminal_rebuilt = _state_diagnostic_row(
        checkpoint_index=int(terminal_saved.get("checkpoint_index", len(trajectory) - 1)),
        time_value=float(terminal_saved["time"]),
        method="generic_avqds_terminal_reconstruction",
        method_kind="avqds",
        state=state,
        exact_state=flow.exact_states[-1],
        hmat=flow.hmat_at_time(float(terminal_saved["time"])),
        **dict(flow.observable_context or {}),
    )
    doublon_abs_difference = None
    if "doublon" in terminal_saved and "doublon" in terminal_rebuilt:
        doublon_abs_difference = abs(
            float(terminal_saved["doublon"])
            - float(terminal_rebuilt["doublon"])
        )
    parity = {
        "schema": "avqds_terminal_reconstruction_parity_v1",
        "runtime_parameter_count_saved": int(terminal_saved["runtime_parameter_count"]),
        "runtime_parameter_count_rebuilt": int(theta_current.size),
        "logical_block_count_saved": int(terminal_saved["logical_block_count"]),
        "logical_block_count_rebuilt": int(layout.logical_parameter_count),
        "energy_total_abs_difference": abs(
            float(terminal_saved["energy_total"])
            - float(terminal_rebuilt["energy_total"])
        ),
        "doublon_abs_difference": doublon_abs_difference,
        "site_occupations_max_abs_difference": _max_abs_difference(
            terminal_saved.get("site_occupations"),
            terminal_rebuilt.get("site_occupations"),
        ),
        "state_norm": float(np.linalg.norm(state)),
    }
    parity["passed"] = bool(
        parity["runtime_parameter_count_saved"]
        == parity["runtime_parameter_count_rebuilt"]
        and parity["logical_block_count_saved"]
        == parity["logical_block_count_rebuilt"]
        and float(parity["energy_total_abs_difference"]) <= 1.0e-10
        and (
            parity["doublon_abs_difference"] is None
            or float(parity["doublon_abs_difference"]) <= 1.0e-10
        )
        and (
            parity["site_occupations_max_abs_difference"] is None
            or float(parity["site_occupations_max_abs_difference"]) <= 1.0e-10
        )
        and abs(float(parity["state_norm"]) - 1.0) <= 1.0e-10
    )
    if not parity["passed"]:
        raise ValueError(f"AVQDS terminal reconstruction parity failed: {parity}")
    return TerminalReconstruction(
        case=case,
        runtime_input=runtime_input,
        flow=flow,
        terms=current_terms,
        layout=layout,
        theta_runtime=theta_current,
        psi_ref=np.asarray(psi_ref, dtype=complex).reshape(-1),
        state=state,
        parity=parity,
        drive_aligned_ansatz=dict(drive_aligned.to_json_dict()),
        diagnostic_redundancy_stress=dict(redundancy_stress),
    )


def _event_rows(payload: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    events_raw = payload.get("append_events", ())
    events: dict[int, dict[str, Any]] = {}
    previous_runtime = None
    trajectory = payload.get("trajectory", ())
    if isinstance(trajectory, Sequence) and trajectory and isinstance(trajectory[0], Mapping):
        previous_runtime = int(trajectory[0].get("runtime_parameter_count", 0))
    for raw in events_raw if isinstance(events_raw, Sequence) else ():
        if not isinstance(raw, Mapping):
            continue
        event = dict(raw)
        current_runtime = int(event.get("runtime_parameter_count", previous_runtime or 0))
        event["appended_runtime_count"] = (
            None if previous_runtime is None else int(current_runtime - previous_runtime)
        )
        previous_runtime = current_runtime
        interval_index = int(event["interval_index"])
        previous = events.get(interval_index)
        if previous is None:
            event["event_count"] = 1
            event["selected_labels"] = list(event.get("pauli_terms", ()))
            events[interval_index] = event
        else:
            previous["event_count"] = int(previous.get("event_count", 1)) + 1
            previous["appended_runtime_count"] = int(
                previous.get("appended_runtime_count", 0) or 0
            ) + int(event.get("appended_runtime_count", 0) or 0)
            previous.setdefault("selected_labels", []).extend(event.get("pauli_terms", ()))
    return events


def _event_plot_label(event: Mapping[str, Any]) -> str:
    appended_count = int(event.get("appended_runtime_count", 0) or 0)
    layer_count = int(event.get("event_count", 1) or 1)
    if event.get("selected_labels") or event.get("pauli_terms"):
        layer_word = "layer" if layer_count == 1 else "layers"
        term_word = "term" if appended_count == 1 else "terms"
        return (
            f"+{appended_count} Pauli {term_word} in "
            f"{layer_count} TETRIS {layer_word}"
        )
    candidate_label = str(event.get("candidate_label", "candidate"))
    return f"+{appended_count} coordinate via {candidate_label}"


def build_results_report_payload(
    *,
    avqds_payload: Mapping[str, Any],
    reference_ap_payload: Mapping[str, Any],
    raw_payload_path: Path,
    reference_ap_path: Path,
    label: str,
    comparison_runs: Sequence[int],
) -> dict[str, Any]:
    schema_config = _schema_config(avqds_payload)
    trajectory = avqds_payload.get("trajectory", ())
    reference_rows = reference_ap_payload.get("plot_rows", ())
    if not isinstance(trajectory, Sequence) or not isinstance(reference_rows, Sequence):
        raise ValueError("Comparator and AP reference trajectories must be sequences.")
    if len(trajectory) != len(reference_rows):
        raise ValueError(
            f"Comparator/AP reporting grids differ: {len(trajectory)} vs "
            f"{len(reference_rows)}."
        )
    events = _event_rows(avqds_payload)
    steps = avqds_payload.get(schema_config["steps_key"], ())
    plot_rows: list[dict[str, Any]] = []
    seed_energy_diffs: list[float] = []
    seed_doublon_diffs: list[float] = []

    for index, (raw_row, ref_row) in enumerate(zip(trajectory, reference_rows)):
        if not isinstance(raw_row, Mapping) or not isinstance(ref_row, Mapping):
            raise ValueError(f"Invalid reporting row at index {index}.")
        time_value = float(raw_row["time"])
        if not np.isclose(time_value, float(ref_row["time"]), atol=1.0e-12, rtol=0.0):
            raise ValueError(f"Comparator/AP reporting times differ at index {index}.")
        energy = float(raw_row["energy_total"])
        reference_energy = _float_or_none(ref_row.get("reference_energy"))
        seed_reference_energy = _float_or_none(raw_row.get("energy_total_exact"))
        doublon = _float_or_none(raw_row.get("doublon"))
        doublon_exact = _float_or_none(ref_row.get("doublon_exact"))
        seed_doublon_exact = _float_or_none(raw_row.get("doublon_exact"))
        if seed_reference_energy is not None and ref_row.get("seed_reference_energy") is not None:
            seed_energy_diffs.append(
                abs(seed_reference_energy - float(ref_row["seed_reference_energy"]))
            )
        if seed_doublon_exact is not None and ref_row.get("seed_doublon_exact") is not None:
            seed_doublon_diffs.append(
                abs(seed_doublon_exact - float(ref_row["seed_doublon_exact"]))
            )
        event = events.get(index)
        theta_dot_l2 = None
        if index < len(steps) and isinstance(steps[index], Mapping):
            theta_dot_l2 = float(
                np.linalg.norm(np.asarray(steps[index].get("theta_dot", ()), dtype=float))
            )
        site_occupations = raw_row.get("site_occupations")
        site_occupations_exact = ref_row.get("site_occupations_exact")
        seed_site_occupations_exact = raw_row.get("site_occupations_exact")
        row = {
            "index": int(index),
            "time": float(time_value),
            "energy_expectation": float(energy),
            "reference_energy": reference_energy,
            "seed_reference_energy": seed_reference_energy,
            "energy_error": None if reference_energy is None else float(energy - reference_energy),
            "abs_energy_error": None if reference_energy is None else abs(float(energy - reference_energy)),
            "seed_energy_error": (
                None
                if seed_reference_energy is None
                else float(energy - seed_reference_energy)
            ),
            "seed_abs_energy_error": (
                None
                if seed_reference_energy is None
                else abs(float(energy - seed_reference_energy))
            ),
            "doublon": doublon,
            "doublon_exact": doublon_exact,
            "seed_doublon_exact": seed_doublon_exact,
            "abs_doublon_error": (
                None
                if doublon is None or doublon_exact is None
                else abs(float(doublon - doublon_exact))
            ),
            "seed_abs_doublon_error": (
                None
                if doublon is None or seed_doublon_exact is None
                else abs(float(doublon - seed_doublon_exact))
            ),
            "site_occupations": site_occupations,
            "site_occupations_exact": site_occupations_exact,
            "seed_site_occupations_exact": seed_site_occupations_exact,
            "site_occupations_abs_error_max": _max_abs_difference(
                site_occupations, site_occupations_exact
            ),
            "seed_site_occupations_abs_error_max": _max_abs_difference(
                site_occupations, seed_site_occupations_exact
            ),
            "runtime_parameter_count": int(raw_row["runtime_parameter_count"]),
            "logical_parameter_count": int(raw_row["logical_block_count"]),
            "mclachlan_rho_expr": None,
            "mclachlan_residual_ratio": _float_or_none(
                raw_row.get(schema_config["residual_key"])
            ),
            "theta_dot_l2": theta_dot_l2,
            "patch_accepted": event is not None,
            "patch_kind": "append" if event is not None else "stay",
            "patch_appended_count": (
                None if event is None else event.get("appended_runtime_count")
            ),
            "patch_selected_label": "" if event is None else _event_plot_label(event),
            "patch_reason": (
                f"{schema_config['family']} residual-triggered append"
                if event is not None
                else ""
            ),
            "patch_rank_score": None,
            "solve_repair_enabled": False,
            "solve_repair_applied": False,
            "solve_repair_unsupported": False,
        }
        plot_rows.append(row)

    def values(key: str) -> list[float]:
        return [float(row[key]) for row in plot_rows if row.get(key) is not None]

    initial_runtime = int(plot_rows[0]["runtime_parameter_count"])
    final_runtime = int(plot_rows[-1]["runtime_parameter_count"])
    initial_logical = int(plot_rows[0]["logical_parameter_count"])
    final_logical = int(plot_rows[-1]["logical_parameter_count"])
    metrics = dict(avqds_payload.get("metrics", {}))
    pool_complete = metrics.get("candidate_pool_complete")
    if pool_complete is None and str(avqds_payload.get("schema_version")) == (
        "generic_avqds_tetris_benchmark_v1"
    ):
        pool_complete = str(metrics.get("pool_source", "")) in {
            "hamiltonian_pauli",
            "runtime_candidate_pool",
        }
    summary = {
        "label": str(label),
        "comparator_family": schema_config["family"],
        "comparison_results_pdf_runs": [int(x) for x in comparison_runs],
        "drive_enabled": bool(
            dict(dict(avqds_payload.get("case", {})).get("metadata", {}))
            .get("drive", {})
            .get("enable_drive", False)
        ),
        "point_count": int(len(plot_rows)),
        "integrator_method": "Euler",
        "logical_parameter_count_initial": int(initial_logical),
        "logical_parameter_count_final": int(final_logical),
        "runtime_parameter_count_initial": int(initial_runtime),
        "runtime_parameter_count_final": int(final_runtime),
        "active_parameter_count_initial": int(initial_runtime),
        "active_parameter_count_final": int(final_runtime),
        "accepted_append_count": int(len(avqds_payload.get("append_events", ()))),
        "accepted_appended_coordinate_count": int(final_runtime - initial_runtime),
        "accepted_delete_count": 0,
        "accepted_deleted_coordinate_count": 0,
        "accepted_exchange_count": 0,
        "accepted_patch_count": int(len(avqds_payload.get("append_events", ()))),
        "final_abs_energy_error": values("abs_energy_error")[-1],
        "max_abs_energy_error": max(values("abs_energy_error")),
        "mean_abs_energy_error": float(np.mean(values("abs_energy_error"))),
        "seed_final_abs_energy_error": values("seed_abs_energy_error")[-1],
        "seed_max_abs_energy_error": max(values("seed_abs_energy_error")),
        "final_abs_doublon_error": values("abs_doublon_error")[-1],
        "max_abs_doublon_error": max(values("abs_doublon_error")),
        "seed_final_abs_doublon_error": values("seed_abs_doublon_error")[-1],
        "seed_max_abs_doublon_error": max(values("seed_abs_doublon_error")),
        "final_site_occupations_abs_error_max": values(
            "site_occupations_abs_error_max"
        )[-1],
        "max_abs_site_occupations_error": max(
            values("site_occupations_abs_error_max")
        ),
        "seed_final_site_occupations_abs_error_max": values(
            "seed_site_occupations_abs_error_max"
        )[-1],
        "seed_max_abs_site_occupations_error": max(
            values("seed_site_occupations_abs_error_max")
        ),
        "final_mclachlan_residual_ratio": values("mclachlan_residual_ratio")[-1],
        "max_mclachlan_residual_ratio": max(values("mclachlan_residual_ratio")),
        "candidate_pool_complete": bool(pool_complete),
        "candidate_pool_term_count": int(metrics.get("candidate_pool_size", 0)),
        "exact_reference_policy": "reporting_only",
        "unsupported_checkpoint_count": int(
            metrics.get("unsupported_checkpoint_count", 0)
        ),
    }
    reporting_audit = {
        "schema": "avqds_results_pdf_reporting_overlay_audit_v1",
        "reference_ap_trajectory_json": str(reference_ap_path),
        "reference_ap_trajectory_sha256": _sha256_file(reference_ap_path),
        "seed_exact_energy_max_abs_difference": max(seed_energy_diffs or [0.0]),
        "seed_exact_doublon_max_abs_difference": max(seed_doublon_diffs or [0.0]),
        "seed_exact_overlay_consistency_tolerance": SEED_EXACT_OVERLAY_TOLERANCE,
        "exact_data_controller_inputs": False,
        "exact_data_scope": "reporting_only",
    }
    reporting_audit["seed_exact_overlay_consistency_passed"] = bool(
        float(reporting_audit["seed_exact_energy_max_abs_difference"])
        <= SEED_EXACT_OVERLAY_TOLERANCE
        and float(reporting_audit["seed_exact_doublon_max_abs_difference"])
        <= SEED_EXACT_OVERLAY_TOLERANCE
    )
    if not reporting_audit["seed_exact_overlay_consistency_passed"]:
        raise ValueError(f"Seed-exact energy overlays do not match: {reporting_audit}")
    return {
        "schema": "avqds_results_pdf_adapter_v1",
        "source_schema": str(avqds_payload.get("schema_version")),
        "source_avqds_payload_json": str(raw_payload_path),
        "source_avqds_payload_sha256": _sha256_file(raw_payload_path),
        "reporting_reference_ap_trajectory_json": str(reference_ap_path),
        "summary": summary,
        "plot_rows": plot_rows,
        "append_events": [dict(event) for event in events.values()],
        "reporting_overlay_audit": reporting_audit,
        "decision_data_flow": dict(avqds_payload.get("provenance", {})),
    }


def compile_terminal_qiskit_cost(
    reconstruction: TerminalReconstruction,
    *,
    backend_name: str,
    seed_transpiler: int,
    optimization_level: int,
) -> dict[str, Any]:
    nq = int(reconstruction.runtime_input.resolved_problem.layout.total_qubits)
    circuit = build_ansatz_circuit(
        reconstruction.layout,
        reconstruction.theta_runtime,
        nq,
        ref_state=reconstruction.psi_ref,
    )
    targets, resolution_audit = resolve_backend_targets(
        requested_names=(str(backend_name),),
        preferred_fake_backends=DEFAULT_PREFERRED_FAKE_BACKENDS,
        allow_preferred_fallback=True,
        fallback_mode="single",
        allow_runtime_lookup=False,
    )
    rows: list[dict[str, Any]] = []
    for target in targets:
        row: dict[str, Any] = {
            "requested_backend_name": str(target.requested_name),
            "backend_name": str(target.resolved_name),
            "resolution_kind": str(target.resolution_kind),
            "using_fake_backend": bool(target.using_fake_backend),
            "seed_transpiler": int(seed_transpiler),
            "optimization_level": int(optimization_level),
            "transpile_status": "not_run",
        }
        try:
            compiled_info = compile_circuit_for_backend(
                circuit,
                target.backend_obj,
                seed_transpiler=int(seed_transpiler),
                optimization_level=int(optimization_level),
            )
            compiled = compiled_info["compiled"]
            row.update(
                {
                    "transpile_status": "ok",
                    "compiled_depth": int(safe_circuit_depth(compiled)),
                    "compiled_size": int(compiled.size()),
                    "compiled_num_qubits": int(compiled.num_qubits),
                    "logical_to_physical": [
                        int(x) for x in compiled_info.get("logical_to_physical", ())
                    ],
                }
            )
            row.update(compiled_gate_stats(compiled))
        except Exception as exc:
            row.update(
                {
                    "transpile_status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        rows.append(row)
    selected = rank_compile_rows(rows)
    if selected is None:
        raise RuntimeError(f"No Qiskit compile target succeeded: {rows}")
    layout_payload = _layout_payload(reconstruction.layout)
    layout_digest = hashlib.sha256(
        json.dumps(layout_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema": "avqds_terminal_qiskit_compile_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "compile_scope": "final_active_avqds_ansatz_at_terminal_time",
        "backend_name": str(selected["backend_name"]),
        "requested_backend_name": str(backend_name),
        "seed_transpiler": int(seed_transpiler),
        "optimization_level": int(optimization_level),
        "runtime_lookup_enabled": False,
        "local_fake_only": True,
        "N2q": int(selected["compiled_count_2q"]),
        "D2q": int(selected["compiled_depth_2q"]),
        "Dc": int(selected["compiled_depth"]),
        "compiled_size": int(selected["compiled_size"]),
        "compiled_count_1q": int(selected["compiled_count_1q"]),
        "compiled_op_counts": dict(selected.get("compiled_op_counts", {})),
        "logical_parameter_count": int(reconstruction.layout.logical_parameter_count),
        "runtime_parameter_count": int(reconstruction.layout.runtime_parameter_count),
        "final_support_layout_sha256": layout_digest,
        "final_theta_sha256": hashlib.sha256(
            np.asarray(reconstruction.theta_runtime, dtype="<f8").tobytes()
        ).hexdigest(),
        "terminal_reconstruction_parity": dict(reconstruction.parity),
        "drive_aligned_ansatz": dict(reconstruction.drive_aligned_ansatz),
        "diagnostic_redundancy_stress": dict(
            reconstruction.diagnostic_redundancy_stress
        ),
        "resolution_audit": resolution_audit,
        "compile_rows": rows,
    }


def build_qiskit_cost_table(
    *,
    report_payload: Mapping[str, Any],
    report_path: Path,
    raw_payload_path: Path,
    reconstruction: TerminalReconstruction,
    compile_result: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    summary = dict(report_payload.get("summary", {}))
    row = {
        "label": str(label),
        "trajectory_json": str(report_path),
        "raw_trajectory_json": str(raw_payload_path),
        "logical_parameter_count": int(summary["logical_parameter_count_final"]),
        "runtime_parameter_count": int(summary["runtime_parameter_count_final"]),
        "accepted_append_count": int(summary["accepted_append_count"]),
        "accepted_appended_coordinate_count": int(
            summary["accepted_appended_coordinate_count"]
        ),
        "final_abs_energy_error": float(summary["final_abs_energy_error"]),
        "final_abs_doublon_error": float(summary["final_abs_doublon_error"]),
        "N2q": int(compile_result["N2q"]),
        "D2q": int(compile_result["D2q"]),
        "Dc": int(compile_result["Dc"]),
        "qiskit_cost_status": "ok",
        "qiskit_cost_source": "final_active_avqds_ansatz_qiskit_compile",
        "final_support_layout_sha256": str(
            compile_result["final_support_layout_sha256"]
        ),
        "terminal_reconstruction_parity": dict(reconstruction.parity),
    }
    return {
        "schema": "avqds_results_pdf_qiskit_cost_table_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "compile_defaults": {
            "backend_name": str(compile_result["backend_name"]),
            "requested_backend_name": str(compile_result["requested_backend_name"]),
            "seed_transpiler": int(compile_result["seed_transpiler"]),
            "optimization_level": int(compile_result["optimization_level"]),
            "local_fake_only": True,
        },
        "rows": [row],
        "compile_result": dict(compile_result),
    }


def _parse_runs(raw: str) -> tuple[int, ...]:
    return tuple(int(chunk.strip()) for chunk in raw.split(",") if chunk.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-payload", type=Path, required=True)
    parser.add_argument("--reference-ap-trajectory", type=Path, required=True)
    parser.add_argument("--output-report-json", type=Path, required=True)
    parser.add_argument("--output-cost-table-json", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--comparison-runs", default="")
    parser.add_argument("--backend-name", default=DEFAULT_BACKEND)
    parser.add_argument("--seed-transpiler", type=int, default=7)
    parser.add_argument("--optimization-level", type=int, default=2)
    args = parser.parse_args()

    raw_path = Path(args.raw_payload).resolve()
    reference_path = Path(args.reference_ap_trajectory).resolve()
    output_report = Path(args.output_report_json).resolve()
    output_cost = Path(args.output_cost_table_json).resolve()
    raw_payload = _load_json(raw_path)
    reference_payload = _load_json(reference_path)
    reconstruction = reconstruct_terminal_avqds(raw_payload)
    report_payload = build_results_report_payload(
        avqds_payload=raw_payload,
        reference_ap_payload=reference_payload,
        raw_payload_path=raw_path,
        reference_ap_path=reference_path,
        label=str(args.label),
        comparison_runs=_parse_runs(str(args.comparison_runs)),
    )
    report_payload["terminal_reconstruction_parity"] = dict(reconstruction.parity)
    report_payload["final_support_layout"] = _layout_payload(reconstruction.layout)
    report_payload["final_theta_runtime"] = [
        float(x) for x in reconstruction.theta_runtime.tolist()
    ]
    _write_json(output_report, report_payload)
    compile_result = compile_terminal_qiskit_cost(
        reconstruction,
        backend_name=str(args.backend_name),
        seed_transpiler=int(args.seed_transpiler),
        optimization_level=int(args.optimization_level),
    )
    cost_payload = build_qiskit_cost_table(
        report_payload=report_payload,
        report_path=output_report,
        raw_payload_path=raw_path,
        reconstruction=reconstruction,
        compile_result=compile_result,
        label=str(args.label),
    )
    _write_json(output_cost, cost_payload)
    print(
        json.dumps(
            {
                "report_json": str(output_report),
                "cost_table_json": str(output_cost),
                "terminal_reconstruction_parity": reconstruction.parity,
                "N2q": compile_result["N2q"],
                "D2q": compile_result["D2q"],
                "Dc": compile_result["Dc"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
