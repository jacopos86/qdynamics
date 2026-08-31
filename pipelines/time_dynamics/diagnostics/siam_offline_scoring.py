"""Offline accuracy scoring and matched-pair classification for SIAM.

Raw SIAM trajectories are exact-reference blind.  This module is the only
place where their recorded checkpoint statevectors meet the immutable exact
reference.  The resulting document is JSON-safe and digest-bound so later
reporting can consume it without rediscovering or silently changing the
scoring inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.time_dynamics.diagnostics.siam_exact_reference_store import (
    SIAM_EXACT_REFERENCE_STORE_V2,
    SIAMExactReferenceArtifact,
    SIAMExactReferencePrefixView,
    SIAMExactReferenceSliceView,
    SIAMExactReferenceStore,
)
from pipelines.time_dynamics.paper_ii_runs import (
    SIAM_NB4_CONTINUATION_TIMES,
    SIAM_NB4_FULL_TIMES,
    SIAM_NB4_SCREEN_TIMES,
)
from pipelines.time_dynamics.resume import (
    CONTROLLER_RESUME_SCHEMA_V1,
    RESUME_SCHEMA_V2,
    ControllerResumeState,
    runtime_input_from_resume_state,
)
from pipelines.time_dynamics.run_lock import assert_comparable
from pipelines.time_dynamics.siam_seed import load_siam_seed_manifest
from src.quantum.siam_model import SIAMLayout, project_to_sector, sector_indices
from src.quantum.siam_observables import (
    SIAM_OBSERVABLES_V2,
    screening_observable_operators,
)
from src.quantum.siam_reference import SIAM_REFERENCE_V2, sector_hamiltonian
from src.quantum.vqe_latex_python_pairs import hamiltonian_matrix


SIAM_OFFLINE_SCORE_V1 = "siam_offline_accuracy_score_v1"
SIAM_OFFLINE_SCORE_V2 = "siam_offline_accuracy_score_v2"
SIAM_OFFLINE_SCORE_V3 = "siam_offline_accuracy_score_v3"
SIAM_MATCHED_PAIR_V1 = "siam_matched_accuracy_pair_v1"
SIAM_MATCHED_PAIR_V2 = "siam_matched_accuracy_pair_v2"
SIAM_MATCHED_PAIR_V3 = "siam_matched_accuracy_pair_v3"
SIAM_SIGMA_V2_TRANSITION_V1 = "siam_sigma_v2_reporting_transition_v1"
SIAM_ACCURACY_POLICY_V2 = "siam_response_matched_accuracy_policy_v2"
SIAM_NB4_PRIMITIVE_NAMES = ("m_d", "D_d", "C_1", "C_2", "C_3", "C_4")

INDIVIDUAL_RESPONSE_TARGET = 0.05
BROAD_RESPONSE_GUARD = 0.15
MEAN_INFIDELITY_CEILING = 1.0e-4
AGGREGATE_BURDEN_RATIO_CEILING = 2.0
COMPONENT_RATIO_CEILING = 2.0
COMPONENT_RATIO_DEADBAND = 0.10
RESPONSE_FLOOR = 1.0e-12
# Historical v1 absolute guard. Retained only so already-written v1 receipts
# remain intelligible; v2 scoring never reads it.
ENERGY_DRIFT_CEILING_V1 = 1.0e-4
# Backward-compatible name for code that only reads historical v1 receipts.
ENERGY_DRIFT_CEILING = ENERGY_DRIFT_CEILING_V1
RELATIVE_ENERGY_DRIFT_CEILING = 1.0e-3
INITIAL_ENERGY_ERROR_CEILING = 1.0e-10
INITIAL_STATE_INFIDELITY_CEILING = 1.0e-12
STATE_NORM_ERROR_CEILING = 1.0e-8
SECTOR_LEAKAGE_CEILING = 1.0e-8


def _accuracy_policy_v2_record() -> dict[str, Any]:
    """Return the one canonical policy-v2 identity carried by every score."""

    return {
        "schema": SIAM_ACCURACY_POLICY_V2,
        "energy_guard": "sector_spectral_width_normalized_max_drift",
        "energy_in_aggregate_burden": False,
        "energy_in_componentwise_matching": True,
    }


def _accuracy_thresholds_v2_record() -> dict[str, float]:
    """Return the immutable numerical constants defining policy v2."""

    return {
        "response_floor": RESPONSE_FLOOR,
        "individual_response_target": INDIVIDUAL_RESPONSE_TARGET,
        "broad_response_guard": BROAD_RESPONSE_GUARD,
        "mean_infidelity_ceiling": MEAN_INFIDELITY_CEILING,
        "relative_energy_drift_ceiling": RELATIVE_ENERGY_DRIFT_CEILING,
        "initial_state_infidelity_ceiling": INITIAL_STATE_INFIDELITY_CEILING,
        "initial_energy_error_ceiling": INITIAL_ENERGY_ERROR_CEILING,
        "aggregate_burden_ratio_ceiling": AGGREGATE_BURDEN_RATIO_CEILING,
        "componentwise_ratio_deadband": COMPONENT_RATIO_DEADBAND,
        "componentwise_ratio_ceiling": COMPONENT_RATIO_CEILING,
    }


def _observable_contract_v2_record() -> dict[str, Any]:
    """The corrected reporting algebra; deliberately outside admission gates."""

    return {
        "schema": SIAM_OBSERVABLES_V2,
        "reporting_role": "offline_derived_from_checkpoint_statevectors",
        "impurity_number": "N_d=n_d_up+n_d_down",
        "impurity_spin_square": "S_d2=3/4*(N_d-2*D_d)",
        "integrated_screening": "Sigma_r=S_d2+sum_{j<=r}C_j",
        "primitive_gate_set_unchanged": True,
    }


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest_document(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(dict(value))).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _time_grid_sha256(times: Sequence[float]) -> str:
    return hashlib.sha256(
        json.dumps(
            [float(value) for value in times],
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _finite_series(values: Sequence[float], *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if not array.size or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be nonempty and finite.")
    return array


def _recorded_state(row: Mapping[str, Any], *, expected_dimension: int) -> np.ndarray:
    if "psi_real" not in row or "psi_imag" not in row:
        raise ValueError("Every SIAM checkpoint must record its statevector.")
    real = np.asarray(row["psi_real"], dtype=float).reshape(-1)
    imag = np.asarray(row["psi_imag"], dtype=float).reshape(-1)
    if real.size != expected_dimension or imag.size != expected_dimension:
        raise ValueError(
            "SIAM checkpoint state dimension mismatch: "
            f"expected {expected_dimension}, got {real.size}/{imag.size}."
        )
    state = real + 1.0j * imag
    if not np.all(np.isfinite(state)):
        raise ValueError("SIAM checkpoint statevector must be finite.")
    return state


def _infidelity(left: np.ndarray, right: np.ndarray) -> float:
    lhs = np.asarray(left, dtype=complex).reshape(-1)
    rhs = np.asarray(right, dtype=complex).reshape(-1)
    lhs_norm = float(np.real(np.vdot(lhs, lhs)))
    rhs_norm = float(np.real(np.vdot(rhs, rhs)))
    if lhs.size != rhs.size or lhs_norm <= 0.0 or rhs_norm <= 0.0:
        raise ValueError("State fidelity requires compatible nonzero states.")
    fidelity = float(abs(np.vdot(lhs, rhs)) ** 2 / (lhs_norm * rhs_norm))
    if not math.isfinite(fidelity):
        raise ValueError("State fidelity is nonfinite.")
    return float(np.clip(1.0 - fidelity, 0.0, 1.0))


def _primitive_names(n_bath: int) -> tuple[str, ...]:
    return ("m_d", "D_d", *(f"C_{site}" for site in range(1, n_bath + 1)))


def _derived_names(n_bath: int) -> tuple[str, ...]:
    return tuple(f"Sigma_{site}" for site in range(1, n_bath + 1))


def _aggregate_accuracy_burden(
    quality_components: Mapping[str, float],
) -> float:
    """Scalar burden after every component is normalized by its own target."""

    correlation_names = sorted(
        (name for name in quality_components if name.startswith("C_")),
        key=lambda name: int(name.split("_", 1)[1]),
    )
    if not correlation_names:
        raise ValueError("SIAM accuracy burden requires at least one C_j component.")
    required = {"m_d", "D_d", "fidelity", *correlation_names}
    if set(quality_components) != required:
        raise ValueError("SIAM accuracy burden component schema mismatch.")
    values = {
        name: float(value) for name, value in quality_components.items()
    }
    if any(not math.isfinite(value) or value < 0.0 for value in values.values()):
        raise ValueError("SIAM normalized accuracy components must be finite and nonnegative.")
    correlation_squared_mean = float(
        np.mean([values[name] ** 2 for name in correlation_names])
    )
    return math.sqrt(
        0.25
        * (
            values["m_d"] ** 2
            + values["D_d"] ** 2
            + correlation_squared_mean
            + values["fidelity"] ** 2
        )
    )


def _quality_components_from_score(score: Mapping[str, Any]) -> dict[str, float]:
    thresholds = score["thresholds"]
    response_target = float(thresholds["individual_response_target"])
    fidelity_ceiling = float(thresholds["mean_infidelity_ceiling"])
    if response_target <= 0.0 or fidelity_ceiling <= 0.0:
        raise ValueError("SIAM accuracy normalization targets must be positive.")
    metrics = score["primitive_response_metrics"]
    components = {
        str(name): float(metric["response_normalized_mean_absolute_error"])
        / response_target
        for name, metric in metrics.items()
    }
    components["fidelity"] = (
        float(score["summary"]["mean_infidelity"]) / fidelity_ceiling
    )
    return components


def _relative_energy_guard(
    observed_energy: Sequence[float],
    *,
    spectral_minimum: float,
    spectral_maximum: float,
) -> dict[str, Any]:
    """Shift- and scale-invariant conservation guard for policy v2."""

    energy = _finite_series(observed_energy, name="observed SIAM energy")
    minimum = float(spectral_minimum)
    maximum = float(spectral_maximum)
    width = maximum - minimum
    if not all(math.isfinite(value) for value in (minimum, maximum, width)):
        raise ValueError("SIAM sector spectral bounds must be finite.")
    if width <= 0.0:
        raise ValueError("SIAM sector spectral width must be positive.")
    absolute_drift = np.abs(energy - energy[0])
    max_absolute_drift = float(np.max(absolute_drift))
    relative_drift = max_absolute_drift / width
    if not math.isfinite(relative_drift):
        raise ValueError("SIAM relative energy drift is nonfinite.")
    return {
        "sector_spectral_minimum": minimum,
        "sector_spectral_maximum": maximum,
        "sector_spectral_width": width,
        "mean_absolute_drift": float(np.mean(absolute_drift)),
        "max_absolute_drift": max_absolute_drift,
        "g_E": relative_drift,
        "relative_spectral_drift_ceiling": RELATIVE_ENERGY_DRIFT_CEILING,
        "pass": bool(relative_drift <= RELATIVE_ENERGY_DRIFT_CEILING),
    }


def _pair_quality_components_from_score(
    score: Mapping[str, Any],
) -> dict[str, float]:
    components = _quality_components_from_score(score)
    ceiling = float(score["thresholds"]["relative_energy_drift_ceiling"])
    if ceiling <= 0.0 or not math.isfinite(ceiling):
        raise ValueError("SIAM relative energy-drift ceiling must be positive.")
    components["energy"] = (
        float(score["diagnostics"]["energy"]["g_E"])
        / ceiling
    )
    return components


def _terminal_success_evidence(
    run: Mapping[str, Any], summary: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate a runner-provided completion marker without inventing one."""

    sources: list[str] = []
    for container_name, container in (("run", run), ("summary", summary)):
        for field in ("terminal_success", "run_success", "success"):
            if field not in container:
                continue
            if container[field] is not True:
                raise ValueError(
                    f"SIAM terminal-success marker {container_name}.{field} "
                    "is not true."
                )
            sources.append(f"{container_name}.{field}")
        if "status" in container:
            status = str(container["status"]).strip().lower()
            if status not in {"complete", "completed", "success", "succeeded"}:
                raise ValueError(
                    f"SIAM terminal status {container_name}.status={status!r} "
                    "does not certify success."
                )
            sources.append(f"{container_name}.status")
    receipt = run.get("completion_receipt")
    if receipt is not None:
        if not isinstance(receipt, Mapping):
            raise ValueError("SIAM completion_receipt must be a mapping.")
        receipt_success = receipt.get("success")
        receipt_status = str(receipt.get("status", "")).strip().lower()
        if receipt_success is not True and receipt_status not in {
            "complete",
            "completed",
            "success",
            "succeeded",
        }:
            raise ValueError("SIAM completion receipt does not certify success.")
        sources.append("run.completion_receipt")
    return {
        "runner_marker_provided": bool(sources),
        "validated_sources": sources,
        "implicit_boundary_checks_required": not bool(sources),
    }


def _validate_resume_contract(
    run: Mapping[str, Any],
    *,
    seed_path: Path,
    terminal_time: float,
) -> dict[str, Any]:
    resume = run.get("resume_state")
    if not isinstance(resume, Mapping):
        raise ValueError("SIAM raw run is missing its resume payload.")
    if str(resume.get("schema")) != RESUME_SCHEMA_V2:
        raise ValueError(f"SIAM raw run requires resume schema {RESUME_SCHEMA_V2!r}.")
    if str(resume.get("problem_family", "")).strip().lower() != "siam":
        raise ValueError("SIAM resume payload has the wrong problem family.")
    if str(resume.get("parameterization_mode")) != "logical_shared":
        raise ValueError("SIAM resume V2 must preserve logical-shared coordinates.")
    resume_time = float(resume.get("time", float("nan")))
    if not math.isfinite(resume_time) or not math.isclose(
        resume_time,
        float(terminal_time),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("SIAM resume time does not match the terminal checkpoint.")
    controller = resume.get("controller_state")
    if not isinstance(controller, Mapping) or str(controller.get("schema")) != (
        CONTROLLER_RESUME_SCHEMA_V1
    ):
        raise ValueError("SIAM resume V2 requires valid controller resume state.")
    ControllerResumeState.from_json_dict(controller)
    reconstructed = runtime_input_from_resume_state(
        resume,
        seed_artifact_json=str(seed_path),
    )
    theta_runtime = np.asarray(reconstructed.theta_runtime, dtype=float).reshape(-1)
    theta_logical = np.asarray(reconstructed.theta_logical, dtype=float).reshape(-1)
    if not np.all(np.isfinite(theta_runtime)) or not np.all(np.isfinite(theta_logical)):
        raise ValueError("SIAM reconstructed resume angles must be finite.")
    return {
        "schema": RESUME_SCHEMA_V2,
        "controller_schema": CONTROLLER_RESUME_SCHEMA_V1,
        "terminal_time": resume_time,
        "logical_coordinate_count": int(theta_logical.size),
        "reconstruction_pass": True,
    }


def _verified_run_contract(
    run: Mapping[str, Any],
    *,
    seed_sha256: str,
) -> tuple[
    Mapping[str, Any],
    tuple[Mapping[str, Any], ...],
    tuple[float, ...],
    str,
    Mapping[str, Any],
]:
    lock = run.get("run_lock")
    if not isinstance(lock, Mapping):
        raise ValueError("SIAM raw run is missing its run lock.")
    physics = lock.get("physics")
    if not isinstance(physics, Mapping) or physics.get("family_key") != "siam":
        raise ValueError("Offline SIAM scoring requires family_key='siam'.")
    if physics.get("seed_sha256") != seed_sha256:
        raise ValueError("SIAM raw run and verified seed digest do not match.")
    if physics.get("drive_profile") is not None:
        raise ValueError("SIAM hybridization quench must not claim an external drive.")
    if physics.get("exact_reference_json") is not None or physics.get(
        "exact_reference_sha256"
    ) is not None:
        raise ValueError("SIAM raw trajectory was not exact-reference blind.")
    flow = run.get("decision_data_flow")
    if not isinstance(flow, Mapping):
        raise ValueError("SIAM raw trajectory lacks an exact-blind decision contract.")
    for field in (
        "uses_reference_for_decision",
        "uses_exact_reference_for_decision",
        "uses_future_exact_forecast_for_decision",
    ):
        if flow.get(field) is not False:
            raise ValueError(
                "SIAM raw trajectory lacks an exact-blind decision contract: "
                f"{field} must be explicitly false."
            )
    trajectory = run.get("trajectory")
    trajectory_metadata = (
        trajectory.get("metadata") if isinstance(trajectory, Mapping) else None
    )
    if trajectory_metadata is not None:
        if not isinstance(trajectory_metadata, Mapping):
            raise ValueError("SIAM trajectory metadata must be a mapping.")
        for field in (
            "uses_reference_for_decision",
            "uses_exact_reference_for_decision",
            "uses_future_exact_forecast_for_decision",
        ):
            if trajectory_metadata.get(field) is not False:
                raise ValueError(
                    "SIAM raw trajectory lacks an exact-blind trajectory contract: "
                    f"trajectory.metadata.{field} must be explicitly false."
                )
    summary = run.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("SIAM raw run is missing its completion summary.")
    uncured = summary.get("uncured_local_subdivision_count")
    if isinstance(uncured, bool) or not isinstance(uncured, int):
        raise ValueError(
            "SIAM summary.uncured_local_subdivision_count must be present as an integer."
        )
    if uncured != 0:
        raise ValueError(
            "SIAM run has nonzero summary.uncured_local_subdivision_count."
        )
    terminal_evidence = _terminal_success_evidence(run, summary)
    rows = tuple(run.get("plot_rows") or ())
    if not rows or any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("SIAM raw run needs nonempty checkpoint rows.")
    times = tuple(float(row["time"]) for row in rows)
    if any(not math.isfinite(value) for value in times):
        raise ValueError("SIAM raw-run times must be finite.")
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError("SIAM raw-run times must be strictly increasing.")
    grid = physics.get("time_grid")
    if not isinstance(grid, Mapping):
        raise ValueError("SIAM run lock lacks its complete ordered time grid.")
    if int(grid.get("point_count", -1)) != len(times):
        raise ValueError("SIAM run-lock time-grid point count mismatch.")
    if float(grid.get("t_initial", float("nan"))) != times[0] or float(
        grid.get("t_final", float("nan"))
    ) != times[-1]:
        raise ValueError("SIAM run-lock time-grid endpoint mismatch.")
    observed_time_digest = _time_grid_sha256(times)
    if grid.get("ordered_values_sha256") != observed_time_digest:
        raise ValueError("SIAM run-lock ordered time-grid digest mismatch.")
    comparison_fingerprint = assert_comparable((lock,))
    code_revision = str(lock.get("code_revision", "")).strip()
    if not code_revision:
        raise ValueError("SIAM raw-run lock requires a nonempty code revision.")
    return lock, rows, times, comparison_fingerprint, terminal_evidence


def _observable_matrices(
    *, n_bath: int, u: float, v: float, hopping: float
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    h_sector = sector_hamiltonian(
        n_bath,
        u=u,
        v=v,
        hopping=hopping,
    )
    operators = {
        name: project_to_sector(
            np.asarray(hamiltonian_matrix(operator), dtype=complex),
            n_bath,
        )
        for name, operator in screening_observable_operators(
            SIAMLayout(n_bath), v=v
        ).items()
    }
    return h_sector, operators


def _expectations(states: np.ndarray, operator: np.ndarray) -> np.ndarray:
    return np.asarray(
        [float(np.real(np.vdot(state, operator @ state))) for state in states],
        dtype=float,
    )


def _score_core(
    *,
    run_path: Path,
    run_sha256: str,
    seed_path: Path,
    run: Mapping[str, Any],
    reference: (
        SIAMExactReferenceArtifact
        | SIAMExactReferencePrefixView
        | SIAMExactReferenceSliceView
    ),
    lock: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    times: Sequence[float],
    comparison_fingerprint: str,
    terminal_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    if reference.manifest.get("schema") != SIAM_EXACT_REFERENCE_STORE_V2 or (
        reference.manifest.get("reference_schema") != SIAM_REFERENCE_V2
        or reference.manifest.get("observable_schema") != SIAM_OBSERVABLES_V2
    ):
        raise ValueError(
            "SIAM Sigma-v2 scoring requires the versioned exact/reporting contract."
        )
    seed_manifest = load_siam_seed_manifest(seed_path)
    physical = dict(seed_manifest.physical)
    n_bath = int(physical["n_bath"])
    n_qubits = int(physical["n_qubits"])
    full_dimension = 1 << n_qubits
    indices = np.asarray(sector_indices(n_bath), dtype=int)
    resume_evidence = _validate_resume_contract(
        run,
        seed_path=seed_path,
        terminal_time=float(times[-1]),
    )

    sector_states: list[np.ndarray] = []
    norm_errors: list[float] = []
    sector_leakages: list[float] = []
    for row in rows:
        full_state = _recorded_state(row, expected_dimension=full_dimension)
        norm_sq = float(np.real(np.vdot(full_state, full_state)))
        if not math.isfinite(norm_sq) or norm_sq <= 0.0:
            raise ValueError("SIAM checkpoint state norm must be positive and finite.")
        norm_errors.append(abs(math.sqrt(norm_sq) - 1.0))
        selected = np.asarray(full_state[indices], dtype=complex)
        sector_weight = float(np.real(np.vdot(selected, selected))) / norm_sq
        sector_leakages.append(max(0.0, 1.0 - sector_weight))
        selected_norm = float(np.linalg.norm(selected))
        if selected_norm <= 0.0:
            raise ValueError("SIAM checkpoint has no target-sector support.")
        sector_states.append(selected / selected_norm)
    if max(norm_errors) > STATE_NORM_ERROR_CEILING:
        raise ValueError("SIAM checkpoint state norm exceeds the validity ceiling.")
    if max(sector_leakages) > SECTOR_LEAKAGE_CEILING:
        raise ValueError("SIAM checkpoint leaks outside its fixed-number sector.")
    observed_states = np.asarray(sector_states, dtype=complex)
    exact_states = np.asarray(reference.sector_states, dtype=complex)
    if exact_states.shape != observed_states.shape:
        raise ValueError("SIAM raw and exact state arrays have incompatible shapes.")

    h_sector, operators = _observable_matrices(
        n_bath=n_bath,
        u=float(physical["u"]),
        v=float(physical["v_post_quench"]),
        hopping=float(physical["hopping"]),
    )
    observed_energy = _expectations(observed_states, h_sector)
    recorded_energy = _finite_series(
        [float(row["energy_expectation"]) for row in rows],
        name="recorded SIAM energy",
    )
    if not np.allclose(recorded_energy, observed_energy, rtol=1.0e-9, atol=1.0e-9):
        raise ValueError("Recorded SIAM energy does not match its statevector.")
    exact_energy = np.asarray(reference.arrays["energy"], dtype=float)
    observed = {
        name: _expectations(observed_states, operator)
        for name, operator in operators.items()
    }
    exact = {name: reference.observable(name) for name in operators}
    infidelity = np.asarray(
        [_infidelity(left, right) for left, right in zip(observed_states, exact_states)],
        dtype=float,
    )
    initial_state_infidelity = float(infidelity[0])
    initial_energy_error = float(abs(observed_energy[0] - exact_energy[0]))
    starts_at_seed = math.isclose(
        float(times[0]), 0.0, rel_tol=0.0, abs_tol=1.0e-12
    )
    if starts_at_seed:
        if initial_state_infidelity > INITIAL_STATE_INFIDELITY_CEILING:
            raise ValueError("SIAM t=0 state does not match the immutable seed.")
        if initial_energy_error > INITIAL_ENERGY_ERROR_CEILING:
            raise ValueError("SIAM t=0 energy does not match the immutable seed.")
        initial_boundary = {
            "boundary_time": float(times[0]),
            "boundary_role": "immutable_seed_boundary",
            "exact_seed_equality_required": True,
            "state_infidelity": initial_state_infidelity,
            "absolute_energy_error": initial_energy_error,
            "pass": True,
        }
    else:
        initial_boundary = {
            "boundary_time": float(times[0]),
            "boundary_role": "approximate_resume_boundary",
            "exact_seed_equality_required": False,
            "state_infidelity": initial_state_infidelity,
            "absolute_energy_error": initial_energy_error,
            "pass": None,
        }

    primitive_metrics: dict[str, dict[str, float]] = {}
    for name in _primitive_names(n_bath):
        scale = float(np.sqrt(np.mean(np.square(exact[name] - exact[name][0]))))
        mean_abs_error = float(np.mean(np.abs(observed[name] - exact[name])))
        primitive_metrics[name] = {
            "response_scale": scale,
            "mean_absolute_error": mean_abs_error,
            "response_normalized_mean_absolute_error": mean_abs_error
            / max(scale, RESPONSE_FLOOR),
        }
    derived_metrics: dict[str, dict[str, float]] = {}
    for name in _derived_names(n_bath):
        scale = float(np.sqrt(np.mean(np.square(exact[name] - exact[name][0]))))
        mean_abs_error = float(np.mean(np.abs(observed[name] - exact[name])))
        derived_metrics[name] = {
            "response_scale": scale,
            "mean_absolute_error": mean_abs_error,
            "response_normalized_mean_absolute_error": mean_abs_error
            / max(scale, RESPONSE_FLOOR),
        }

    mean_infidelity = float(np.mean(infidelity))
    max_normalized_error = max(
        metric["response_normalized_mean_absolute_error"]
        for metric in primitive_metrics.values()
    )
    quality_components = {
        name: metric["response_normalized_mean_absolute_error"]
        / INDIVIDUAL_RESPONSE_TARGET
        for name, metric in primitive_metrics.items()
    }
    quality_components["fidelity"] = (
        mean_infidelity / MEAN_INFIDELITY_CEILING
    )
    burden = _aggregate_accuracy_burden(quality_components)
    energy_error = observed_energy - exact_energy
    sector_eigenvalues = np.linalg.eigvalsh(h_sector)
    energy_guard = _relative_energy_guard(
        observed_energy,
        spectral_minimum=float(sector_eigenvalues[0]),
        spectral_maximum=float(sector_eigenvalues[-1]),
    )
    energy_conservation_pass = bool(energy_guard["pass"])
    pair_quality_components = {
        **quality_components,
        "energy": float(energy_guard["g_E"])
        / RELATIVE_ENERGY_DRIFT_CEILING,
    }
    fidelity_pass = bool(mean_infidelity <= MEAN_INFIDELITY_CEILING)
    individual_pass = bool(
        max_normalized_error <= INDIVIDUAL_RESPONSE_TARGET
        and fidelity_pass
        and energy_conservation_pass
    )
    broad_pass = bool(
        max_normalized_error <= BROAD_RESPONSE_GUARD
        and fidelity_pass
        and energy_conservation_pass
    )

    policy = lock.get("policy") or {}
    method = str(policy.get("structural_policy", ""))
    if method not in {"exchange", "avqds"}:
        raise ValueError(f"Unsupported SIAM structural arm {method!r}.")
    time_values = [float(value) for value in times]
    exact_series = {name: [float(value) for value in exact[name]] for name in sorted(exact)}
    observed_series = {
        name: [float(value) for value in observed[name]] for name in sorted(observed)
    }
    absolute_error_series = {
        name: [
            float(value)
            for value in np.abs(observed[name] - exact[name])
        ]
        for name in sorted(observed)
    }
    return {
        "schema": SIAM_OFFLINE_SCORE_V3,
        "accuracy_policy": _accuracy_policy_v2_record(),
        "observable_contract": _observable_contract_v2_record(),
        "reporting_only": True,
        "source": {
            "run_path": str(run_path.resolve()),
            "run_sha256": str(run_sha256),
            "seed_path": str(seed_path.resolve()),
            "seed_sha256": str(reference.manifest["seed_sha256"]),
            "post_quench_hamiltonian_sha256": str(
                seed_manifest.digests["post_quench_hamiltonian_sha256"]
            ),
        },
        "method": {
            "structural_policy": method,
            "comparison_fingerprint": comparison_fingerprint,
            "physics_fingerprint": str(lock.get("physics_fingerprint", "")),
            "run_lock_schema": str(lock.get("schema", "")),
            "code_revision": str(lock["code_revision"]),
        },
        "reference": {
            "key": reference.key,
            "manifest_path": str(reference.manifest_path.resolve()),
            "manifest_sha256": reference.manifest_sha256,
            "data_path": str(reference.data_path.resolve()),
            "data_sha256": reference.data_sha256,
            "store_schema": str(reference.manifest["schema"]),
            "reference_schema": str(reference.manifest["reference_schema"]),
            "observable_schema": str(reference.manifest["observable_schema"]),
            "propagator_schema": str(reference.manifest["propagator_schema"]),
        },
        "grid": {
            "times": time_values,
            "ordered_values_sha256": _time_grid_sha256(time_values),
            "point_count": len(time_values),
        },
        "thresholds": _accuracy_thresholds_v2_record(),
        "primitive_response_metrics": primitive_metrics,
        "derived_screening_metrics": derived_metrics,
        "diagnostics": {
            "I_d": {
                "mean_absolute_error": float(np.mean(np.abs(observed["I_d"] - exact["I_d"]))),
                "max_absolute_value": float(np.max(np.abs(observed["I_d"]))),
                "gate_role": "diagnostic_only_symmetry_identity",
            },
            "N_d": {
                "mean_absolute_error": float(
                    np.mean(np.abs(observed["N_d"] - exact["N_d"]))
                ),
                "gate_role": "diagnostic_only_sigma_v2_input",
            },
            "S_d2": {
                "mean_absolute_error": float(
                    np.mean(np.abs(observed["S_d2"] - exact["S_d2"]))
                ),
                "gate_role": "derived_diagnostic_only_sigma_v2_self_term",
            },
            "energy": {
                "mean_absolute_error": float(np.mean(np.abs(energy_error))),
                "max_absolute_error": float(np.max(np.abs(energy_error))),
                **energy_guard,
                "gate_role": (
                    "sector_spectral_width_normalized_conservation_guard_"
                    "excluded_from_aggregate"
                ),
            },
            "state": {
                "max_norm_error": float(max(norm_errors)),
                "max_sector_leakage": float(max(sector_leakages)),
            },
            "initial_boundary": initial_boundary,
            "resume": dict(resume_evidence),
            "terminal_success": dict(terminal_evidence),
        },
        "series": {
            "time": time_values,
            "observed": observed_series,
            "exact": exact_series,
            "absolute_error": absolute_error_series,
            "observed_energy": [float(value) for value in observed_energy],
            "exact_energy": [float(value) for value in exact_energy],
            "infidelity": [float(value) for value in infidelity],
        },
        "summary": {
            "mean_infidelity": mean_infidelity,
            "max_primitive_response_normalized_error": max_normalized_error,
            "aggregate_accuracy_burden": burden,
            "normalized_accuracy_components": quality_components,
            "pair_normalized_accuracy_components": pair_quality_components,
            "energy_conservation_pass": energy_conservation_pass,
            "fidelity_pass": fidelity_pass,
            "individual_target_pass": individual_pass,
            "broad_response_guard_pass": broad_pass,
            "classification": (
                "qualified"
                if individual_pass
                else "candidate_broad_guard"
                if broad_pass
                else "candidate_outside_broad_guard"
            ),
        },
    }


def _verified_reference_for_grid(
    *,
    store: SIAMExactReferenceStore,
    reference: (
        SIAMExactReferenceArtifact
        | SIAMExactReferencePrefixView
        | SIAMExactReferenceSliceView
    ),
    seed_sha256: str,
    times: Sequence[float],
) -> (
    SIAMExactReferenceArtifact
    | SIAMExactReferencePrefixView
    | SIAMExactReferenceSliceView
):
    """Re-verify an explicit reference and bind it to the raw-run grid.

    The prefix route deliberately never calls ``get_or_build``: a short screen
    must remain a view of the one immutable full-horizon artifact rather than
    creating a second exact trajectory under a shorter content key.
    """

    if isinstance(reference, SIAMExactReferencePrefixView):
        if reference.receipt_path is None:
            verified = store.prefix_view(reference.source, times=reference.times)
        else:
            verified = store.load_prefix_view(reference.receipt_path)
        if dict(verified.receipt) != dict(reference.receipt):
            raise ValueError("SIAM exact-prefix view receipt digest/source mismatch.")
        resolved: (
            SIAMExactReferenceArtifact
            | SIAMExactReferencePrefixView
            | SIAMExactReferenceSliceView
        ) = verified
    elif isinstance(reference, SIAMExactReferenceSliceView):
        if reference.receipt_path is None:
            verified_slice = store.slice_view(
                reference.source, times=reference.times
            )
        else:
            verified_slice = store.load_slice_view(reference.receipt_path)
        if dict(verified_slice.receipt) != dict(reference.receipt):
            raise ValueError("SIAM exact-slice view receipt digest/source mismatch.")
        resolved = verified_slice
    elif isinstance(reference, SIAMExactReferenceArtifact):
        # ``prefix_view`` re-verifies the manifest, NPZ digest, every array
        # digest, and store ownership without materializing new data.  Taking
        # its source retains full-artifact semantics for the score receipt.
        resolved = store.prefix_view(reference, times=reference.times).source
    else:
        raise TypeError(
            "verified_reference must be a SIAM exact artifact, prefix view, "
            "or contiguous slice view."
        )

    if str(resolved.manifest.get("seed_sha256", "")) != str(seed_sha256):
        raise ValueError("SIAM raw run and explicit exact-reference seed digest differ.")
    raw_times = np.asarray(tuple(float(value) for value in times), dtype=float)
    reference_times = np.asarray(resolved.times, dtype=float)
    if not np.array_equal(reference_times, raw_times):
        raise ValueError(
            "SIAM raw-run grid does not exactly match the verified reference view."
        )
    if _time_grid_sha256(reference_times) != _time_grid_sha256(raw_times):
        raise ValueError("SIAM raw/reference ordered time-grid digest mismatch.")
    return resolved


def score_siam_raw_run(
    *,
    run_path: str | Path,
    seed_path: str | Path,
    reference_store_root: str | Path,
    verified_reference: (
        SIAMExactReferenceArtifact
        | SIAMExactReferencePrefixView
        | SIAMExactReferenceSliceView
        | None
    ) = None,
) -> dict[str, Any]:
    """Attach an immutable exact reference to one completed raw trajectory.

    Existing callers may omit ``verified_reference`` and retain the original
    exact-grid ``get_or_build`` behavior.  Prefix screens must instead pass a
    prefix view loaded or created by ``SIAMExactReferenceStore``; that route
    re-verifies the full source bytes and cannot create a prefix NPZ.
    """

    run_source = Path(run_path)
    seed_source = Path(seed_path)
    run_bytes = run_source.read_bytes()
    run_sha256 = _bytes_sha256(run_bytes)
    run = json.loads(run_bytes.decode("utf-8"))
    if not isinstance(run, Mapping):
        raise ValueError("SIAM raw run must decode to a JSON object.")
    seed_sha256_before = _file_sha256(seed_source)
    seed_manifest = load_siam_seed_manifest(
        seed_source
    )  # deterministic tamper verification
    seed_sha256 = _file_sha256(seed_source)
    if seed_sha256 != seed_sha256_before:
        raise ValueError("SIAM seed changed while offline scoring was loading it.")
    (
        lock,
        rows,
        times,
        comparison_fingerprint,
        terminal_evidence,
    ) = _verified_run_contract(
        run,
        seed_sha256=seed_sha256,
    )
    store = SIAMExactReferenceStore(reference_store_root)
    n_bath = int(seed_manifest.physical["n_bath"])
    raw_grid = tuple(float(value) for value in times)
    if n_bath == 4:
        registered_nb4_grids = {
            tuple(SIAM_NB4_SCREEN_TIMES): SIAMExactReferencePrefixView,
            tuple(SIAM_NB4_CONTINUATION_TIMES): SIAMExactReferenceSliceView,
            tuple(SIAM_NB4_FULL_TIMES): SIAMExactReferenceArtifact,
        }
        expected_reference_type = registered_nb4_grids.get(raw_grid)
        if expected_reference_type is None:
            raise ValueError(
                "SIAM Nb4 scoring requires the registered T=1 screen, T=1..5 "
                "continuation, or full T=5 grid."
            )
        if raw_grid != tuple(SIAM_NB4_FULL_TIMES) and verified_reference is None:
            raise ValueError(
                "SIAM Nb4 segment scoring requires a verified view of the one "
                "canonical full-T5 exact reference."
            )
        if verified_reference is not None and not isinstance(
            verified_reference, expected_reference_type
        ):
            raise ValueError(
                "SIAM Nb4 exact-reference view type does not match the registered "
                "segment grid."
            )
    if verified_reference is None:
        reference: (
            SIAMExactReferenceArtifact
            | SIAMExactReferencePrefixView
            | SIAMExactReferenceSliceView
        ) = store.get_or_build(
            seed_path=seed_source,
            times=times,
        )
    else:
        reference = _verified_reference_for_grid(
            store=store,
            reference=verified_reference,
            seed_sha256=seed_sha256,
            times=times,
        )
    if n_bath == 4:
        source_reference = (
            reference.source
            if isinstance(
                reference,
                (SIAMExactReferencePrefixView, SIAMExactReferenceSliceView),
            )
            else reference
        )
        if not np.array_equal(
            np.asarray(source_reference.times, dtype=float),
            np.asarray(SIAM_NB4_FULL_TIMES, dtype=float),
        ):
            raise ValueError(
                "SIAM Nb4 segment reference is not derived from the canonical "
                "full-T5 grid."
            )
    core = _score_core(
        run_path=run_source,
        run_sha256=run_sha256,
        seed_path=seed_source,
        run=run,
        reference=reference,
        lock=lock,
        rows=rows,
        times=times,
        comparison_fingerprint=comparison_fingerprint,
        terminal_evidence=terminal_evidence,
    )
    if isinstance(reference, SIAMExactReferencePrefixView):
        reference_record = core.get("reference")
        if not isinstance(reference_record, dict):
            raise ValueError("SIAM score core lacks its exact-reference record.")
        prefix_receipt = dict(reference.receipt)
        reference_record["prefix_view"] = {
            "schema": str(prefix_receipt["schema"]),
            "receipt_path": (
                None
                if reference.receipt_path is None
                else str(reference.receipt_path.resolve())
            ),
            "receipt_sha256": str(prefix_receipt["receipt_sha256"]),
            "ordered_values_sha256": str(
                prefix_receipt["prefix"]["ordered_values_sha256"]
            ),
            "point_count": int(prefix_receipt["prefix"]["point_count"]),
        }
    elif isinstance(reference, SIAMExactReferenceSliceView):
        reference_record = core.get("reference")
        if not isinstance(reference_record, dict):
            raise ValueError("SIAM score core lacks its exact-reference record.")
        slice_receipt = dict(reference.receipt)
        reference_record["slice_view"] = {
            "schema": str(slice_receipt["schema"]),
            "receipt_path": (
                None
                if reference.receipt_path is None
                else str(reference.receipt_path.resolve())
            ),
            "receipt_sha256": str(slice_receipt["receipt_sha256"]),
            "ordered_values_sha256": str(
                slice_receipt["slice"]["ordered_values_sha256"]
            ),
            "point_count": int(slice_receipt["slice"]["point_count"]),
            "start_index": int(slice_receipt["slice"]["start_index"]),
            "stop_index_exclusive": int(
                slice_receipt["slice"]["stop_index_exclusive"]
            ),
        }
    return {**core, "score_sha256": _digest_document(core)}


def _expected_primitive_names(
    score: Mapping[str, Any],
    *,
    required_names: Sequence[str] | None,
) -> tuple[str, ...]:
    """Resolve the complete primitive family without trusting missing fields.

    The registered Nb4 grids identify the paper-facing pilot even if a
    re-digested document deletes one or more trailing ``C_j`` metrics.  Other
    grids retain the shared scorer's Nb2 (and future even-bath) behavior, but
    must still carry a complete contiguous even-bath correlation family.
    """

    if required_names is not None:
        required = tuple(str(name) for name in required_names)
        if required != SIAM_NB4_PRIMITIVE_NAMES:
            raise ValueError(
                "The SIAM Nb4 verifier requires exactly the frozen primitive set."
            )
        return required

    grid = score.get("grid")
    if not isinstance(grid, Mapping):
        raise ValueError("SIAM offline score lacks its grid mapping.")
    try:
        times = tuple(float(value) for value in grid.get("times") or ())
    except (TypeError, ValueError) as exc:
        raise ValueError("SIAM offline score has a malformed time grid.") from exc
    if times in {
        tuple(SIAM_NB4_SCREEN_TIMES),
        tuple(SIAM_NB4_CONTINUATION_TIMES),
        tuple(SIAM_NB4_FULL_TIMES),
    }:
        return SIAM_NB4_PRIMITIVE_NAMES

    metrics = score.get("primitive_response_metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("SIAM offline score lacks primitive response metrics.")
    correlation_indices: list[int] = []
    for name in metrics:
        text = str(name)
        if not text.startswith("C_"):
            continue
        suffix = text[2:]
        if not suffix.isdigit() or int(suffix) < 1 or str(int(suffix)) != suffix:
            raise ValueError("SIAM primitive correlation names must be canonical C_j.")
        correlation_indices.append(int(suffix))
    if not correlation_indices:
        raise ValueError("SIAM primitive metrics require a complete C_j family.")
    n_bath = max(correlation_indices)
    if n_bath % 2:
        raise ValueError("SIAM primitive metrics require an even bath size.")
    return _primitive_names(n_bath)


def _validate_policy_v2_derived_gates(
    score: Mapping[str, Any],
    *,
    required_primitive_names: Sequence[str] | None = None,
) -> None:
    """Recompute every reader-facing gate and label from scored metrics."""

    summary = score.get("summary")
    metrics = score.get("primitive_response_metrics")
    diagnostics = score.get("diagnostics")
    thresholds = score.get("thresholds")
    if not isinstance(summary, Mapping):
        raise ValueError("SIAM offline score lacks its summary mapping.")
    if not isinstance(metrics, Mapping) or not metrics:
        raise ValueError("SIAM offline score lacks primitive response metrics.")
    if not isinstance(diagnostics, Mapping) or not isinstance(thresholds, Mapping):
        raise ValueError("SIAM offline score lacks diagnostics or thresholds.")

    expected_names = _expected_primitive_names(
        score,
        required_names=required_primitive_names,
    )
    if set(metrics) != set(expected_names):
        raise ValueError(
            "SIAM primitive response metrics must contain exactly "
            f"{', '.join(expected_names)}."
        )

    normalized_errors: list[float] = []
    for name in expected_names:
        metric = metrics[name]
        if not isinstance(metric, Mapping):
            raise ValueError(f"SIAM primitive metric {name!r} is not a mapping.")
        value = float(
            metric.get("response_normalized_mean_absolute_error", float("nan"))
        )
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                "SIAM primitive response-normalized errors must be finite and "
                "nonnegative."
            )
        normalized_errors.append(value)
    max_normalized_error = max(normalized_errors)
    declared_max = float(
        summary.get("max_primitive_response_normalized_error", float("nan"))
    )
    if not math.isclose(
        declared_max,
        max_normalized_error,
        rel_tol=1.0e-12,
        abs_tol=1.0e-15,
    ):
        raise ValueError(
            "SIAM summary.max_primitive_response_normalized_error is "
            "inconsistent with primitive metrics."
        )

    mean_infidelity = float(summary.get("mean_infidelity", float("nan")))
    if (
        not math.isfinite(mean_infidelity)
        or mean_infidelity < 0.0
        or mean_infidelity > 1.0
    ):
        raise ValueError("SIAM summary.mean_infidelity must lie in [0, 1].")

    energy = diagnostics.get("energy")
    if not isinstance(energy, Mapping):
        raise ValueError("SIAM offline score lacks diagnostics.energy.")
    width = float(energy.get("sector_spectral_width", float("nan")))
    max_drift = float(energy.get("max_absolute_drift", float("nan")))
    g_e = float(energy.get("g_E", float("nan")))
    if (
        not math.isfinite(width)
        or width <= 0.0
        or not math.isfinite(max_drift)
        or max_drift < 0.0
        or not math.isfinite(g_e)
        or g_e < 0.0
    ):
        raise ValueError("SIAM diagnostics.energy metrics are invalid.")
    recomputed_g_e = max_drift / width
    if not math.isclose(g_e, recomputed_g_e, rel_tol=1.0e-12, abs_tol=1.0e-15):
        raise ValueError(
            "SIAM diagnostics.energy.g_E is inconsistent with its spectral width "
            "and maximum drift."
        )

    response_target = float(thresholds["individual_response_target"])
    broad_guard = float(thresholds["broad_response_guard"])
    fidelity_ceiling = float(thresholds["mean_infidelity_ceiling"])
    energy_ceiling = float(thresholds["relative_energy_drift_ceiling"])
    declared_energy_ceiling = float(
        energy.get("relative_spectral_drift_ceiling", float("nan"))
    )
    if declared_energy_ceiling != energy_ceiling:
        raise ValueError(
            "SIAM diagnostics.energy relative ceiling is inconsistent with policy."
        )

    energy_pass = bool(g_e <= energy_ceiling)
    fidelity_pass = bool(mean_infidelity <= fidelity_ceiling)
    individual_pass = bool(
        max_normalized_error <= response_target
        and fidelity_pass
        and energy_pass
    )
    broad_pass = bool(
        max_normalized_error <= broad_guard
        and fidelity_pass
        and energy_pass
    )
    expected_flags = {
        "energy_conservation_pass": energy_pass,
        "fidelity_pass": fidelity_pass,
        "individual_target_pass": individual_pass,
        "broad_response_guard_pass": broad_pass,
    }
    if energy.get("pass") is not energy_pass:
        raise ValueError(
            "SIAM diagnostics.energy.pass is inconsistent with g_E and policy."
        )
    for field, expected in expected_flags.items():
        if summary.get(field) is not expected:
            raise ValueError(
                f"SIAM summary.{field} is inconsistent with metrics and policy."
            )
    expected_classification = (
        "qualified"
        if individual_pass
        else "candidate_broad_guard"
        if broad_pass
        else "candidate_outside_broad_guard"
    )
    if summary.get("classification") != expected_classification:
        raise ValueError(
            "SIAM summary.classification is inconsistent with metrics and policy."
        )

    grid = score.get("grid")
    initial_boundary = diagnostics.get("initial_boundary")
    if not isinstance(grid, Mapping) or not isinstance(initial_boundary, Mapping):
        raise ValueError("SIAM offline score lacks its initial-boundary contract.")
    times = tuple(float(value) for value in grid.get("times") or ())
    if not times:
        raise ValueError("SIAM offline score has an empty time grid.")
    if grid.get("ordered_values_sha256") != _time_grid_sha256(times):
        raise ValueError("SIAM offline-score ordered time-grid digest mismatch.")
    starts_at_seed = math.isclose(times[0], 0.0, rel_tol=0.0, abs_tol=1.0e-12)
    expected_role = (
        "immutable_seed_boundary"
        if starts_at_seed
        else "approximate_resume_boundary"
    )
    if initial_boundary.get("boundary_role") != expected_role:
        raise ValueError("SIAM initial-boundary role is inconsistent with its grid.")
    boundary_time = float(initial_boundary.get("boundary_time", float("nan")))
    if not math.isclose(
        boundary_time, times[0], rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError("SIAM initial-boundary time is inconsistent with its grid.")
    if initial_boundary.get("exact_seed_equality_required") is not starts_at_seed:
        raise ValueError(
            "SIAM initial-boundary exact-seed requirement is inconsistent with its grid."
        )
    boundary_state_infidelity = float(
        initial_boundary.get("state_infidelity", float("nan"))
    )
    boundary_energy_error = float(
        initial_boundary.get("absolute_energy_error", float("nan"))
    )
    if (
        not math.isfinite(boundary_state_infidelity)
        or boundary_state_infidelity < 0.0
        or not math.isfinite(boundary_energy_error)
        or boundary_energy_error < 0.0
    ):
        raise ValueError("SIAM initial-boundary diagnostics must be nonnegative.")
    if starts_at_seed and (
        boundary_state_infidelity
        > float(thresholds["initial_state_infidelity_ceiling"])
        or boundary_energy_error
        > float(thresholds["initial_energy_error_ceiling"])
    ):
        raise ValueError("SIAM immutable seed boundary does not pass its ceilings.")
    expected_boundary_pass: bool | None = True if starts_at_seed else None
    if initial_boundary.get("pass") is not expected_boundary_pass:
        raise ValueError("SIAM initial-boundary pass field is inconsistent with its role.")


def _validate_sigma_v2_reporting(score: Mapping[str, Any]) -> None:
    """Fail closed on mixed Sigma-v1/v2 series and reference identities."""

    contract = score.get("observable_contract")
    if not isinstance(contract, Mapping) or dict(contract) != (
        _observable_contract_v2_record()
    ):
        raise ValueError("SIAM score lacks the exact Sigma-v2 observable contract.")
    reference = score.get("reference")
    if not isinstance(reference, Mapping) or {
        "store_schema": reference.get("store_schema"),
        "reference_schema": reference.get("reference_schema"),
        "observable_schema": reference.get("observable_schema"),
    } != {
        "store_schema": SIAM_EXACT_REFERENCE_STORE_V2,
        "reference_schema": SIAM_REFERENCE_V2,
        "observable_schema": SIAM_OBSERVABLES_V2,
    }:
        raise ValueError("SIAM score mixes incompatible Sigma-v2 reference schemas.")
    series = score.get("series")
    if not isinstance(series, Mapping):
        raise ValueError("SIAM Sigma-v2 score lacks its series contract.")
    times = np.asarray(series.get("time") or (), dtype=float)
    if not times.size:
        raise ValueError("SIAM Sigma-v2 score has an empty series grid.")
    primitive_names = _expected_primitive_names(score, required_names=None)
    correlation_names = tuple(
        name for name in primitive_names if name.startswith("C_")
    )
    n_bath = len(correlation_names)
    expected_observables = {
        *primitive_names,
        "N_d",
        "S_d2",
        "I_d",
        *(f"Sigma_{site}" for site in range(1, n_bath + 1)),
    }
    records: dict[str, Mapping[str, Any]] = {}
    for side in ("observed", "exact", "absolute_error"):
        record = series.get(side)
        if not isinstance(record, Mapping) or set(record) != expected_observables:
            raise ValueError(f"SIAM Sigma-v2 {side} observable schema mismatch.")
        records[side] = record
    for side in ("observed", "exact"):
        record = records[side]
        arrays = {
            name: np.asarray(record[name], dtype=float).reshape(-1)
            for name in expected_observables
        }
        if any(array.shape != times.shape for array in arrays.values()):
            raise ValueError(f"SIAM Sigma-v2 {side} series shape mismatch.")
        s_d2 = arrays["S_d2"]
        if not np.allclose(
            s_d2,
            0.75 * (arrays["N_d"] - 2.0 * arrays["D_d"]),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(f"SIAM Sigma-v2 {side} S_d2 identity mismatch.")
        cumulative = s_d2.copy()
        for site in range(1, n_bath + 1):
            cumulative = cumulative + arrays[f"C_{site}"]
            if not np.allclose(
                arrays[f"Sigma_{site}"], cumulative, rtol=0.0, atol=1.0e-12
            ):
                raise ValueError(
                    f"SIAM Sigma-v2 {side} Sigma_{site} identity mismatch."
                )
    for name in expected_observables:
        expected_error = np.abs(
            np.asarray(records["observed"][name], dtype=float)
            - np.asarray(records["exact"][name], dtype=float)
        )
        if not np.allclose(
            np.asarray(records["absolute_error"][name], dtype=float),
            expected_error,
            rtol=0.0,
            atol=1.0e-15,
        ):
            raise ValueError(
                f"SIAM Sigma-v2 absolute-error series mismatch for {name}."
            )
    derived = score.get("derived_screening_metrics")
    if not isinstance(derived, Mapping) or set(derived) != {
        *(f"Sigma_{site}" for site in range(1, n_bath + 1))
    }:
        raise ValueError("SIAM Sigma-v2 derived-screening metric schema mismatch.")


def _verified_score(
    score: Mapping[str, Any],
    *,
    require_policy_v2: bool = True,
    require_sigma_v2: bool = True,
    required_primitive_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    payload = dict(score)
    digest = payload.pop("score_sha256", None)
    if digest != _digest_document(payload):
        raise ValueError("SIAM offline-score digest mismatch.")
    schema = payload.get("schema")
    if schema == SIAM_OFFLINE_SCORE_V1:
        if require_policy_v2:
            raise ValueError(
                "Matched comparison requires policy-v2 SIAM offline scores."
            )
        # Historical v1 receipts remain immutable and writable, but are never
        # silently reinterpreted under the v2 energy-normalization policy.
        if require_sigma_v2:
            raise ValueError("Canonical reporting requires Sigma-v2 SIAM scores.")
    elif schema not in {SIAM_OFFLINE_SCORE_V2, SIAM_OFFLINE_SCORE_V3}:
        raise ValueError("Matched comparison requires SIAM offline-score documents.")
    else:
        if schema == SIAM_OFFLINE_SCORE_V2 and require_sigma_v2:
            raise ValueError("Canonical reporting requires Sigma-v2 SIAM scores.")
        policy = payload.get("accuracy_policy")
        if not isinstance(policy, Mapping) or dict(policy) != (
            _accuracy_policy_v2_record()
        ):
            raise ValueError(
                "SIAM offline score does not carry the exact policy-v2 identity."
            )
        thresholds = payload.get("thresholds")
        if not isinstance(thresholds, Mapping) or dict(thresholds) != (
            _accuracy_thresholds_v2_record()
        ):
            raise ValueError(
                "SIAM offline score does not carry the exact policy-v2 thresholds."
            )
        _validate_policy_v2_derived_gates(
            payload,
            required_primitive_names=required_primitive_names,
        )
        if schema == SIAM_OFFLINE_SCORE_V3:
            _validate_sigma_v2_reporting(payload)
    components = _quality_components_from_score(payload)
    declared_components = payload["summary"].get(
        "normalized_accuracy_components"
    )
    if declared_components != components:
        raise ValueError("SIAM offline-score normalized components are inconsistent.")
    expected_burden = _aggregate_accuracy_burden(components)
    if not math.isclose(
        float(payload["summary"]["aggregate_accuracy_burden"]),
        expected_burden,
        rel_tol=1.0e-12,
        abs_tol=1.0e-15,
    ):
        raise ValueError("SIAM offline-score aggregate burden is inconsistent.")
    if schema in {SIAM_OFFLINE_SCORE_V2, SIAM_OFFLINE_SCORE_V3}:
        pair_components = _pair_quality_components_from_score(payload)
        declared_pair_components = payload["summary"].get(
            "pair_normalized_accuracy_components"
        )
        if declared_pair_components != pair_components:
            raise ValueError(
                "SIAM offline-score pair-normalized components are inconsistent."
            )
    return payload


def classify_siam_matched_pair(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> dict[str, Any]:
    """Classify one exchange/AVQDS pair after proving comparability."""

    left = _verified_score(first)
    right = _verified_score(second)
    methods = {
        str(left["method"]["structural_policy"]): left,
        str(right["method"]["structural_policy"]): right,
    }
    if set(methods) != {"exchange", "avqds"}:
        raise ValueError("Matched SIAM pair requires exactly exchange and AVQDS arms.")
    apm = methods["exchange"]
    avqds = methods["avqds"]
    comparable_fields = (
        ("method", "comparison_fingerprint"),
        ("method", "code_revision"),
        ("source", "seed_sha256"),
        ("reference", "key"),
        ("reference", "manifest_sha256"),
        ("reference", "data_sha256"),
        ("reference", "store_schema"),
        ("reference", "reference_schema"),
        ("reference", "observable_schema"),
        ("grid", "ordered_values_sha256"),
        ("grid", "point_count"),
    )
    for section, field in comparable_fields:
        if apm[section][field] != avqds[section][field]:
            raise ValueError(
                f"SIAM arms are incomparable: {section}.{field} differs."
            )
    if apm["thresholds"] != avqds["thresholds"]:
        raise ValueError("SIAM arms are incomparable: scoring thresholds differ.")
    if apm["accuracy_policy"] != avqds["accuracy_policy"]:
        raise ValueError("SIAM arms are incomparable: accuracy policies differ.")

    apm_burden = float(apm["summary"]["aggregate_accuracy_burden"])
    avqds_burden = float(avqds["summary"]["aggregate_accuracy_burden"])
    if min(apm_burden, avqds_burden) == 0.0:
        burden_ratio: float | None = (
            1.0 if max(apm_burden, avqds_burden) == 0.0 else None
        )
    else:
        burden_ratio = max(apm_burden / avqds_burden, avqds_burden / apm_burden)
    both_broad = bool(
        apm["summary"]["broad_response_guard_pass"]
        and avqds["summary"]["broad_response_guard_pass"]
    )
    both_energy_safe = bool(
        apm["summary"]["energy_conservation_pass"]
        and avqds["summary"]["energy_conservation_pass"]
    )
    both_fidelity_safe = bool(
        apm["summary"]["fidelity_pass"]
        and avqds["summary"]["fidelity_pass"]
    )
    burdens_matched = bool(
        burden_ratio is not None
        and burden_ratio <= AGGREGATE_BURDEN_RATIO_CEILING
    )
    apm_components = _pair_quality_components_from_score(apm)
    avqds_components = _pair_quality_components_from_score(avqds)
    if set(apm_components) != set(avqds_components):
        raise ValueError("SIAM arms are incomparable: accuracy components differ.")
    component_ratios: dict[str, float] = {}
    for name in sorted(apm_components):
        apm_value = max(float(apm_components[name]), COMPONENT_RATIO_DEADBAND)
        avqds_value = max(
            float(avqds_components[name]), COMPONENT_RATIO_DEADBAND
        )
        component_ratios[name] = max(
            apm_value / avqds_value,
            avqds_value / apm_value,
        )
    components_matched = bool(
        all(
            ratio <= COMPONENT_RATIO_CEILING
            for ratio in component_ratios.values()
        )
    )
    admissible = bool(
        both_broad
        and both_energy_safe
        and both_fidelity_safe
        and burdens_matched
        and components_matched
    )
    both_individual = bool(
        apm["summary"]["individual_target_pass"]
        and avqds["summary"]["individual_target_pass"]
    )
    core = {
        "schema": SIAM_MATCHED_PAIR_V3,
        "observable_contract": dict(apm["observable_contract"]),
        "accuracy_policy": dict(apm["accuracy_policy"]),
        "reference": dict(apm["reference"]),
        "grid": dict(apm["grid"]),
        "comparison_fingerprint": str(
            apm["method"]["comparison_fingerprint"]
        ),
        "arms": {
            "full_apm": {
                "score_sha256": str(first["score_sha256"])
                if first["method"]["structural_policy"] == "exchange"
                else str(second["score_sha256"]),
                "aggregate_accuracy_burden": apm_burden,
                "individual_target_pass": bool(
                    apm["summary"]["individual_target_pass"]
                ),
                "broad_response_guard_pass": bool(
                    apm["summary"]["broad_response_guard_pass"]
                ),
                "g_E": float(apm["diagnostics"]["energy"]["g_E"]),
            },
            "avqds": {
                "score_sha256": str(first["score_sha256"])
                if first["method"]["structural_policy"] == "avqds"
                else str(second["score_sha256"]),
                "aggregate_accuracy_burden": avqds_burden,
                "individual_target_pass": bool(
                    avqds["summary"]["individual_target_pass"]
                ),
                "broad_response_guard_pass": bool(
                    avqds["summary"]["broad_response_guard_pass"]
                ),
                "g_E": float(avqds["diagnostics"]["energy"]["g_E"]),
            },
        },
        "matched_accuracy": {
            "both_inside_broad_guard": both_broad,
            "both_pass_energy_conservation": both_energy_safe,
            "both_pass_fidelity": both_fidelity_safe,
            "burden_ratio": burden_ratio,
            "burden_ratio_ceiling": AGGREGATE_BURDEN_RATIO_CEILING,
            "burdens_matched": burdens_matched,
            "componentwise_deadband": COMPONENT_RATIO_DEADBAND,
            "componentwise_ratio_ceiling": COMPONENT_RATIO_CEILING,
            "componentwise_ratios": component_ratios,
            "components_matched": components_matched,
            "cost_comparison_admissible": admissible,
            "classification": (
                "qualified_pair"
                if admissible and both_individual
                else "matched_candidate_pair"
                if admissible
                else "rejected_accuracy_confounded_pair"
            ),
        },
    }
    return {**core, "pair_sha256": _digest_document(core)}


def build_siam_sigma_v2_transition_receipt(
    superseded_score: Mapping[str, Any],
    replacement_score: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind an offline-only Sigma correction to one unchanged raw trajectory.

    The old score remains immutable evidence.  The receipt proves that only the
    reporting algebra/reference bytes changed: the run, seed, method, grid,
    primitive gates, accuracy policy, and gate decisions are identical.
    """

    old = _verified_score(
        superseded_score,
        require_policy_v2=True,
        require_sigma_v2=False,
    )
    if old.get("schema") != SIAM_OFFLINE_SCORE_V2:
        raise ValueError("Sigma-v2 transition requires a policy-v2 predecessor.")
    new = _verified_score(replacement_score)
    if old["grid"]["times"] != new["grid"]["times"]:
        raise ValueError("Sigma-v2 transition changes the full time grid.")
    stable_paths = (
        ("source", "run_sha256"),
        ("source", "seed_sha256"),
        ("source", "post_quench_hamiltonian_sha256"),
        ("method", "structural_policy"),
        ("method", "comparison_fingerprint"),
        ("method", "physics_fingerprint"),
        ("method", "run_lock_schema"),
        ("method", "code_revision"),
        ("grid", "ordered_values_sha256"),
        ("grid", "point_count"),
    )
    for section, field in stable_paths:
        if old[section][field] != new[section][field]:
            raise ValueError(
                f"Sigma-v2 transition changes {section}.{field}; rerun identity differs."
            )
    if old["accuracy_policy"] != new["accuracy_policy"] or (
        old["thresholds"] != new["thresholds"]
    ):
        raise ValueError("Sigma-v2 transition changes the accuracy policy.")
    if old["primitive_response_metrics"] != new["primitive_response_metrics"]:
        raise ValueError("Sigma-v2 transition changes primitive gate metrics.")
    stable_summary_fields = (
        "mean_infidelity",
        "max_primitive_response_normalized_error",
        "aggregate_accuracy_burden",
        "normalized_accuracy_components",
        "pair_normalized_accuracy_components",
        "energy_conservation_pass",
        "fidelity_pass",
        "individual_target_pass",
        "broad_response_guard_pass",
        "classification",
    )
    if any(old["summary"][field] != new["summary"][field] for field in stable_summary_fields):
        raise ValueError("Sigma-v2 transition changes a primitive admission decision.")
    if old["diagnostics"]["energy"] != new["diagnostics"]["energy"]:
        raise ValueError("Sigma-v2 transition changes the energy guard.")
    if old["diagnostics"]["initial_boundary"] != new["diagnostics"][
        "initial_boundary"
    ]:
        raise ValueError("Sigma-v2 transition initial-boundary diagnostics differ.")
    core = {
        "schema": SIAM_SIGMA_V2_TRANSITION_V1,
        "transition_role": "offline_reporting_semantics_correction",
        "trajectory_rerun_required": False,
        "primitive_gate_policy_changed": False,
        "raw_trajectory": {
            "run_sha256": str(new["source"]["run_sha256"]),
            "seed_sha256": str(new["source"]["seed_sha256"]),
        },
        "supersedes": {
            "score_schema": SIAM_OFFLINE_SCORE_V2,
            "score_sha256": str(superseded_score["score_sha256"]),
            "reporting_semantics": "Sigma_r=sum_{j<=r}C_j",
            "artifact_retained": True,
        },
        "replacement": {
            "score_schema": SIAM_OFFLINE_SCORE_V3,
            "score_sha256": str(replacement_score["score_sha256"]),
            "observable_schema": SIAM_OBSERVABLES_V2,
            "reporting_semantics": "Sigma_r=S_d2+sum_{j<=r}C_j",
        },
    }
    return {**core, "transition_sha256": _digest_document(core)}


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    content = json.dumps(
        dict(payload), indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == content:
            return
        raise FileExistsError(f"Refusing to replace immutable scoring artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise FileExistsError(
                    f"Refusing to replace immutable scoring artifact: {path}"
                ) from None
    finally:
        temporary.unlink(missing_ok=True)


def write_siam_run_score(path: str | Path, score: Mapping[str, Any]) -> None:
    """Atomically install one already digest-verified score document."""

    _verified_score(score, require_policy_v2=False, require_sigma_v2=False)
    _write_immutable_json(Path(path), score)


def write_siam_matched_pair(path: str | Path, pair: Mapping[str, Any]) -> None:
    """Atomically install one digest-verified matched-pair document."""

    payload = dict(pair)
    digest = payload.pop("pair_sha256", None)
    schema = payload.get("schema")
    if schema not in {
        SIAM_MATCHED_PAIR_V1,
        SIAM_MATCHED_PAIR_V2,
        SIAM_MATCHED_PAIR_V3,
    } or (
        digest != _digest_document(payload)
    ):
        raise ValueError("SIAM matched-pair digest mismatch.")
    if schema in {SIAM_MATCHED_PAIR_V2, SIAM_MATCHED_PAIR_V3}:
        policy = payload.get("accuracy_policy")
        if not isinstance(policy, Mapping) or policy.get("schema") != (
            SIAM_ACCURACY_POLICY_V2
        ):
            raise ValueError("SIAM matched pair lacks the policy-v2 identity.")
    if schema == SIAM_MATCHED_PAIR_V3 and payload.get("observable_contract") != (
        _observable_contract_v2_record()
    ):
        raise ValueError("SIAM matched pair lacks the Sigma-v2 observable identity.")
    _write_immutable_json(Path(path), pair)


def write_siam_sigma_v2_transition_receipt(
    path: str | Path, receipt: Mapping[str, Any]
) -> None:
    """Install one digest-verified immutable Sigma-v2 transition receipt."""

    payload = dict(receipt)
    digest = payload.pop("transition_sha256", None)
    if payload.get("schema") != SIAM_SIGMA_V2_TRANSITION_V1 or digest != (
        _digest_document(payload)
    ):
        raise ValueError("SIAM Sigma-v2 transition digest mismatch.")
    _write_immutable_json(Path(path), receipt)


__all__ = [
    "AGGREGATE_BURDEN_RATIO_CEILING",
    "BROAD_RESPONSE_GUARD",
    "COMPONENT_RATIO_CEILING",
    "COMPONENT_RATIO_DEADBAND",
    "ENERGY_DRIFT_CEILING",
    "ENERGY_DRIFT_CEILING_V1",
    "INDIVIDUAL_RESPONSE_TARGET",
    "MEAN_INFIDELITY_CEILING",
    "RELATIVE_ENERGY_DRIFT_CEILING",
    "RESPONSE_FLOOR",
    "SIAM_MATCHED_PAIR_V1",
    "SIAM_MATCHED_PAIR_V2",
    "SIAM_MATCHED_PAIR_V3",
    "SIAM_NB4_PRIMITIVE_NAMES",
    "SIAM_OFFLINE_SCORE_V1",
    "SIAM_OFFLINE_SCORE_V2",
    "SIAM_OFFLINE_SCORE_V3",
    "SIAM_SIGMA_V2_TRANSITION_V1",
    "SIAM_ACCURACY_POLICY_V2",
    "build_siam_sigma_v2_transition_receipt",
    "classify_siam_matched_pair",
    "score_siam_raw_run",
    "write_siam_matched_pair",
    "write_siam_run_score",
    "write_siam_sigma_v2_transition_receipt",
]
