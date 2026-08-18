#!/usr/bin/env python3
"""Local QSE-root -> promoted circuit -> driven AP-McLachlan diagnostic.

This module is deliberately diagnostic-only.  It consumes the already locked
Paper-III Hubbard--Holstein QSE root and driven exact trajectory, constructs an
honest HF -> RA -> compact excited-root circuit, validates that circuit through
the scaffold runtime contract, and then propagates its statevector handoff with
adaptive, append-only AP-McLachlan.  Exact states are reconstructed only after
each variational trajectory exists and are used only for reporting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.excited_dynamics.paper_iii_advisor_demo import (
    _half_filled_sector_indices,
    _midpoint_step,
)
from pipelines.qse_spectra.hh_response_observables import (
    HHResponseLayout,
    build_hh_neutral_response_observable_bundle,
)
from pipelines.qse_spectra.io import load_polynomial_json, load_state_json
from pipelines.contracts.scaffold import CandidatePoolSource
from pipelines.scaffold.qse_compact_root_refit import (
    CompactQSERootRefitConfig,
    run_compact_qse_root_refit,
)
from pipelines.scaffold.qse_root_refit import reconstruct_qse_root_target
from pipelines.scaffold.qse_runtime_promotion import (
    QSERuntimePromotionConfig,
    promote_qse_root_refit,
)
from pipelines.scaffold.runtime_loader import load_scaffold_runtime_input_from_payload
from pipelines.time_dynamics.ap_mclachlan.drive_aligned import (
    augment_state_with_drive_aligned_generator,
)
from pipelines.time_dynamics.ap_mclachlan.adaptive_trajectory import (
    SupportPatchControllerConfig,
    run_append_mclachlan_trajectory,
)
from pipelines.time_dynamics.ap_mclachlan.hamiltonian import (
    time_dependent_hamiltonian_from_runtime_input,
)
from pipelines.time_dynamics.ap_mclachlan.fixed_step import SolveRepairConfig
from pipelines.time_dynamics.ap_mclachlan.inverse import McLachlanInversePolicy
from pipelines.time_dynamics.ap_mclachlan.state import (
    APMcLachlanState,
    AP_PARAMETERIZATION_LOGICAL_SHARED,
    AP_PARAMETERIZATION_PER_PAULI_TERM,
    state_from_scaffold_runtime_input,
)
from pipelines.time_dynamics.ap_mclachlan.trajectory import run_fixed_mclachlan_trajectory
from pipelines.time_dynamics.normalized_pauli_pool import (
    NORMALIZED_POOL_FULL_META_CHILDREN,
    NORMALIZED_POOL_HAMILTONIAN_DRIVE,
    build_normalized_pauli_pool,
    normalized_pool_candidate_terms,
)
from src.quantum.drives_time_potential import gaussian_sinusoid_waveform
from src.quantum.ansatz_parameterization import build_parameter_layout
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.pauli_actions import compile_pauli_action_exyz
from src.quantum.vqe_latex_python_pairs import hamiltonian_matrix


SCHEMA_VERSION = "paper_iii_hh_promoted_ap_demo_v1"
PIPELINE = "paper_iii_hh_promoted_ap_demo"
LOCKED_INITIAL_POOL_COUNT = 30
LOCKED_INITIAL_POOL_SHA256 = "979ab84b91e061c008375b1941d26c084b4873b09447625f29afbb4117789442"
LOCKED_FUTURE_POOL_COUNT = 948
LOCKED_FUTURE_POOL_SHA256 = "61822951f13318abd380dcafd4fb8793342abd1fd97a2d8197df7cac4ce88410"
LOCKED_FULL_META_PARENT_COUNT = 127


class PromotedAPDemoError(ValueError):
    """Raised when a source lock or scientific acceptance gate fails."""


@dataclass(frozen=True)
class PromotedAPDemoConfig:
    qse_result_json: Path
    source_seed_json: Path
    locked_advisor_result_json: Path
    output_dir: Path
    state_index: int = 0
    max_selected_paulis: int = 40
    refit_target_infidelity: float = 1.0e-8
    refit_max_energy_error: float = 1.0e-6
    refit_max_physical_residual: float = 1.0e-3
    refit_optimizer_maxiter: int = 2000
    time_steps: tuple[float, ...] = (0.05, 0.025)
    pinv_rcond: float = 1.0e-10
    ridge_lambda: float = 1.0e-7
    solve_damping: float = 0.0
    maximum_density_abs_error: float = 2.0e-2
    maximum_phonon_abs_error: float = 5.0e-2
    minimum_state_fidelity: float = 0.999
    maximum_convergence_density_delta: float = 5.0e-3
    maximum_convergence_phonon_delta: float = 1.0e-2


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PromotedAPDemoError(f"Could not read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PromotedAPDemoError(f"Expected a JSON object at {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(_json_safe(dict(payload)), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_config(config: PromotedAPDemoConfig) -> None:
    for path in (
        config.qse_result_json,
        config.source_seed_json,
        config.locked_advisor_result_json,
    ):
        if not Path(path).is_file():
            raise PromotedAPDemoError(f"Required source artifact not found: {path}")
    if int(config.state_index) < 0:
        raise PromotedAPDemoError("state_index must be non-negative")
    if int(config.max_selected_paulis) < 1:
        raise PromotedAPDemoError("max_selected_paulis must be positive")
    if not config.time_steps or any((not math.isfinite(float(dt)) or float(dt) <= 0.0) for dt in config.time_steps):
        raise PromotedAPDemoError("time_steps must contain positive finite values")


def _normalize(state: np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(state, dtype=complex).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise PromotedAPDemoError(f"{name} has non-positive norm")
    return vector / norm


def _expectation(state: np.ndarray, operator: np.ndarray) -> float:
    psi = _normalize(state, name="expectation state")
    value = complex(np.vdot(psi, np.asarray(operator, dtype=complex) @ psi))
    if abs(value.imag) > 1.0e-9 * max(1.0, abs(value.real)):
        raise PromotedAPDemoError(f"Expectation has non-negligible imaginary part {value.imag}")
    return float(value.real)


def _state_fidelity(left: np.ndarray, right: np.ndarray) -> float:
    a = _normalize(left, name="fidelity left state")
    b = _normalize(right, name="fidelity right state")
    return float(max(0.0, min(1.0, abs(np.vdot(a, b)) ** 2)))


def _observable_matrices(
    *,
    source_seed_json: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hamiltonian_poly, _ = load_polynomial_json(Path(source_seed_json))
    hamiltonian = np.asarray(hamiltonian_matrix(hamiltonian_poly), dtype=complex)
    nq = int(round(math.log2(int(hamiltonian.shape[0]))))
    prepared_state, _ = load_state_json(
        Path(source_seed_json),
        expected_nq=nq,
        state_key="initial_state",
    )
    layout = HHResponseLayout(
        num_sites=2,
        n_ph_max=3,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        total_qubits=nq,
        num_particles=(1, 1),
        source_metadata={"source_seed_json": str(source_seed_json)},
    )
    bundle = build_hh_neutral_response_observable_bundle(
        layout=layout,
        channels=("nn", "XX"),
        form_factor="staggered",
        prepared_state=prepared_state,
    )
    by_family = {
        str(record.metadata.get("channel_family")): record
        for record in bundle.observables
        if isinstance(record.metadata, Mapping)
    }
    if "n" not in by_family or "X" not in by_family:
        raise PromotedAPDemoError("Could not reconstruct locked HH staggered observables")
    density = np.asarray(hamiltonian_matrix(by_family["n"].polynomial), dtype=complex)
    phonon = np.asarray(hamiltonian_matrix(by_family["X"].polynomial), dtype=complex)
    sector = _half_filled_sector_indices(
        num_sites=2,
        n_ph_max=3,
        boson_encoding="binary",
        ordering="blocked",
        nq_total=nq,
    )
    return hamiltonian, density, phonon, np.asarray(sector, dtype=int)


def _locked_drive(locked: Mapping[str, Any]) -> dict[str, Any]:
    dynamics = locked.get("dynamics")
    if not isinstance(dynamics, Mapping):
        raise PromotedAPDemoError("Locked advisor artifact is missing dynamics")
    drive = dynamics.get("drive")
    metrics = dynamics.get("metrics")
    trajectory = dynamics.get("trajectory")
    if not isinstance(drive, Mapping) or not isinstance(metrics, Mapping) or not isinstance(trajectory, Sequence):
        raise PromotedAPDemoError("Locked advisor artifact has incomplete dynamics blocks")
    expected = {
        "amplitude": 0.05,
        "tbar": 4.0,
        "phi": 0.0,
        "operator": "hh_n[staggered]",
        "spatial_pattern": "staggered",
    }
    for key, value in expected.items():
        actual = drive.get(key)
        if isinstance(value, float):
            if actual is None or not math.isclose(float(actual), value, rel_tol=0.0, abs_tol=1.0e-14):
                raise PromotedAPDemoError(f"Locked drive {key}={actual!r}; expected {value!r}")
        elif actual != value:
            raise PromotedAPDemoError(f"Locked drive {key}={actual!r}; expected {value!r}")
    if str(metrics.get("exact_reference_method")) != "fixed_sector_exponential_midpoint_magnus2_order2":
        raise PromotedAPDemoError("Locked exact trajectory uses an unexpected propagation method")
    if bool(metrics.get("exact_reference_used_for_controller_or_drive_selection")):
        raise PromotedAPDemoError("Locked exact trajectory violates diagnostic-only data flow")
    return {
        "amplitude": float(drive["amplitude"]),
        "omega": float(drive["omega"]),
        "tbar": float(drive["tbar"]),
        "phi": float(drive["phi"]),
        "t_final": float(trajectory[-1]["time"]),
        "locked_dt": float(metrics["dt"]),
        "locked_rows": [dict(row) for row in trajectory],
        "payload": dict(drive),
    }


def _drive_config(drive: Mapping[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        enabled=True,
        n_sites=2,
        ordering="blocked",
        drive_A=float(drive["amplitude"]),
        drive_omega=float(drive["omega"]),
        drive_tbar=float(drive["tbar"]),
        drive_phi=float(drive["phi"]),
        drive_pattern="staggered",
        drive_custom_weights=None,
        drive_include_identity=False,
        drive_time_sampling="midpoint",
        drive_t0=0.0,
    )


def _controller_drive(drive: Mapping[str, Any]) -> dict[str, float]:
    """Return the scalar drive contract without any locked reference rows."""

    return {
        "amplitude": float(drive["amplitude"]),
        "omega": float(drive["omega"]),
        "tbar": float(drive["tbar"]),
        "phi": float(drive["phi"]),
        "t_final": float(drive["t_final"]),
    }


def _exact_reference_states(
    *,
    initial_state_full: np.ndarray,
    sector_indices: np.ndarray,
    hamiltonian_full: np.ndarray,
    drive_full: np.ndarray,
    drive: Mapping[str, Any],
    times: np.ndarray,
) -> tuple[np.ndarray, ...]:
    h0 = np.asarray(hamiltonian_full[np.ix_(sector_indices, sector_indices)], dtype=complex)
    dmat = np.asarray(drive_full[np.ix_(sector_indices, sector_indices)], dtype=complex)
    current = _normalize(initial_state_full[sector_indices], name="locked exact initial state")
    states: list[np.ndarray] = []
    for index, time_value in enumerate(np.asarray(times, dtype=float)):
        states.append(np.asarray(current, dtype=complex).copy())
        if index + 1 == int(times.size):
            continue
        dt = float(times[index + 1] - time_value)
        midpoint = float(time_value) + 0.5 * dt
        coefficient = gaussian_sinusoid_waveform(
            midpoint,
            A=float(drive["amplitude"]),
            omega=float(drive["omega"]),
            tbar=float(drive["tbar"]),
            phi=float(drive["phi"]),
        )
        current = _midpoint_step(current, h0 + coefficient * dmat, dt)
    return tuple(states)


def _assert_locked_reference_replay(
    *,
    exact_states: Sequence[np.ndarray],
    locked_rows: Sequence[Mapping[str, Any]],
    density_sector: np.ndarray,
    phonon_sector: np.ndarray,
    hamiltonian_sector: np.ndarray,
    atol: float = 2.0e-12,
) -> dict[str, float]:
    if len(exact_states) != len(locked_rows):
        raise PromotedAPDemoError("Replayed locked exact trajectory has the wrong point count")
    deltas = {"density": 0.0, "phonon": 0.0, "static_energy": 0.0}
    for state, row in zip(exact_states, locked_rows):
        values = {
            "density": _expectation(state, density_sector),
            "phonon": _expectation(state, phonon_sector),
            "static_energy": _expectation(state, hamiltonian_sector),
        }
        expected = {
            "density": float(row["staggered_density_exact"]),
            "phonon": float(row["staggered_phonon_displacement_exact"]),
            "static_energy": float(row["static_energy_exact"]),
        }
        for key in deltas:
            deltas[key] = max(float(deltas[key]), abs(float(values[key]) - float(expected[key])))
    if max(deltas.values()) > float(atol):
        raise PromotedAPDemoError(f"Locked exact trajectory replay mismatch: {deltas}")
    return {f"maximum_{key}_replay_delta": float(value) for key, value in deltas.items()}


def _validate_midpoint_reference_against_dop853(
    *,
    initial_state_full: np.ndarray,
    sector_indices: np.ndarray,
    hamiltonian_full: np.ndarray,
    drive_full: np.ndarray,
    phonon_full: np.ndarray,
    drive: Mapping[str, Any],
    times: np.ndarray,
    midpoint_states: Sequence[np.ndarray],
) -> dict[str, Any]:
    """Post-run independent integration check for the locked midpoint reference."""

    from scipy.integrate import solve_ivp

    h_sector = np.asarray(
        hamiltonian_full[np.ix_(sector_indices, sector_indices)], dtype=complex
    )
    d_sector = np.asarray(drive_full[np.ix_(sector_indices, sector_indices)], dtype=complex)
    x_sector = np.asarray(phonon_full[np.ix_(sector_indices, sector_indices)], dtype=complex)
    y0 = _normalize(initial_state_full[sector_indices], name="DOP853 reference initial state")
    t_eval = np.asarray(times, dtype=float)

    def rhs(time_value: float, state: np.ndarray) -> np.ndarray:
        coefficient = gaussian_sinusoid_waveform(
            float(time_value),
            A=float(drive["amplitude"]),
            omega=float(drive["omega"]),
            tbar=float(drive["tbar"]),
            phi=float(drive["phi"]),
        )
        return -1.0j * ((h_sector + float(coefficient) * d_sector) @ state)

    solution = solve_ivp(
        rhs,
        (float(t_eval[0]), float(t_eval[-1])),
        y0,
        method="DOP853",
        t_eval=t_eval,
        rtol=1.0e-12,
        atol=1.0e-13,
    )
    if not bool(solution.success) or int(solution.y.shape[1]) != int(t_eval.size):
        raise PromotedAPDemoError(f"DOP853 reference validation failed: {solution.message}")
    deltas = {
        "density": 0.0,
        "phonon": 0.0,
        "static_energy": 0.0,
        "state_infidelity": 0.0,
    }
    for midpoint, dop853 in zip(midpoint_states, np.asarray(solution.y, dtype=complex).T):
        left = _normalize(midpoint, name="midpoint reference state")
        right = _normalize(dop853, name="DOP853 reference state")
        deltas["density"] = max(
            deltas["density"], abs(_expectation(left, d_sector) - _expectation(right, d_sector))
        )
        deltas["phonon"] = max(
            deltas["phonon"], abs(_expectation(left, x_sector) - _expectation(right, x_sector))
        )
        deltas["static_energy"] = max(
            deltas["static_energy"],
            abs(_expectation(left, h_sector) - _expectation(right, h_sector)),
        )
        deltas["state_infidelity"] = max(
            deltas["state_infidelity"], 1.0 - _state_fidelity(left, right)
        )
    return {
        "schema": "paper_iii_locked_reference_dop853_validation_v1",
        "method": "scipy_solve_ivp_DOP853",
        "rtol": 1.0e-12,
        "atol": 1.0e-13,
        "point_count": int(t_eval.size),
        "maximum_density_abs_delta": float(deltas["density"]),
        "maximum_phonon_abs_delta": float(deltas["phonon"]),
        "maximum_static_energy_abs_delta": float(deltas["static_energy"]),
        "maximum_state_infidelity": float(deltas["state_infidelity"]),
    }


def _run_ap_grid(
    *,
    runtime_input: Any,
    source_seed_json: Path,
    drive_config: Any,
    drive: Mapping[str, Any],
    dt: float,
    initial_target_full: np.ndarray,
    hamiltonian_full: np.ndarray,
    density_full: np.ndarray,
    phonon_full: np.ndarray,
    sector_indices: np.ndarray,
    inverse_policy: McLachlanInversePolicy,
    progress_callback: Any | None = None,
) -> tuple[dict[str, Any], tuple[np.ndarray, ...]]:
    steps_float = float(drive["t_final"]) / float(dt)
    steps = int(round(steps_float))
    if not math.isclose(steps_float, steps, rel_tol=0.0, abs_tol=1.0e-10):
        raise PromotedAPDemoError(f"t_final={drive['t_final']} is not divisible by dt={dt}")
    times = np.linspace(0.0, float(drive["t_final"]), steps + 1)
    promoted_state = state_from_scaffold_runtime_input(
        runtime_input,
        parameterization_mode=AP_PARAMETERIZATION_LOGICAL_SHARED,
    )
    hamiltonian = time_dependent_hamiltonian_from_runtime_input(
        runtime_input,
        drive_config=drive_config,
    )
    state, root_handoff = _promoted_root_adaptive_state(
        promoted_state,
        source_seed_json=Path(source_seed_json),
        hamiltonian=hamiltonian,
        sector_indices=sector_indices,
    )
    augmentation = augment_state_with_drive_aligned_generator(
        state,
        hamiltonian=hamiltonian,
        enabled=False,
    )
    state = augmentation.state
    support_config = SupportPatchControllerConfig(
        parameterization_mode_default=AP_PARAMETERIZATION_PER_PAULI_TERM,
        exchange_enabled=False,
        prune_enabled=False,
        prune_commit_enabled=False,
        append_ladder_mode="combinatorial",
        append_occurrence_policy="layer_reuse",
        max_append_batch_size=10,
        append_rung_set_cap=64,
        append_prefilter_size=12,
        append_gain_threshold=1.0e-10,
        append_batch_score_threshold=1.0e-10,
        append_min_time=float(dt),
        residual_ratio_threshold=1.0e-3,
        allow_incomplete_candidate_pool=False,
        uses_reference_for_decision=False,
        uses_future_exact_forecast_for_decision=False,
    )
    solve_repair_config = SolveRepairConfig(
        enabled=True,
        # The locked pulse starts at exactly zero, so the relative temporal-kink
        # denominator is singular during the physical zero-to-driven startup.
        # Keep the finite/rho/state-motion repair lanes active and record this
        # explicit local calibration instead of treating that startup as a
        # numerical solve defect.
        state_space_kink_eta_max=None,
    )
    trajectory = run_append_mclachlan_trajectory(
        state=state,
        hamiltonian=hamiltonian,
        times=times,
        inverse_policy=inverse_policy,
        integrator_method="rk4",
        support_patch_config=support_config,
        solve_repair_config=solve_repair_config,
        progress_callback=progress_callback,
        metadata={
            "diagnostic_schema": SCHEMA_VERSION,
            "exact_reference_visible_to_integrator": False,
            "parameterization_mode": AP_PARAMETERIZATION_PER_PAULI_TERM,
            "root_handoff": dict(root_handoff),
        },
    )
    fixed_support_baseline = run_fixed_mclachlan_trajectory(
        state=state,
        hamiltonian=hamiltonian,
        times=times,
        inverse_policy=inverse_policy,
        integrator_method="rk4",
        metadata={
            "diagnostic_schema": SCHEMA_VERSION,
            "baseline_role": "matched_30_atom_hamiltonian_drive_fixed_support",
            "exact_reference_visible_to_integrator": False,
        },
    )
    exact_states = _exact_reference_states(
        initial_state_full=initial_target_full,
        sector_indices=sector_indices,
        hamiltonian_full=hamiltonian_full,
        drive_full=density_full,
        drive=drive,
        times=times,
    )
    rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    ap_states: list[np.ndarray] = []
    h_sector = hamiltonian_full[np.ix_(sector_indices, sector_indices)]
    d_sector = density_full[np.ix_(sector_indices, sector_indices)]
    x_sector = phonon_full[np.ix_(sector_indices, sector_indices)]
    for point, baseline_point, exact in zip(
        trajectory.points,
        fixed_support_baseline.points,
        exact_states,
    ):
        ap = _normalize(point.geometry.psi, name=f"AP state at t={point.time}")
        ap_states.append(ap.copy())
        sector_weight = float(np.linalg.norm(ap[sector_indices]) ** 2)
        ap_sector = _normalize(ap[sector_indices], name=f"AP sector state at t={point.time}")
        coefficient = gaussian_sinusoid_waveform(
            float(point.time),
            A=float(drive["amplitude"]),
            omega=float(drive["omega"]),
            tbar=float(drive["tbar"]),
            phi=float(drive["phi"]),
        )
        rows.append(
            {
                "step_index": int(point.index),
                "time": float(point.time),
                "drive_coefficient": float(coefficient),
                "ap_exact_state_fidelity": _state_fidelity(ap_sector, exact),
                "ap_sector_weight": sector_weight,
                "staggered_density_ap": _expectation(ap, density_full),
                "staggered_density_exact": _expectation(exact, d_sector),
                "staggered_phonon_displacement_ap": _expectation(ap, phonon_full),
                "staggered_phonon_displacement_exact": _expectation(exact, x_sector),
                "static_energy_ap": _expectation(ap, hamiltonian_full),
                "static_energy_exact": _expectation(exact, h_sector),
                "instantaneous_energy_ap": float(point.energy_expectation),
                "mclachlan_residual_sq": float(point.fixed_step.residual_sq),
                "mclachlan_residual_ratio": float(point.fixed_step.residual_ratio),
                "mclachlan_rank": int(point.fixed_step.rank),
                "mclachlan_condition_number": (
                    None
                    if point.fixed_step.condition_number is None
                    else float(point.fixed_step.condition_number)
                ),
                "solve_repair_enabled": bool(point.fixed_step.solve_repair_enabled),
                "solve_repair_applied": bool(point.fixed_step.solve_repair_applied),
                "solve_repair_unsupported": bool(point.fixed_step.solve_repair_unsupported),
                "theta_l2": float(np.linalg.norm(point.theta_runtime)),
                "theta_dot_l2": float(np.linalg.norm(point.fixed_step.theta_dot)),
                "runtime_parameter_count": int(point.runtime_parameter_count),
                "logical_parameter_count": int(point.logical_parameter_count),
                "support_patch": _compact_patch_decision(point.patch_decision),
            }
        )
        baseline = _normalize(
            baseline_point.geometry.psi,
            name=f"fixed-support baseline state at t={baseline_point.time}",
        )
        baseline_sector_weight = float(np.linalg.norm(baseline[sector_indices]) ** 2)
        baseline_sector = _normalize(
            baseline[sector_indices],
            name=f"fixed-support baseline sector state at t={baseline_point.time}",
        )
        baseline_rows.append(
            {
                "step_index": int(baseline_point.index),
                "time": float(baseline_point.time),
                "drive_coefficient": float(coefficient),
                "ap_exact_state_fidelity": _state_fidelity(baseline_sector, exact),
                "ap_sector_weight": baseline_sector_weight,
                "staggered_density_ap": _expectation(baseline, density_full),
                "staggered_density_exact": _expectation(exact, d_sector),
                "staggered_phonon_displacement_ap": _expectation(baseline, phonon_full),
                "staggered_phonon_displacement_exact": _expectation(exact, x_sector),
                "static_energy_ap": _expectation(baseline, hamiltonian_full),
                "static_energy_exact": _expectation(exact, h_sector),
                "instantaneous_energy_ap": float(baseline_point.energy_expectation),
                "mclachlan_residual_sq": float(baseline_point.fixed_step.residual_sq),
                "mclachlan_residual_ratio": float(baseline_point.fixed_step.residual_ratio),
                "mclachlan_rank": int(baseline_point.fixed_step.rank),
                "mclachlan_condition_number": (
                    None
                    if baseline_point.fixed_step.condition_number is None
                    else float(baseline_point.fixed_step.condition_number)
                ),
                "theta_l2": float(np.linalg.norm(baseline_point.theta_runtime)),
                "theta_dot_l2": float(np.linalg.norm(baseline_point.fixed_step.theta_dot)),
                "runtime_parameter_count": int(state.runtime_parameter_count),
                "logical_parameter_count": int(state.logical_parameter_count),
            }
        )
    metrics = _trajectory_metrics(rows)
    baseline_metrics = _trajectory_metrics(baseline_rows)
    metrics["initial_runtime_parameter_count"] = int(state.runtime_parameter_count)
    metrics["final_runtime_parameter_count"] = int(trajectory.final_state.runtime_parameter_count)
    metrics["accepted_support_patch_count"] = int(
        sum(bool(point.patch_decision.accepted) for point in trajectory.points)
    )
    metrics["solve_repair_enabled_at_all_points"] = bool(
        all(bool(point.fixed_step.solve_repair_enabled) for point in trajectory.points)
    )
    metrics["solve_repair_applied_point_count"] = int(
        sum(bool(point.fixed_step.solve_repair_applied) for point in trajectory.points)
    )
    metrics["solve_repair_unsupported_point_count"] = int(
        sum(bool(point.fixed_step.solve_repair_unsupported) for point in trajectory.points)
    )
    final_theta_runtime = np.asarray(trajectory.final_theta_runtime, dtype=float).reshape(-1)
    terminal_replay = _normalize(
        trajectory.final_state.prepare_state(final_theta_runtime),
        name="terminal AP topology/theta replay",
    )
    terminal_replay_error = float(np.linalg.norm(terminal_replay - ap_states[-1]))
    if terminal_replay_error > 1.0e-10:
        raise PromotedAPDemoError(
            "Final AP topology/theta pair does not replay the terminal state: "
            f"error={terminal_replay_error:.3e}"
        )
    return (
        {
            "dt": float(dt),
            "point_count": int(len(rows)),
            "integrator": "rk4",
            "parameterization_mode": AP_PARAMETERIZATION_PER_PAULI_TERM,
            "inverse_policy": {
                "pinv_rcond": float(inverse_policy.pinv_rcond),
                "ridge_lambda": float(inverse_policy.ridge_lambda),
                "solve_damping": float(inverse_policy.solve_damping),
            },
            "initial_state": state.to_json_dict(),
            "final_state_topology": trajectory.final_state.to_json_dict(),
            "final_theta_runtime": [float(value) for value in final_theta_runtime],
            "final_topology_theta_replay_error": float(terminal_replay_error),
            "root_handoff": root_handoff,
            "support_patch_controller": support_config.to_json_dict(),
            "solve_repair": solve_repair_config.to_json_dict(),
            "drive_aligned_ansatz": {
                **augmentation.to_json_dict(),
                "explicit_macro_disabled_to_avoid_duplicate_coordinates": True,
                "drive_pauli_directions_present_in_initial_normalized_pool": True,
            },
            "hamiltonian": hamiltonian.to_json_dict(),
            "trajectory_contract": _compact_adaptive_trajectory_contract(trajectory),
            "fixed_support_baseline": {
                "schema": "paper_iii_matched_fixed_support_baseline_v1",
                "support_profile": NORMALIZED_POOL_HAMILTONIAN_DRIVE,
                "runtime_parameter_count": int(state.runtime_parameter_count),
                "integrator": "rk4",
                "inverse_policy": {
                    "pinv_rcond": float(inverse_policy.pinv_rcond),
                    "ridge_lambda": float(inverse_policy.ridge_lambda),
                    "solve_damping": float(inverse_policy.solve_damping),
                },
                "trajectory_contract": {
                    "schema": "fixed_mclachlan_trajectory_summary_v1",
                    "integrator_method": str(fixed_support_baseline.integrator_method),
                    "point_count": int(len(fixed_support_baseline.points)),
                    "metadata": _json_safe(dict(fixed_support_baseline.metadata)),
                },
                "metrics": baseline_metrics,
                "trajectory": baseline_rows,
            },
            "metrics": metrics,
            "trajectory": rows,
        },
        tuple(ap_states),
    )


def _compact_patch_decision(decision: Any) -> dict[str, Any]:
    selected_score = decision.selected_score
    return {
        "patch_kind": str(decision.patch_kind),
        "accepted": bool(decision.accepted),
        "candidate_count": int(decision.candidate_count),
        "scored_count": int(decision.scored_count),
        "selected_label": None if decision.selected_label is None else str(decision.selected_label),
        "reason": str(decision.reason),
        "selected_score": (
            None if selected_score is None else selected_score.to_json_dict()
        ),
    }


def _compact_adaptive_trajectory_contract(trajectory: Any) -> dict[str, Any]:
    return {
        "schema": "ap_mclachlan_append_trajectory_summary_v1",
        "integrator_method": str(trajectory.integrator_method),
        "point_count": int(len(trajectory.points)),
        "inverse_policy": {
            "policy_id": str(trajectory.inverse_policy.policy_id),
            "pinv_rcond": float(trajectory.inverse_policy.pinv_rcond),
            "ridge_lambda": float(trajectory.inverse_policy.ridge_lambda),
            "solve_damping": float(trajectory.inverse_policy.solve_damping),
        },
        "controller_config": trajectory.controller_config.to_json_dict(),
        "support_patch_config": (
            None
            if trajectory.support_patch_config is None
            else trajectory.support_patch_config.to_json_dict()
        ),
        "solve_repair_config": (
            None
            if trajectory.solve_repair_config is None
            else trajectory.solve_repair_config.to_json_dict()
        ),
        "accepted_support_patch_count": int(
            sum(bool(point.patch_decision.accepted) for point in trajectory.points)
        ),
        "metadata": _json_safe(dict(trajectory.metadata)),
    }


def _promoted_root_adaptive_state(
    promoted_state: APMcLachlanState,
    *,
    source_seed_json: Path,
    hamiltonian: Any,
    sector_indices: np.ndarray,
) -> tuple[APMcLachlanState, dict[str, Any]]:
    """Materialize the validated promoted root and attach Paper-II AP support.

    The promoted HF -> RA -> compact-refit circuit is replayed and validated
    before this boundary.  AP then consumes that circuit output as its static
    statevector handoff.  The preparation parameters are intentionally not live
    McLachlan coordinates; the serialized AP state contains only target-neutral
    Hamiltonian/drive support plus future source-locked full-meta candidates.
    """

    prepared_root = _normalize(
        promoted_state.prepare_state(promoted_state.theta_runtime),
        name="validated promoted root circuit output",
    )
    promoted_initial = _normalize(
        promoted_state.psi_initial,
        name="promoted runtime initial state",
    )
    promoted_replay_fidelity = _state_fidelity(prepared_root, promoted_initial)
    if 1.0 - float(promoted_replay_fidelity) > 1.0e-12:
        raise PromotedAPDemoError(
            "Promoted circuit replay does not match its runtime handoff state: "
            f"infidelity={1.0 - promoted_replay_fidelity:.3e}"
        )

    initial_pool = build_normalized_pauli_pool(
        profile=NORMALIZED_POOL_HAMILTONIAN_DRIVE,
        static_poly=hamiltonian.static_poly,
        drive_poly=hamiltonian.drive_poly,
    )
    source_payload = dict(_read_json(Path(source_seed_json)))
    source_payload["replay_candidate_pool_mode"] = "diagnostic_replay_family_pool"
    source_runtime = load_scaffold_runtime_input_from_payload(
        source_payload,
        artifact_json=Path(source_seed_json),
    )
    if not bool(source_runtime.candidate_pool_source.candidate_pool_complete):
        raise PromotedAPDemoError("Could not reconstruct a complete source full-meta pool")
    future_pool = build_normalized_pauli_pool(
        profile=NORMALIZED_POOL_FULL_META_CHILDREN,
        static_poly=hamiltonian.static_poly,
        drive_poly=hamiltonian.drive_poly,
        candidate_pool_terms=tuple(source_runtime.candidate_pool_terms),
    )
    source_modes = [
        str(getattr(term, "execution_mode", "termwise_product") or "termwise_product")
        .strip()
        .lower()
        for term in tuple(source_runtime.candidate_pool_terms)
    ]
    grouped_exact_parent_count = int(sum(mode == "grouped_exact" for mode in source_modes))
    locked_receipts = {
        "initial_atom_count": int(len(initial_pool.atoms)),
        "initial_ordered_unique_pauli_sha256": str(initial_pool.ordered_unique_pauli_sha256),
        "future_atom_count": int(len(future_pool.atoms)),
        "future_ordered_unique_pauli_sha256": str(future_pool.ordered_unique_pauli_sha256),
        "source_full_meta_parent_count": int(len(tuple(source_runtime.candidate_pool_terms))),
        "source_grouped_exact_parent_count": grouped_exact_parent_count,
    }
    expected_receipts = {
        "initial_atom_count": LOCKED_INITIAL_POOL_COUNT,
        "initial_ordered_unique_pauli_sha256": LOCKED_INITIAL_POOL_SHA256,
        "future_atom_count": LOCKED_FUTURE_POOL_COUNT,
        "future_ordered_unique_pauli_sha256": LOCKED_FUTURE_POOL_SHA256,
        "source_full_meta_parent_count": LOCKED_FULL_META_PARENT_COUNT,
        "source_grouped_exact_parent_count": 0,
    }
    if locked_receipts != expected_receipts:
        raise PromotedAPDemoError(
            "Locked normalized AP pool receipts changed: "
            f"resolved={locked_receipts}, expected={expected_receipts}"
        )
    initial_labels = set(initial_pool.ordered_paulis)
    future_labels = set(future_pool.ordered_paulis)
    warm_overlap_count = int(len(initial_labels.intersection(future_labels)))
    if warm_overlap_count != int(len(initial_labels)):
        raise PromotedAPDemoError(
            "The full-meta layer-reuse pool does not contain every warm-support atom"
        )
    initial_sector_legality = _assert_normalized_pool_preserves_sector(
        initial_pool,
        sector_indices=sector_indices,
    )
    future_sector_legality = _assert_normalized_pool_preserves_sector(
        future_pool,
        sector_indices=sector_indices,
    )
    dynamic_terms = normalized_pool_candidate_terms(initial_pool)
    layout = build_parameter_layout(
        dynamic_terms,
        ignore_identity=True,
        coefficient_tolerance=1.0e-12,
        sort_terms=True,
    )
    theta = np.zeros(int(layout.runtime_parameter_count), dtype=float)
    executor = CompiledAnsatzExecutor(
        dynamic_terms,
        ignore_identity=True,
        coefficient_tolerance=1.0e-12,
        sort_terms=True,
        parameterization_mode=AP_PARAMETERIZATION_PER_PAULI_TERM,
        parameterization_layout=layout,
    )
    future_pool_terms = normalized_pool_candidate_terms(future_pool)
    candidate_source = CandidatePoolSource(
        source_kind="resolved_pool",
        pool_key=f"normalized::{NORMALIZED_POOL_FULL_META_CHILDREN}",
        completeness="complete",
        pool_build_kwargs=dict(source_runtime.candidate_pool_source.pool_build_kwargs),
        filter_payload={
            "normalized_pauli_pool": future_pool.to_json_dict(include_atoms=False),
            "source_full_meta_reconstruction": dict(
                source_runtime.candidate_pool_source.filter_payload
            ),
        },
    )
    state = APMcLachlanState(
        terms=dynamic_terms,
        layout=layout,
        theta_runtime=theta,
        psi_ref=prepared_root,
        psi_initial=prepared_root,
        executor=executor,
        static_hamiltonian=promoted_state.static_hamiltonian,
        resolved_problem=promoted_state.resolved_problem,
        parameterization_mode=AP_PARAMETERIZATION_PER_PAULI_TERM,
        exact_energy=None,
        candidate_pool_terms=future_pool_terms,
        candidate_pool_source=candidate_source,
        provenance={
            **dict(promoted_state.provenance),
            "ap_initialization": "validated_promoted_circuit_statevector_handoff",
            "promoted_prefix_logical_parameter_count": int(promoted_state.logical_parameter_count),
            "initial_support_profile": str(initial_pool.profile),
            "future_pool_profile": str(future_pool.profile),
        },
        extensions={
            **dict(promoted_state.extensions),
            "promoted_root_handoff": {
                "schema": "ap_validated_promoted_root_statevector_handoff_v1",
                "preparation_circuit_order": "HF_to_RA_to_compact_qse_root_refit",
                "preparation_circuit_replayed_and_validated": True,
                "statevector_handoff_used": True,
                "promoted_preparation_parameters_live_in_ap": False,
                "promoted_preparation_circuit_serialized_inside_ap_state": False,
                "prefix_logical_parameter_count": int(promoted_state.logical_parameter_count),
                "prefix_runtime_pauli_parameter_count": int(promoted_state.runtime_pauli_parameter_count),
                "promoted_replay_fidelity": float(promoted_replay_fidelity),
            },
            "initial_support": initial_pool.to_json_dict(include_atoms=False),
            "future_candidate_pool": future_pool.to_json_dict(include_atoms=False),
            "initial_support_sector_legality": initial_sector_legality,
            "future_candidate_pool_sector_legality": future_sector_legality,
            "locked_pool_receipts": locked_receipts,
            "warm_support_overlap_with_future_pool_count": warm_overlap_count,
        },
    )
    replay_error = float(np.linalg.norm(state.prepare_state(theta) - prepared_root))
    if replay_error > 1.0e-12:
        raise PromotedAPDemoError(
            f"Zero-angle initial AP support changed the promoted root by {replay_error:.3e}"
        )
    return state, {
        "schema": "ap_validated_promoted_root_statevector_handoff_v1",
        "preparation_circuit_order": "HF_to_RA_to_compact_qse_root_refit",
        "preparation_circuit_replayed_and_validated": True,
        "statevector_handoff_used": True,
        "promoted_preparation_parameters_live_in_ap": False,
        "promoted_preparation_circuit_serialized_inside_ap_state": False,
        "promoted_prefix_logical_parameter_count": int(promoted_state.logical_parameter_count),
        "promoted_prefix_runtime_pauli_parameter_count": int(promoted_state.runtime_pauli_parameter_count),
        "promoted_replay_fidelity": float(promoted_replay_fidelity),
        "zero_angle_initial_support_replay_error": float(replay_error),
        "initial_support": initial_pool.to_json_dict(include_atoms=False),
        "future_candidate_pool": future_pool.to_json_dict(include_atoms=False),
        "initial_support_sector_legality": initial_sector_legality,
        "future_candidate_pool_sector_legality": future_sector_legality,
        "locked_pool_receipts": locked_receipts,
        "warm_support_overlap_with_future_pool_count": warm_overlap_count,
        "source_candidate_pool_reconstructed_from_typed_settings": True,
        "qse_or_exact_target_used_for_support_selection": False,
    }


def _assert_normalized_pool_preserves_sector(
    contract: Any,
    *,
    sector_indices: np.ndarray,
) -> dict[str, Any]:
    """Fail closed unless every singleton Pauli maps the full legal sector to itself."""

    indices = np.asarray(sector_indices, dtype=np.int64).reshape(-1)
    if indices.size == 0:
        raise PromotedAPDemoError("Cannot validate Pauli-pool legality against an empty sector")
    sector = {int(index) for index in indices}
    rejected: list[str] = []
    for atom in tuple(contract.atoms):
        action = compile_pauli_action_exyz(str(atom.pauli_exyz), int(atom.nq))
        if any((int(index) ^ int(action.flip_mask)) not in sector for index in indices):
            rejected.append(str(atom.pauli_exyz))
    if rejected:
        raise PromotedAPDemoError(
            f"Normalized pool {contract.profile!r} contains {len(rejected)} "
            "singleton Pauli generators that leave the locked particle sector"
        )
    return {
        "criterion": "every_pauli_maps_every_locked_sector_basis_state_inside_sector",
        "sector_dimension": int(indices.size),
        "audited_atom_count": int(len(tuple(contract.atoms))),
        "rejected_atom_count": 0,
        "status": "passed",
    }


def _trajectory_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise PromotedAPDemoError("Cannot summarize an empty AP trajectory")
    fidelities = np.asarray([float(row["ap_exact_state_fidelity"]) for row in rows])
    density_error = np.asarray(
        [float(row["staggered_density_ap"]) - float(row["staggered_density_exact"]) for row in rows]
    )
    phonon_error = np.asarray(
        [
            float(row["staggered_phonon_displacement_ap"])
            - float(row["staggered_phonon_displacement_exact"])
            for row in rows
        ]
    )
    energy_error = np.asarray(
        [float(row["static_energy_ap"]) - float(row["static_energy_exact"]) for row in rows]
    )
    residual_ratio = np.asarray([float(row["mclachlan_residual_ratio"]) for row in rows])
    return {
        "minimum_ap_exact_state_fidelity": float(np.min(fidelities)),
        "final_ap_exact_state_fidelity": float(fidelities[-1]),
        "maximum_staggered_density_abs_error": float(np.max(np.abs(density_error))),
        "maximum_staggered_phonon_abs_error": float(np.max(np.abs(phonon_error))),
        "maximum_static_energy_abs_error": float(np.max(np.abs(energy_error))),
        "maximum_mclachlan_residual_ratio": float(np.max(residual_ratio)),
        "final_mclachlan_residual_ratio": float(residual_ratio[-1]),
        "minimum_sector_weight": float(min(float(row["ap_sector_weight"]) for row in rows)),
        "maximum_rank": int(max(int(row["mclachlan_rank"]) for row in rows)),
        "maximum_condition_number": float(
            max(
                float(row["mclachlan_condition_number"])
                for row in rows
                if row.get("mclachlan_condition_number") is not None
            )
        ),
    }


def _convergence_metrics(
    *,
    coarse: Mapping[str, Any],
    fine: Mapping[str, Any],
    coarse_states: Sequence[np.ndarray],
    fine_states: Sequence[np.ndarray],
) -> dict[str, Any]:
    coarse_rows = list(coarse["trajectory"])
    fine_rows = list(fine["trajectory"])
    ratio_float = float(coarse["dt"]) / float(fine["dt"])
    ratio = int(round(ratio_float))
    if ratio < 1 or not math.isclose(ratio_float, ratio, rel_tol=0.0, abs_tol=1.0e-12):
        raise PromotedAPDemoError("Fine AP grid does not evenly refine the coarse grid")
    if len(fine_rows[::ratio]) != len(coarse_rows):
        raise PromotedAPDemoError("Fine/coarse AP point counts do not align")
    density_delta: list[float] = []
    phonon_delta: list[float] = []
    state_fidelity: list[float] = []
    for idx, (left, right) in enumerate(zip(coarse_rows, fine_rows[::ratio])):
        if not math.isclose(float(left["time"]), float(right["time"]), rel_tol=0.0, abs_tol=1.0e-12):
            raise PromotedAPDemoError(f"Fine/coarse AP time mismatch at index {idx}")
        density_delta.append(abs(float(left["staggered_density_ap"]) - float(right["staggered_density_ap"])))
        phonon_delta.append(
            abs(
                float(left["staggered_phonon_displacement_ap"])
                - float(right["staggered_phonon_displacement_ap"])
            )
        )
        state_fidelity.append(_state_fidelity(coarse_states[idx], fine_states[idx * ratio]))
    return {
        "coarse_dt": float(coarse["dt"]),
        "fine_dt": float(fine["dt"]),
        "maximum_density_abs_delta": float(max(density_delta)),
        "maximum_phonon_abs_delta": float(max(phonon_delta)),
        "minimum_state_fidelity": float(min(state_fidelity)),
    }


def _attach_locked_frozen_rows(
    ap_grid: dict[str, Any],
    *,
    locked_rows: Sequence[Mapping[str, Any]],
    locked_dt: float,
) -> None:
    if not math.isclose(float(ap_grid["dt"]), float(locked_dt), rel_tol=0.0, abs_tol=1.0e-14):
        return
    rows = list(ap_grid["trajectory"])
    if len(rows) != len(locked_rows):
        raise PromotedAPDemoError("Locked frozen-QSE and AP trajectories have different point counts")
    for ap, frozen in zip(rows, locked_rows):
        if not math.isclose(float(ap["time"]), float(frozen["time"]), rel_tol=0.0, abs_tol=1.0e-12):
            raise PromotedAPDemoError("Locked frozen-QSE and AP time grids differ")
        ap["staggered_density_frozen_qse"] = float(frozen["staggered_density_qse"])
        ap["staggered_phonon_displacement_frozen_qse"] = float(
            frozen["staggered_phonon_displacement_qse"]
        )
        ap["frozen_qse_exact_state_fidelity"] = float(frozen["qse_exact_state_fidelity"])


def _science_gate(
    *,
    config: PromotedAPDemoConfig,
    root_refit: Mapping[str, Any],
    promotion: Mapping[str, Any],
    coarse_grid: Mapping[str, Any],
    fine_grid: Mapping[str, Any],
    convergence: Mapping[str, Any],
    reference_validation: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = fine_grid["metrics"]
    coarse_metrics = coarse_grid["metrics"]
    baseline_metrics = fine_grid["fixed_support_baseline"]["metrics"]
    handoff = fine_grid["root_handoff"]
    future_pool = handoff["future_candidate_pool"]
    checks = {
        "root_refit_all_thresholds": bool(root_refit["fit_summary"]["passes"]["all_thresholds"]),
        "runtime_contract_validated": str(promotion["runtime_contract"]["status"]) == "validated",
        "controller_usable_promoted_payload": bool(promotion["controller_boundary"]["controller_usable"]),
        "promoted_circuit_replay": float(handoff["promoted_replay_fidelity"])
        >= 1.0 - 1.0e-12,
        "complete_full_meta_future_pool": bool(fine_grid["initial_state"]["candidate_pool_complete"])
        and str(future_pool["profile"]) == NORMALIZED_POOL_FULL_META_CHILDREN
        and not bool(future_pool["truncated"])
        and int(future_pool["atom_count"]) == int(future_pool["untruncated_atom_count"]),
        "locked_pool_receipts_match": dict(handoff["locked_pool_receipts"])
        == {
            "initial_atom_count": LOCKED_INITIAL_POOL_COUNT,
            "initial_ordered_unique_pauli_sha256": LOCKED_INITIAL_POOL_SHA256,
            "future_atom_count": LOCKED_FUTURE_POOL_COUNT,
            "future_ordered_unique_pauli_sha256": LOCKED_FUTURE_POOL_SHA256,
            "source_full_meta_parent_count": LOCKED_FULL_META_PARENT_COUNT,
            "source_grouped_exact_parent_count": 0,
        }
        and int(handoff["warm_support_overlap_with_future_pool_count"])
        == LOCKED_INITIAL_POOL_COUNT,
        "sector_legal_singleton_pools": str(
            handoff["initial_support_sector_legality"]["status"]
        )
        == "passed"
        and str(handoff["future_candidate_pool_sector_legality"]["status"])
        == "passed",
        "adaptive_support_changed": int(metrics["accepted_support_patch_count"]) > 0
        and int(coarse_metrics["accepted_support_patch_count"]) > 0,
        "adaptive_no_worse_than_fixed_support": float(
            metrics["maximum_staggered_density_abs_error"]
        )
        <= float(baseline_metrics["maximum_staggered_density_abs_error"]) + 1.0e-10
        and float(metrics["maximum_staggered_phonon_abs_error"])
        <= float(baseline_metrics["maximum_staggered_phonon_abs_error"]) + 1.0e-10
        and float(metrics["minimum_ap_exact_state_fidelity"])
        >= float(baseline_metrics["minimum_ap_exact_state_fidelity"]) - 1.0e-10,
        "adaptive_materially_improves_fixed_support": (
            float(baseline_metrics["maximum_staggered_density_abs_error"])
            - float(metrics["maximum_staggered_density_abs_error"])
            >= 1.0e-4
            or float(baseline_metrics["maximum_staggered_phonon_abs_error"])
            - float(metrics["maximum_staggered_phonon_abs_error"])
            >= 1.0e-4
            or float(metrics["minimum_ap_exact_state_fidelity"])
            - float(baseline_metrics["minimum_ap_exact_state_fidelity"])
            >= 1.0e-5
        ),
        "solve_repair_enabled": bool(metrics["solve_repair_enabled_at_all_points"]),
        "solve_repair_supported": int(metrics["solve_repair_unsupported_point_count"]) == 0,
        "locked_inverse_policy_preserved": int(metrics["solve_repair_applied_point_count"])
        == 0,
        "sector_weight_preserved": float(metrics["minimum_sector_weight"]) >= 1.0 - 1.0e-10,
        "terminal_topology_theta_replay": float(fine_grid["final_topology_theta_replay_error"])
        <= 1.0e-10,
        "independent_reference_validation": float(
            reference_validation["maximum_density_abs_delta"]
        )
        <= 5.0e-5
        and float(reference_validation["maximum_phonon_abs_delta"]) <= 5.0e-5
        and float(reference_validation["maximum_static_energy_abs_delta"]) <= 5.0e-6
        and float(reference_validation["maximum_state_infidelity"]) <= 1.0e-8,
        "density_accuracy": float(metrics["maximum_staggered_density_abs_error"])
        <= float(config.maximum_density_abs_error),
        "phonon_accuracy": float(metrics["maximum_staggered_phonon_abs_error"])
        <= float(config.maximum_phonon_abs_error),
        "state_fidelity": float(metrics["minimum_ap_exact_state_fidelity"])
        >= float(config.minimum_state_fidelity),
        "density_integrator_convergence": float(convergence["maximum_density_abs_delta"])
        <= float(config.maximum_convergence_density_delta),
        "phonon_integrator_convergence": float(convergence["maximum_phonon_abs_delta"])
        <= float(config.maximum_convergence_phonon_delta),
        "state_integrator_convergence": float(convergence["minimum_state_fidelity"])
        >= 0.999,
    }
    return {
        "thresholds": {
            "maximum_density_abs_error": float(config.maximum_density_abs_error),
            "maximum_phonon_abs_error": float(config.maximum_phonon_abs_error),
            "minimum_state_fidelity": float(config.minimum_state_fidelity),
            "maximum_convergence_density_delta": float(config.maximum_convergence_density_delta),
            "maximum_convergence_phonon_delta": float(config.maximum_convergence_phonon_delta),
            "minimum_convergence_state_fidelity": 0.999,
            "minimum_sector_weight": 1.0 - 1.0e-10,
            "reference_maximum_density_abs_delta": 5.0e-5,
            "reference_maximum_phonon_abs_delta": 5.0e-5,
            "reference_maximum_static_energy_abs_delta": 5.0e-6,
            "reference_maximum_state_infidelity": 1.0e-8,
        },
        "checks": checks,
        "algorithm_stack_result": bool(all(checks.values())),
        "evidence_classification": "local_diagnostic_not_paper_facing",
    }


def _write_plot(path: Path, *, payload: Mapping[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grids = sorted(payload["ap_trajectories"].values(), key=lambda item: float(item["dt"]), reverse=True)
    coarse = grids[0]
    fine = grids[-1]
    coarse_rows = list(coarse["trajectory"])
    fine_rows = list(fine["trajectory"])
    fixed_rows = list(fine["fixed_support_baseline"]["trajectory"])
    t_fine = np.asarray([row["time"] for row in fine_rows], dtype=float)
    t_coarse = np.asarray([row["time"] for row in coarse_rows], dtype=float)

    fig, axes = plt.subplots(4, 1, figsize=(8.2, 10.5), sharex=True, constrained_layout=True)
    axes[0].plot(t_fine, [row["staggered_density_exact"] for row in fine_rows], "k-", label="exact")
    axes[0].plot(t_fine, [row["staggered_density_ap"] for row in fine_rows], color="#c44e52", label=f"AP RK4 dt={fine['dt']}")
    axes[0].plot(t_fine, [row["staggered_density_ap"] for row in fixed_rows], "-.", color="#777777", label="fixed 30-atom support")
    axes[0].plot(t_coarse, [row["staggered_density_ap"] for row in coarse_rows], "--", color="#dd8452", label=f"AP RK4 dt={coarse['dt']}")
    if "staggered_density_frozen_qse" in coarse_rows[0]:
        axes[0].plot(t_coarse, [row["staggered_density_frozen_qse"] for row in coarse_rows], ":", color="#2673b8", label="frozen QSE")
    axes[0].set_ylabel(r"$n_0-n_1$")
    axes[0].legend(ncol=2, fontsize=8)

    axes[1].plot(t_fine, [row["staggered_phonon_displacement_exact"] for row in fine_rows], "k-", label="exact")
    axes[1].plot(t_fine, [row["staggered_phonon_displacement_ap"] for row in fine_rows], color="#7a3e9d", label="AP fine")
    axes[1].plot(t_fine, [row["staggered_phonon_displacement_ap"] for row in fixed_rows], "-.", color="#777777", label="fixed 30-atom support")
    axes[1].plot(t_coarse, [row["staggered_phonon_displacement_ap"] for row in coarse_rows], "--", color="#dd8452", label="AP coarse")
    if "staggered_phonon_displacement_frozen_qse" in coarse_rows[0]:
        axes[1].plot(t_coarse, [row["staggered_phonon_displacement_frozen_qse"] for row in coarse_rows], ":", color="#2673b8", label="frozen QSE")
    axes[1].set_ylabel(r"$X_0-X_1$")
    axes[1].legend(ncol=2, fontsize=8)

    axes[2].plot(t_fine, [row["ap_exact_state_fidelity"] for row in fine_rows], color="#55a868", label="AP/exact fidelity")
    axes[2].plot(t_fine, [row["ap_exact_state_fidelity"] for row in fixed_rows], "-.", color="#777777", label="fixed-support/exact")
    if "frozen_qse_exact_state_fidelity" in coarse_rows[0]:
        axes[2].plot(t_coarse, [row["frozen_qse_exact_state_fidelity"] for row in coarse_rows], ":", color="#2673b8", label="frozen-QSE/exact")
    axes[2].set_ylabel("state fidelity")
    fidelity_floor = min(
        [float(row["ap_exact_state_fidelity"]) for row in fine_rows]
        + [float(row["ap_exact_state_fidelity"]) for row in fixed_rows]
        + (
            [float(row["frozen_qse_exact_state_fidelity"]) for row in coarse_rows]
            if "frozen_qse_exact_state_fidelity" in coarse_rows[0]
            else []
        )
    )
    axes[2].set_ylim(max(0.0, fidelity_floor - 0.005), 1.0002)
    axes[2].legend(fontsize=8)

    axes[3].semilogy(t_fine, np.maximum(1.0e-16, [row["mclachlan_residual_ratio"] for row in fine_rows]), color="#8172b2")
    axes[3].set_ylabel("McLachlan residual ratio")
    axes[3].set_xlabel("time")
    fig.suptitle("HH excited QSE root → promoted circuit → adaptive driven AP-McLachlan")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_readme(path: Path, *, payload: Mapping[str, Any]) -> None:
    gate = payload["science_gate"]
    fine = min(payload["ap_trajectories"].values(), key=lambda item: float(item["dt"]))
    metrics = fine["metrics"]
    baseline_metrics = fine["fixed_support_baseline"]["metrics"]
    refit = payload["root_refit_summary"]
    handoff = fine["root_handoff"]
    lines = [
        "# Paper III promoted-root AP-McLachlan advisor diagnostic",
        "",
        "This is a local, ideal-statevector diagnostic. It is not Paper-III evidence.",
        "",
        f"- Honest preparation: HF → {refit['base_runtime_parameter_count']} RA rotations → {refit['excitation_runtime_parameter_count']} compact excitation rotations.",
        f"- Root-refit fidelity: {refit['fidelity']:.12f}.",
        f"- Root-refit physical residual: {refit['physical_residual_norm']:.6e}.",
        f"- Runtime promotion status: {payload['promotion_summary']['runtime_contract_status']}.",
        f"- AP handoff: validated promoted-circuit output materialized as the initial state; preparation angles are not live coordinates.",
        f"- Initial AP support: {handoff['initial_support']['atom_count']} zero-angle normalized Hamiltonian+drive Pauli generators.",
        f"- Future AP pool: {handoff['future_candidate_pool']['atom_count']} complete, sector-legal normalized full-meta Pauli children; combinatorial layer-reuse append.",
        f"- Driven AP integrator: RK4, fine dt={fine['dt']}; solve repair enabled.",
        f"- Accepted adaptive support patches: {metrics['accepted_support_patch_count']}.",
        f"- Matched fixed-support minimum fidelity: {baseline_metrics['minimum_ap_exact_state_fidelity']:.8f}.",
        f"- Matched fixed-support maximum density/phonon errors: {baseline_metrics['maximum_staggered_density_abs_error']:.6e} / {baseline_metrics['maximum_staggered_phonon_abs_error']:.6e}.",
        f"- Minimum AP/exact fidelity: {metrics['minimum_ap_exact_state_fidelity']:.8f}.",
        f"- Maximum density error: {metrics['maximum_staggered_density_abs_error']:.6e}.",
        f"- Maximum phonon error: {metrics['maximum_staggered_phonon_abs_error']:.6e}.",
        f"- Algorithm-stack gate: {'PASS' if gate['algorithm_stack_result'] else 'FAIL'}.",
        "",
        "The exact trajectory is the locked same-QSE-initial-state midpoint reference from the earlier advisor diagnostic. It is reconstructed only after AP propagation for comparison and never enters the refit selection, McLachlan solve, drive, or integrator decisions.",
        "",
        "The compact root compiler is an offline, QSE-target-aware ideal-statevector compiler. It does not use an exact eigensystem, but it is not claimed as a scalable hardware state-preparation algorithm.",
        "The exact comparator begins from the QSE Ritz state, while AP begins from its promoted circuit refit; their initial infidelity is reported and is negligible at the locked threshold.",
        "",
        "The prior frozen-QSE curves are retained only as a comparator. They were inaccurate because Q0 projection excluded the resonantly coupled ground-state channel.",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_promoted_ap_demo(config: PromotedAPDemoConfig) -> dict[str, Any]:
    _validate_config(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    root_refit_path = output_dir / "qse_compact_root_refit.json"
    promoted_path = output_dir / "qse_runtime_promoted_ansatz.json"
    result_path = output_dir / "promoted_ap_result.json"
    plot_path = output_dir / "promoted_ap_comparison.png"
    readme_path = output_dir / "README.md"

    qse_payload = _read_json(Path(config.qse_result_json))
    target, _prepared, _basis, _nq = reconstruct_qse_root_target(
        qse_payload,
        qse_result_json=Path(config.qse_result_json),
        state_index=int(config.state_index),
        allow_ground_state=False,
        amplitude_cutoff=1.0e-12,
    )
    locked = _read_json(Path(config.locked_advisor_result_json))
    drive = _locked_drive(locked)
    controller_drive = _controller_drive(drive)
    hamiltonian, density, phonon, sector_indices = _observable_matrices(
        source_seed_json=Path(config.source_seed_json)
    )

    root_refit = run_compact_qse_root_refit(
        CompactQSERootRefitConfig(
            qse_result_json=Path(config.qse_result_json),
            state_index=int(config.state_index),
            output_json=root_refit_path,
            base_scaffold_json=Path(config.source_seed_json),
            hamiltonian_json=Path(config.source_seed_json),
            max_selected_paulis=int(config.max_selected_paulis),
            target_infidelity=float(config.refit_target_infidelity),
            max_energy_error=float(config.refit_max_energy_error),
            max_physical_residual=float(config.refit_max_physical_residual),
            optimizer_maxiter=int(config.refit_optimizer_maxiter),
        )
    )
    promotion = promote_qse_root_refit(
        QSERuntimePromotionConfig(
            qse_root_refit_json=root_refit_path,
            output_json=promoted_path,
            runtime_template_json=Path(config.source_seed_json),
            require_runtime_contract=True,
            max_reconstruction_error=1.0e-10,
        )
    )
    if promotion.get("runtime_payload") is None:
        raise PromotedAPDemoError("Validated promotion omitted runtime_payload")
    runtime_input = load_scaffold_runtime_input_from_payload(
        promotion["runtime_payload"],
        artifact_json=promoted_path,
    )
    promoted_initial_fidelity = _state_fidelity(runtime_input.psi_initial, target.state)
    if promoted_initial_fidelity < 1.0 - float(config.refit_target_infidelity):
        raise PromotedAPDemoError(
            f"Promoted runtime initial fidelity {promoted_initial_fidelity:.12g} misses refit threshold"
        )

    inverse_policy = McLachlanInversePolicy(
        pinv_rcond=float(config.pinv_rcond),
        ridge_lambda=float(config.ridge_lambda),
        solve_damping=float(config.solve_damping),
    )
    grids: dict[str, dict[str, Any]] = {}
    states_by_key: dict[str, tuple[np.ndarray, ...]] = {}
    for dt in sorted({float(value) for value in config.time_steps}, reverse=True):
        key = f"dt_{dt:g}"
        grid, states = _run_ap_grid(
            runtime_input=runtime_input,
            source_seed_json=Path(config.source_seed_json),
            drive_config=_drive_config(controller_drive),
            drive=controller_drive,
            dt=float(dt),
            initial_target_full=target.state,
            hamiltonian_full=hamiltonian,
            density_full=density,
            phonon_full=phonon,
            sector_indices=sector_indices,
            inverse_policy=inverse_policy,
        )
        _attach_locked_frozen_rows(
            grid,
            locked_rows=drive["locked_rows"],
            locked_dt=float(drive["locked_dt"]),
        )
        grids[key] = grid
        states_by_key[key] = states
    locked_times = np.asarray([float(row["time"]) for row in drive["locked_rows"]], dtype=float)
    locked_exact = _exact_reference_states(
        initial_state_full=target.state,
        sector_indices=sector_indices,
        hamiltonian_full=hamiltonian,
        drive_full=density,
        drive=drive,
        times=locked_times,
    )
    reference_replay = _assert_locked_reference_replay(
        exact_states=locked_exact,
        locked_rows=drive["locked_rows"],
        density_sector=density[np.ix_(sector_indices, sector_indices)],
        phonon_sector=phonon[np.ix_(sector_indices, sector_indices)],
        hamiltonian_sector=hamiltonian[np.ix_(sector_indices, sector_indices)],
    )
    independent_reference_validation = _validate_midpoint_reference_against_dop853(
        initial_state_full=target.state,
        sector_indices=sector_indices,
        hamiltonian_full=hamiltonian,
        drive_full=density,
        phonon_full=phonon,
        drive=controller_drive,
        times=locked_times,
        midpoint_states=locked_exact,
    )
    ordered_keys = sorted(grids, key=lambda key: float(grids[key]["dt"]), reverse=True)
    if len(ordered_keys) < 2:
        raise PromotedAPDemoError("At least two nested time steps are required for convergence")
    coarse_key, fine_key = ordered_keys[0], ordered_keys[-1]
    convergence = _convergence_metrics(
        coarse=grids[coarse_key],
        fine=grids[fine_key],
        coarse_states=states_by_key[coarse_key],
        fine_states=states_by_key[fine_key],
    )
    science_gate = _science_gate(
        config=config,
        root_refit=root_refit,
        promotion=promotion,
        coarse_grid=grids[coarse_key],
        fine_grid=grids[fine_key],
        convergence=convergence,
        reference_validation=independent_reference_validation,
    )

    compact = root_refit.get("compact_refit", {})
    composition = root_refit.get("base_scaffold_composition", {})
    physical_residual = root_refit["fit_summary"].get("physical_residual_norm")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "pipeline": PIPELINE,
        "generated_utc": _utc_now(),
        "backend": "ideal_statevector_diagnostic",
        "paper_facing": False,
        "config": asdict(config),
        "source_locks": {
            "qse_result_json": str(Path(config.qse_result_json)),
            "qse_result_sha256": _sha256_file(Path(config.qse_result_json)),
            "source_seed_json": str(Path(config.source_seed_json)),
            "source_seed_sha256": _sha256_file(Path(config.source_seed_json)),
            "locked_advisor_result_json": str(Path(config.locked_advisor_result_json)),
            "locked_advisor_result_sha256": _sha256_file(Path(config.locked_advisor_result_json)),
            "root_refit_json": str(root_refit_path),
            "root_refit_sha256": _sha256_file(root_refit_path),
            "promoted_ansatz_json": str(promoted_path),
            "promoted_ansatz_sha256": _sha256_file(promoted_path),
        },
        "root_refit_summary": {
            "fidelity": float(root_refit["fit_summary"]["fidelity"]),
            "infidelity": float(root_refit["fit_summary"]["infidelity"]),
            "physical_residual_norm": None if physical_residual is None else float(physical_residual),
            "base_runtime_parameter_count": int(composition.get("base_runtime_parameter_count", 0)),
            "excitation_runtime_parameter_count": int(
                compact.get("selected_pauli_count", composition.get("excitation_runtime_parameter_count", 0))
            ),
            "total_runtime_parameter_count": int(
                root_refit["ansatz_payload"]["parameterization"]["runtime_parameter_count"]
            ),
            "promoted_initial_fidelity_to_qse_root": float(promoted_initial_fidelity),
        },
        "promotion_summary": {
            "runtime_contract_status": str(promotion["runtime_contract"]["status"]),
            "controller_usable": bool(promotion["controller_boundary"]["controller_usable"]),
            "prepared_state_reconstruction_error": promotion["runtime_contract"].get(
                "prepared_state_reconstruction_error"
            ),
            "problem_key": promotion["runtime_contract"].get("problem_key"),
        },
        "drive": dict(drive["payload"]),
        "reference_trajectory": {
            "method": "fixed_sector_exponential_midpoint_magnus2_order2",
            "initial_state": "locked_qse_root_zero",
            "identical_to_prior_advisor_reference_on_locked_dt_0p05_grid": True,
            "fine_grid_uses_same_state_drive_and_midpoint_method": True,
            "replay_validation": reference_replay,
            "independent_dop853_validation": independent_reference_validation,
            "used_for_refit_or_ap_decisions": False,
        },
        "ap_trajectories": grids,
        "integrator_convergence": convergence,
        "science_gate": science_gate,
        "decision_data_flow": {
            "qse_target_used_only_by_offline_refit": True,
            "runtime_initial_state_from_sanitized_promoted_payload": True,
            "runtime_future_pool_from_source_locked_full_meta_settings": True,
            "runtime_support_selection_uses_qse_or_exact_target": False,
            "exact_reference_used_for_controller_or_drive_selection": False,
            "exact_reference_used_for_mclachlan_solve": False,
            "exact_reference_constructed_only_after_each_ap_trajectory": True,
            "exact_reference_scope": "post_run_diagnostic_only",
        },
        "observable_audit": {
            "density": "raw_unnormalized_n0_minus_n1",
            "phonon": "raw_unnormalized_X0_minus_X1",
            "ap_builtin_normalized_staggered_field_used": False,
            "operators_evaluated_directly_from_locked_qse_polynomials": True,
        },
    }
    _write_json(result_path, payload)
    _write_plot(plot_path, payload=payload)
    _write_readme(readme_path, payload=payload)
    return payload


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the locked HH QSE-root -> promoted-circuit -> driven AP diagnostic."
    )
    parser.add_argument("--qse-result-json", type=Path, required=True)
    parser.add_argument("--source-seed-json", type=Path, required=True)
    parser.add_argument("--locked-advisor-result-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-selected-paulis", type=int, default=40)
    parser.add_argument("--time-steps", default="0.05,0.025")
    parser.add_argument("--pinv-rcond", type=float, default=1.0e-10)
    parser.add_argument("--ridge-lambda", type=float, default=1.0e-7)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    time_steps = tuple(float(chunk.strip()) for chunk in str(args.time_steps).split(",") if chunk.strip())
    payload = run_promoted_ap_demo(
        PromotedAPDemoConfig(
            qse_result_json=Path(args.qse_result_json),
            source_seed_json=Path(args.source_seed_json),
            locked_advisor_result_json=Path(args.locked_advisor_result_json),
            output_dir=Path(args.output_dir),
            max_selected_paulis=int(args.max_selected_paulis),
            time_steps=time_steps,
            pinv_rcond=float(args.pinv_rcond),
            ridge_lambda=float(args.ridge_lambda),
        )
    )
    fine = min(payload["ap_trajectories"].values(), key=lambda item: float(item["dt"]))
    print(f"output_dir: {args.output_dir}")
    print(f"root_refit_fidelity: {payload['root_refit_summary']['fidelity']:.12g}")
    print(f"runtime_contract_status: {payload['promotion_summary']['runtime_contract_status']}")
    print(f"minimum_ap_exact_fidelity: {fine['metrics']['minimum_ap_exact_state_fidelity']:.12g}")
    print(f"maximum_density_abs_error: {fine['metrics']['maximum_staggered_density_abs_error']:.12g}")
    print(f"maximum_phonon_abs_error: {fine['metrics']['maximum_staggered_phonon_abs_error']:.12g}")
    print(f"algorithm_stack_result: {str(payload['science_gate']['algorithm_stack_result']).lower()}")
    return 0


__all__ = [
    "PIPELINE",
    "SCHEMA_VERSION",
    "PromotedAPDemoConfig",
    "PromotedAPDemoError",
    "build_parser",
    "main",
    "run_promoted_ap_demo",
]


if __name__ == "__main__":
    raise SystemExit(main())
