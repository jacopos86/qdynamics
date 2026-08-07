"""Build the stable Paper V scalar-stability diagnostic PDF.

The report is a working diagnostic artifact, not promoted manuscript evidence.
All numerical tables and plots are regenerated from the scalar harness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from paper5.repo import REPO_ROOT

from .cone_correction import (
    closed_cone_projected_rhs,
    structured_boson_barrier_correction,
)
from .hubbard_dimer import (
    EXTENDED_FAN_MIGDAL_STATE_NAMES,
    FAN_MIGDAL_STATE_NAMES,
    DimerParameters,
    fan_migdal_rhs,
    fan_migdal_with_anomalous_rhs,
    finite_difference_jacobian,
    hartree_fock_zero_correlation_state,
    integrate_rk4,
)
from .initial_conditions import (
    closed_boson_moment_eigenvalues,
    closed_electron_eigenvalues,
    closed_phonon_eigenvalues,
    closed_residual_subtracted_rhs,
    electron_density_eigenvalues,
    exact_ground_closed_scalar_coordinates,
    exact_ground_extended_scalar_coordinates,
    exact_ground_scalar_coordinates,
    extended_residual_subtracted_rhs,
    hartree_fock_closed_scalar_coordinates,
    phonon_density_eigenvalues,
    relative_boson_uncertainty_margin,
    residual_subtracted_rhs,
    source_connected_stationary_state,
)
from .exact_reference import exact_holstein_ground_state
from .matrix_reference import (
    CLOSED_SCALAR_STATE_NAMES,
    MatrixDimerState,
    boson_boundary_flux_decomposition,
    closed_eq14d_history_flux_decomposition,
    closed_scalar_rhs,
    closed_scalar_to_matrix_state,
    discover_invariant_closure,
    extended_scalar_embedding_normal_residual,
    extended_scalar_to_matrix_state,
    matrix_derivative_to_extended_scalar,
    matrix_derivative_to_scalar,
    matrix_dimer_rhs,
    matrix_total_energy,
    pack_matrix_state,
    scalar_embedding_normal_residual,
    scalar_to_matrix_state,
    unpack_matrix_state,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

REPORT_STEM = "paper_v_scalar_stability_diagnostic"
TEX_SOURCE = (
    REPO_ROOT
    / "MATH"
    / "paper_facing"
    / "paper_V_high_u_gkba"
    / f"{REPORT_STEM}.tex"
)
WORKING_NOTE = (
    REPO_ROOT
    / "MATH"
    / "paper_facing"
    / "paper_V_high_u_gkba"
    / "stability_quantum_workflow_working_notes.md"
)
HUBBARD_DIMER_PDF = Path(
    "/Users/jakestrobel/Downloads/Dynamics_on_the_Hubbard_DIMER.pdf"
)
CHIRAL_PHONON_PDF = Path(
    "/Users/jakestrobel/Downloads/Electron_phonon_interactions___chiral_phonons.pdf"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _rk4_trace(
    parameters: DimerParameters,
    *,
    time_step: float,
    final_time: float,
    failure_threshold: float,
) -> dict[str, Any]:
    state = hartree_fock_zero_correlation_state()
    rhs = lambda time, value: fan_migdal_rhs(time, value, parameters)
    time = 0.0
    times = [time]
    states = [state.copy()]
    maxima = [float(np.max(np.abs(state)))]
    failure_time: float | None = None
    failure_component: str | None = None

    while time < final_time:
        step = min(time_step, final_time - time)
        k1 = rhs(time, state)
        k2 = rhs(time + 0.5 * step, state + 0.5 * step * k1)
        k3 = rhs(time + 0.5 * step, state + 0.5 * step * k2)
        k4 = rhs(time + step, state + step * k3)
        state = state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        time += step
        times.append(time)
        states.append(state.copy())
        maxima.append(float(np.max(np.abs(state))))

        if not np.all(np.isfinite(state)) or maxima[-1] > failure_threshold:
            failure_time = time
            component = int(np.argmax(np.abs(state)))
            failure_component = FAN_MIGDAL_STATE_NAMES[component]
            break

    return {
        "times": np.asarray(times),
        "states": np.asarray(states),
        "maxima": np.asarray(maxima),
        "failure_time": failure_time,
        "failure_component": failure_component,
    }


def _adaptive_failure(
    method: str,
    parameters: DimerParameters,
    *,
    final_time: float,
    failure_threshold: float,
) -> dict[str, Any]:
    initial_state = hartree_fock_zero_correlation_state()
    rhs = lambda time, state: fan_migdal_rhs(time, state, parameters)

    def threshold_event(_time: float, state: np.ndarray) -> float:
        return failure_threshold - float(np.max(np.abs(state)))

    threshold_event.terminal = True  # type: ignore[attr-defined]
    threshold_event.direction = -1  # type: ignore[attr-defined]
    solution = solve_ivp(
        rhs,
        (0.0, final_time),
        initial_state,
        method=method,
        rtol=1e-10,
        atol=1e-12,
        max_step=0.1,
        events=threshold_event,
    )
    event_times = solution.t_events[0]
    failure_time = float(event_times[0]) if len(event_times) else None
    return {
        "method": method,
        "rtol": 1e-10,
        "atol": 1e-12,
        "max_step": 0.1,
        "success": bool(solution.success),
        "failure_time": failure_time,
        "function_evaluations": int(solution.nfev),
        "message": solution.message,
    }


def _first_crossing_time(
    times: np.ndarray,
    values: np.ndarray,
    dense_state,
    metric,
) -> float | None:
    if values[0] < -1e-10:
        return float(times[0])
    for index in range(1, len(times)):
        if values[index] < -1e-10:
            return float(
                brentq(
                    lambda time: float(metric(dense_state(time))),
                    float(times[index - 1]),
                    float(times[index]),
                )
            )
    return None


def _scalar_protocol(
    label: str,
    initial_state: np.ndarray,
    rhs,
    *,
    final_time: float,
    failure_threshold: float,
    sample_step: float = 0.05,
    max_step: float = 0.05,
) -> dict[str, Any]:
    def threshold_event(_time: float, state: np.ndarray) -> float:
        return failure_threshold - float(np.max(np.abs(state)))

    threshold_event.terminal = True  # type: ignore[attr-defined]
    threshold_event.direction = -1  # type: ignore[attr-defined]
    sample_times = np.arange(0.0, final_time + 0.5 * sample_step, sample_step)
    solution = solve_ivp(
        rhs,
        (0.0, final_time),
        initial_state,
        method="DOP853",
        t_eval=sample_times,
        dense_output=True,
        rtol=1e-10,
        atol=1e-12,
        max_step=max_step,
        events=threshold_event,
    )
    if solution.sol is None:
        raise RuntimeError("dense scalar protocol solution was not produced")

    times = np.asarray(solution.t)
    states = np.asarray(solution.y)
    electron_minima = np.asarray(
        [
            electron_density_eigenvalues(states[:, index])[0]
            for index in range(states.shape[1])
        ]
    )
    phonon_minima = np.asarray(
        [
            phonon_density_eigenvalues(states[:, index])[0]
            for index in range(states.shape[1])
        ]
    )
    uncertainty_margins = np.asarray(
        [
            relative_boson_uncertainty_margin(states[:, index])
            for index in range(states.shape[1])
        ]
    )
    electron_crossing = _first_crossing_time(
        times,
        electron_minima,
        solution.sol,
        lambda state: electron_density_eigenvalues(state)[0],
    )
    phonon_crossing = _first_crossing_time(
        times,
        phonon_minima,
        solution.sol,
        lambda state: phonon_density_eigenvalues(state)[0],
    )
    uncertainty_crossing = _first_crossing_time(
        times,
        uncertainty_margins,
        solution.sol,
        relative_boson_uncertainty_margin,
    )
    event_times = solution.t_events[0]
    amplitude_failure_time = (
        float(event_times[0]) if len(event_times) else None
    )
    maxima = np.max(np.abs(states), axis=0)
    if amplitude_failure_time is not None:
        event_state = solution.sol(amplitude_failure_time)
        maxima = np.append(maxima, np.max(np.abs(event_state)))
        times = np.append(times, amplitude_failure_time)

    return {
        "label": label,
        "times": times,
        "maxima": maxima,
        "electron_minima": electron_minima,
        "phonon_minima": phonon_minima,
        "uncertainty_margins": uncertainty_margins,
        "amplitude_failure_time": amplitude_failure_time,
        "electron_positivity_loss_time": electron_crossing,
        "phonon_positivity_loss_time": phonon_crossing,
        "boson_uncertainty_loss_time": uncertainty_crossing,
        "minimum_electron_eigenvalue": float(np.min(electron_minima)),
        "minimum_phonon_eigenvalue": float(np.min(phonon_minima)),
        "minimum_boson_uncertainty_margin": float(
            np.min(uncertainty_margins)
        ),
        "max_abs_state": float(np.max(maxima)),
        "requested_final_time": final_time,
        "completed_time": float(
            amplitude_failure_time
            if amplitude_failure_time is not None
            else solution.t[-1]
        ),
        "function_evaluations": int(solution.nfev),
        "rtol": 1e-10,
        "atol": 1e-12,
        "max_step": max_step,
    }


def _closed_scalar_protocol(
    label: str,
    initial_state: np.ndarray,
    rhs,
    *,
    final_time: float,
    failure_threshold: float,
    sample_step: float = 0.05,
    max_step: float = 0.1,
) -> dict[str, Any]:
    """Propagate the invariant 31D closure with full physicality metrics."""

    def threshold_event(_time: float, state: np.ndarray) -> float:
        return failure_threshold - float(np.max(np.abs(state)))

    threshold_event.terminal = True  # type: ignore[attr-defined]
    threshold_event.direction = -1  # type: ignore[attr-defined]
    sample_times = np.arange(
        0.0,
        final_time + 0.5 * sample_step,
        sample_step,
    )
    solution = solve_ivp(
        rhs,
        (0.0, final_time),
        initial_state,
        method="DOP853",
        t_eval=sample_times,
        dense_output=True,
        rtol=1e-9,
        atol=1e-11,
        max_step=max_step,
        events=threshold_event,
    )
    if solution.sol is None:
        raise RuntimeError("dense 31D scalar solution was not produced")

    times = np.asarray(solution.t)
    states = np.asarray(solution.y)
    electron_minima = np.asarray(
        [
            closed_electron_eigenvalues(states[:, index])[0]
            for index in range(states.shape[1])
        ]
    )
    phonon_minima = np.asarray(
        [
            closed_phonon_eigenvalues(states[:, index])[0]
            for index in range(states.shape[1])
        ]
    )
    boson_moment_minima = np.asarray(
        [
            closed_boson_moment_eigenvalues(states[:, index])[0]
            for index in range(states.shape[1])
        ]
    )
    electron_crossing = _first_crossing_time(
        times,
        electron_minima,
        solution.sol,
        lambda state: closed_electron_eigenvalues(state)[0],
    )
    phonon_crossing = _first_crossing_time(
        times,
        phonon_minima,
        solution.sol,
        lambda state: closed_phonon_eigenvalues(state)[0],
    )
    boson_moment_crossing = _first_crossing_time(
        times,
        boson_moment_minima,
        solution.sol,
        lambda state: closed_boson_moment_eigenvalues(state)[0],
    )
    boson_moment_crossing_state = (
        np.asarray(solution.sol(boson_moment_crossing), dtype=float)
        if boson_moment_crossing is not None
        else None
    )
    event_times = solution.t_events[0]
    amplitude_failure_time = (
        float(event_times[0]) if len(event_times) else None
    )
    maxima = np.max(np.abs(states), axis=0)
    if amplitude_failure_time is not None:
        event_state = solution.sol(amplitude_failure_time)
        maxima = np.append(maxima, np.max(np.abs(event_state)))
        times = np.append(times, amplitude_failure_time)

    return {
        "label": label,
        "times": times,
        "maxima": maxima,
        "electron_minima": electron_minima,
        "phonon_minima": phonon_minima,
        "boson_moment_minima": boson_moment_minima,
        "amplitude_failure_time": amplitude_failure_time,
        "electron_positivity_loss_time": electron_crossing,
        "phonon_positivity_loss_time": phonon_crossing,
        "boson_moment_positivity_loss_time": boson_moment_crossing,
        "boson_moment_positivity_loss_state": (
            boson_moment_crossing_state
        ),
        "minimum_electron_eigenvalue": float(np.min(electron_minima)),
        "minimum_phonon_eigenvalue": float(np.min(phonon_minima)),
        "minimum_boson_moment_eigenvalue": float(
            np.min(boson_moment_minima)
        ),
        "max_abs_state": float(np.max(maxima)),
        "requested_final_time": final_time,
        "completed_time": float(
            amplitude_failure_time
            if amplitude_failure_time is not None
            else solution.t[-1]
        ),
        "initial_residual_norm": float(np.linalg.norm(rhs(0.0, initial_state))),
        "function_evaluations": int(solution.nfev),
        "rtol": 1e-9,
        "atol": 1e-11,
        "max_step": max_step,
    }


def _cone_correction_audit(
    parameters: DimerParameters,
    initial_state: np.ndarray,
    *,
    label: str,
    correction_mode: str,
    final_time: float = 20.0,
    sample_step: float = 0.05,
) -> dict[str, Any]:
    """Audit the direct cone barrier with and without energy neutrality."""

    if correction_mode not in {"none", "direct", "energy_neutral"}:
        raise ValueError(f"unsupported correction mode: {correction_mode}")

    base_rhs = closed_residual_subtracted_rhs(parameters, initial_state)
    energy_neutral = correction_mode == "energy_neutral"
    rhs = (
        base_rhs
        if correction_mode == "none"
        else closed_cone_projected_rhs(
            parameters,
            initial_state,
            activation_margin=1e-5,
            target_flux=0.0,
            barrier_rate=5.0,
            energy_neutral=energy_neutral,
        )
    )
    sample_times = np.arange(
        0.0,
        final_time + 0.5 * sample_step,
        sample_step,
    )
    solution = solve_ivp(
        rhs,
        (0.0, final_time),
        initial_state,
        method="DOP853",
        t_eval=sample_times,
        rtol=1e-9,
        atol=1e-11,
        max_step=0.05,
    )
    if not solution.success:
        raise RuntimeError(
            f"{label} cone-correction audit failed: {solution.message}"
        )

    times = np.asarray(solution.t)
    states = np.asarray(solution.y)
    electron_minima = np.asarray(
        [
            closed_electron_eigenvalues(states[:, index])[0]
            for index in range(states.shape[1])
        ]
    )
    phonon_minima = np.asarray(
        [
            closed_phonon_eigenvalues(states[:, index])[0]
            for index in range(states.shape[1])
        ]
    )
    boson_minima = np.asarray(
        [
            closed_boson_moment_eigenvalues(states[:, index])[0]
            for index in range(states.shape[1])
        ]
    )
    energies = np.asarray(
        [
            matrix_total_energy(
                closed_scalar_to_matrix_state(states[:, index]),
                parameters,
            )
            for index in range(states.shape[1])
        ]
    )

    if correction_mode == "none":
        correction_norms = np.zeros(times.size, dtype=float)
        constraint_counts = np.zeros(times.size, dtype=int)
        converged = np.ones(times.size, dtype=bool)
        correction_energy_flux = np.zeros(times.size, dtype=float)
    else:
        corrections = [
            structured_boson_barrier_correction(
                states[:, index],
                base_rhs(float(times[index]), states[:, index]),
                activation_margin=1e-5,
                target_flux=0.0,
                barrier_rate=5.0,
                energy_neutral=energy_neutral,
            )
            for index in range(states.shape[1])
        ]
        correction_norms = np.asarray(
            [item.correction_norm for item in corrections]
        )
        constraint_counts = np.asarray(
            [item.constraint_count for item in corrections],
            dtype=int,
        )
        converged = np.asarray(
            [item.converged for item in corrections],
            dtype=bool,
        )
        correction_energy_flux = parameters.omega_ph * np.asarray(
            [
                item.correction_coordinates[0]
                + item.correction_coordinates[1]
                for item in corrections
            ]
        )

    active = correction_norms > 1e-10
    post_pulse = times >= 4.0
    post_pulse_energies = energies[post_pulse]
    first_active_time = (
        float(times[np.flatnonzero(active)[0]]) if np.any(active) else None
    )
    return {
        "label": label,
        "correction_mode": correction_mode,
        "times": times,
        "electron_minima": electron_minima,
        "phonon_minima": phonon_minima,
        "boson_moment_minima": boson_minima,
        "energies": energies,
        "correction_norms": correction_norms,
        "minimum_electron_eigenvalue": float(np.min(electron_minima)),
        "minimum_phonon_eigenvalue": float(np.min(phonon_minima)),
        "minimum_boson_moment_eigenvalue": float(np.min(boson_minima)),
        "max_abs_state": float(np.max(np.abs(states))),
        "first_active_sample_time": first_active_time,
        "active_sample_fraction": float(np.mean(active)),
        "maximum_correction_norm": float(np.max(correction_norms)),
        "maximum_constraint_count": int(np.max(constraint_counts)),
        "nonconverged_sample_count": int(np.sum(~converged)),
        "maximum_abs_correction_energy_flux": float(
            np.max(np.abs(correction_energy_flux))
        ),
        "energy_initial": float(energies[0]),
        "energy_final": float(energies[-1]),
        "post_pulse_energy_range": float(
            np.max(post_pulse_energies) - np.min(post_pulse_energies)
        ),
        "post_pulse_max_drift_from_t4": float(
            np.max(
                np.abs(post_pulse_energies - post_pulse_energies[0])
            )
        ),
        "requested_final_time": final_time,
        "function_evaluations": int(solution.nfev),
        "rtol": 1e-9,
        "atol": 1e-11,
        "max_step": 0.05,
        "activation_margin": 1e-5,
        "barrier_rate": 5.0,
    }


def _protocol_manifest(protocol: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in protocol.items()
        if key
        not in {
            "times",
            "maxima",
            "electron_minima",
            "phonon_minima",
            "uncertainty_margins",
            "boson_moment_minima",
            "boson_moment_positivity_loss_state",
            "energies",
            "correction_norms",
        }
    }


def _boundary_flux_manifest(
    crossing_time: float,
    decomposition: dict[str, object],
) -> dict[str, Any]:
    """Convert the complex-valued boundary audit into JSON-safe evidence."""

    null_mode = np.asarray(decomposition["null_eigenvector"], dtype=complex)
    term_fluxes = {
        name: float(value)
        for name, value in dict(decomposition["term_fluxes"]).items()
    }
    equation_group_fluxes = {
        "eq14b_correlation_source": float(
            decomposition["eq14b_correlation_source_flux"]
        ),
        "eq14c_correlation_source": float(
            decomposition["eq14c_correlation_source_flux"]
        ),
        "eq14d_direct": float(decomposition["eq14d_direct_flux"]),
        "eq112_residual_subtraction": float(
            decomposition["eq112_residual_subtraction_flux"]
        ),
    }
    total_flux = float(decomposition["total_flux"])
    return {
        "protocol": "31D exact contractions plus Eq. (112)",
        "crossing_time": crossing_time,
        "minimum_eigenvalue": float(decomposition["minimum_eigenvalue"]),
        "next_eigenvalue": float(decomposition["next_eigenvalue"]),
        "spectral_gap": float(decomposition["spectral_gap"]),
        "null_eigenvector_real": null_mode.real.tolist(),
        "null_eigenvector_imag": null_mode.imag.tolist(),
        "term_fluxes": term_fluxes,
        "equation_group_fluxes": equation_group_fluxes,
        "total_flux": total_flux,
        "direct_rhs_flux": float(decomposition["direct_rhs_flux"]),
        "reconstruction_error": float(
            decomposition["reconstruction_error"]
        ),
        "finite_difference_flux": float(
            decomposition["finite_difference_flux"]
        ),
        "finite_difference_error": float(
            decomposition["finite_difference_error"]
        ),
        "relative_to_net_flux": {
            name: value / total_flux
            for name, value in equation_group_fluxes.items()
        },
        "dominant_outward_term": "eq14b_correlation_source",
        "eq14d_role": (
            "zero direct first-derivative contribution; indirect through "
            "the correlation field entering Eqs. (14b) and (14c)"
        ),
    }


def _eq14d_history_manifest(
    decomposition: dict[str, object],
    correlation_ablated: dict[str, object],
    initial_residual: np.ndarray,
) -> dict[str, Any]:
    """Convert the Eq. (14d) causal histories into JSON-safe evidence."""

    eq14b_histories = {
        name: float(value)
        for name, value in dict(
            decomposition["eq14b_flux_by_history"]
        ).items()
    }
    eq14c_histories = {
        name: float(value)
        for name, value in dict(
            decomposition["eq14c_flux_by_history"]
        ).items()
    }
    correlation_norms = {
        name: float(value)
        for name, value in dict(
            decomposition["correlation_norm_by_history"]
        ).items()
    }
    instantaneous_eq14b_rates = {
        name: float(value)
        for name, value in dict(
            decomposition["instantaneous_eq14b_flux_rate_by_term"]
        ).items()
    }
    instantaneous_eq14c_rates = {
        name: float(value)
        for name, value in dict(
            decomposition["instantaneous_eq14c_flux_rate_by_term"]
        ).items()
    }
    realized_eq14b_flux = float(decomposition["realized_eq14b_flux"])
    absolute_flux_sum = float(sum(abs(value) for value in eq14b_histories.values()))
    eq112_history = eq14b_histories["eq112_correlation_subtraction"]
    bare_history = eq14b_histories["eq14d_bare_pauli_source"]
    dominant_pair_net = eq112_history + bare_history
    residual = np.asarray(initial_residual, dtype=float)
    return {
        "protocol": "31D exact contractions plus Eq. (112)",
        "decomposition_convention": (
            "variation of constants along the realized trajectory; "
            "Eq. (14d) transport is the common homogeneous propagator"
        ),
        "crossing_time": float(decomposition["crossing_time"]),
        "minimum_eigenvalue": float(decomposition["minimum_eigenvalue"]),
        "spectral_gap": float(decomposition["spectral_gap"]),
        "correlation_norm_by_history": correlation_norms,
        "correlation_reconstruction_error": float(
            decomposition["correlation_reconstruction_error"]
        ),
        "eq14b_flux_by_history": eq14b_histories,
        "eq14c_flux_by_history": eq14c_histories,
        "realized_eq14b_flux": realized_eq14b_flux,
        "realized_eq14c_flux": float(decomposition["realized_eq14c_flux"]),
        "eq14b_flux_reconstruction_error": float(
            decomposition["eq14b_flux_reconstruction_error"]
        ),
        "eq14c_flux_reconstruction_error": float(
            decomposition["eq14c_flux_reconstruction_error"]
        ),
        "instantaneous_eq14b_flux_rate_by_term": instantaneous_eq14b_rates,
        "instantaneous_eq14c_flux_rate_by_term": instantaneous_eq14c_rates,
        "dominant_outward_history": str(
            decomposition["dominant_outward_history"]
        ),
        "dominant_inward_history": max(
            eq14b_histories,
            key=eq14b_histories.__getitem__,
        ),
        "absolute_flux_sum": absolute_flux_sum,
        "cancellation_fraction": (
            1.0 - abs(realized_eq14b_flux) / absolute_flux_sum
        ),
        "dominant_pair_net_flux": dominant_pair_net,
        "dominant_pair_fraction_of_realized_flux": (
            dominant_pair_net / realized_eq14b_flux
        ),
        "eq112_outward_excess_over_bare_pauli_fraction": (
            (abs(eq112_history) - bare_history) / bare_history
        ),
        "residual_sector_norms": {
            "electronic": float(np.linalg.norm(residual[:3])),
            "coherent_phonon": float(np.linalg.norm(residual[3:7])),
            "normal_phonon": float(np.linalg.norm(residual[7:11])),
            "anomalous_phonon": float(np.linalg.norm(residual[11:17])),
            "electron_phonon_correlation": float(
                np.linalg.norm(residual[17:])
            ),
        },
        "correlation_subtraction_ablation": {
            "full_eq112_crossing_time": float(
                decomposition["crossing_time"]
            ),
            "eq112_without_correlation_sector_crossing_time": float(
                correlation_ablated["crossing_time"]
            ),
            "crossing_delay": float(
                correlation_ablated["crossing_time"]
                - decomposition["crossing_time"]
            ),
        },
    }


def _matrix_protocol(
    label: str,
    initial_state: MatrixDimerState,
    parameters: DimerParameters,
    *,
    final_time: float,
    failure_threshold: float,
    subtract_initial_residual: bool,
) -> dict[str, Any]:
    initial_vector = pack_matrix_state(initial_state)
    undriven = replace(parameters, drive_amplitude=0.0)
    correction = (
        pack_matrix_state(matrix_dimer_rhs(0.0, initial_state, undriven))
        if subtract_initial_residual
        else np.zeros_like(initial_vector)
    )

    def rhs(time: float, vector: np.ndarray) -> np.ndarray:
        state = unpack_matrix_state(vector)
        return (
            pack_matrix_state(matrix_dimer_rhs(time, state, parameters))
            - correction
        )

    def threshold_event(_time: float, vector: np.ndarray) -> float:
        return failure_threshold - float(np.max(np.abs(vector)))

    threshold_event.terminal = True  # type: ignore[attr-defined]
    threshold_event.direction = -1  # type: ignore[attr-defined]
    sample_times = np.arange(0.0, final_time + 0.05, 0.1)
    solution = solve_ivp(
        rhs,
        (0.0, final_time),
        initial_vector,
        method="DOP853",
        t_eval=sample_times,
        dense_output=True,
        rtol=1e-9,
        atol=1e-11,
        max_step=0.05,
        events=threshold_event,
    )
    if solution.sol is None:
        raise RuntimeError("dense matrix protocol solution was not produced")

    electron_minima = []
    phonon_minima = []
    for index in range(solution.y.shape[1]):
        state = unpack_matrix_state(solution.y[:, index])
        electron = 0.5 * (
            state.electron_density + state.electron_density.conjugate().T
        )
        phonon = 0.5 * (
            state.phonon_density + state.phonon_density.conjugate().T
        )
        electron_minima.append(float(np.linalg.eigvalsh(electron)[0]))
        phonon_minima.append(float(np.linalg.eigvalsh(phonon)[0]))
    electron_minima_array = np.asarray(electron_minima)
    phonon_minima_array = np.asarray(phonon_minima)

    def matrix_electron_minimum(vector: np.ndarray) -> float:
        state = unpack_matrix_state(vector)
        electron = 0.5 * (
            state.electron_density + state.electron_density.conjugate().T
        )
        return float(np.linalg.eigvalsh(electron)[0])

    def matrix_phonon_minimum(vector: np.ndarray) -> float:
        state = unpack_matrix_state(vector)
        phonon = 0.5 * (
            state.phonon_density + state.phonon_density.conjugate().T
        )
        return float(np.linalg.eigvalsh(phonon)[0])

    electron_crossing = _first_crossing_time(
        solution.t,
        electron_minima_array,
        solution.sol,
        matrix_electron_minimum,
    )
    phonon_crossing = _first_crossing_time(
        solution.t,
        phonon_minima_array,
        solution.sol,
        matrix_phonon_minimum,
    )
    event_times = solution.t_events[0]
    amplitude_failure_time = (
        float(event_times[0]) if len(event_times) else None
    )
    max_abs_state = float(np.max(np.abs(solution.y)))
    if amplitude_failure_time is not None:
        max_abs_state = max(
            max_abs_state,
            float(np.max(np.abs(solution.sol(amplitude_failure_time)))),
        )
    return {
        "label": label,
        "subtract_initial_residual": subtract_initial_residual,
        "requested_final_time": final_time,
        "amplitude_failure_time": amplitude_failure_time,
        "electron_positivity_loss_time": electron_crossing,
        "phonon_positivity_loss_time": phonon_crossing,
        "minimum_electron_eigenvalue": float(
            np.min(electron_minima_array)
        ),
        "minimum_phonon_eigenvalue": float(np.min(phonon_minima_array)),
        "max_abs_state": max_abs_state,
        "initial_residual_norm": float(np.linalg.norm(rhs(0.0, initial_vector))),
        "function_evaluations": int(solution.nfev),
    }


def _matrix_scalar_parity(
    parameters: DimerParameters,
    *,
    samples: int = 100,
) -> dict[str, Any]:
    rng = np.random.default_rng(260622233)
    maximum_error = 0.0
    for _ in range(samples):
        state = rng.normal(scale=0.15, size=len(FAN_MIGDAL_STATE_NAMES))
        bloch = rng.normal(size=3)
        bloch *= rng.uniform(0.0, 0.9) / np.linalg.norm(bloch)
        state[0] = bloch[2]
        state[1] = 0.5 * bloch[0]
        state[2] = 0.5 * bloch[1]
        state[5] = rng.uniform(0.05, 0.5)
        state[6] = rng.uniform(-state[5], state[5])
        time = float(rng.uniform(0.0, 3.0))
        matrix_derivative = matrix_dimer_rhs(
            time,
            scalar_to_matrix_state(state),
            parameters,
        )
        projected = matrix_derivative_to_scalar(matrix_derivative)
        scalar = fan_migdal_rhs(time, state, parameters)
        maximum_error = max(
            maximum_error,
            float(np.max(np.abs(projected - scalar))),
        )

    initial = hartree_fock_zero_correlation_state()
    first_order_step = 1e-3
    first_order_state = initial + first_order_step * fan_migdal_rhs(
        0.0, initial, parameters
    )
    first_order_matrix_derivative = matrix_dimer_rhs(
        first_order_step,
        scalar_to_matrix_state(first_order_state),
        parameters,
    )
    normal = scalar_embedding_normal_residual(first_order_matrix_derivative)
    return {
        "samples": samples,
        "random_seed": 260622233,
        "maximum_component_error": maximum_error,
        "retained_equations": ["14a", "14b", "14d", "14e"],
        "omitted_equation": "14c",
        "first_order_probe_step": first_order_step,
        "normal_residual_after_first_order_probe": normal,
    }


def _extended_matrix_scalar_parity(
    parameters: DimerParameters,
    *,
    samples: int = 100,
) -> dict[str, Any]:
    """Audit the two-coordinate Eq. (14c) extension and remaining normal sector."""

    rng = np.random.default_rng(140315)
    maximum_error = 0.0
    maximum_anomalous_normal = 0.0
    maximum_preexisting_normal = 0.0
    for _ in range(samples):
        state = rng.normal(
            scale=0.15,
            size=len(EXTENDED_FAN_MIGDAL_STATE_NAMES),
        )
        bloch = rng.normal(size=3)
        bloch *= rng.uniform(0.0, 0.9) / np.linalg.norm(bloch)
        state[0] = bloch[2]
        state[1] = 0.5 * bloch[0]
        state[2] = 0.5 * bloch[1]
        state[5] = rng.uniform(0.05, 0.5)
        state[6] = rng.uniform(-state[5], state[5])
        time = float(rng.uniform(0.0, 3.0))

        matrix_derivative = matrix_dimer_rhs(
            time,
            extended_scalar_to_matrix_state(state),
            parameters,
        )
        projected = matrix_derivative_to_extended_scalar(matrix_derivative)
        scalar = fan_migdal_with_anomalous_rhs(
            time,
            state,
            parameters,
        )
        maximum_error = max(
            maximum_error,
            float(np.max(np.abs(projected - scalar))),
        )
        normal = extended_scalar_embedding_normal_residual(matrix_derivative)
        for name, value in normal.items():
            if name.startswith("anomalous_"):
                maximum_anomalous_normal = max(
                    maximum_anomalous_normal,
                    value,
                )
            else:
                maximum_preexisting_normal = max(
                    maximum_preexisting_normal,
                    value,
                )

    return {
        "samples": samples,
        "random_seed": 140315,
        "state_dimension": len(EXTENDED_FAN_MIGDAL_STATE_NAMES),
        "added_coordinates": [
            "anomalous_relative_real",
            "anomalous_relative_imag",
        ],
        "maximum_component_error": maximum_error,
        "maximum_anomalous_normal_residual": maximum_anomalous_normal,
        "maximum_preexisting_normal_residual": maximum_preexisting_normal,
        "closure_status": (
            "eq14c_projection_exact_but_full_closure_false_due_to_"
            "correlation_trace_and_mode_sum_sector"
        ),
    }


def _jacobian_series(
    trace: dict[str, Any],
    parameters: DimerParameters,
    *,
    sample_interval: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    times: np.ndarray = trace["times"]
    states: np.ndarray = trace["states"]
    time_step = float(times[1] - times[0])
    stride = max(1, round(sample_interval / time_step))
    sample_indices = np.arange(0, len(times), stride)
    if sample_indices[-1] != len(times) - 1:
        sample_indices = np.append(sample_indices, len(times) - 1)

    sampled_times = times[sample_indices]
    max_real_parts = []
    rhs = lambda time, state: fan_migdal_rhs(time, state, parameters)
    for index in sample_indices:
        jacobian = finite_difference_jacobian(rhs, float(times[index]), states[index])
        eigenvalues = np.linalg.eigvals(jacobian)
        max_real_parts.append(float(np.max(np.real(eigenvalues))))
    return sampled_times, np.asarray(max_real_parts)


def _state_at(trace: dict[str, Any], target_time: float) -> np.ndarray:
    times: np.ndarray = trace["times"]
    index = int(np.argmin(np.abs(times - target_time)))
    return trace["states"][index]


def _critical_terms(
    time: float,
    state: np.ndarray,
    parameters: DimerParameters,
) -> dict[str, float]:
    (
        delta_n,
        rho_real,
        rho_imag,
        delta_b_real,
        _delta_b_imag,
        phonon_population,
        phonon_coherence,
        delta_corr_real,
        delta_corr_imag,
        delta_corr_imag_plus,
        delta_corr_imag_minus,
        delta_corr_real_plus,
        delta_corr_real_minus,
    ) = state
    coupling = parameters.coupling
    omega_ph = parameters.omega_ph
    drive = parameters.drive_difference(time)
    phonon_factor = 1.0 + 2.0 * phonon_population - 2.0 * phonon_coherence
    return {
        "Eq. 95: density-blocking source": float(
            -0.25 * coupling * (1.0 - delta_n**2)
        ),
        "Eq. 97: coherent-field feedback": float(
            -2.0 * coupling * delta_b_real * delta_corr_real_plus
        ),
        "Eq. 97: population feedback": float(
            -coupling * rho_real * phonon_factor
        ),
        "Eq. 98: coherent-field feedback": float(
            2.0 * coupling * delta_b_real * delta_corr_imag_minus
        ),
        "Eq. 98: population feedback": float(
            coupling * rho_imag * phonon_factor
        ),
        "Eq. 99: coherent-field feedback": float(
            2.0 * coupling * delta_b_real * delta_corr_imag_plus
        ),
        "Eq. 99: density feedback": float(-coupling * rho_imag * delta_n),
        "Eq. 97: remaining linear terms": float(
            -4.0 * parameters.hopping * delta_corr_real
            - delta_corr_real_plus * drive
            - omega_ph * delta_corr_real_minus
        ),
        "Eq. 98: remaining linear terms": float(
            delta_corr_imag_minus * drive
            + omega_ph * delta_corr_imag_plus
        ),
        "Eq. 99: remaining linear terms": float(
            4.0 * parameters.hopping * delta_corr_imag
            + delta_corr_imag_plus * drive
            + omega_ph * delta_corr_imag_minus
        ),
    }


def _plot_trajectory(
    strong_trace: dict[str, Any],
    weak_trace: dict[str, Any],
    *,
    failure_threshold: float,
    output: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 3.2))
    axis.semilogy(
        strong_trace["times"],
        strong_trace["maxima"],
        color="#A33A2B",
        linewidth=1.8,
        label=r"strong coupling: $\lambda=1.5$",
    )
    axis.semilogy(
        weak_trace["times"],
        weak_trace["maxima"],
        color="#236C5B",
        linewidth=1.8,
        label=r"weak control: $\lambda=0.5$",
    )
    axis.axhline(
        failure_threshold,
        color="#4A4F57",
        linestyle="--",
        linewidth=1.0,
        label=r"declared failure threshold $10^4$",
    )
    axis.set_xlim(0.0, 140.0)
    axis.set_ylim(1e-1, 3e4)
    axis.set_xlabel(r"time $t\,t_{\mathrm{hop}}$")
    axis.set_ylabel(r"$\max_i |x_i(t)|$")
    axis.grid(True, which="both", alpha=0.22)
    axis.legend(loc="upper left", frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def _plot_jacobian(
    times: np.ndarray,
    max_real_parts: np.ndarray,
    *,
    failure_time: float,
    output: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 3.0))
    axis.plot(times, max_real_parts, color="#4E2A84", linewidth=1.8)
    axis.axhline(0.0, color="#4A4F57", linewidth=0.8)
    axis.axvline(
        failure_time,
        color="#A33A2B",
        linestyle="--",
        linewidth=1.0,
        label="threshold crossing",
    )
    axis.set_xlim(0.0, 131.0)
    axis.set_xlabel(r"time $t\,t_{\mathrm{hop}}$")
    axis.set_ylabel(r"$\max\,\mathrm{Re}\,\sigma[J(x(t),t)]$")
    axis.grid(True, alpha=0.22)
    axis.legend(loc="upper left", frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def _plot_ablations(ablations: list[dict[str, Any]], output: Path) -> None:
    labels = ["full", "Eq. 95 off", "Eq. 97 off", "both off"]
    values = [item["failure_time"] for item in ablations]
    colors = ["#17365D", "#D48A28", "#B04A7A", "#777C83"]
    figure, axis = plt.subplots(figsize=(7.2, 2.8))
    bars = axis.bar(labels, values, color=colors, width=0.62)
    axis.set_ylabel(r"failure time $t\,t_{\mathrm{hop}}$")
    axis.set_ylim(0.0, 145.0)
    axis.grid(True, axis="y", alpha=0.22)
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 3.0,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def _plot_initialization_protocols(
    protocols: list[dict[str, Any]],
    *,
    failure_threshold: float,
    output: Path,
) -> None:
    colors = ["#A33A2B", "#4E2A84", "#236C5B"]
    figure, axis = plt.subplots(figsize=(7.2, 3.2))
    for protocol, color in zip(protocols, colors, strict=True):
        axis.semilogy(
            protocol["times"],
            protocol["maxima"],
            color=color,
            linewidth=1.6,
            label=protocol["label"],
        )
    axis.axhline(
        failure_threshold,
        color="#4A4F57",
        linestyle="--",
        linewidth=1.0,
        label=r"amplitude threshold $10^4$",
    )
    axis.set_xlim(0.0, 400.0)
    axis.set_ylim(1e-2, 3e4)
    axis.set_xlabel(r"time $t\,t_{\mathrm{hop}}$")
    axis.set_ylabel(r"$\max_i |x_i(t)|$")
    axis.grid(True, which="both", alpha=0.22)
    axis.legend(loc="upper left", frameon=False, fontsize=7.5)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def _plot_regularized_physicality(
    protocol: dict[str, Any],
    output: Path,
) -> None:
    times = protocol["times"][: len(protocol["electron_minima"])]
    figure, axis = plt.subplots(figsize=(7.2, 2.9))
    axis.plot(
        times,
        protocol["electron_minima"],
        color="#17365D",
        linewidth=1.5,
        label="minimum electronic 1-RDM eigenvalue",
    )
    axis.plot(
        times,
        protocol["phonon_minima"],
        color="#236C5B",
        linewidth=1.5,
        label="minimum retained phonon-density eigenvalue",
    )
    axis.set_yscale("log")
    axis.set_ylim(1e-5, 4e-1)
    axis.set_xlim(0.0, float(protocol["requested_final_time"]))
    axis.set_xlabel(r"time $t\,t_{\mathrm{hop}}$")
    axis.set_ylabel("minimum eigenvalue")
    axis.grid(True, alpha=0.22)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.17),
        ncol=2,
        frameon=False,
        fontsize=7.5,
    )
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def _plot_eq14c_protocols(
    baseline_13: dict[str, Any],
    protocols_15: list[dict[str, Any]],
    *,
    failure_threshold: float,
    output: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 3.2))
    axis.semilogy(
        baseline_13["times"],
        baseline_13["maxima"],
        color="#A33A2B",
        linewidth=1.4,
        linestyle="--",
        label="13D HF/zero",
    )
    colors = ["#17365D", "#4E2A84", "#236C5B"]
    for protocol, color in zip(protocols_15, colors, strict=True):
        axis.semilogy(
            protocol["times"],
            protocol["maxima"],
            color=color,
            linewidth=1.6,
            label=protocol["label"],
        )
    axis.axhline(
        failure_threshold,
        color="#4A4F57",
        linestyle=":",
        linewidth=1.0,
        label=r"amplitude threshold $10^4$",
    )
    axis.set_xlim(0.0, 400.0)
    axis.set_ylim(1e-2, 3e4)
    axis.set_xlabel(r"time $t\,t_{\mathrm{hop}}$")
    axis.set_ylabel(r"$\max_i |x_i(t)|$")
    axis.grid(True, which="both", alpha=0.22)
    axis.legend(loc="upper left", frameon=False, fontsize=7.0, ncol=2)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def _plot_eq14c_uncertainty(
    protocols: list[dict[str, Any]],
    output: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 3.0))
    colors = ["#17365D", "#4E2A84", "#236C5B"]
    for protocol, color in zip(protocols, colors, strict=True):
        count = len(protocol["uncertainty_margins"])
        axis.plot(
            protocol["times"][:count],
            protocol["uncertainty_margins"],
            color=color,
            linewidth=1.5,
            label=protocol["label"],
        )
    axis.axhline(
        0.0,
        color="#A33A2B",
        linestyle="--",
        linewidth=1.0,
        label="one-mode uncertainty boundary",
    )
    axis.set_xlim(0.0, 50.0)
    axis.set_ylim(-3.0, 4.5)
    axis.set_xlabel(r"time $t\,t_{\mathrm{hop}}$")
    axis.set_ylabel(r"$n_-(n_-+1)-|m_-|^2$")
    axis.grid(True, alpha=0.22)
    axis.legend(loc="lower left", frameon=False, fontsize=7.0)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def _plot_closed_scalar_protocols(
    protocols: list[dict[str, Any]],
    *,
    failure_threshold: float,
    output: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 3.2))
    colors = ["#A33A2B", "#4E2A84", "#236C5B"]
    for protocol, color in zip(protocols, colors, strict=True):
        axis.semilogy(
            protocol["times"],
            protocol["maxima"],
            color=color,
            linewidth=1.6,
            label=protocol["label"],
        )
    axis.axhline(
        failure_threshold,
        color="#4A4F57",
        linestyle="--",
        linewidth=1.0,
        label=r"amplitude threshold $10^4$",
    )
    axis.set_xlim(0.0, 400.0)
    axis.set_ylim(1e-2, 3e4)
    axis.set_xlabel(r"time $t\,t_{\mathrm{hop}}$")
    axis.set_ylabel(r"$\max_i |x_i(t)|$")
    axis.grid(True, which="both", alpha=0.22)
    axis.legend(loc="upper right", frameon=False, fontsize=7.2)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def _plot_closed_scalar_physicality(
    protocol: dict[str, Any],
    output: Path,
) -> None:
    count = len(protocol["boson_moment_minima"])
    times = protocol["times"][:count]
    figure, axis = plt.subplots(figsize=(7.2, 3.0))
    axis.plot(
        times,
        protocol["electron_minima"],
        color="#17365D",
        linewidth=1.5,
        label="electronic 1-RDM minimum",
    )
    axis.plot(
        times,
        protocol["phonon_minima"],
        color="#4E2A84",
        linewidth=1.5,
        label="normal-phonon minimum",
    )
    axis.plot(
        times,
        protocol["boson_moment_minima"],
        color="#236C5B",
        linewidth=1.5,
        label="full boson-moment minimum",
    )
    axis.axhline(
        0.0,
        color="#A33A2B",
        linestyle="--",
        linewidth=1.0,
        label="physicality boundary",
    )
    axis.set_xlim(0.0, 20.0)
    axis.set_ylim(-0.8, 0.25)
    axis.set_xlabel(r"time $t\,t_{\mathrm{hop}}$")
    axis.set_ylabel("minimum eigenvalue")
    axis.grid(True, alpha=0.22)
    axis.legend(loc="lower left", frameon=False, fontsize=7.0)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def _plot_boson_boundary_flux(
    boundary_flux: dict[str, Any],
    output: Path,
) -> None:
    """Plot signed equation-group contributions to the boundary flux."""

    groups = boundary_flux["equation_group_fluxes"]
    labels = [
        "Eq. (14b), correlation source",
        "Eq. (14c), correlation source",
        "Eq. (14d), direct",
        "Eq. (112), subtraction",
    ]
    values = np.asarray(
        [
            groups["eq14b_correlation_source"],
            groups["eq14c_correlation_source"],
            groups["eq14d_direct"],
            groups["eq112_residual_subtraction"],
        ],
        dtype=float,
    )
    colors = [
        "#A33A2B" if value < 0.0 else "#236C5B"
        for value in values
    ]
    figure, axis = plt.subplots(figsize=(7.2, 3.0))
    positions = np.arange(len(labels))
    axis.barh(positions, 1e3 * values, color=colors, height=0.62)
    axis.axvline(0.0, color="#4A4F57", linewidth=0.9)
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlim(-4.4, 0.55)
    axis.set_xlabel(r"projected flux $10^3\dot{\lambda}_{\min}$")
    axis.grid(True, axis="x", alpha=0.22)
    for position, value in zip(positions, values, strict=True):
        offset = 5
        horizontal_alignment = "left"
        text_color = "white" if value < 0.0 else "black"
        axis.annotate(
            f"{value:+.3e}",
            (1e3 * value, position),
            xytext=(offset, 0),
            textcoords="offset points",
            ha=horizontal_alignment,
            va="center",
            fontsize=7.2,
            color=text_color,
        )
    axis.set_title(
        "First full-boson boundary crossing, "
        + rf"$t={boundary_flux['crossing_time']:.5f}$"
    )
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def _plot_eq14d_history_flux(
    history: dict[str, Any],
    output: Path,
) -> None:
    """Plot the large causal cancellation and its smaller source pairs."""

    fluxes = history["eq14b_flux_by_history"]
    dominant_names = [
        "eq112_correlation_subtraction",
        "eq14d_bare_pauli_source",
    ]
    smaller_names = [
        "eq14d_anomalous_first_source",
        "eq14d_anomalous_second_source",
        "eq14d_normal_particle_source",
        "eq14d_normal_hole_source",
        "initial_correlation",
    ]
    labels = {
        "eq112_correlation_subtraction": "Eq. (112), correlation sector",
        "eq14d_bare_pauli_source": "Eq. (14d), bare Pauli source",
        "eq14d_anomalous_first_source": "Eq. (14d), anomalous source 1",
        "eq14d_anomalous_second_source": "Eq. (14d), anomalous source 2",
        "eq14d_normal_particle_source": "Eq. (14d), normal particle source",
        "eq14d_normal_hole_source": "Eq. (14d), normal hole source",
        "initial_correlation": "propagated initial correlation",
    }

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(7.2, 4.7),
        gridspec_kw={"height_ratios": [1.0, 1.9]},
    )
    for axis, names, scale, xlabel in (
        (
            axes[0],
            dominant_names,
            1.0,
            r"causal contribution to Eq. (14b) boundary flux",
        ),
        (
            axes[1],
            smaller_names,
            1e4,
            r"smaller contributions $10^4\times$ boundary flux",
        ),
    ):
        values = np.asarray([scale * fluxes[name] for name in names])
        positions = np.arange(len(names))
        colors = [
            "#A33A2B" if value < 0.0 else "#236C5B"
            for value in values
        ]
        axis.barh(positions, values, color=colors, height=0.62)
        axis.axvline(0.0, color="#4A4F57", linewidth=0.9)
        axis.set_yticks(positions, [labels[name] for name in names])
        axis.invert_yaxis()
        axis.set_xlabel(xlabel)
        axis.grid(True, axis="x", alpha=0.22)
        span = max(abs(values))
        axis.set_xlim(-1.18 * span, 1.18 * span)
        for position, raw_value, plotted_value in zip(
            positions,
            [fluxes[name] for name in names],
            values,
            strict=True,
        ):
            place_inside = abs(plotted_value) > 0.12 * span
            if place_inside:
                offset = 4 if plotted_value < 0.0 else -4
                alignment = "left" if plotted_value < 0.0 else "right"
                text_color = "white"
            else:
                offset = -4 if plotted_value < 0.0 else 4
                alignment = "right" if plotted_value < 0.0 else "left"
                text_color = "black"
            axis.annotate(
                f"{raw_value:+.3e}",
                (plotted_value, position),
                xytext=(offset, 0),
                textcoords="offset points",
                ha=alignment,
                va="center",
                fontsize=7.0,
                color=text_color,
            )
    axes[0].set_title(
        "Eq. (14d) causal histories at the first bosonic crossing"
    )
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def _plot_cone_correction_audit(
    protocols: list[dict[str, Any]],
    output: Path,
) -> None:
    """Compare cone margin, energy drift, and correction activity."""

    colors = {
        "none": "#A33A2B",
        "direct": "#1F5A94",
        "energy_neutral": "#236C5B",
    }
    figure, axes = plt.subplots(3, 1, figsize=(7.2, 7.1), sharex=True)
    for protocol in protocols:
        mode = protocol["correction_mode"]
        times = protocol["times"]
        axes[0].plot(
            times,
            protocol["boson_moment_minima"],
            color=colors[mode],
            linewidth=1.35,
            label=protocol["label"],
        )
        reference_index = int(np.searchsorted(times, 4.0))
        energy_change = (
            protocol["energies"] - protocol["energies"][reference_index]
        )
        axes[1].plot(
            times,
            energy_change,
            color=colors[mode],
            linewidth=1.35,
            label=protocol["label"],
        )
        if mode != "none":
            axes[2].plot(
                times,
                protocol["correction_norms"],
                color=colors[mode],
                linewidth=1.2,
                label=protocol["label"],
            )

    axes[0].axhline(0.0, color="#4A4F57", linewidth=0.8)
    axes[0].set_yscale("symlog", linthresh=1e-4)
    axes[0].set_ylabel(
        r"$\lambda_{\min}(\mathcal{M}_{\mathrm{B}})$"
    )
    axes[0].set_title(
        "Structured cone correction, pinned 31D protocol",
        pad=30,
    )
    axes[0].legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        frameon=False,
        fontsize=7.4,
        ncol=3,
    )
    axes[1].axhline(0.0, color="#4A4F57", linewidth=0.8)
    axes[1].set_yscale("symlog", linthresh=1e-6)
    axes[1].set_ylabel(r"$E(t)-E(4)$")
    axes[2].set_ylabel(r"$\|y_\ast\|_2$")
    axes[2].set_xlabel(r"time $t\,t_{\rm hop}$")
    axes[2].legend(frameon=False, fontsize=7.4, ncol=2)
    for axis in axes:
        axis.grid(True, alpha=0.22)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def _tex_scientific(value: float, digits: int = 3) -> str:
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return rf"{mantissa}\times 10^{{{int(exponent)}}}"


def _tex_fragment(manifest: dict[str, Any]) -> str:
    baseline = manifest["baseline"]
    weak = manifest["weak_control"]
    ablations = manifest["ablations"]
    critical = manifest["critical_terms_at_t130"]
    adaptive = manifest["adaptive_integrators"]
    rk4 = manifest["rk4_refinement"]
    residual = manifest["initial_residual"]
    parity = manifest["matrix_scalar_parity"]
    stationary = manifest["source_connected_stationary_state"]
    protocols = manifest["initialization_protocols"]
    baseline_protocol = protocols["hartree_fock_zero_correlation"]
    exact_protocol = protocols["exact_ground_contractions"]
    regularized_protocol = protocols["exact_ground_residual_subtracted"]
    matrix_protocols = manifest["complete_matrix_protocols"]
    matrix_baseline = matrix_protocols["hartree_fock_zero_correlation"]
    matrix_regularized = matrix_protocols["exact_ground_residual_subtracted"]
    extended = manifest["eq14c_projection"]
    extended_parity = extended["matrix_scalar_parity"]
    extended_protocols = extended["protocols"]
    extended_baseline = extended_protocols[
        "hartree_fock_zero_correlation"
    ]
    extended_exact = extended_protocols["exact_ground_contractions"]
    extended_regularized = extended_protocols[
        "exact_ground_residual_subtracted"
    ]
    closed = manifest["closed_scalar_closure"]
    closed_discovery = closed["subspace_discovery"]
    closed_protocols = closed["protocols"]
    closed_baseline = closed_protocols["hartree_fock_zero_correlation"]
    closed_exact = closed_protocols["exact_ground_contractions"]
    closed_regularized = closed_protocols[
        "exact_ground_residual_subtracted"
    ]
    boundary_flux = closed["boundary_flux_at_first_crossing"]
    boundary_groups = boundary_flux["equation_group_fluxes"]
    boundary_terms = boundary_flux["term_fluxes"]
    eq14d_history = closed["eq14d_history_at_first_crossing"]
    eq14d_history_fluxes = eq14d_history["eq14b_flux_by_history"]
    eq14d_ablation = eq14d_history["correlation_subtraction_ablation"]
    cone_audit = closed["cone_correction_audit"]
    cone_uncorrected = cone_audit["none"]
    cone_direct = cone_audit["direct"]
    cone_energy_neutral = cone_audit["energy_neutral"]

    def time_cell(item: dict[str, Any], key: str) -> str:
        value = item[key]
        if value is None:
            return rf"$>{item['requested_final_time']:.0f}$"
        return f"{value:.2f}"

    integrator_rows = [
        f"RK4 & $\\Delta t={item['time_step']:.3f}$ & {item['failure_time']:.5f} \\\\"
        for item in rk4
    ]
    integrator_rows.extend(
        f"{item['method']} & $10^{{-10}}/10^{{-12}}$ & "
        f"{item['failure_time']:.7f} \\\\"
        for item in adaptive
    )
    ablation_rows = [
        f"{item['label']} & {item['eq95_source_scale']:.0f} & "
        f"{item['eq97_source_scale']:.0f} & "
        f"{item['initial_residual_norm']:.4f} & "
        f"{item['failure_time']:.2f} \\\\"
        for item in ablations
    ]
    residual_rows = [
        f"\\texttt{{{name.replace('_', r'\_')}}} & {value:.10f} \\\\"
        for name, value in residual["nonzero_components"].items()
    ]
    critical_rows = [
        f"{label} & {value:+.2f} \\\\"
        for label, value in sorted(
            critical.items(), key=lambda item: abs(item[1]), reverse=True
        )
    ]
    cone_correction_rows = [
        (
            f"{item['label']} & "
            f"{item['minimum_boson_moment_eigenvalue']:.3e} & "
            f"{item['minimum_electron_eigenvalue']:.3e} & "
            f"{item['max_abs_state']:.3f} & "
            f"{item['post_pulse_max_drift_from_t4']:.3e} \\\\"
        )
        for item in (
            cone_uncorrected,
            cone_direct,
            cone_energy_neutral,
        )
    ]
    initialization_rows = [
        (
            "HF/zero correlations"
            f" & {residual['norm']:.4f}"
            f" & {time_cell(baseline_protocol, 'amplitude_failure_time')}"
            f" & {time_cell(baseline_protocol, 'electron_positivity_loss_time')}"
            f" & {baseline_protocol['max_abs_state']:.2g} \\\\"
        ),
        (
            "source-connected fixed point"
            f" & {stationary['residual_norm']:.2e}"
            f" & {time_cell(protocols['source_connected_stationary'], 'amplitude_failure_time')}"
            f" & {time_cell(protocols['source_connected_stationary'], 'electron_positivity_loss_time')}"
            f" & {protocols['source_connected_stationary']['max_abs_state']:.3f} \\\\"
        ),
        (
            "exact contractions"
            f" & {stationary['exact_seed_residual_norm']:.4f}"
            f" & {time_cell(exact_protocol, 'amplitude_failure_time')}"
            f" & {time_cell(exact_protocol, 'electron_positivity_loss_time')}"
            f" & {exact_protocol['max_abs_state']:.2g} \\\\"
        ),
        (
            r"exact + Eq.\ (112)"
            " & 0"
            f" & {time_cell(regularized_protocol, 'amplitude_failure_time')}"
            f" & {time_cell(regularized_protocol, 'electron_positivity_loss_time')}"
            f" & {regularized_protocol['max_abs_state']:.3f} \\\\"
        ),
    ]
    extended_rows = [
        (
            "15D HF/zero"
            f" & {time_cell(extended_baseline, 'amplitude_failure_time')}"
            f" & {time_cell(extended_baseline, 'electron_positivity_loss_time')}"
            f" & {time_cell(extended_baseline, 'phonon_positivity_loss_time')}"
            f" & {time_cell(extended_baseline, 'boson_uncertainty_loss_time')} \\\\"
        ),
        (
            "15D exact contractions"
            f" & {time_cell(extended_exact, 'amplitude_failure_time')}"
            f" & {time_cell(extended_exact, 'electron_positivity_loss_time')}"
            f" & {time_cell(extended_exact, 'phonon_positivity_loss_time')}"
            f" & {time_cell(extended_exact, 'boson_uncertainty_loss_time')} \\\\"
        ),
        (
            r"15D exact + Eq.\ (112)"
            f" & {time_cell(extended_regularized, 'amplitude_failure_time')}"
            f" & {time_cell(extended_regularized, 'electron_positivity_loss_time')}"
            f" & {time_cell(extended_regularized, 'phonon_positivity_loss_time')}"
            f" & {time_cell(extended_regularized, 'boson_uncertainty_loss_time')} \\\\"
        ),
    ]
    closed_rows = [
        (
            "31D HF/zero"
            f" & {time_cell(closed_baseline, 'amplitude_failure_time')}"
            f" & {time_cell(closed_baseline, 'electron_positivity_loss_time')}"
            f" & {time_cell(closed_baseline, 'phonon_positivity_loss_time')}"
            f" & {time_cell(closed_baseline, 'boson_moment_positivity_loss_time')} \\\\"
        ),
        (
            "31D exact contractions"
            f" & {time_cell(closed_exact, 'amplitude_failure_time')}"
            f" & {time_cell(closed_exact, 'electron_positivity_loss_time')}"
            f" & {time_cell(closed_exact, 'phonon_positivity_loss_time')}"
            f" & {time_cell(closed_exact, 'boson_moment_positivity_loss_time')} \\\\"
        ),
        (
            r"31D exact + Eq.\ (112)"
            f" & {time_cell(closed_regularized, 'amplitude_failure_time')}"
            f" & {time_cell(closed_regularized, 'electron_positivity_loss_time')}"
            f" & {time_cell(closed_regularized, 'phonon_positivity_loss_time')}"
            f" & {time_cell(closed_regularized, 'boson_moment_positivity_loss_time')} \\\\"
        ),
    ]
    return "\n".join(
        [
            "% Generated by paper5.stability.report; do not edit by hand.",
            rf"\newcommand{{\ReportGeneratedAt}}{{{manifest['generated_at_utc']}}}",
            rf"\newcommand{{\ReportGitCommit}}{{\texttt{{{manifest['git_commit'][:12]}}}}}",
            rf"\newcommand{{\BaselineFailureTime}}{{{baseline['failure_time']:.5f}}}",
            rf"\newcommand{{\BaselineFailureComponent}}{{\texttt{{{baseline['failure_component'].replace('_', r'\_')}}}}}",
            rf"\newcommand{{\BaselineResidualNorm}}{{{residual['norm']:.10f}}}",
            rf"\newcommand{{\BaselineInitialMaxReal}}{{{_tex_scientific(residual['initial_jacobian_max_real'])}}}",
            rf"\newcommand{{\BaselinePositivityLoss}}{{{baseline_protocol['electron_positivity_loss_time']:.5f}}}",
            rf"\newcommand{{\WeakMaximum}}{{{weak['max_abs_state']:.4f}}}",
            rf"\newcommand{{\ParityMaximumError}}{{{_tex_scientific(parity['maximum_component_error'])}}}",
            rf"\newcommand{{\ParitySamples}}{{{parity['samples']}}}",
            rf"\newcommand{{\EqFourteenCNormal}}{{{_tex_scientific(parity['normal_residual_after_first_order_probe']['anomalous_phonon_rhs_norm'])}}}",
            rf"\newcommand{{\ExtendedParityMaximumError}}{{{_tex_scientific(extended_parity['maximum_component_error'])}}}",
            rf"\newcommand{{\ExtendedPreexistingNormal}}{{{_tex_scientific(extended_parity['maximum_preexisting_normal_residual'])}}}",
            rf"\newcommand{{\ExtendedBaselineFailure}}{{{extended_baseline['amplitude_failure_time']:.5f}}}",
            rf"\newcommand{{\ExtendedRegularizedMaximum}}{{{extended_regularized['max_abs_state']:.6f}}}",
            rf"\newcommand{{\ExtendedRegularizedUncertaintyLoss}}{{{extended_regularized['boson_uncertainty_loss_time']:.5f}}}",
            rf"\newcommand{{\ExtendedRegularizedUncertaintyMinimum}}{{{extended_regularized['minimum_boson_uncertainty_margin']:.6f}}}",
            rf"\newcommand{{\ClosedAmbientDimension}}{{{closed_discovery['ambient_real_dimension']}}}",
            rf"\newcommand{{\ClosedScalarDimension}}{{{closed_discovery['closure_dimension']}}}",
            rf"\newcommand{{\ClosedValidationResidual}}{{{_tex_scientific(closed_discovery['maximum_validation_residual'])}}}",
            rf"\newcommand{{\ClosedRegularizedMaximum}}{{{closed_regularized['max_abs_state']:.6f}}}",
            rf"\newcommand{{\ClosedRegularizedElectronMinimum}}{{{closed_regularized['minimum_electron_eigenvalue']:.6f}}}",
            rf"\newcommand{{\ClosedRegularizedPhononMinimum}}{{{closed_regularized['minimum_phonon_eigenvalue']:.6f}}}",
            rf"\newcommand{{\ClosedRegularizedBosonMinimum}}{{{closed_regularized['minimum_boson_moment_eigenvalue']:.6f}}}",
            rf"\newcommand{{\ClosedRegularizedBosonLoss}}{{{closed_regularized['boson_moment_positivity_loss_time']:.5f}}}",
            rf"\newcommand{{\BoundaryFluxTime}}{{{boundary_flux['crossing_time']:.8f}}}",
            rf"\newcommand{{\BoundaryFluxMinimum}}{{{_tex_scientific(boundary_flux['minimum_eigenvalue'])}}}",
            rf"\newcommand{{\BoundaryFluxGap}}{{{_tex_scientific(boundary_flux['spectral_gap'])}}}",
            rf"\newcommand{{\BoundaryFluxTotal}}{{{_tex_scientific(boundary_flux['total_flux'])}}}",
            rf"\newcommand{{\BoundaryFluxEqFourteenB}}{{{_tex_scientific(boundary_groups['eq14b_correlation_source'])}}}",
            rf"\newcommand{{\BoundaryFluxEqFourteenBMinus}}{{{_tex_scientific(boundary_terms['eq14b_minus_correlation'])}}}",
            rf"\newcommand{{\BoundaryFluxEqFourteenBPlus}}{{{_tex_scientific(boundary_terms['eq14b_plus_conjugate_correlation'])}}}",
            rf"\newcommand{{\BoundaryFluxEqFourteenC}}{{{_tex_scientific(boundary_groups['eq14c_correlation_source'])}}}",
            rf"\newcommand{{\BoundaryFluxEqFourteenD}}{{{_tex_scientific(boundary_groups['eq14d_direct'])}}}",
            rf"\newcommand{{\BoundaryFluxEqOneTwelve}}{{{_tex_scientific(boundary_groups['eq112_residual_subtraction'])}}}",
            rf"\newcommand{{\BoundaryFluxFiniteDifference}}{{{_tex_scientific(boundary_flux['finite_difference_flux'])}}}",
            rf"\newcommand{{\BoundaryFluxFiniteDifferenceError}}{{{_tex_scientific(boundary_flux['finite_difference_error'])}}}",
            rf"\newcommand{{\BoundaryFluxReconstructionError}}{{{_tex_scientific(boundary_flux['reconstruction_error'])}}}",
            rf"\newcommand{{\EqFourteenDHistoryEqOneTwelve}}{{{_tex_scientific(eq14d_history_fluxes['eq112_correlation_subtraction'])}}}",
            rf"\newcommand{{\EqFourteenDHistoryBarePauli}}{{{_tex_scientific(eq14d_history_fluxes['eq14d_bare_pauli_source'])}}}",
            rf"\newcommand{{\EqFourteenDHistoryAnomalousOne}}{{{_tex_scientific(eq14d_history_fluxes['eq14d_anomalous_first_source'])}}}",
            rf"\newcommand{{\EqFourteenDHistoryAnomalousTwo}}{{{_tex_scientific(eq14d_history_fluxes['eq14d_anomalous_second_source'])}}}",
            rf"\newcommand{{\EqFourteenDHistoryNormalParticle}}{{{_tex_scientific(eq14d_history_fluxes['eq14d_normal_particle_source'])}}}",
            rf"\newcommand{{\EqFourteenDHistoryNormalHole}}{{{_tex_scientific(eq14d_history_fluxes['eq14d_normal_hole_source'])}}}",
            rf"\newcommand{{\EqFourteenDHistoryInitial}}{{{_tex_scientific(eq14d_history_fluxes['initial_correlation'])}}}",
            rf"\newcommand{{\EqFourteenDHistoryReconstruction}}{{{_tex_scientific(eq14d_history['correlation_reconstruction_error'])}}}",
            rf"\newcommand{{\EqFourteenDHistoryFluxReconstruction}}{{{_tex_scientific(eq14d_history['eq14b_flux_reconstruction_error'])}}}",
            rf"\newcommand{{\EqFourteenDHistoryCancellationPercent}}{{{100.0 * eq14d_history['cancellation_fraction']:.4f}\%}}",
            rf"\newcommand{{\EqFourteenDHistoryPairNet}}{{{_tex_scientific(eq14d_history['dominant_pair_net_flux'])}}}",
            rf"\newcommand{{\EqFourteenDHistoryResidualNorm}}{{{_tex_scientific(eq14d_history['residual_sector_norms']['electron_phonon_correlation'])}}}",
            rf"\newcommand{{\EqFourteenDAblatedCrossing}}{{{eq14d_ablation['eq112_without_correlation_sector_crossing_time']:.8f}}}",
            rf"\newcommand{{\EqFourteenDAblationDelay}}{{{eq14d_ablation['crossing_delay']:.8f}}}",
            rf"\newcommand{{\ConeDirectActiveFraction}}{{{100.0 * cone_direct['active_sample_fraction']:.2f}\%}}",
            rf"\newcommand{{\ConeDirectMaximumCorrection}}{{{cone_direct['maximum_correction_norm']:.6f}}}",
            rf"\newcommand{{\ConeDirectEnergyDrift}}{{{_tex_scientific(cone_direct['post_pulse_max_drift_from_t4'])}}}",
            rf"\newcommand{{\ConeEnergyActiveFraction}}{{{100.0 * cone_energy_neutral['active_sample_fraction']:.2f}\%}}",
            rf"\newcommand{{\ConeEnergyMaximumCorrection}}{{{cone_energy_neutral['maximum_correction_norm']:.6f}}}",
            rf"\newcommand{{\ConeEnergyDrift}}{{{_tex_scientific(cone_energy_neutral['post_pulse_max_drift_from_t4'])}}}",
            rf"\newcommand{{\ConeEnergyFlux}}{{{_tex_scientific(cone_energy_neutral['maximum_abs_correction_energy_flux'])}}}",
            rf"\newcommand{{\ConeEnergyNonconverged}}{{{cone_energy_neutral['nonconverged_sample_count']}}}",
            rf"\newcommand{{\StationaryResidual}}{{{_tex_scientific(stationary['residual_norm'])}}}",
            rf"\newcommand{{\StationaryEnergy}}{{{stationary['energy']:.8f}}}",
            rf"\newcommand{{\ExactSeedEnergy}}{{{stationary['exact_seed_energy']:.8f}}}",
            rf"\newcommand{{\RegularizedMaximum}}{{{regularized_protocol['max_abs_state']:.6f}}}",
            rf"\newcommand{{\RegularizedElectronMinimum}}{{{regularized_protocol['minimum_electron_eigenvalue']:.6f}}}",
            rf"\newcommand{{\RegularizedPhononMinimum}}{{{regularized_protocol['minimum_phonon_eigenvalue']:.8f}}}",
            rf"\newcommand{{\FullMatrixBaselineFailure}}{{{matrix_baseline['amplitude_failure_time']:.5f}}}",
            rf"\newcommand{{\FullMatrixRegularizedElectronMinimum}}{{{matrix_regularized['minimum_electron_eigenvalue']:.6f}}}",
            rf"\newcommand{{\FullMatrixRegularizedPhononMinimum}}{{{matrix_regularized['minimum_phonon_eigenvalue']:.6f}}}",
            rf"\newcommand{{\IntegratorRows}}{{{' '.join(integrator_rows)}}}",
            rf"\newcommand{{\AblationRows}}{{{' '.join(ablation_rows)}}}",
            rf"\newcommand{{\ResidualRows}}{{{' '.join(residual_rows)}}}",
            rf"\newcommand{{\CriticalRows}}{{{' '.join(critical_rows)}}}",
            rf"\newcommand{{\InitializationRows}}{{{' '.join(initialization_rows)}}}",
            rf"\newcommand{{\ExtendedRows}}{{{' '.join(extended_rows)}}}",
            rf"\newcommand{{\ClosedRows}}{{{' '.join(closed_rows)}}}",
            rf"\newcommand{{\ConeCorrectionRows}}{{{' '.join(cone_correction_rows)}}}",
            rf"\newcommand{{\CodeHashShort}}{{\texttt{{{manifest['source_hashes']['hubbard_dimer_py'][:16]}}}}}",
            rf"\newcommand{{\NoteHashShort}}{{\texttt{{{manifest['source_hashes']['working_note'][:16]}}}}}",
            rf"\newcommand{{\HubbardPdfHashShort}}{{\texttt{{{manifest['source_hashes']['hubbard_dimer_pdf'][:16]}}}}}",
            rf"\newcommand{{\ChiralPdfHashShort}}{{\texttt{{{manifest['source_hashes']['chiral_phonon_pdf'][:16]}}}}}",
        ]
    )


def _build_manifest(build_dir: Path) -> dict[str, Any]:
    failure_threshold = 1e4
    strong_parameters = DimerParameters(
        gamma=0.5,
        lambda_ep=1.5,
        drive_amplitude=1.0,
    )
    weak_parameters = DimerParameters(
        gamma=0.5,
        lambda_ep=0.5,
        drive_amplitude=1.0,
    )
    initial_state = hartree_fock_zero_correlation_state()
    strong_rhs = lambda time, state: fan_migdal_rhs(
        time, state, strong_parameters
    )
    initial_residual = strong_rhs(0.0, initial_state)
    initial_jacobian = finite_difference_jacobian(
        strong_rhs, 0.0, initial_state
    )
    initial_eigenvalues = np.linalg.eigvals(initial_jacobian)
    parity = _matrix_scalar_parity(strong_parameters)
    extended_parity = _extended_matrix_scalar_parity(strong_parameters)

    exact_ground = exact_holstein_ground_state(
        replace(strong_parameters, drive_amplitude=0.0),
        phonon_cutoff=16,
    )
    exact_scalar_state = exact_ground_scalar_coordinates(
        strong_parameters,
        phonon_cutoff=16,
    )
    exact_extended_state = exact_ground_extended_scalar_coordinates(
        strong_parameters,
        phonon_cutoff=16,
    )
    stationary = source_connected_stationary_state(
        strong_parameters,
        phonon_cutoff=16,
    )
    baseline_protocol = _scalar_protocol(
        "HF/zero correlations",
        initial_state,
        strong_rhs,
        final_time=140.0,
        failure_threshold=failure_threshold,
    )
    stationary_protocol = _scalar_protocol(
        "source-connected fixed point",
        stationary.state,
        strong_rhs,
        final_time=400.0,
        failure_threshold=failure_threshold,
    )
    exact_protocol = _scalar_protocol(
        "exact contractions",
        exact_scalar_state,
        strong_rhs,
        final_time=400.0,
        failure_threshold=failure_threshold,
    )
    regularized_protocol = _scalar_protocol(
        r"exact contractions + Eq. (112)",
        exact_scalar_state,
        residual_subtracted_rhs(strong_parameters, exact_scalar_state),
        final_time=400.0,
        failure_threshold=failure_threshold,
    )
    extended_initial_state = np.concatenate(
        [initial_state, np.zeros(2, dtype=float)]
    )
    extended_rhs = lambda time, state: fan_migdal_with_anomalous_rhs(
        time,
        state,
        strong_parameters,
    )
    extended_baseline_protocol = _scalar_protocol(
        "15D HF/zero",
        extended_initial_state,
        extended_rhs,
        final_time=400.0,
        failure_threshold=failure_threshold,
        max_step=0.025,
    )
    extended_exact_protocol = _scalar_protocol(
        "15D exact contractions",
        exact_extended_state,
        extended_rhs,
        final_time=400.0,
        failure_threshold=failure_threshold,
    )
    extended_regularized_protocol = _scalar_protocol(
        r"15D exact + Eq. (112)",
        exact_extended_state,
        extended_residual_subtracted_rhs(
            strong_parameters,
            exact_extended_state,
        ),
        final_time=400.0,
        failure_threshold=failure_threshold,
    )
    closed_initial_state = hartree_fock_closed_scalar_coordinates(
        strong_parameters
    )
    closed_exact_state = exact_ground_closed_scalar_coordinates(
        strong_parameters,
        phonon_cutoff=16,
    )
    closed_rhs = lambda time, state: closed_scalar_rhs(
        time,
        state,
        strong_parameters,
    )
    closed_baseline_protocol = _closed_scalar_protocol(
        "31D HF/zero",
        closed_initial_state,
        closed_rhs,
        final_time=400.0,
        failure_threshold=failure_threshold,
    )
    closed_exact_protocol = _closed_scalar_protocol(
        "31D exact contractions",
        closed_exact_state,
        closed_rhs,
        final_time=400.0,
        failure_threshold=failure_threshold,
    )
    closed_regularized_protocol = _closed_scalar_protocol(
        r"31D exact + Eq. (112)",
        closed_exact_state,
        closed_residual_subtracted_rhs(
            strong_parameters,
            closed_exact_state,
        ),
        final_time=400.0,
        failure_threshold=failure_threshold,
    )
    closed_regularized_crossing_time = closed_regularized_protocol[
        "boson_moment_positivity_loss_time"
    ]
    closed_regularized_crossing_state = closed_regularized_protocol[
        "boson_moment_positivity_loss_state"
    ]
    if (
        closed_regularized_crossing_time is None
        or closed_regularized_crossing_state is None
    ):
        raise RuntimeError(
            "regularized 31D protocol did not reach the bosonic boundary"
        )
    closed_initial_residual = closed_scalar_rhs(
        0.0,
        closed_exact_state,
        replace(strong_parameters, drive_amplitude=0.0),
    )
    closed_boundary_flux = _boundary_flux_manifest(
        float(closed_regularized_crossing_time),
        boson_boundary_flux_decomposition(
            float(closed_regularized_crossing_time),
            closed_scalar_to_matrix_state(
                np.asarray(closed_regularized_crossing_state, dtype=float)
            ),
            strong_parameters,
            residual_subtraction=closed_initial_residual,
        ),
    )
    closed_eq14d_history_raw = closed_eq14d_history_flux_decomposition(
        strong_parameters,
        closed_exact_state,
        residual_subtraction=closed_initial_residual,
        maximum_time=2.0,
    )
    residual_without_correlation = closed_initial_residual.copy()
    residual_without_correlation[17:] = 0.0
    closed_eq14d_correlation_ablated_raw = (
        closed_eq14d_history_flux_decomposition(
            strong_parameters,
            closed_exact_state,
            residual_subtraction=residual_without_correlation,
            maximum_time=4.0,
        )
    )
    closed_eq14d_history = _eq14d_history_manifest(
        closed_eq14d_history_raw,
        closed_eq14d_correlation_ablated_raw,
        closed_initial_residual,
    )
    cone_correction_protocols = [
        _cone_correction_audit(
            strong_parameters,
            closed_exact_state,
            label=r"Eq. (112), no cone correction",
            correction_mode="none",
        ),
        _cone_correction_audit(
            strong_parameters,
            closed_exact_state,
            label="direct matrix barrier",
            correction_mode="direct",
        ),
        _cone_correction_audit(
            strong_parameters,
            closed_exact_state,
            label="energy-neutral matrix barrier",
            correction_mode="energy_neutral",
        ),
    ]
    closure_discovery = discover_invariant_closure(
        strong_parameters,
        samples_per_iteration=200,
        validation_samples=300,
    )

    baseline_matrix = scalar_to_matrix_state(initial_state)
    equilibrium_center = np.full(
        2,
        -strong_parameters.coupling / strong_parameters.omega_ph,
        dtype=complex,
    )
    baseline_matrix = MatrixDimerState(
        electron_density=baseline_matrix.electron_density,
        coherent_phonon=(
            baseline_matrix.coherent_phonon + equilibrium_center
        ),
        phonon_density=baseline_matrix.phonon_density,
        anomalous_phonon_density=(
            baseline_matrix.anomalous_phonon_density
        ),
        electron_phonon_correlation=(
            baseline_matrix.electron_phonon_correlation
        ),
    )
    full_matrix_baseline = _matrix_protocol(
        "complete matrix EOM, HF/zero correlations",
        baseline_matrix,
        strong_parameters,
        final_time=140.0,
        failure_threshold=failure_threshold,
        subtract_initial_residual=False,
    )
    full_matrix_regularized = _matrix_protocol(
        "complete matrix EOM, exact contractions + residual subtraction",
        exact_ground.matrix_state,
        strong_parameters,
        final_time=140.0,
        failure_threshold=failure_threshold,
        subtract_initial_residual=True,
    )

    strong_trace = _rk4_trace(
        strong_parameters,
        time_step=0.01,
        final_time=140.0,
        failure_threshold=failure_threshold,
    )
    weak_trace = _rk4_trace(
        weak_parameters,
        time_step=0.01,
        final_time=140.0,
        failure_threshold=failure_threshold,
    )
    jacobian_times, jacobian_max_real = _jacobian_series(
        strong_trace, strong_parameters
    )

    rk4_refinement = []
    for time_step in (0.02, 0.01, 0.005):
        result = integrate_rk4(
            strong_rhs,
            initial_state,
            final_time=140.0,
            time_step=time_step,
            failure_threshold=failure_threshold,
            state_names=FAN_MIGDAL_STATE_NAMES,
        )
        rk4_refinement.append(
            {
                "time_step": time_step,
                "failure_time": result.failure_time,
                "failure_component": result.failure_component,
                "max_abs_state": result.max_abs_state,
            }
        )

    adaptive = [
        _adaptive_failure(
            method,
            strong_parameters,
            final_time=140.0,
            failure_threshold=failure_threshold,
        )
        for method in ("DOP853", "Radau", "BDF")
    ]

    ablation_specs = [
        ("full source", 1.0, 1.0),
        ("Eq. (95) source off", 0.0, 1.0),
        ("Eq. (97) source off", 1.0, 0.0),
        ("both sources off", 0.0, 0.0),
    ]
    ablations = []
    for label, eq95_scale, eq97_scale in ablation_specs:
        parameters = DimerParameters(
            gamma=0.5,
            lambda_ep=1.5,
            drive_amplitude=1.0,
            eq95_source_scale=eq95_scale,
            eq97_source_scale=eq97_scale,
        )
        rhs = lambda time, state, p=parameters: fan_migdal_rhs(time, state, p)
        residual = rhs(0.0, initial_state)
        result = integrate_rk4(
            rhs,
            initial_state,
            final_time=140.0,
            time_step=0.01,
            failure_threshold=failure_threshold,
            state_names=FAN_MIGDAL_STATE_NAMES,
        )
        ablations.append(
            {
                "label": label,
                "eq95_source_scale": eq95_scale,
                "eq97_source_scale": eq97_scale,
                "initial_residual_norm": float(np.linalg.norm(residual)),
                "failure_time": result.failure_time,
                "failure_component": result.failure_component,
            }
        )

    state_t130 = _state_at(strong_trace, 130.0)
    critical_terms = _critical_terms(130.0, state_t130, strong_parameters)

    _plot_trajectory(
        strong_trace,
        weak_trace,
        failure_threshold=failure_threshold,
        output=build_dir / "trajectory_max_abs.pdf",
    )
    _plot_jacobian(
        jacobian_times,
        jacobian_max_real,
        failure_time=float(strong_trace["failure_time"]),
        output=build_dir / "jacobian_growth.pdf",
    )
    _plot_ablations(
        ablations,
        output=build_dir / "ablation_failure_times.pdf",
    )
    _plot_initialization_protocols(
        [
            baseline_protocol,
            stationary_protocol,
            regularized_protocol,
        ],
        failure_threshold=failure_threshold,
        output=build_dir / "initialization_protocols.pdf",
    )
    _plot_regularized_physicality(
        regularized_protocol,
        output=build_dir / "regularized_physicality.pdf",
    )
    _plot_eq14c_protocols(
        baseline_protocol,
        [
            extended_baseline_protocol,
            extended_exact_protocol,
            extended_regularized_protocol,
        ],
        failure_threshold=failure_threshold,
        output=build_dir / "eq14c_protocols.pdf",
    )
    _plot_eq14c_uncertainty(
        [
            extended_baseline_protocol,
            extended_exact_protocol,
            extended_regularized_protocol,
        ],
        output=build_dir / "eq14c_uncertainty.pdf",
    )
    _plot_closed_scalar_protocols(
        [
            closed_baseline_protocol,
            closed_exact_protocol,
            closed_regularized_protocol,
        ],
        failure_threshold=failure_threshold,
        output=build_dir / "closed_scalar_protocols.pdf",
    )
    _plot_closed_scalar_physicality(
        closed_regularized_protocol,
        output=build_dir / "closed_scalar_physicality.pdf",
    )
    _plot_boson_boundary_flux(
        closed_boundary_flux,
        output=build_dir / "boson_boundary_flux.pdf",
    )
    _plot_eq14d_history_flux(
        closed_eq14d_history,
        output=build_dir / "eq14d_history_flux.pdf",
    )
    _plot_cone_correction_audit(
        cone_correction_protocols,
        output=build_dir / "cone_correction_audit.pdf",
    )

    source_hashes = {
        "hubbard_dimer_py": _sha256(Path(__file__).with_name("hubbard_dimer.py")),
        "matrix_reference_py": _sha256(
            Path(__file__).with_name("matrix_reference.py")
        ),
        "exact_reference_py": _sha256(
            Path(__file__).with_name("exact_reference.py")
        ),
        "initial_conditions_py": _sha256(
            Path(__file__).with_name("initial_conditions.py")
        ),
        "cone_correction_py": _sha256(
            Path(__file__).with_name("cone_correction.py")
        ),
        "report_py": _sha256(Path(__file__)),
        "working_note": _sha256(WORKING_NOTE),
        "tex_source": _sha256(TEX_SOURCE),
        "hubbard_dimer_pdf": _sha256(HUBBARD_DIMER_PDF),
        "chiral_phonon_pdf": _sha256(CHIRAL_PHONON_PDF),
    }
    return {
        "schema": "paper_v_scalar_stability_diagnostic/v7",
        "status": "working_diagnostic_not_promoted_paper_evidence",
        "generated_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "git_commit": _git_commit(),
        "source_hashes": source_hashes,
        "source_equations": {
            "ehrenfest": "Dynamics_on_the_Hubbard_DIMER.pdf Eqs. (78)-(82)",
            "fan_migdal_scalar": "Dynamics_on_the_Hubbard_DIMER.pdf Eqs. (87)-(99)",
            "primary_matrix": "arXiv:2606.22233 Eqs. (14a)-(14e)",
            "matrix_to_scalar_mapping_status": (
                "verified_for_13d_15d_and_invariant_31d_coordinates"
            ),
            "complete_equivalence_status": (
                "31d_is_invariant_and_matrix_exact_within_physical_"
                "hermiticity_symmetry_and_equal_correlation_trace_constraints"
            ),
        },
        "baseline": {
            "parameters": asdict(strong_parameters),
            "initial_state": "hartree_fock_electronic_plus_zero_correlations",
            "failure_threshold": failure_threshold,
            "failure_time": strong_trace["failure_time"],
            "failure_component": strong_trace["failure_component"],
            "electron_positivity_loss_time": baseline_protocol[
                "electron_positivity_loss_time"
            ],
        },
        "weak_control": {
            "parameters": asdict(weak_parameters),
            "failure_time": weak_trace["failure_time"],
            "max_abs_state": float(np.max(weak_trace["maxima"])),
        },
        "initial_residual": {
            "norm": float(np.linalg.norm(initial_residual)),
            "nonzero_components": {
                name: float(value)
                for name, value in zip(
                    FAN_MIGDAL_STATE_NAMES, initial_residual, strict=True
                )
                if abs(value) > 1e-14
            },
            "initial_jacobian_max_real": float(
                np.max(np.real(initial_eigenvalues))
            ),
        },
        "rk4_refinement": rk4_refinement,
        "adaptive_integrators": adaptive,
        "jacobian_along_trajectory": {
            "sample_times": jacobian_times.tolist(),
            "max_real_parts": jacobian_max_real.tolist(),
            "interpretation": "instantaneous local diagnostic, not Lyapunov exponents",
        },
        "ablations": ablations,
        "critical_terms_at_t130": critical_terms,
        "matrix_scalar_parity": parity,
        "eq14c_projection": {
            "definition": (
                "m_minus=A_00-A_01; two real coordinates added to the "
                "thirteen-scalar projection"
            ),
            "boson_uncertainty_condition": (
                "n_minus*(n_minus+1)-abs(m_minus)**2 >= 0"
            ),
            "matrix_scalar_parity": extended_parity,
            "protocols": {
                "hartree_fock_zero_correlation": _protocol_manifest(
                    extended_baseline_protocol
                ),
                "exact_ground_contractions": _protocol_manifest(
                    extended_exact_protocol
                ),
                "exact_ground_residual_subtracted": _protocol_manifest(
                    extended_regularized_protocol
                ),
            },
        },
        "closed_scalar_closure": {
            "state_dimension": len(CLOSED_SCALAR_STATE_NAMES),
            "state_names": list(CLOSED_SCALAR_STATE_NAMES),
            "coordinate_groups": {
                "electronic_hermitian_trace_one": 3,
                "coherent_phonon_complex_two_mode": 4,
                "normal_phonon_hermitian": 4,
                "anomalous_phonon_complex_symmetric": 6,
                "electron_phonon_equal_trace_pair": 14,
            },
            "invariant_constraints": [
                "electron density Hermitian with trace one",
                "normal phonon density Hermitian",
                "anomalous phonon density complex symmetric",
                "trace(C_0) equals trace(C_1)",
            ],
            "subspace_discovery": closure_discovery,
            "boson_physicality_condition": (
                "minimum eigenvalue of [[N.T,A.conj()],[A,I+N]] "
                "must be nonnegative"
            ),
            "protocols": {
                "hartree_fock_zero_correlation": _protocol_manifest(
                    closed_baseline_protocol
                ),
                "exact_ground_contractions": _protocol_manifest(
                    closed_exact_protocol
                ),
                "exact_ground_residual_subtracted": _protocol_manifest(
                    closed_regularized_protocol
                ),
            },
            "boundary_flux_at_first_crossing": closed_boundary_flux,
            "eq14d_history_at_first_crossing": closed_eq14d_history,
            "cone_correction_audit": {
                item["correction_mode"]: _protocol_manifest(item)
                for item in cone_correction_protocols
            },
        },
        "source_connected_stationary_state": {
            "state": {
                name: float(value)
                for name, value in zip(
                    FAN_MIGDAL_STATE_NAMES,
                    stationary.state,
                    strict=True,
                )
            },
            "residual_norm": stationary.residual_norm,
            "energy": stationary.energy,
            "exact_seed_energy": stationary.exact_seed_energy,
            "exact_seed_residual_norm": stationary.exact_seed_residual_norm,
            "electron_eigenvalues": stationary.electron_eigenvalues.tolist(),
            "phonon_eigenvalues": stationary.phonon_eigenvalues.tolist(),
            "phonon_cutoff": stationary.phonon_cutoff,
            "selection_status": (
                "exact_seed_connected_basic_psd_not_global_N_representability_proof"
            ),
        },
        "initialization_protocols": {
            "hartree_fock_zero_correlation": _protocol_manifest(
                baseline_protocol
            ),
            "source_connected_stationary": _protocol_manifest(
                stationary_protocol
            ),
            "exact_ground_contractions": _protocol_manifest(exact_protocol),
            "exact_ground_residual_subtracted": _protocol_manifest(
                regularized_protocol
            ),
        },
        "complete_matrix_protocols": {
            "hartree_fock_zero_correlation": full_matrix_baseline,
            "exact_ground_residual_subtracted": full_matrix_regularized,
        },
        "diagnosis": {
            "ruled_out_for_scalar_transcription": [
                "coarse fixed-step RK4 as the sole failure cause",
                "either Eq. (95) or Eq. (97) initial source acting alone as the sole cause",
                "sign or normalization mismatch in the thirteen retained matrix-projected derivatives",
            ],
            "supported": [
                "strong-pulse entry into a rapidly amplifying nonlinear-feedback region",
                "stabilizing cancellation among coupled source terms",
                "material dependence on fixed-point branch and correlated initialization",
                "electronic positivity loss precedes the baseline amplitude divergence",
                "Eq. (112) residual subtraction keeps the strong scalar protocol bounded and basic-PSD through t=400",
                "the thirteen-scalar model omits a dynamically generated Eq. (14c) field",
                "the fifteen-scalar Eq. (14c) projection delays the HF amplitude threshold from about 130.47 to about 366.17",
                "the fifteen-scalar Eq. (14c) projection violates the relative-mode bosonic uncertainty condition before late amplitude growth",
                "the smallest sampled linear invariant closure containing the fifteen-scalar projection has 31 real coordinates",
                "the 31-scalar closure reproduces the complete matrix trajectory and its early bosonic physicality loss",
                "Eq. (112) keeps the 31-scalar closure bounded and electronically positive through t=400 but not boson-positive",
                "at the first regularized bosonic boundary crossing the Eq. (14b) correlation source supplies the net outward first-derivative flux",
                "Eq. (14d) has no direct contribution to the first derivative of the boson moment matrix and acts indirectly through the correlation field",
                "the boundary-driving Eq. (14b) correlation field is a 98.9956-percent causal cancellation dominated by the Eq. (112) correlation subtraction and the opposing bare Pauli-blocking source in Eq. (14d)",
                "removing only the correlation-sector component of Eq. (112) delays the first bosonic moment crossing from about 1.487 to about 3.683",
                "a ten-real-coordinate full-matrix barrier correction preserves the bosonic cone and boundedness through t=20 for the pinned protocol",
                "restricting the correction to dN_00+dN_11=0 preserves the direct correction's instantaneous Eq. (22) energy contribution exactly",
                "the energy-neutral barrier preserves the sampled electronic and bosonic cones through t=20 and limits post-pulse energy drift to the uncorrected Eq. (112) scale",
            ],
            "unresolved": [
                "minimum-energy physically admissible full-EOM fixed point",
                "higher-moment N-representability of candidate scalar roots",
                "long-time and parameter-robust validation of the short-time positivity-preserving correction",
                "whether a correlation-sector correction can preserve both the physical cone and the late-time amplitude stabilization supplied by Eq. (112)",
                "exact-observable error introduced by the direct moment correction",
                "high-lambda evidence versus a distinct high-U mechanism",
                "chaos or strange-attractor classification",
            ],
        },
        "commands": {
            "build": "cd paper_5 && PYTHONPATH=src python3 -m paper5.stability.report",
            "red_capable_reproduction": (
                "cd paper_5 && PYTHONPATH=src python3 -m "
                "paper5.stability.reproduce --lambda-ep 1.5 --gamma 0.5 "
                "--drive 1.0 --time-step 0.01 --final-time 140 "
                "--expect-bounded"
            ),
            "tests": "cd paper_5 && python3 -m pytest -q",
        },
    }


def build_report(
    *,
    build_dir: Path,
    output_pdf: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    build_dir.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    manifest = _build_manifest(build_dir)

    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (build_dir / f"{REPORT_STEM}_manifest.json").write_text(
        manifest_text, encoding="utf-8"
    )
    manifest_path.write_text(manifest_text, encoding="utf-8")
    (build_dir / "generated_results.tex").write_text(
        _tex_fragment(manifest) + "\n", encoding="utf-8"
    )
    local_tex = build_dir / TEX_SOURCE.name
    shutil.copy2(TEX_SOURCE, local_tex)
    isolated_texmf = build_dir / "empty-texmf"
    isolated_texmf.mkdir(exist_ok=True)

    subprocess.run(
        [
            "latexmk",
            "-g",
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            local_tex.name,
        ],
        cwd=build_dir,
        check=True,
        env={**os.environ, "TEXMFHOME": str(isolated_texmf)},
    )
    built_pdf = build_dir / f"{REPORT_STEM}.pdf"
    shutil.copy2(built_pdf, output_pdf)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=REPO_ROOT / "tmp" / "pdfs" / REPORT_STEM,
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=REPO_ROOT / "output" / "pdf" / f"{REPORT_STEM}.pdf",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            REPO_ROOT
            / "output"
            / "pdf"
            / f"{REPORT_STEM}_manifest.json"
        ),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    manifest = build_report(
        build_dir=args.build_dir.resolve(),
        output_pdf=args.output_pdf.resolve(),
        manifest_path=args.manifest.resolve(),
    )
    print(
        json.dumps(
            {
                "pdf": str(args.output_pdf.resolve()),
                "manifest": str(args.manifest.resolve()),
                "status": manifest["status"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
