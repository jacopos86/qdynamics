"""Long-horizon propagation and scoring for the structured cone barrier.

This module carries the pieces plan step 9 needs that are not already in
``cone_correction`` (which it does not touch): a DOP853 stepping loop that can
be stopped on a wall-clock budget while keeping the samples already produced,
the six reported observables, the two RMS conventions the record uses, and the
barrier-activity census of ``report._cone_correction_audit``.

Provenance of the copied pieces:

* the observable definitions are copied from
  ``pipelines/open_dynamics/plot_high_coupling_controller_observables.py``
  (``OBSERVABLE_NAMES``, ``observable_trajectories``); the copy exists so the
  ``paper5`` package does not import a plotting pipeline, and the driver
  checks the two against each other at run time;
* ``time_rms`` is copied from
  ``pipelines/open_dynamics/score_coupling_controller_exact_reference.py``;
* the active-sample census and the post-pulse drift are copied from
  ``paper5.stability.report._cone_correction_audit``;
* the pinned protocol and its settings come from
  ``output/local_runs/paper_v_cone_ref_t2_20260914_v1/run_pinned_energy_neutral_barrier.py``
  (overnight round 1, 2026-09-14).

Nothing here changes a default, an acceptance rule, or an existing routine.
``integrate_sampled_dop853`` with the default unlimited wall budget reproduces
``scipy.integrate.solve_ivp(..., method="DOP853", t_eval=...)`` exactly,
because it repeats that function's own stepping and dense-output loop.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import DOP853

from paper5.stability.cone_correction import structured_boson_barrier_correction
from paper5.stability.matrix_reference import (
    closed_scalar_to_matrix_state,
    local_holstein_couplings,
)

FloatArray = NDArray[np.float64]
RhsFunction = Callable[[float, FloatArray], FloatArray]

#: Copied from ``plot_high_coupling_controller_observables.OBSERVABLE_NAMES``.
OBSERVABLE_NAMES = (
    "site_0_occupation",
    "site_1_occupation",
    "electronic_energy",
    "phonon_energy",
    "electron_phonon_energy",
    "internal_energy",
)

#: Sample times at or after this time are the post-pulse window of
#: ``report._cone_correction_audit``.
POST_PULSE_START = 4.0

#: A sampled correction of norm above this is an active barrier sample, as in
#: ``report._cone_correction_audit``.
BARRIER_ACTIVITY_NORM = 1e-10


@dataclass(frozen=True)
class SampledTrajectory:
    """Sampled states of one propagation and how far it actually reached."""

    times: FloatArray
    states: FloatArray
    wall_seconds: float
    function_evaluations: int
    step_count: int
    reached_final_time: bool
    stop_reason: str
    final_time_reached: float


def integrate_sampled_dop853(
    rhs: RhsFunction,
    initial_state: FloatArray,
    *,
    final_time: float,
    sample_times: FloatArray,
    relative_tolerance: float = 1e-9,
    absolute_tolerance: float = 1e-11,
    maximum_step: float = 0.05,
    wall_budget_seconds: float | None = None,
) -> SampledTrajectory:
    """Propagate with DOP853 and keep every sample reached inside the budget.

    The loop is ``scipy.integrate.solve_ivp``'s own: step the solver, then
    interpolate every requested sample time that the completed step covers
    from that step's dense output.  With ``wall_budget_seconds=None`` the
    result is the same trajectory ``solve_ivp`` returns; with a budget the
    integration stops after the first completed step whose end time is past
    the budget, and the samples already interpolated are returned.
    """

    times = np.asarray(sample_times, dtype=float)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("sample_times must contain at least two times")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("sample_times must be strictly increasing")
    if times[0] < 0.0 or times[-1] > final_time + 1e-12:
        raise ValueError("sample_times must lie inside the integration span")
    if wall_budget_seconds is not None and wall_budget_seconds <= 0.0:
        raise ValueError("wall_budget_seconds must be positive when given")

    state = np.asarray(initial_state, dtype=float).copy()
    solver = DOP853(
        rhs,
        0.0,
        state,
        final_time,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
        max_step=maximum_step,
    )
    collected_times: list[FloatArray] = []
    collected_states: list[FloatArray] = []
    sample_index = 0
    step_count = 0
    stop_reason = "final_time"
    started = _time.perf_counter()
    while True:
        message = solver.step()
        if solver.status == "failed":
            stop_reason = f"solver_failed:{message}"
            break
        step_count += 1
        next_index = int(np.searchsorted(times, solver.t, side="right"))
        if next_index > sample_index:
            interpolant = solver.dense_output()
            step_times = times[sample_index:next_index]
            collected_times.append(step_times)
            collected_states.append(np.asarray(interpolant(step_times)).T)
            sample_index = next_index
        if solver.status == "finished":
            break
        if (
            wall_budget_seconds is not None
            and _time.perf_counter() - started > wall_budget_seconds
        ):
            stop_reason = "wall_budget"
            break
    wall_seconds = _time.perf_counter() - started

    if not collected_times:
        raise RuntimeError("no sample time was reached before the budget ran out")
    sampled_times = np.concatenate(collected_times)
    sampled_states = np.concatenate(collected_states, axis=0)
    return SampledTrajectory(
        times=sampled_times,
        states=sampled_states,
        wall_seconds=wall_seconds,
        function_evaluations=int(solver.nfev),
        step_count=step_count,
        reached_final_time=stop_reason == "final_time",
        stop_reason=stop_reason,
        final_time_reached=float(sampled_times[-1]),
    )


def closed_observable_trajectories(
    coordinates: FloatArray,
    parameters: Any,
) -> FloatArray:
    """Return the six reported observables at every sampled coordinate row.

    Copied from ``plot_high_coupling_controller_observables``; the columns are
    ``OBSERVABLE_NAMES`` in order.
    """

    rows = np.asarray(coordinates, dtype=float)
    if rows.ndim != 2 or rows.shape[1] != 31:
        raise ValueError("coordinates must have shape (samples, 31)")
    couplings = local_holstein_couplings(parameters)
    bare_electron = np.array(
        [[0.0, -parameters.hopping], [-parameters.hopping, 0.0]],
        dtype=complex,
    )
    values = np.empty((rows.shape[0], len(OBSERVABLE_NAMES)), dtype=float)
    for index, row in enumerate(rows):
        state = closed_scalar_to_matrix_state(row)
        rho = np.asarray(state.electron_density, dtype=complex)
        coherent = np.asarray(state.coherent_phonon, dtype=complex)
        normal = np.asarray(state.phonon_density, dtype=complex)
        correlation = np.asarray(state.electron_phonon_correlation, dtype=complex)

        electronic = 2.0 * np.trace(bare_electron @ rho).real
        phonon = parameters.omega_ph * (
            np.vdot(coherent, coherent).real + np.trace(normal).real
        )
        interaction_amplitude = 0.0j
        for mode in range(2):
            for one in range(2):
                for two in range(2):
                    interaction_amplitude += couplings[mode, one, two] * (
                        coherent[mode] * rho[two, one]
                        + correlation[mode, two, one]
                    )
        electron_phonon = 4.0 * interaction_amplitude.real
        values[index] = (
            2.0 * rho[0, 0].real,
            2.0 * rho[1, 1].real,
            electronic,
            phonon,
            electron_phonon,
            electronic + phonon + electron_phonon,
        )
    return values


def time_rms(times: FloatArray, values: FloatArray) -> float:
    """Trapezoidal time-RMS over the sampled interval.

    Copied from ``score_coupling_controller_exact_reference._time_rms``.
    """

    time_array = np.asarray(times, dtype=float)
    value_array = np.asarray(values, dtype=float)
    duration = float(time_array[-1] - time_array[0])
    if duration <= 0.0:
        raise ValueError("the scored interval must have positive duration")
    return float(np.sqrt(np.trapezoid(value_array**2, time_array) / duration))


def sample_rms(values: FloatArray) -> float:
    """Unweighted root mean square over the sampled values.

    This is the convention behind the adaptive-packet route's reported
    observable RMS values (``analyze_adaptive_packet_event_ablation``).
    """

    return float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2)))


def window_indices(
    times: FloatArray,
    window: tuple[float, float],
    *,
    tolerance: float = 1e-9,
) -> NDArray[np.bool_]:
    """Return the mask of samples inside a closed scoring window."""

    time_array = np.asarray(times, dtype=float)
    start, stop = float(window[0]), float(window[1])
    if not start < stop:
        raise ValueError("a scoring window must have positive duration")
    mask = (time_array >= start - tolerance) & (time_array <= stop + tolerance)
    if int(np.count_nonzero(mask)) < 2:
        raise ValueError("a scoring window must contain at least two samples")
    return mask


def window_scores(
    times: FloatArray,
    lane_states: FloatArray,
    exact_states: FloatArray,
    parameters: Any,
    *,
    window: tuple[float, float],
    coordinate_scales: FloatArray | None = None,
) -> dict[str, Any]:
    """Score one propagated lane against the exact reference on one window.

    Both RMS conventions of the record are returned side by side:
    ``coordinate_l2_time_rms`` is the trapezoidal time-RMS of the unscaled
    31-coordinate Euclidean distance, as in
    ``score_coupling_controller_exact_reference``; ``all31_scaled_rms`` is the
    unweighted root mean square of the coordinate difference divided by the
    supplied per-coordinate scales, as in
    ``analyze_adaptive_packet_event_ablation``.  Observable errors carry both
    the trapezoidal time-RMS and the unweighted sample RMS.
    """

    mask = window_indices(times, window)
    selected_times = np.asarray(times, dtype=float)[mask]
    lane = np.asarray(lane_states, dtype=float)[mask]
    exact = np.asarray(exact_states, dtype=float)[mask]
    if lane.shape != exact.shape:
        raise ValueError("lane and exact trajectories must have the same shape")

    difference = lane - exact
    coordinate_l2 = np.linalg.norm(difference, axis=1)
    lane_observables = closed_observable_trajectories(lane, parameters)
    exact_observables = closed_observable_trajectories(exact, parameters)
    observable_difference = lane_observables - exact_observables

    scores: dict[str, Any] = {
        "window": [float(selected_times[0]), float(selected_times[-1])],
        "sample_count": int(selected_times.size),
        "coordinate_l2_time_rms": time_rms(selected_times, coordinate_l2),
        "coordinate_l2_maximum": float(np.max(coordinate_l2)),
        "observables": {
            name: {
                "time_rms": time_rms(selected_times, observable_difference[:, index]),
                "sample_rms": sample_rms(observable_difference[:, index]),
                "maximum_absolute": float(
                    np.max(np.abs(observable_difference[:, index]))
                ),
            }
            for index, name in enumerate(OBSERVABLE_NAMES)
        },
    }
    if coordinate_scales is not None:
        scales = np.asarray(coordinate_scales, dtype=float)
        if scales.shape != (31,) or np.any(scales <= 0.0):
            raise ValueError("coordinate_scales must be 31 positive numbers")
        scaled = difference / scales
        scores["all31_scaled_rms"] = sample_rms(scaled)
        scores["c_block_scaled_rms"] = sample_rms(scaled[:, 17:31])
    return scores


def barrier_activity(
    times: FloatArray,
    states: FloatArray,
    base_rhs: RhsFunction,
    *,
    activation_margin: float = 1e-5,
    target_flux: float = 0.0,
    barrier_rate: float = 5.0,
    energy_neutral: bool = True,
    window: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Census of the structured barrier at the sampled states.

    This repeats ``report._cone_correction_audit``'s census: the correction is
    re-solved at each stored sample from the uncorrected velocity there, and a
    sample counts as active when the correction norm exceeds
    ``BARRIER_ACTIVITY_NORM``.
    """

    time_array = np.asarray(times, dtype=float)
    state_array = np.asarray(states, dtype=float)
    mask = (
        np.ones(time_array.size, dtype=bool)
        if window is None
        else window_indices(time_array, window)
    )
    norms = np.zeros(int(np.count_nonzero(mask)), dtype=float)
    constraint_counts = np.zeros(norms.size, dtype=int)
    nonconverged = 0
    for position, index in enumerate(np.flatnonzero(mask)):
        state = state_array[index]
        derivative = base_rhs(float(time_array[index]), state)
        result = structured_boson_barrier_correction(
            state,
            derivative,
            activation_margin=activation_margin,
            target_flux=target_flux,
            barrier_rate=barrier_rate,
            energy_neutral=energy_neutral,
        )
        norms[position] = float(result.correction_norm)
        constraint_counts[position] = int(result.constraint_count)
        if not result.converged:
            nonconverged += 1
    active = norms > BARRIER_ACTIVITY_NORM
    selected_times = time_array[mask]
    return {
        "sample_count": int(norms.size),
        "active_sample_fraction": float(np.mean(active)),
        "first_active_sample_time": (
            float(selected_times[np.flatnonzero(active)[0]])
            if np.any(active)
            else None
        ),
        "maximum_correction_norm": float(np.max(norms)),
        "maximum_constraint_count": int(np.max(constraint_counts)),
        "nonconverged_sample_count": int(nonconverged),
    }


def post_pulse_energy_drift(
    times: FloatArray,
    energies: FloatArray,
    *,
    window: tuple[float, float] | None = None,
    post_pulse_start: float = POST_PULSE_START,
) -> dict[str, Any]:
    """Maximum departure of the internal energy from its value at E(4).

    Copied from ``report._cone_correction_audit``'s
    ``post_pulse_max_drift_from_t4``: the reference value is the first stored
    sample at or after ``post_pulse_start``.
    """

    time_array = np.asarray(times, dtype=float)
    energy_array = np.asarray(energies, dtype=float)
    mask = (
        np.ones(time_array.size, dtype=bool)
        if window is None
        else window_indices(time_array, window)
    )
    mask = mask & (time_array >= post_pulse_start)
    if not np.any(mask):
        raise ValueError("the window contains no post-pulse sample")
    selected = energy_array[mask]
    return {
        "post_pulse_start": float(time_array[mask][0]),
        "post_pulse_sample_count": int(selected.size),
        "post_pulse_max_drift_from_t4": float(np.max(np.abs(selected - selected[0]))),
        "post_pulse_energy_range": float(np.max(selected) - np.min(selected)),
    }
