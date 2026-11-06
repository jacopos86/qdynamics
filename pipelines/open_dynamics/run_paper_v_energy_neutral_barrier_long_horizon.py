"""Plan step 9: the energy-neutral matrix barrier through the late horizon.

The working notes declared this calculation next on 2026-08-04 and never ran
it (``paper_5/notes/electron_phonon_closure_worklog_20260803.md``; plan
``agent_guidance/paper-v/paper-v-state-and-plan-20260914.md:276-282``):
propagate the ten-coordinate energy-neutral structured boson barrier on the
31-coordinate archive closure through the late amplitude-validation horizon
and compare selected observables with exact driven-dimer propagation.

Protocol, pinned to the working notes' strong-coupling table and copied from
``output/local_runs/paper_v_cone_ref_t2_20260914_v1/run_pinned_energy_neutral_barrier.py``
(overnight round 1, 2026-09-14), which in turn copied it from
``paper5.stability.report._cone_correction_audit`` with
``correction_mode="energy_neutral"``: ``lambda_ep = 1.5``, ``gamma = 0.5``,
drive amplitude 1, exact ground-state contractions at phonon cutoff 16,
``activation_margin = 1e-5``, ``target_flux = 0``, ``barrier_rate = 5``,
DOP853 with ``rtol = 1e-9``, ``atol = 1e-11``, maximum step 0.05.

Three lanes are produced on one sample grid:

``corrected``
    ``closed_cone_projected_rhs(..., energy_neutral=True)``, the ten
    ``(dN, dA)`` derivative coordinates 7--16 corrected.
``raw``
    ``closed_residual_subtracted_rhs``, the uncorrected archive EOM of
    Eq. (112), the same right-hand side the default corrected lane corrects
    and the same one ``_cone_correction_audit`` calls
    ``correction_mode="none"``.  The named ``joint_gram_barrier`` lane runs
    with ``subtract_initial_residual=False``, so the velocity *it* corrects is
    the plain archive EOM ``closed_scalar_rhs`` and not this lane; the summary
    records which velocity the barrier census was solved from
    (``corrected_lane.census_base_velocity``).
``exact``
    ``exact_holstein_driven_trajectory`` at phonon cutoff 16, contracted to
    the same 31 coordinates.

Every number this driver writes is a diagnostic; nothing here promotes
evidence, and no default, profile, or acceptance rule is changed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from paper5.stability import (  # noqa: E402
    DimerParameters,
    closed_boson_moment_eigenvalues,
    closed_cone_projected_rhs,
    closed_electron_eigenvalues,
    closed_electron_phonon_cone_projected_rhs,
    closed_residual_subtracted_rhs,
    exact_ground_closed_scalar_coordinates,
)
from paper5.stability import cone_correction as _cone_correction  # noqa: E402
from paper5.stability.cone_long_horizon import (  # noqa: E402
    OBSERVABLE_NAMES,
    SampledTrajectory,
    barrier_activity,
    barrier_census_base_rhs,
    closed_observable_trajectories,
    integrate_sampled_dop853,
    joint_gram_barrier_activity,
    post_pulse_energy_drift,
    window_indices,
    window_scores,
)
from paper5.stability.exact_reference import (  # noqa: E402
    exact_holstein_driven_trajectory,
)
from paper5.stability.matrix_reference import (  # noqa: E402
    closed_scalar_to_matrix_state,
    matrix_state_to_closed_scalar_coordinates,
    matrix_total_energy,
)
from paper5.stability.multi_coherent_scores import (  # noqa: E402
    development_coordinate_scales,
)
from pipelines.open_dynamics.analyze_archive_observable_spectra import (  # noqa: E402
    _band_peak,
    _band_power_ratio,
    _hellinger,
    _power_ratio,
    _spectrum,
    _time_grid,
)

LANES = ("exact", "raw", "corrected")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--final-time", type=float, default=100.0)
    parser.add_argument("--sample-step", type=float, default=0.05)
    parser.add_argument("--phonon-cutoff", type=int, default=16)
    parser.add_argument(
        "--corrected-wall-budget-seconds",
        type=float,
        default=1500.0,
        help=(
            "stop the corrected lane after the first completed step past this "
            "wall time and score the horizon it reached"
        ),
    )
    parser.add_argument(
        "--raw-wall-budget-seconds",
        type=float,
        default=1500.0,
    )
    parser.add_argument(
        "--score-window",
        type=float,
        nargs=2,
        action="append",
        default=None,
        metavar=("START", "STOP"),
        help="repeatable; defaults to 0 20 and 0 100",
    )
    parser.add_argument("--spectrum-start", type=float, default=10.0)
    parser.add_argument("--spectrum-stop", type=float, default=100.0)
    parser.add_argument("--spectrum-sample-step", type=float, default=0.2)
    parser.add_argument(
        "--skip-barrier-census",
        action="store_true",
        help="skip re-solving the correction at every stored sample",
    )
    parser.add_argument(
        "--corrected-controller",
        choices=("boson_matrix_barrier", "joint_gram_barrier"),
        default="boson_matrix_barrier",
        help=(
            "which structured barrier the corrected lane carries: the "
            "recorded ten-coordinate energy-neutral structured boson matrix "
            "barrier on derivative coordinates 7--16 "
            "(structured_boson_barrier_correction through "
            "closed_cone_projected_rhs, the default), or the 31-coordinate "
            "joint-Gram barrier "
            "(structured_electron_phonon_barrier_correction through "
            "closed_electron_phonon_cone_projected_rhs)"
        ),
    )
    parser.add_argument(
        "--halfspace-backend",
        choices=("slsqp", "active_set"),
        default=None,
        help=(
            "minimum-norm-point solver inside the constraint-generation "
            "loop; the default None resolves to "
            "cone_correction.DEFAULT_HALFSPACE_BACKEND ('slsqp'), the "
            "backend every recorded Paper V number was produced with"
        ),
    )
    parser.add_argument(
        "--corrected-on-rhs-error",
        choices=("raise", "stop"),
        default="raise",
        help=(
            "what the corrected lane does when its right-hand side raises "
            "(the convergence contract of the joint-Gram barrier): 'raise' "
            "is the recorded behaviour; 'stop' keeps the samples already "
            "produced and records the exception"
        ),
    )
    parser.add_argument(
        "--progress-seconds",
        type=float,
        default=0.0,
        help=(
            "print a corrected-lane progress line no more often than this "
            "many wall seconds; 0 prints nothing"
        ),
    )
    parser.add_argument(
        "--score-matched-horizon",
        action="store_true",
        help=(
            "also score and census the window every lane actually reached, "
            "under its own key, when a lane stopped before the requested "
            "window's stop time; off by default, which is the recorded "
            "behaviour of scoring only the requested windows"
        ),
    )
    parser.add_argument(
        "--spectrum-stop-at-common-horizon",
        action="store_true",
        help=(
            "when a lane stopped before --spectrum-stop, take the spectral "
            "window's stop time to be the last sample every lane reached "
            "instead of declaring the spectrum unavailable"
        ),
    )
    return parser


def _trajectory_record(trajectory: SampledTrajectory) -> dict[str, Any]:
    return {
        "wall_seconds": trajectory.wall_seconds,
        "function_evaluations": trajectory.function_evaluations,
        "step_count": trajectory.step_count,
        "reached_final_time": trajectory.reached_final_time,
        "stop_reason": trajectory.stop_reason,
        "final_time_reached": trajectory.final_time_reached,
        "sample_count": int(trajectory.times.size),
    }


def _physicality_and_energy(
    states: np.ndarray,
    parameters: DimerParameters,
) -> dict[str, np.ndarray]:
    boson = np.asarray(
        [closed_boson_moment_eigenvalues(row)[0] for row in states],
        dtype=float,
    )
    electron = np.asarray(
        [closed_electron_eigenvalues(row)[0] for row in states],
        dtype=float,
    )
    energies = np.asarray(
        [
            matrix_total_energy(closed_scalar_to_matrix_state(row), parameters)
            for row in states
        ],
        dtype=float,
    )
    return {
        "boson_moment_minima": boson,
        "electron_minima": electron,
        "internal_energies": energies,
    }


def _band_peak_record(
    spectrum: Any,
    *,
    minimum_angular_frequency: float,
    maximum_angular_frequency: float,
) -> dict[str, Any]:
    """Peak record, or the reason the finite window cannot resolve the band."""

    try:
        peak = _band_peak(
            spectrum,
            minimum_angular_frequency=minimum_angular_frequency,
            maximum_angular_frequency=maximum_angular_frequency,
        )
    except ValueError as error:
        return {"unavailable": str(error)}
    return _peak_record(peak)


def _peak_record(peak: Any) -> dict[str, float]:
    return {
        "angular_frequency": float(peak.angular_frequency),
        "fwhm": float(peak.fwhm),
        "normalized_power_density": float(peak.normalized_power_density),
        "band_power": float(peak.band_power),
        "frequency_resolution": float(peak.frequency_resolution),
    }


def _polarization(times: np.ndarray, states: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Polarization ``P = -Delta n / 2`` interpolated onto the spectrum grid.

    The definition and the interpolation are those of
    ``analyze_archive_polarization_peaks._weak_polarization``: coordinate 0 of
    the closed scalar state is ``Delta n = rho_00 - rho_11``.
    """

    tolerance = 1e-10
    if grid[0] < times[0] - tolerance or grid[-1] > times[-1] + tolerance:
        raise ValueError("the requested spectral interval is not recorded")
    return np.interp(grid, times, -0.5 * np.asarray(states, dtype=float)[:, 0])


def main() -> int:
    args = _parser().parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    windows = (
        [(0.0, 20.0), (0.0, args.final_time)]
        if args.score_window is None
        else [(float(start), float(stop)) for start, stop in args.score_window]
    )

    parameters = DimerParameters(gamma=0.5, lambda_ep=1.5, drive_amplitude=1.0)
    initial_state = exact_ground_closed_scalar_coordinates(
        parameters,
        phonon_cutoff=args.phonon_cutoff,
    )
    sample_times = np.arange(
        0.0,
        args.final_time + 0.5 * args.sample_step,
        args.sample_step,
    )
    # ``np.arange`` can overshoot the final time by one ulp; a sample outside
    # the integration span is not a sample of this protocol.
    sample_times = sample_times[sample_times <= args.final_time]

    joint_gram_settings = {
        "activation_margin": 1e-5,
        "target_flux": 0.0,
        "barrier_rate": 5.0,
        "energy_neutral": True,
        "preserve_correlation_trace": True,
        "subtract_initial_residual": False,
        "require_convergence": True,
        "cone_tolerance": 1e-8,
        "projection_tolerance": 1e-12,
        "correction_metric": "euclidean",
    }
    if args.corrected_controller == "joint_gram_barrier":
        corrected_rhs = closed_electron_phonon_cone_projected_rhs(
            parameters,
            initial_state,
            halfspace_backend=args.halfspace_backend,
            **joint_gram_settings,
        )
    else:
        corrected_rhs = closed_cone_projected_rhs(
            parameters,
            initial_state,
            activation_margin=1e-5,
            target_flux=0.0,
            barrier_rate=5.0,
            energy_neutral=True,
            halfspace_backend=args.halfspace_backend,
        )
    raw_rhs = closed_residual_subtracted_rhs(parameters, initial_state)
    # A census of the corrected lane's barrier has to solve the correction
    # from the velocity that lane's right-hand side corrects.  The joint-Gram
    # lane runs with ``subtract_initial_residual=False``, so its base velocity
    # is the plain archive EOM, not the Eq. (112) archive EOM of the raw lane.
    census_base_rhs, census_base_velocity = barrier_census_base_rhs(
        parameters,
        initial_state,
        subtract_initial_residual=bool(
            joint_gram_settings["subtract_initial_residual"]
        )
        if args.corrected_controller == "joint_gram_barrier"
        else True,
    )

    progress_state = {"last": 0.0}

    def _progress(solver_time: float, step_count: int, elapsed: float) -> None:
        if args.progress_seconds <= 0.0:
            return
        if elapsed - progress_state["last"] < args.progress_seconds:
            return
        progress_state["last"] = elapsed
        projections = _cone_correction.SOLVER_COUNTERS.halfspace_projections
        print(
            f"corrected lane progress: t={solver_time:.4f} "
            f"steps={step_count} halfspace_projections={projections} "
            f"elapsed={elapsed:.1f} s",
            flush=True,
        )

    _cone_correction.SOLVER_COUNTERS.reset()
    corrected = integrate_sampled_dop853(
        corrected_rhs,
        initial_state,
        final_time=args.final_time,
        sample_times=sample_times,
        wall_budget_seconds=args.corrected_wall_budget_seconds,
        step_callback=_progress,
        on_rhs_error=args.corrected_on_rhs_error,
    )
    corrected_counters = _cone_correction.SOLVER_COUNTERS.as_dict()
    print(
        "corrected lane: "
        f"{corrected.stop_reason} at t={corrected.final_time_reached} "
        f"in {corrected.wall_seconds:.1f} s",
        flush=True,
    )
    # Insurance for a long corrected lane: the propagated samples are on disk
    # before any scoring runs.
    np.savez_compressed(
        args.output_directory / "corrected_trajectory_raw.npz",
        times=corrected.times,
        states=corrected.states,
        initial_state=initial_state,
        wall_seconds=corrected.wall_seconds,
        function_evaluations=corrected.function_evaluations,
        step_count=corrected.step_count,
        stop_reason=corrected.stop_reason,
    )
    raw = integrate_sampled_dop853(
        raw_rhs,
        initial_state,
        final_time=args.final_time,
        sample_times=sample_times,
        wall_budget_seconds=args.raw_wall_budget_seconds,
    )
    print(
        "raw lane: "
        f"{raw.stop_reason} at t={raw.final_time_reached} "
        f"in {raw.wall_seconds:.1f} s",
        flush=True,
    )

    exact = exact_holstein_driven_trajectory(
        parameters,
        sample_times=sample_times,
        phonon_cutoff=args.phonon_cutoff,
        eigensolver_tolerance=1e-12,
        relative_tolerance=1e-9,
        absolute_tolerance=1e-11,
        maximum_step=0.05,
    )
    exact_states = np.stack(
        [
            matrix_state_to_closed_scalar_coordinates(state)
            for state in exact.matrix_states
        ]
    )
    initial_difference = float(np.linalg.norm(exact_states[0] - initial_state))
    if initial_difference > 1e-9:
        raise RuntimeError(
            "the exact reference does not reproduce the run initial moments: "
            f"{initial_difference}"
        )

    lane_trajectories = {
        "exact": (np.asarray(exact.times, dtype=float), exact_states),
        "raw": (raw.times, raw.states),
        "corrected": (corrected.times, corrected.states),
    }
    diagnostics = {
        lane: _physicality_and_energy(states, parameters)
        for lane, (_, states) in lane_trajectories.items()
    }

    # Cross-check the copied observable definitions against the pipeline they
    # were copied from, on the exact lane.
    from pipelines.open_dynamics.plot_high_coupling_controller_observables import (
        observable_trajectories as _pipeline_observables,
    )

    observable_copy_difference = float(
        np.max(
            np.abs(
                closed_observable_trajectories(exact_states, parameters)
                - _pipeline_observables(exact_states, parameters)
            )
        )
    )

    scales = development_coordinate_scales(
        exact_states,
        phonon_cutoff=args.phonon_cutoff,
    )

    # The window every lane actually reached, so the matched-horizon table is
    # regenerated by the driver itself rather than by a later script.
    common_horizon = min(float(lane_trajectories[lane][0][-1]) for lane in LANES)
    matched_window = (0.0, common_horizon)

    comparisons: dict[str, Any] = {}
    for lane in ("raw", "corrected"):
        lane_times, lane_states = lane_trajectories[lane]
        reached = float(lane_times[-1])
        lane_result: dict[str, Any] = {}
        lane_windows = list(windows)
        if args.score_matched_horizon and all(
            abs(stop - common_horizon) > 1e-9 for _, stop in lane_windows
        ):
            lane_windows.append(matched_window)
        scored: dict[tuple[float, float], dict[str, Any]] = {}
        for start, stop in lane_windows:
            key = f"t_{start:g}_to_{stop:g}"
            effective_stop = min(stop, reached)
            if effective_stop <= start:
                lane_result[key] = {"unavailable": "the lane did not reach this window"}
                continue
            already = scored.get((start, effective_stop))
            if already is not None:
                # Two requested windows truncated to the same effective window
                # are one calculation; only the requested interval differs.
                repeated = dict(already)
                repeated["requested_window"] = [start, stop]
                lane_result[key] = repeated
                continue
            mask = window_indices(lane_times, (start, effective_stop))
            scores = window_scores(
                lane_times,
                lane_states,
                exact_states[: lane_times.size],
                parameters,
                window=(start, effective_stop),
                coordinate_scales=scales,
            )
            scores["requested_window"] = [start, stop]
            selected = diagnostics[lane]
            scores["minimum_boson_moment_eigenvalue"] = float(
                np.min(selected["boson_moment_minima"][mask])
            )
            scores["minimum_electron_eigenvalue"] = float(
                np.min(selected["electron_minima"][mask])
            )
            exact_selected = diagnostics["exact"]
            scores["exact_minimum_boson_moment_eigenvalue"] = float(
                np.min(exact_selected["boson_moment_minima"][: lane_times.size][mask])
            )
            scores["exact_minimum_electron_eigenvalue"] = float(
                np.min(exact_selected["electron_minima"][: lane_times.size][mask])
            )
            if effective_stop >= 4.0:
                scores.update(
                    post_pulse_energy_drift(
                        lane_times,
                        selected["internal_energies"],
                        window=(start, effective_stop),
                    )
                )
            scored[(start, effective_stop)] = scores
            lane_result[key] = scores
        lane_result["trajectory"] = _trajectory_record(
            corrected if lane == "corrected" else raw
        )
        comparisons[lane] = lane_result

    if not args.skip_barrier_census:
        for lane in ("raw", "corrected"):
            lane_times, lane_states = lane_trajectories[lane]
            reached = float(lane_times[-1])
            census: dict[str, Any] = {}
            census_windows = list(windows)
            if args.score_matched_horizon and all(
                abs(stop - common_horizon) > 1e-9 for _, stop in census_windows
            ):
                census_windows.append(matched_window)
            censused: dict[tuple[float, float], dict[str, Any]] = {}
            for start, stop in census_windows:
                effective_stop = min(stop, reached)
                if effective_stop <= start:
                    continue
                already = censused.get((start, effective_stop))
                if already is not None:
                    census[f"t_{start:g}_to_{stop:g}"] = already
                    continue
                if args.corrected_controller == "joint_gram_barrier":
                    census[f"t_{start:g}_to_{stop:g}"] = joint_gram_barrier_activity(
                        lane_times,
                        lane_states,
                        census_base_rhs,
                        parameters,
                        activation_margin=joint_gram_settings["activation_margin"],
                        target_flux=joint_gram_settings["target_flux"],
                        barrier_rate=joint_gram_settings["barrier_rate"],
                        energy_neutral=joint_gram_settings["energy_neutral"],
                        preserve_correlation_trace=joint_gram_settings[
                            "preserve_correlation_trace"
                        ],
                        cone_tolerance=joint_gram_settings["cone_tolerance"],
                        projection_tolerance=joint_gram_settings[
                            "projection_tolerance"
                        ],
                        correction_metric=joint_gram_settings["correction_metric"],
                        halfspace_backend=args.halfspace_backend,
                        window=(start, effective_stop),
                    )
                else:
                    census[f"t_{start:g}_to_{stop:g}"] = barrier_activity(
                        lane_times,
                        lane_states,
                        census_base_rhs,
                        window=(start, effective_stop),
                    )
                censused[(start, effective_stop)] = census[
                    f"t_{start:g}_to_{stop:g}"
                ]
                print(
                    f"barrier census {lane} {start}-{effective_stop}: "
                    f"{census[f't_{start:g}_to_{stop:g}']['active_sample_fraction']}",
                    flush=True,
                )
            comparisons[lane]["barrier_activity"] = census

    spectrum_stop = args.spectrum_stop
    if args.spectrum_stop_at_common_horizon:
        spectrum_stop = min(spectrum_stop, common_horizon)
    spectra_available = (
        spectrum_stop > args.spectrum_start
        and all(
            float(lane_trajectories[lane][0][-1]) >= spectrum_stop - 1e-9
            for lane in LANES
        )
    )
    spectrum_summary: dict[str, Any]
    if spectra_available:
        grid = _time_grid(
            args.spectrum_start,
            spectrum_stop,
            args.spectrum_sample_step,
        )
        spectra = {
            lane: _spectrum(
                _polarization(lane_trajectories[lane][0], lane_trajectories[lane][1], grid),
                args.spectrum_sample_step,
            )
            for lane in LANES
        }
        reference = spectra["exact"]
        spectrum_summary = {
            "interval": [args.spectrum_start, float(grid[-1])],
            "requested_interval": [args.spectrum_start, args.spectrum_stop],
            "sample_step": args.spectrum_sample_step,
            "observable": "polarization P = -Delta n / 2",
            "mean_subtracted": True,
            "window": "Hann",
            "electronic_peak_band": [1.5, 3.5],
            "low_frequency_peak_band": [0.05, 1.0],
            "width_definition": (
                "observed half-maximum width of the finite-window spectrum; "
                "no lifetime deconvolution"
            ),
            "lanes": {
                lane: {
                    "hellinger_distance_to_exact": _hellinger(spectra[lane], reference),
                    "total_hann_power_ratio_to_exact": _power_ratio(
                        spectra[lane], reference
                    ),
                    "oscillation_rms_ratio_to_exact": (
                        spectra[lane].oscillation_rms / reference.oscillation_rms
                    ),
                    "electronic_band_hann_power_ratio_to_exact": _band_power_ratio(
                        spectra[lane],
                        reference,
                        minimum_angular_frequency=1.5,
                        maximum_angular_frequency=3.5,
                    ),
                    "dominant_angular_frequency": (
                        spectra[lane].dominant_angular_frequency
                    ),
                    "electronic_band_peak": _band_peak_record(
                        spectra[lane],
                        minimum_angular_frequency=1.5,
                        maximum_angular_frequency=3.5,
                    ),
                    "low_frequency_peak": _band_peak_record(
                        spectra[lane],
                        minimum_angular_frequency=0.05,
                        maximum_angular_frequency=1.0,
                    ),
                }
                for lane in LANES
            },
        }
        np.savez_compressed(
            args.output_directory / "polarization_spectra.npz",
            angular_frequency=reference.angular_frequency,
            **{
                f"{lane}_normalized_power": spectra[lane].normalized_power
                for lane in LANES
            },
            **{f"{lane}_power": spectra[lane].power for lane in LANES},
            grid=grid,
            **{
                f"{lane}_polarization": _polarization(
                    lane_trajectories[lane][0], lane_trajectories[lane][1], grid
                )
                for lane in LANES
            },
        )
    else:
        spectrum_summary = {
            "unavailable": (
                "one lane did not reach the spectral stop time; "
                "the spectrum window is not recorded"
            ),
            "reached": {
                lane: float(lane_trajectories[lane][0][-1]) for lane in LANES
            },
        }

    for lane in LANES:
        lane_times, lane_states = lane_trajectories[lane]
        np.savez_compressed(
            args.output_directory / f"{lane}_trajectory.npz",
            times=lane_times,
            states=lane_states,
            **{key: value for key, value in diagnostics[lane].items()},
            observables=closed_observable_trajectories(lane_states, parameters),
            initial_state=initial_state,
        )

    summary: dict[str, Any] = {
        "schema": "paper5.energy_neutral_barrier_long_horizon.v1",
        "classification": "diagnostic_exploratory_not_promoted",
        "plan_step": (
            "agent_guidance/paper-v/paper-v-state-and-plan-20260914.md:276-282"
        ),
        "protocol": {
            "lambda_ep": 1.5,
            "gamma": 0.5,
            "drive_amplitude": 1.0,
            "pulse_width": parameters.pulse_width,
            "phonon_cutoff": args.phonon_cutoff,
            "activation_margin": 1e-5,
            "target_flux": 0.0,
            "barrier_rate": 5.0,
            "energy_neutral": True,
            "corrected_derivative_coordinates": (
                # The joint-Gram barrier solves its correction in all 31
                # closed scalar coordinates; the default boson matrix barrier
                # corrects the ten ``(dN, dA)`` coordinates 7--16.
                [0, 31]
                if args.corrected_controller == "joint_gram_barrier"
                else [7, 17]
            ),
            "integrator": "DOP853",
            "rtol": 1e-9,
            "atol": 1e-11,
            "max_step": 0.05,
            "requested_final_time": args.final_time,
            "sample_step": args.sample_step,
            "raw_lane": "closed_residual_subtracted_rhs (Eq. 112 archive EOM)",
        },
        "exact_reference": {
            "method": "cutoff-truncated full-wavefunction DOP853 propagation",
            "phonon_cutoff": args.phonon_cutoff,
            "function_evaluations": exact.function_evaluations,
            "maximum_state_norm_error": float(np.max(np.abs(exact.state_norms - 1.0))),
            "initial_coordinate_l2_difference": initial_difference,
        },
        "corrected_lane": {
            "controller": args.corrected_controller,
            "right_hand_side": (
                "closed_electron_phonon_cone_projected_rhs"
                if args.corrected_controller == "joint_gram_barrier"
                else "closed_cone_projected_rhs"
            ),
            "correction": (
                "structured_electron_phonon_barrier_correction"
                if args.corrected_controller == "joint_gram_barrier"
                else "structured_boson_barrier_correction"
            ),
            "settings": (
                joint_gram_settings
                if args.corrected_controller == "joint_gram_barrier"
                else {
                    "activation_margin": 1e-5,
                    "target_flux": 0.0,
                    "barrier_rate": 5.0,
                    "energy_neutral": True,
                }
            ),
            "halfspace_backend": (
                args.halfspace_backend
                if args.halfspace_backend is not None
                else _cone_correction.DEFAULT_HALFSPACE_BACKEND
            ),
            "halfspace_backend_named_explicitly": args.halfspace_backend is not None,
            "on_rhs_error": args.corrected_on_rhs_error,
            "census_base_velocity": census_base_velocity,
            "wall_budget_seconds": args.corrected_wall_budget_seconds,
            "solver_counters": corrected_counters,
        },
        "observable_names": list(OBSERVABLE_NAMES),
        "observable_copy_maximum_difference": observable_copy_difference,
        "rms_conventions": {
            "coordinate_l2_time_rms": (
                "trapezoidal time-RMS of the unscaled 31-coordinate Euclidean "
                "distance from the exact cutoff-16 reference"
            ),
            "all31_scaled_rms": (
                "unweighted root mean square over samples and coordinates of "
                "(lane - exact) / development_coordinate_scales; the "
                "adaptive-packet route's t=0 to 98 line uses this convention "
                "with scales frozen from its own development path"
            ),
            "observable sample_rms": (
                "unweighted root mean square over samples of the observable "
                "difference; the packet line's observable RMS convention"
            ),
            "observable time_rms": "trapezoidal time-RMS of the same difference",
        },
        "adaptive_packet_reference_line": {
            "source": (
                "paper_5/notes/electron_phonon_closure_worklog_20260803.md:"
                "2509-2516"
            ),
            "window": [0.0, 98.0],
            "all31_scaled_rms": 0.063649,
            "site_0_occupation_rms": 0.013600,
            "electronic_energy_rms": 0.023320,
            "phonon_energy_rms": 0.028350,
            "electron_phonon_energy_rms": 0.033785,
            "internal_energy_rms": 0.000579,
        },
        "comparisons": comparisons,
        "polarization_spectrum": spectrum_summary,
    }
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
