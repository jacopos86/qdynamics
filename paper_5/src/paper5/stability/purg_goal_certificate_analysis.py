"""Frozen-space numerical pilot for the amended PURG output certificate.

This command opens no exact driven trajectory.  It keeps the frozen online
rank-128 PURG model, builds an offline residual-conditioned correction space
and a goal-conditioned enriched dual space, and evaluates the amended
centered-derivative intervals as numerical a posteriori estimates.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import eye
from scipy.sparse.linalg import splu

from .exact_reference import _build_exact_dimer_model
from .hubbard_dimer import DimerParameters
from .krylov_memory_closure import (
    _build_raw_moment_basis_from_model,
    raw_to_closed_jacobian,
)
from .purg import (
    _CENTERED_BLOCKS,
    _CENTERING_HESSIAN,
    _make_projection,
    _rrqr_append,
    build_purg_operator_bounds,
)
from .purg_goal_certificate import (
    _full_goal_actions,
    build_dual_leakage_envelope,
    estimate_centered_derivative_intervals,
    propagate_forward_remainder,
    propagate_purg_error_correction,
)

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class PurgGoalCertificatePilotSettings:
    """Fixed exploratory settings; these are not amended Gate-B settings."""

    online_cap: int = 128
    correction_rank: int = 192
    dual_rank: int = 256
    step: float = 0.01
    score_step: float = 0.04
    shift: float = 0.5
    rrqr_relative_tolerance: float = 1e-12
    quadrature_absolute_tolerance: float = 1e-7


def _packet_expand(
    basis: ComplexArray,
    candidates: ComplexArray,
    *,
    target_rank: int,
    shifted_hamiltonian: object,
    drive_hamiltonian: object,
    factorization: object,
    relative_tolerance: float,
) -> tuple[ComplexArray, dict[str, object]]:
    columns_needed = target_rank - basis.shape[1]
    if columns_needed <= 0:
        raise ValueError("target_rank must exceed the current basis rank")
    seed_count = int(np.ceil(columns_needed / 4.0))
    seed_result = _rrqr_append(
        basis,
        candidates,
        relative_tolerance=relative_tolerance,
        maximum_new_columns=seed_count,
    )
    seeds = seed_result.basis[:, basis.shape[1] :]
    packet: list[ComplexArray] = []
    solve_residuals: list[float] = []
    for column in range(seeds.shape[1]):
        seed = seeds[:, column]
        inverse = np.asarray(factorization.solve(seed), dtype=complex)
        solve_residuals.append(
            float(
                np.linalg.norm(shifted_hamiltonian @ inverse - seed)
                / max(np.linalg.norm(seed), 1e-30)
            )
        )
        packet.extend(
            (
                seed,
                inverse,
                shifted_hamiltonian @ seed,
                drive_hamiltonian @ seed,
            )
        )
    if not packet:
        raise RuntimeError("every enrichment seed deflated")
    append = _rrqr_append(
        basis,
        np.column_stack(packet),
        relative_tolerance=relative_tolerance,
        maximum_new_columns=columns_needed,
    )
    if append.basis.shape[1] != target_rank:
        raise RuntimeError(
            f"requested rank {target_rank}, reached {append.basis.shape[1]}"
        )
    return append.basis, {
        "candidate_columns": int(candidates.shape[1]),
        "selected_seeds": int(seeds.shape[1]),
        "seed_deflated": int(seed_result.deflated),
        "seed_truncated": int(seed_result.truncated),
        "packet_deflated": int(append.deflated),
        "packet_truncated": int(append.truncated),
        "maximum_shifted_solve_relative_residual": max(
            solve_residuals,
            default=0.0,
        ),
    }


def _goal_candidates(
    projection: object,
    parameters: DimerParameters,
    correction_basis: ComplexArray,
    correction: object,
    operator_bounds: object,
    *,
    sample_stride: int,
) -> tuple[ComplexArray, dict[str, object]]:
    budgets = {
        "rho": 1e-4,
        "B": 0.0025,
        "N": 0.0025,
        "A": 0.0025,
        "C": 0.025,
    }
    component_budget = np.empty(31, dtype=float)
    for block, block_slice in _CENTERED_BLOCKS.items():
        component_budget[block_slice] = budgets[block]
    candidates: list[ComplexArray] = []
    priorities: list[float] = []
    labels: list[tuple[int, int]] = []
    for trajectory_index in range(0, correction.times.size, sample_stride):
        time = float(correction.times[trajectory_index])
        drive = parameters.drive_difference(time)
        reduced_state = correction.primal_states[trajectory_index]
        represented = projection.lift(reduced_state)
        lifted = correction_basis @ correction.correction_states[trajectory_index]
        observable_actions, derivative_actions = _full_goal_actions(
            projection,
            drive,
            np.column_stack((represented, lifted)),
        )
        f_center = observable_actions[:, :, 0] + observable_actions[:, :, 1]
        k_center = derivative_actions[:, :, 0] + derivative_actions[:, :, 1]
        raw = projection.model.raw_coordinates(reduced_state)
        physical_velocity = np.asarray(
            [
                np.vdot(represented, derivative_actions[index, :, 0]).real
                for index in range(derivative_actions.shape[0])
            ],
            dtype=float,
        )
        jacobian = raw_to_closed_jacobian(raw)
        value_coefficients = np.einsum(
            "iab,b->ia",
            _CENTERING_HESSIAN,
            physical_velocity,
        )
        derivative_norms = (
            operator_bounds.static_derivative
            + abs(drive) * operator_bounds.drive_derivative
        )
        hamiltonian = (
            projection.static_hamiltonian
            + drive * projection.drive_hamiltonian
        ).tocsc()
        beta = float(correction.unresolved_error_bound[trajectory_index])
        for component in range(31):
            action = (
                jacobian[component] @ k_center
                + value_coefficients[component] @ f_center
            )
            action_norm = float(np.linalg.norm(action))
            if action_norm <= 1e-14:
                continue
            operator_norm = float(
                np.abs(jacobian[component]) @ derivative_norms
                + np.abs(value_coefficients[component]) @ operator_bounds.raw
            )
            priority = (
                2.0 * action_norm * beta + operator_norm * beta**2
            ) / component_budget[component]
            projected_action = correction_basis @ (
                correction_basis.conj().T @ action
            )
            terminal_seed = action - projected_action
            propagated_action = hamiltonian @ projected_action
            dual_leakage_seed = propagated_action - correction_basis @ (
                correction_basis.conj().T @ propagated_action
            )
            for seed in (terminal_seed, dual_leakage_seed):
                seed_norm = float(np.linalg.norm(seed))
                if seed_norm <= 1e-14:
                    continue
                candidates.append((priority / seed_norm) * seed)
                priorities.append(priority)
                labels.append((component, trajectory_index))
    if not candidates:
        raise RuntimeError("no nonzero direct-goal candidate was produced")
    maximum = int(np.argmax(priorities))
    return np.column_stack(candidates), {
        "candidate_columns": len(candidates),
        "maximum_priority": float(priorities[maximum]),
        "maximum_priority_component": int(labels[maximum][0]),
        "maximum_priority_time": float(correction.times[labels[maximum][1]]),
    }


def run_frozen_goal_certificate_pilot(
    input_artifact: Path,
    output_directory: Path,
    *,
    settings: PurgGoalCertificatePilotSettings | None = None,
) -> dict[str, object]:
    """Run the no-exact-data residual/goal-space pilot and write its artifact."""

    resolved = settings or PurgGoalCertificatePilotSettings()
    parameters = DimerParameters(
        lambda_ep=1.5,
        gamma=0.5,
        drive_amplitude=1.0,
    )
    arrays_path = input_artifact / "arrays.npz"
    with np.load(arrays_path) as arrays:
        online_basis = arrays[f"cap_{resolved.online_cap}_basis"].copy()
        frozen_states = arrays[f"cap_{resolved.online_cap}_states"].copy()
        frozen_times = arrays[f"cap_{resolved.online_cap}_times"].copy()
        ground_state = arrays["ground_state"].copy()

    full_model = _build_exact_dimer_model(parameters, phonon_cutoff=16)
    raw_basis = _build_raw_moment_basis_from_model(
        full_model,
        phonon_cutoff=16,
    )
    static_hamiltonian = full_model.static_hamiltonian.tocsc()
    drive_hamiltonian = full_model.drive_operator.tocsc()
    projection, _ = _make_projection(
        basis=online_basis,
        cap_label=resolved.online_cap,
        phonon_cutoff=16,
        static_hamiltonian=static_hamiltonian,
        drive_hamiltonian=drive_hamiltonian,
        raw_observables=raw_basis.observables,
        reference_initial_state=ground_state,
    )
    ground_energy = float(
        np.vdot(ground_state, static_hamiltonian @ ground_state).real
    )
    identity = eye(static_hamiltonian.shape[0], format="csc", dtype=complex)
    shifted = (
        static_hamiltonian
        - ground_energy * identity
        + resolved.shift * identity
    ).tocsc()
    factorization = splu(shifted)

    ratio = int(round(resolved.step / (frozen_times[1] - frozen_times[0])))
    if ratio <= 0 or not np.isclose(
        ratio * (frozen_times[1] - frozen_times[0]),
        resolved.step,
        atol=1e-13,
    ):
        raise ValueError("pilot step is incompatible with frozen state grid")
    residual_candidates = np.column_stack(
        [
            projection.projection_residual(
                frozen_states[index],
                drive_value=parameters.drive_difference(float(frozen_times[index])),
            )
            for index in range(0, frozen_times.size, ratio)
        ]
    )
    correction_basis, correction_build = _packet_expand(
        online_basis,
        residual_candidates,
        target_rank=resolved.correction_rank,
        shifted_hamiltonian=shifted,
        drive_hamiltonian=drive_hamiltonian,
        factorization=factorization,
        relative_tolerance=resolved.rrqr_relative_tolerance,
    )
    correction = propagate_purg_error_correction(
        projection,
        parameters,
        correction_basis,
        final_time=4.0,
        step=resolved.step,
        quadrature_absolute_tolerance=resolved.quadrature_absolute_tolerance,
    )
    operator_bounds = build_purg_operator_bounds(projection)
    goal_stride = max(1, int(round(0.1 / resolved.step)))
    goal_candidates, goal_build = _goal_candidates(
        projection,
        parameters,
        correction_basis,
        correction,
        operator_bounds,
        sample_stride=goal_stride,
    )
    residual_candidates = correction.correction_residuals[::goal_stride].T
    residual_norms = np.linalg.norm(residual_candidates, axis=0)
    nonzero = residual_norms > 1e-14
    if np.any(nonzero):
        residual_candidates = residual_candidates[:, nonzero]
        residual_candidates = residual_candidates / residual_norms[nonzero]
        goal_candidates = np.column_stack(
            (goal_candidates, residual_candidates)
        )
    dual_basis, dual_build = _packet_expand(
        correction_basis,
        goal_candidates,
        target_rank=resolved.dual_rank,
        shifted_hamiltonian=shifted,
        drive_hamiltonian=drive_hamiltonian,
        factorization=factorization,
        relative_tolerance=resolved.rrqr_relative_tolerance,
    )
    forward = propagate_forward_remainder(
        projection,
        parameters,
        correction_basis,
        correction,
        dual_basis,
        quadrature_absolute_tolerance=resolved.quadrature_absolute_tolerance,
    )
    envelope = build_dual_leakage_envelope(
        projection,
        parameters,
        dual_basis,
        correction.times,
        quadrature_absolute_tolerance=resolved.quadrature_absolute_tolerance,
    )
    score_stride = int(round(resolved.score_step / resolved.step))
    if score_stride <= 0 or not np.isclose(
        score_stride * resolved.step,
        resolved.score_step,
        atol=1e-13,
    ):
        raise ValueError("score_step must be an integer multiple of step")
    estimate = estimate_centered_derivative_intervals(
        projection,
        parameters,
        correction_basis,
        correction,
        dual_basis,
        forward,
        envelope,
        operator_bounds,
        sample_stride=score_stride,
    )
    summary: dict[str, object] = {
        "status": "numerical_a_posteriori_pilot_not_formal_certificate",
        "exact_driven_scorer_opened": False,
        "settings": asdict(resolved),
        "input_artifact": str(input_artifact),
        "ranks": {
            "online": int(online_basis.shape[1]),
            "correction": int(correction_basis.shape[1]),
            "dual": int(dual_basis.shape[1]),
        },
        "correction_build": correction_build,
        "goal_build": goal_build,
        "dual_build": dual_build,
        "correction": {
            "final_unresolved_error_bound": float(
                correction.unresolved_error_bound[-1]
            ),
            "maximum_residual_norm": float(
                np.max(correction.correction_residual_norms)
            ),
            "quadrature_error_estimate": correction.quadrature_error_estimate,
            "maximum_residual_identity_error": (
                correction.maximum_residual_identity_error
            ),
        },
        "forward_remainder": {
            "final_numerical_residual_integral": float(
                forward.cumulative_numerical_residual[-1]
            ),
            "quadrature_error_estimate": forward.quadrature_error_estimate,
        },
        "dual_envelope": {
            "final_cumulative_leakage": float(
                envelope.cumulative_leakage[-1]
            ),
            "static_leakage_norm": envelope.static_leakage_norm,
            "drive_leakage_norm": envelope.drive_leakage_norm,
            "reduced_drive_norm": envelope.reduced_drive_norm,
            "quadrature_error_estimate": envelope.quadrature_error_estimate,
        },
        "block_derivative_metrics": estimate.block_metrics,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        output_directory / "arrays.npz",
        online_basis=online_basis,
        correction_basis=correction_basis,
        dual_basis=dual_basis,
        correction_times=correction.times,
        correction_states=correction.correction_states,
        unresolved_error_bound=correction.unresolved_error_bound,
        correction_residuals=correction.correction_residuals,
        midpoint_correction_residuals=(
            correction.midpoint_correction_residuals
        ),
        correction_residual_norms=correction.correction_residual_norms,
        forward_remainder_states=forward.states,
        forward_numerical_residual=forward.cumulative_numerical_residual,
        dual_cumulative_leakage=envelope.cumulative_leakage,
        score_times=estimate.times,
        score_lower=estimate.lower,
        score_upper=estimate.upper,
        score_centers=estimate.centers,
        score_absolute_bounds=estimate.absolute_bounds,
        score_terminal_defects=estimate.direct_goal_terminal_defects,
        score_bilinear_radii=estimate.bilinear_radii,
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_artifact", type=Path)
    parser.add_argument("output_directory", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    summary = run_frozen_goal_certificate_pilot(
        args.input_artifact,
        args.output_directory,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
