"""Command-line reproduction for the scalar Fan--Migdal divergence."""

from __future__ import annotations

import argparse
import json

import numpy as np

from .hubbard_dimer import (
    FAN_MIGDAL_STATE_NAMES,
    DimerParameters,
    fan_migdal_rhs,
    finite_difference_jacobian,
    hartree_fock_zero_correlation_state,
    integrate_rk4,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambda-ep", type=float, default=1.5)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--drive", type=float, default=1.0)
    parser.add_argument("--time-step", type=float, default=0.01)
    parser.add_argument("--final-time", type=float, default=150.0)
    parser.add_argument("--failure-threshold", type=float, default=1e4)
    parser.add_argument("--eq95-source-scale", type=float, default=1.0)
    parser.add_argument("--eq97-source-scale", type=float, default=1.0)
    parser.add_argument(
        "--expect-bounded",
        action="store_true",
        help="Return a nonzero status when the declared failure threshold is crossed.",
    )
    parser.add_argument(
        "--expect-divergence",
        action="store_true",
        help="Return a nonzero status when no declared failure occurs.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.expect_bounded and args.expect_divergence:
        raise SystemExit("choose at most one expectation")

    parameters = DimerParameters(
        gamma=args.gamma,
        lambda_ep=args.lambda_ep,
        drive_amplitude=args.drive,
        eq95_source_scale=args.eq95_source_scale,
        eq97_source_scale=args.eq97_source_scale,
    )
    initial_state = hartree_fock_zero_correlation_state()
    rhs = lambda time, state: fan_migdal_rhs(time, state, parameters)
    residual = rhs(0.0, initial_state)
    jacobian = finite_difference_jacobian(rhs, 0.0, initial_state)
    eigenvalues = np.linalg.eigvals(jacobian)
    result = integrate_rk4(
        rhs,
        initial_state,
        final_time=args.final_time,
        time_step=args.time_step,
        failure_threshold=args.failure_threshold,
        state_names=FAN_MIGDAL_STATE_NAMES,
    )

    nonzero_residual = {
        name: float(value)
        for name, value in zip(FAN_MIGDAL_STATE_NAMES, residual, strict=True)
        if abs(value) > 1e-14
    }
    report = {
        "parameters": {
            "lambda_ep": parameters.lambda_ep,
            "gamma": parameters.gamma,
            "coupling": parameters.coupling,
            "drive_amplitude": parameters.drive_amplitude,
            "time_step": args.time_step,
            "final_time": args.final_time,
            "failure_threshold": args.failure_threshold,
            "eq95_source_scale": parameters.eq95_source_scale,
            "eq97_source_scale": parameters.eq97_source_scale,
        },
        "initial_state": "hartree_fock_electronic_plus_zero_correlations",
        "initial_residual_norm": float(np.linalg.norm(residual)),
        "nonzero_initial_residual": nonzero_residual,
        "initial_jacobian_max_real_eigenvalue": float(
            np.max(np.real(eigenvalues))
        ),
        "integration": {
            "diverged": result.diverged,
            "failure_time": result.failure_time,
            "failure_component": result.failure_component,
            "max_abs_state": result.max_abs_state,
            "last_time": result.final_time,
            "steps": result.steps,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))

    if args.expect_bounded and result.diverged:
        return 1
    if args.expect_divergence and not result.diverged:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
