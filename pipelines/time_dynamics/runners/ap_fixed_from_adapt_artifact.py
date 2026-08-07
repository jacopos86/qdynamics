"""Run fixed-support AP-McLachlan from a static scaffold artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.scaffold.runtime_loader import load_scaffold_runtime_input
from pipelines.time_dynamics.ap_mclachlan.drive_aligned import (
    augment_state_with_drive_aligned_generator,
)
from pipelines.time_dynamics.ap_mclachlan.hamiltonian import (
    time_dependent_hamiltonian_from_runtime_input,
)
from pipelines.time_dynamics.ap_mclachlan.integrators import (
    INTEGRATOR_EULER,
    SUPPORTED_INTEGRATORS,
)
from pipelines.time_dynamics.ap_mclachlan.inverse import (
    DEFAULT_MCLACHLAN_RIDGE_LAMBDA,
    DEFAULT_MCLACHLAN_SOLVE_DAMPING,
    McLachlanInversePolicy,
)
from pipelines.time_dynamics.ap_mclachlan.observables import (
    build_site_doublon_observable_plan,
    observable_row_fields,
)
from pipelines.time_dynamics.ap_mclachlan.reference_diagnostics import (
    ReferenceEnergyTrajectory,
    attach_reference_energy_diagnostics,
    attach_reference_energy_diagnostics_with_prefix,
    load_reference_energy_trajectory,
    reference_energy_summary,
    reference_energy_trajectory_from_payload,
)
from pipelines.time_dynamics.ap_mclachlan.state import (
    AP_PARAMETERIZATION_PER_PAULI_TERM,
    AP_SUPPORTED_PARAMETERIZATION_MODES,
    state_from_scaffold_runtime_input,
)
from pipelines.time_dynamics.ap_mclachlan.trajectory import run_fixed_mclachlan_trajectory


RUNNER_SCHEMA_V1 = "ap_mclachlan_fixed_from_adapt_artifact_v1"


def run_fixed_ap_mclachlan_from_runtime_input(
    runtime_input: Any,
    *,
    times: Sequence[float],
    integrator_method: str = INTEGRATOR_EULER,
    pinv_rcond: float = 1.0e-10,
    ridge_lambda: float = DEFAULT_MCLACHLAN_RIDGE_LAMBDA,
    solve_damping: float = DEFAULT_MCLACHLAN_SOLVE_DAMPING,
    enable_drive: bool = False,
    drive_config: Any | None = None,
    drive_aligned_ansatz: bool = True,
    parameterization_mode: str = AP_PARAMETERIZATION_PER_PAULI_TERM,
    reference_energy_trajectory: Any | None = None,
    reference_energy_atol: float = 1.0e-12,
    seed_reference_energy_trajectory: Any | None = None,
    seed_reference_energy_atol: float = 1.0e-12,
    runner_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run fixed-support AP-McLachlan from an already-loaded scaffold contract."""

    if bool(enable_drive) and drive_config is None:
        raise ValueError("enable_drive=True requires a drive_config.")
    state = state_from_scaffold_runtime_input(
        runtime_input,
        parameterization_mode=str(parameterization_mode),
    )
    hamiltonian = time_dependent_hamiltonian_from_runtime_input(
        runtime_input,
        drive_config=(drive_config if bool(enable_drive) else None),
    )
    drive_augmentation = augment_state_with_drive_aligned_generator(
        state,
        hamiltonian=hamiltonian,
        enabled=bool(enable_drive) and bool(drive_aligned_ansatz),
    )
    state = drive_augmentation.state
    inverse_policy = McLachlanInversePolicy(
        pinv_rcond=float(pinv_rcond),
        ridge_lambda=float(ridge_lambda),
        solve_damping=float(solve_damping),
    )
    trajectory = run_fixed_mclachlan_trajectory(
        state=state,
        hamiltonian=hamiltonian,
        times=tuple(float(t) for t in times),
        inverse_policy=inverse_policy,
        integrator_method=str(integrator_method),
        metadata={
            "runner_schema": RUNNER_SCHEMA_V1,
            **dict(runner_metadata or {}),
        },
    )
    reference = _coerce_reference_energy_trajectory(reference_energy_trajectory)
    rows = attach_reference_energy_diagnostics(
        plot_rows=_plot_rows(trajectory, state=state),
        reference=reference,
        atol=float(reference_energy_atol),
    )
    seed_reference = _coerce_reference_energy_trajectory(seed_reference_energy_trajectory)
    rows = attach_reference_energy_diagnostics_with_prefix(
        plot_rows=rows,
        reference=seed_reference,
        atol=float(seed_reference_energy_atol),
        field_prefix="seed_",
    )
    return {
        "schema": RUNNER_SCHEMA_V1,
        "state": state.to_json_dict(),
        "drive_aligned_ansatz": drive_augmentation.to_json_dict(),
        "hamiltonian": hamiltonian.to_json_dict(),
        "trajectory": trajectory.to_json_dict(),
        "plot_rows": rows,
        "summary": _summary_from_rows(
            rows,
            state=state,
            hamiltonian=hamiltonian,
            integrator_method=str(integrator_method),
            inverse_policy=inverse_policy,
        ),
        "decision_data_flow": {
            "uses_reference_for_decision": False,
            "uses_exact_reference_for_decision": False,
            "uses_future_exact_forecast_for_decision": False,
            "uses_statevector_as_ideal_observable_estimator": True,
            "reference_energy_error_scope": "post_run_reporting",
            "seed_reference_energy_error_scope": "post_run_reporting",
        },
    }


def run_fixed_ap_mclachlan_from_artifact(
    *,
    artifact_json: str | Path,
    times: Sequence[float],
    loader_mode: str | None = None,
    tag: str | None = None,
    generator_family: str = "match_adapt",
    fallback_family: str = "full_meta",
    integrator_method: str = INTEGRATOR_EULER,
    pinv_rcond: float = 1.0e-10,
    ridge_lambda: float = DEFAULT_MCLACHLAN_RIDGE_LAMBDA,
    solve_damping: float = DEFAULT_MCLACHLAN_SOLVE_DAMPING,
    enable_drive: bool = False,
    drive_config: Any | None = None,
    drive_aligned_ansatz: bool = True,
    parameterization_mode: str = AP_PARAMETERIZATION_PER_PAULI_TERM,
    reference_energy_trajectory: Any | None = None,
    reference_energy_atol: float = 1.0e-12,
    seed_reference_energy_trajectory: Any | None = None,
    seed_reference_energy_atol: float = 1.0e-12,
) -> dict[str, Any]:
    artifact_path = Path(artifact_json)
    runtime_input = _load_runtime_input_or_raise(
        artifact_path,
        loader_mode=loader_mode,
        tag=tag,
        generator_family=str(generator_family),
        fallback_family=str(fallback_family),
    )
    payload = run_fixed_ap_mclachlan_from_runtime_input(
        runtime_input,
        times=times,
        integrator_method=str(integrator_method),
        pinv_rcond=float(pinv_rcond),
        ridge_lambda=float(ridge_lambda),
        solve_damping=float(solve_damping),
        enable_drive=bool(enable_drive),
        drive_config=drive_config,
        drive_aligned_ansatz=bool(drive_aligned_ansatz),
        parameterization_mode=str(parameterization_mode),
        reference_energy_trajectory=reference_energy_trajectory,
        reference_energy_atol=float(reference_energy_atol),
        seed_reference_energy_trajectory=seed_reference_energy_trajectory,
        seed_reference_energy_atol=float(seed_reference_energy_atol),
        runner_metadata={
            "artifact_json": str(artifact_path),
            "loader_mode": loader_mode,
            "tag": tag,
            "generator_family": str(generator_family),
            "fallback_family": str(fallback_family),
            "parameterization_mode": str(parameterization_mode),
        },
    )
    payload["source_artifact_json"] = str(artifact_path)
    return payload


def _plot_rows(trajectory: Any, *, state: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    observable_plan = build_site_doublon_observable_plan(
        state.resolved_problem,
        dimension=int(np.asarray(state.psi_ref, dtype=complex).reshape(-1).size),
    )
    for point in trajectory.points:
        theta = np.asarray(point.theta_runtime, dtype=float).reshape(-1)
        theta_dot = np.asarray(point.fixed_step.theta_dot, dtype=float).reshape(-1)
        row = {
            "index": int(point.index),
            "time": float(point.time),
            "energy_expectation": float(point.energy_expectation),
            "theta_l2": float(np.linalg.norm(theta)),
            "theta_dot_l2": float(np.linalg.norm(theta_dot)),
            "mclachlan_gamma": float(point.fixed_step.gamma),
            "mclachlan_residual_sq": float(point.fixed_step.residual_sq),
            "mclachlan_residual_ratio": float(point.fixed_step.residual_ratio),
            "rank": int(point.fixed_step.rank),
            "condition_number": (
                None
                if point.fixed_step.condition_number is None
                else float(point.fixed_step.condition_number)
            ),
            "effective_pinv_rcond": float(point.fixed_step.inverse_policy.pinv_rcond),
            "effective_ridge_lambda": float(point.fixed_step.inverse_policy.ridge_lambda),
            "effective_solve_damping": float(point.fixed_step.inverse_policy.solve_damping),
        }
        row.update(
            observable_row_fields(
                np.asarray(point.geometry.psi, dtype=complex),
                plan=observable_plan,
            )
        )
        rows.append(row)
    return rows


def _summary_from_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    state: Any,
    hamiltonian: Any,
    integrator_method: str,
    inverse_policy: McLachlanInversePolicy,
) -> dict[str, Any]:
    if not rows:
        summary = {
            "point_count": 0,
            "parameterization_mode": str(state.parameterization_mode),
            "parameterization_label": str(state.parameterization_label),
            "active_parameter_count": int(state.active_parameter_count),
            "runtime_parameter_count": int(state.runtime_parameter_count),
            "runtime_pauli_parameter_count": int(state.runtime_pauli_parameter_count),
            "logical_parameter_count": int(state.logical_parameter_count),
        }
        summary.update(reference_energy_summary(rows))
        summary.update(reference_energy_summary(rows, field_prefix="seed_", summary_prefix="seed_"))
        return summary
    residuals = [
        float(row["mclachlan_residual_ratio"])
        for row in rows
        if row.get("mclachlan_residual_ratio") is not None
    ]
    summary = {
        "point_count": int(len(rows)),
        "time_initial": float(rows[0]["time"]),
        "time_final": float(rows[-1]["time"]),
        "energy_initial": float(rows[0]["energy_expectation"]),
        "energy_final": float(rows[-1]["energy_expectation"]),
        "parameterization_mode": str(state.parameterization_mode),
        "parameterization_label": str(state.parameterization_label),
        "active_parameter_count": int(state.active_parameter_count),
        "runtime_parameter_count": int(state.runtime_parameter_count),
        "runtime_pauli_parameter_count": int(state.runtime_pauli_parameter_count),
        "logical_parameter_count": int(state.logical_parameter_count),
        "candidate_pool_term_count": int(len(state.candidate_pool_terms)),
        "drive_enabled": bool(hamiltonian.drive_enabled),
        "integrator_method": str(integrator_method).lower(),
        "pinv_rcond": float(inverse_policy.pinv_rcond),
        "ridge_lambda": float(inverse_policy.ridge_lambda),
        "solve_damping": float(inverse_policy.solve_damping),
        "max_mclachlan_residual_ratio": (
            None if not residuals else float(max(residuals))
        ),
        "final_mclachlan_residual_ratio": (
            None if not residuals else float(residuals[-1])
        ),
    }
    summary.update(reference_energy_summary(rows))
    summary.update(reference_energy_summary(rows, field_prefix="seed_", summary_prefix="seed_"))
    return summary


def _parse_times(args: argparse.Namespace) -> tuple[float, ...]:
    raw_times = getattr(args, "times", None)
    if raw_times not in {None, ""}:
        times = tuple(float(chunk.strip()) for chunk in str(raw_times).split(",") if chunk.strip())
    else:
        num_times = int(args.num_times)
        if num_times <= 0:
            raise ValueError("--num-times must be positive.")
        times = tuple(float(x) for x in np.linspace(0.0, float(args.t_final), num_times))
    if not times:
        raise ValueError("time grid must contain at least one value.")
    if any(not np.isfinite(float(t)) for t in times):
        raise ValueError("time grid must contain only finite values.")
    if any(float(right) < float(left) for left, right in zip(times[:-1], times[1:])):
        raise ValueError("time grid must be monotonically nondecreasing.")
    return times


def _parse_custom_weights(raw: str | None) -> tuple[float, ...] | None:
    if raw in {None, ""}:
        return None
    return tuple(float(chunk.strip()) for chunk in str(raw).split(",") if chunk.strip())


def _drive_config_from_args(args: argparse.Namespace, runtime_input: Any) -> Any | None:
    if not bool(args.enable_drive):
        return None
    resolved = getattr(runtime_input, "resolved_problem", None)
    request = getattr(resolved, "request", None)
    return SimpleNamespace(
        enabled=True,
        n_sites=int(args.drive_n_sites if args.drive_n_sites is not None else getattr(request, "num_sites", 1)),
        ordering=str(args.drive_ordering or getattr(request, "ordering", "blocked")),
        drive_A=float(args.drive_A),
        drive_omega=float(args.drive_omega),
        drive_tbar=float(args.drive_tbar),
        drive_phi=float(args.drive_phi),
        drive_pattern=str(args.drive_pattern),
        drive_custom_weights=_parse_custom_weights(args.drive_custom_weights),
        drive_include_identity=bool(args.drive_include_identity),
        drive_time_sampling=str(args.drive_time_sampling),
        drive_t0=float(args.drive_t0),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run fixed-support AP-McLachlan from a static scaffold artifact."
    )
    parser.add_argument("--artifact-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--loader-mode", default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--generator-family", default="match_adapt")
    parser.add_argument("--fallback-family", default="full_meta")
    parser.add_argument("--times", default=None, help="Comma-separated time grid. Overrides --t-final/--num-times.")
    parser.add_argument("--t-final", type=float, default=0.2)
    parser.add_argument("--num-times", type=int, default=3)
    parser.add_argument("--integrator", choices=SUPPORTED_INTEGRATORS, default=INTEGRATOR_EULER)
    parser.add_argument("--pinv-rcond", type=float, default=1.0e-10)
    parser.add_argument("--ridge-lambda", type=float, default=DEFAULT_MCLACHLAN_RIDGE_LAMBDA)
    parser.add_argument("--solve-damping", type=float, default=DEFAULT_MCLACHLAN_SOLVE_DAMPING)
    parser.add_argument(
        "--parameterization-mode",
        choices=AP_SUPPORTED_PARAMETERIZATION_MODES,
        default=AP_PARAMETERIZATION_PER_PAULI_TERM,
        help=(
            "AP variational coordinate mode: per_pauli_term is per Pauli/polynomial "
            "term; logical_shared is per logical/macro generator."
        ),
    )
    parser.add_argument("--enable-drive", action="store_true")
    parser.add_argument("--drive-A", type=float, default=0.0)
    parser.add_argument("--drive-omega", type=float, default=1.0)
    parser.add_argument("--drive-tbar", type=float, default=1.0)
    parser.add_argument("--drive-phi", type=float, default=0.0)
    parser.add_argument("--drive-pattern", default="staggered")
    parser.add_argument("--drive-custom-weights", default=None)
    parser.add_argument("--drive-include-identity", action="store_true")
    parser.add_argument("--drive-time-sampling", default="midpoint")
    parser.add_argument("--drive-t0", type=float, default=0.0)
    parser.add_argument("--drive-n-sites", type=int, default=None)
    parser.add_argument("--drive-ordering", default=None)
    parser.add_argument(
        "--drive-aligned-ansatz",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For driven AP-McLachlan, append the resolved drive operator as a "
            "zero-angle ansatz generator before propagation."
        ),
    )
    parser.add_argument("--reference-energy-json", default=None)
    parser.add_argument("--reference-energy-atol", type=float, default=1.0e-12)
    parser.add_argument(
        "--seed-reference-energy-json",
        default=None,
        help=(
            "Optional same-seed exact propagation reference for reporting only. "
            "This never enters AP decisions."
        ),
    )
    parser.add_argument("--seed-reference-energy-atol", type=float, default=1.0e-12)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    artifact_path = Path(args.artifact_json)
    try:
        runtime_input = _load_runtime_input_or_raise(
            artifact_path,
            loader_mode=args.loader_mode,
            tag=args.tag,
            generator_family=str(args.generator_family),
            fallback_family=str(args.fallback_family),
        )
        drive_config = _drive_config_from_args(args, runtime_input)
        payload = run_fixed_ap_mclachlan_from_runtime_input(
            runtime_input,
            times=_parse_times(args),
            integrator_method=str(args.integrator),
            pinv_rcond=float(args.pinv_rcond),
            ridge_lambda=float(args.ridge_lambda),
            solve_damping=float(args.solve_damping),
            enable_drive=bool(args.enable_drive),
            drive_config=drive_config,
            drive_aligned_ansatz=bool(args.drive_aligned_ansatz),
            parameterization_mode=str(args.parameterization_mode),
            reference_energy_trajectory=(
                None
                if args.reference_energy_json in {None, ""}
                else load_reference_energy_trajectory(args.reference_energy_json)
            ),
            reference_energy_atol=float(args.reference_energy_atol),
            seed_reference_energy_trajectory=(
                None
                if args.seed_reference_energy_json in {None, ""}
                else load_reference_energy_trajectory(args.seed_reference_energy_json)
            ),
            seed_reference_energy_atol=float(args.seed_reference_energy_atol),
            runner_metadata={
                "artifact_json": str(artifact_path),
                "loader_mode": args.loader_mode,
                "tag": args.tag,
                "generator_family": str(args.generator_family),
                "fallback_family": str(args.fallback_family),
                "parameterization_mode": str(args.parameterization_mode),
                "solve_damping": float(args.solve_damping),
                "drive_aligned_ansatz_requested": bool(args.drive_aligned_ansatz),
                "reference_energy_json": args.reference_energy_json,
                "seed_reference_energy_json": args.seed_reference_energy_json,
            },
        )
        payload["source_artifact_json"] = str(artifact_path)
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    except ValueError as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


def _load_runtime_input_or_raise(
    artifact_path: Path,
    *,
    loader_mode: str | None,
    tag: str | None,
    generator_family: str,
    fallback_family: str,
) -> Any:
    try:
        return load_scaffold_runtime_input(
            artifact_path,
            loader_mode=loader_mode,
            tag=tag,
            generator_family=str(generator_family),
            fallback_family=str(fallback_family),
        )
    except ValueError as exc:
        if "missing settings object" in str(exc):
            raise ValueError(
                "AP fixed runner requires a raw scaffold artifact JSON with "
                "`settings` and `adapt_vqe`/parameterization data. The provided "
                f"path looks like a wrapper/provenance JSON, not a runnable seed artifact: {artifact_path}"
            ) from exc
        raise


def _coerce_reference_energy_trajectory(value: Any | None) -> ReferenceEnergyTrajectory | None:
    if value is None:
        return None
    if isinstance(value, ReferenceEnergyTrajectory):
        return value
    if isinstance(value, Mapping):
        return reference_energy_trajectory_from_payload(value)
    raise TypeError("reference_energy_trajectory must be a ReferenceEnergyTrajectory, mapping, or None.")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    try:
        out = float(value)
    except (TypeError, ValueError):
        return str(value)
    return out if np.isfinite(out) else None


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RUNNER_SCHEMA_V1",
    "main",
    "run_fixed_ap_mclachlan_from_artifact",
    "run_fixed_ap_mclachlan_from_runtime_input",
]
