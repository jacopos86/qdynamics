"""Construction-only Gates A--C for the PURG reduced-state model.

This driver intentionally contains no exact driven propagation.  It selects or
rejects a compact candidate from its deterministic residual certificate and
nested-rank stability before any exact-score gate is allowed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .hubbard_dimer import DimerParameters
from .purg import (
    PurgCertificate,
    PurgConstruction,
    PurgConstructionSettings,
    PurgProjection,
    build_purg_construction,
    build_purg_operator_bounds,
    certify_purg_projection,
    purg_gate_a_diagnostics,
)
from .krylov_memory_closure import RAW_MOMENT_NAMES, raw_to_closed_jacobian

FloatArray = NDArray[np.float64]

_DEPLOYABLE_CAPS = (32, 64, 96)
_NEXT_CAP = {32: 64, 64: 96, 96: 128}
_BLOCK_SLICES: dict[str, slice] = {
    "rho": slice(0, 3),
    "B": slice(3, 7),
    "N": slice(7, 11),
    "A": slice(11, 17),
    "C": slice(17, 31),
}
_GATE_B_LIMITS = {
    "rho": (1e-4, 3e-4),
    "B": (0.0025, 0.0075),
    "N": (0.0025, 0.0075),
    "A": (0.0025, 0.0075),
    "C": (0.025, 0.075),
}
_GATE_C_LIMITS = {
    "rho": (0.002, 0.006),
    "B": (0.002, 0.006),
    "N": (0.002, 0.006),
    "A": (0.002, 0.006),
    "C": (0.01, 0.03),
}


@dataclass(frozen=True)
class PurgConstructionGateResult:
    """Machine-readable result of construction-only PURG Gates A--C."""

    construction: PurgConstruction
    gate_a: dict[str, Any]
    gate_b: dict[int, dict[str, Any]]
    gate_c: dict[int, dict[str, Any]]
    certificates: dict[int, PurgCertificate]
    refined_certificates: dict[int, PurgCertificate]
    selected_cap: int | None
    passed: bool


def _vector_metrics(values: FloatArray) -> dict[str, float]:
    norms = np.linalg.norm(np.asarray(values, dtype=float), axis=1)
    return {
        "rms_l2": float(np.sqrt(np.mean(norms**2))),
        "max_l2": float(np.max(norms)),
    }


def _close_enough(first: float, second: float) -> bool:
    return bool(np.isclose(first, second, rtol=1e-4, atol=1e-8))


def _gate_a_passes(diagnostics: dict[str, Any]) -> bool:
    if diagnostics["ground_residual"] >= 1e-11:
        return False
    if diagnostics["maximum_shifted_solve_relative_residual"] >= 1e-11:
        return False
    if diagnostics["coordinate_map_round_trip_residual"] >= 1e-12:
        return False
    if diagnostics["coordinate_jacobian_directional_residual"] >= 1e-12:
        return False
    if not diagnostics["decoupled_force"]["passed"]:
        return False
    if not diagnostics["online_dependency_audit"]["passed"]:
        return False
    for cap in diagnostics["caps"].values():
        if not cap["available"]:
            continue
        if cap["orthogonality_residual"] >= 1e-12:
            return False
        if cap["initial_state_containment_residual"] >= 1e-11:
            return False
        if cap["initial_drive_direction_containment_residual"] >= 1e-11:
            return False
        if cap["static_hermitian_leakage"] >= 1e-12:
            return False
        if cap["drive_hermitian_leakage"] >= 1e-12:
            return False
        if cap["raw_observable_hermitian_leakage"] >= 1e-12:
            return False
    return True


def _directional_derivative_residual(
    projection: PurgProjection,
    *,
    drive_value: float,
    seed: int,
) -> float:
    """Use an exact five-point derivative for the quartic centered map."""

    model = projection.model
    rng = np.random.default_rng(seed)
    state = rng.normal(size=model.dimension) + 1j * rng.normal(
        size=model.dimension
    )
    state /= np.linalg.norm(state)
    velocity = model.rhs(state, drive_value=drive_value)
    step = 1e-3
    values = {
        multiplier: model.centered_coordinates(
            state + multiplier * step * velocity
        )
        for multiplier in (-2.0, -1.0, 1.0, 2.0)
    }
    directional = (
        values[-2.0]
        - 8.0 * values[-1.0]
        + 8.0 * values[1.0]
        - values[2.0]
    ) / (12.0 * step)
    return float(
        np.linalg.norm(
            directional
            - model.centered_velocity(state, drive_value=drive_value)
        )
    )


def _gate_b_result(
    certificate: PurgCertificate,
    refined: PurgCertificate,
) -> dict[str, Any]:
    block_pass = {
        name: (
            certificate.block_derivative_metrics[name]["rms_l2_bound"]
            <= limits[0]
            and certificate.block_derivative_metrics[name]["max_l2_bound"]
            <= limits[1]
        )
        for name, limits in _GATE_B_LIMITS.items()
    }
    repeated_quantities = {
        "final_state_error_bound": (
            float(certificate.state_error_bound[-1]),
            float(refined.state_error_bound[-1]),
        ),
        "maximum_projection_residual": (
            float(np.max(certificate.projection_residual_norms)),
            float(np.max(refined.projection_residual_norms)),
        ),
        "maximum_total_defect": (
            float(np.max(certificate.total_defect_norms)),
            float(np.max(refined.total_defect_norms)),
        ),
    }
    repeat_pass = all(
        _close_enough(first, second)
        for first, second in repeated_quantities.values()
    )
    for name in _GATE_B_LIMITS:
        for metric in ("rms_l2_bound", "max_l2_bound"):
            repeat_pass = repeat_pass and _close_enough(
                certificate.block_derivative_metrics[name][metric],
                refined.block_derivative_metrics[name][metric],
            )
    quadrature_pass = (
        certificate.quadrature_error_estimate <= 1e-12
        and refined.quadrature_error_estimate <= 1e-12
    )
    norm_pass = (
        certificate.continuous_norm_defect <= 1e-13
        and refined.continuous_norm_defect <= 1e-13
    )
    state_pass = bool(certificate.state_error_bound[-1] <= 2.5e-3)
    passed = bool(
        state_pass
        and all(block_pass.values())
        and repeat_pass
        and quadrature_pass
        and norm_pass
    )
    return {
        "passed": passed,
        "state_bound_passed": state_pass,
        "final_state_error_bound": float(certificate.state_error_bound[-1]),
        "block_passed": block_pass,
        "block_derivative_metrics": certificate.block_derivative_metrics,
        "step_repeat_passed": repeat_pass,
        "step_repeat_quantities": {
            name: {"coarse": first, "refined": second}
            for name, (first, second) in repeated_quantities.items()
        },
        "quadrature_passed": quadrature_pass,
        "quadrature_error_estimate": certificate.quadrature_error_estimate,
        "refined_quadrature_error_estimate": (
            refined.quadrature_error_estimate
        ),
        "norm_passed": norm_pass,
        "continuous_norm_defect": certificate.continuous_norm_defect,
        "refined_continuous_norm_defect": refined.continuous_norm_defect,
    }


def _sample_centered_derivatives(
    projection: PurgProjection,
    certificate: PurgCertificate,
    parameters: DimerParameters,
    *,
    sample_step: float,
) -> tuple[FloatArray, FloatArray]:
    ratio = int(round(sample_step / (certificate.times[1] - certificate.times[0])))
    if ratio <= 0 or not np.isclose(
        ratio * (certificate.times[1] - certificate.times[0]),
        sample_step,
        atol=1e-13,
    ):
        raise ValueError("sample_step must be an integer multiple of certificate step")
    indices = np.arange(0, certificate.times.size, ratio, dtype=int)
    times = certificate.times[indices]
    derivatives = np.asarray(
        [
            projection.model.centered_velocity(
                certificate.states[index],
                drive_value=parameters.drive_difference(float(time)),
            )
            for index, time in zip(indices, times, strict=True)
        ],
        dtype=float,
    )
    return times, derivatives


def _gate_c_result(
    candidate_projection: PurgProjection,
    candidate_certificate: PurgCertificate,
    next_projection: PurgProjection,
    next_certificate: PurgCertificate,
    parameters: DimerParameters,
    *,
    sample_step: float,
) -> dict[str, Any]:
    candidate_times, candidate_derivatives = _sample_centered_derivatives(
        candidate_projection,
        candidate_certificate,
        parameters,
        sample_step=sample_step,
    )
    next_times, next_derivatives = _sample_centered_derivatives(
        next_projection,
        next_certificate,
        parameters,
        sample_step=sample_step,
    )
    np.testing.assert_allclose(candidate_times, next_times, atol=1e-13, rtol=0.0)
    difference = candidate_derivatives - next_derivatives
    block_metrics = {
        name: _vector_metrics(difference[:, block_slice])
        for name, block_slice in _BLOCK_SLICES.items()
    }
    block_pass = {
        name: (
            block_metrics[name]["rms_l2"] <= limits[0]
            and block_metrics[name]["max_l2"] <= limits[1]
        )
        for name, limits in _GATE_C_LIMITS.items()
    }
    state_bound_nonincrease = bool(
        next_certificate.state_error_bound[-1]
        <= 1.05 * candidate_certificate.state_error_bound[-1]
    )
    return {
        "passed": bool(all(block_pass.values()) and state_bound_nonincrease),
        "block_passed": block_pass,
        "block_derivative_difference": block_metrics,
        "state_bound_nonincrease_passed": state_bound_nonincrease,
        "candidate_final_state_error_bound": float(
            candidate_certificate.state_error_bound[-1]
        ),
        "next_final_state_error_bound": float(
            next_certificate.state_error_bound[-1]
        ),
    }


def run_purg_construction_gate(
    parameters: DimerParameters,
    *,
    phonon_cutoff: int = 16,
    settings: PurgConstructionSettings | None = None,
    sample_step: float = 0.01,
) -> PurgConstructionGateResult:
    """Build PURG and execute only construction-authorized Gates A--C."""

    resolved = settings or PurgConstructionSettings()
    construction = build_purg_construction(
        parameters,
        phonon_cutoff=phonon_cutoff,
        settings=resolved,
    )
    gate_a = purg_gate_a_diagnostics(construction)
    gate_a["directional_derivative_residuals"] = {}
    for record in construction.records:
        if record.projection is None:
            continue
        gate_a["directional_derivative_residuals"][str(record.cap_label)] = (
            _directional_derivative_residual(
                record.projection,
                drive_value=0.37,
                seed=2026080300 + record.cap_label,
            )
        )
    gate_a["online_model_fields"] = sorted(
        next(
            record.projection.model.__dataclass_fields__
            for record in construction.records
            if record.projection is not None
        )
    )
    gate_a["passed"] = bool(
        _gate_a_passes(gate_a)
        and all(
            residual <= 1e-11
            for residual in gate_a["directional_derivative_residuals"].values()
        )
    )

    first_projection = next(
        (
            record.projection
            for record in construction.records
            if record.projection is not None
        ),
        None,
    )
    if first_projection is None:
        return PurgConstructionGateResult(
            construction=construction,
            gate_a=gate_a,
            gate_b={},
            gate_c={},
            certificates={},
            refined_certificates={},
            selected_cap=None,
            passed=False,
        )
    operator_bounds = build_purg_operator_bounds(first_projection)

    certificates: dict[int, PurgCertificate] = {}
    refined_certificates: dict[int, PurgCertificate] = {}
    gate_b: dict[int, dict[str, Any]] = {}
    for record in construction.records:
        if record.projection is None:
            gate_b[record.cap_label] = {
                "passed": False,
                "available": False,
                "actual_rank": record.actual_rank,
            }
            continue
        certificate = certify_purg_projection(
            record.projection,
            parameters,
            operator_bounds,
            final_time=resolved.final_time,
            step=resolved.construction_step,
        )
        refined = certify_purg_projection(
            record.projection,
            parameters,
            operator_bounds,
            final_time=resolved.final_time,
            step=0.5 * resolved.construction_step,
        )
        certificates[record.cap_label] = certificate
        refined_certificates[record.cap_label] = refined
        gate_b[record.cap_label] = {
            "available": True,
            "actual_rank": record.actual_rank,
            **_gate_b_result(certificate, refined),
        }

    gate_c: dict[int, dict[str, Any]] = {}
    selected_cap: int | None = None
    available_labels = {record.cap_label for record in construction.records}
    for cap_label in _DEPLOYABLE_CAPS:
        next_cap = _NEXT_CAP[cap_label]
        if cap_label not in available_labels or next_cap not in available_labels:
            continue
        candidate_record = construction.record(cap_label)
        next_record = construction.record(next_cap)
        if candidate_record.projection is None or next_record.projection is None:
            continue
        comparison = _gate_c_result(
            candidate_record.projection,
            certificates[cap_label],
            next_record.projection,
            certificates[next_cap],
            parameters,
            sample_step=sample_step,
        )
        gate_c[cap_label] = {"next_cap": next_cap, **comparison}
        if (
            selected_cap is None
            and gate_a["passed"]
            and gate_b[cap_label]["passed"]
            and comparison["passed"]
        ):
            selected_cap = cap_label

    return PurgConstructionGateResult(
        construction=construction,
        gate_a=gate_a,
        gate_b=gate_b,
        gate_c=gate_c,
        certificates=certificates,
        refined_certificates=refined_certificates,
        selected_cap=selected_cap,
        passed=selected_cap is not None,
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_purg_construction_gate_artifact(
    result: PurgConstructionGateResult,
    output_directory: Path,
) -> None:
    """Write a summary, arrays, and source manifest for Gates A--C."""

    output_directory.mkdir(parents=True, exist_ok=False)
    construction = result.construction
    summary = {
        "method": "preparation_conditioned_unitary_residual_galerkin",
        "scope": "construction_only_gates_a_to_c",
        "passed": result.passed,
        "selected_cap": result.selected_cap,
        "parameters": asdict(construction.parameters),
        "phonon_cutoff": construction.phonon_cutoff,
        "settings": asdict(construction.settings),
        "initial_rank": construction.initial_rank,
        "records": [
            {
                "cap_label": record.cap_label,
                "actual_rank": record.actual_rank,
                "available": record.projection is not None,
                "residual_peak": record.residual_peak,
                "residual_peak_time": record.residual_peak_time,
                "greedy_packets": record.greedy_packets,
                "deflated_columns": record.deflated_columns,
                "truncated_columns": record.truncated_columns,
            }
            for record in construction.records
        ],
        "gate_a": result.gate_a,
        "gate_b": result.gate_b,
        "gate_c": result.gate_c,
        "authorization": {
            "exact_score_gates_d_to_f": bool(result.passed),
            "short_rollout_gate_g": False,
            "long_horizon": False,
        },
    }
    (output_directory / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    raw_dimension = len(RAW_MOMENT_NAMES)
    zero = np.zeros(raw_dimension, dtype=float)
    jacobian_at_zero = raw_to_closed_jacobian(zero)
    hessian = np.empty((31, raw_dimension, raw_dimension), dtype=float)
    for column in range(raw_dimension):
        direction = np.zeros(raw_dimension, dtype=float)
        direction[column] = 1.0
        hessian[:, :, column] = (
            raw_to_closed_jacobian(direction) - jacobian_at_zero
        )
    hessian = 0.5 * (hessian + np.swapaxes(hessian, 1, 2))

    arrays: dict[str, np.ndarray] = {
        "ground_state": construction.ground_state,
        "shifted_solve_relative_residuals": (
            construction.solve_relative_residuals
        ),
        "raw_to_centered_jacobian_at_zero": jacobian_at_zero,
        "raw_to_centered_hessian": hessian,
    }
    for record in construction.records:
        if record.projection is None:
            continue
        prefix = f"cap_{record.cap_label}"
        projection = record.projection
        arrays[f"{prefix}_basis"] = projection.basis
        arrays[f"{prefix}_static_hamiltonian"] = (
            projection.model.static_hamiltonian
        )
        arrays[f"{prefix}_drive_hamiltonian"] = (
            projection.model.drive_hamiltonian
        )
        arrays[f"{prefix}_raw_observables"] = (
            projection.model.raw_observables
        )
        arrays[f"{prefix}_initial_state"] = projection.model.initial_state
        arrays[f"{prefix}_static_residual_gram"] = (
            projection.static_residual_gram
        )
        arrays[f"{prefix}_cross_residual_gram"] = (
            projection.cross_residual_gram
        )
        arrays[f"{prefix}_drive_residual_gram"] = (
            projection.drive_residual_gram
        )
    for cap_label, certificate in result.certificates.items():
        prefix = f"cap_{cap_label}"
        arrays[f"{prefix}_times"] = certificate.times
        arrays[f"{prefix}_states"] = certificate.states
        arrays[f"{prefix}_projection_residual_norms"] = (
            certificate.projection_residual_norms
        )
        arrays[f"{prefix}_total_defect_norms"] = certificate.total_defect_norms
        arrays[f"{prefix}_state_error_bound"] = certificate.state_error_bound
        arrays[f"{prefix}_centered_derivative_bounds"] = (
            certificate.centered_derivative_absolute_bounds
        )
        refined = result.refined_certificates[cap_label]
        arrays[f"{prefix}_refined_times"] = refined.times
        arrays[f"{prefix}_refined_state_error_bound"] = (
            refined.state_error_bound
        )
        arrays[f"{prefix}_refined_centered_derivative_bounds"] = (
            refined.centered_derivative_absolute_bounds
        )
    np.savez_compressed(output_directory / "arrays.npz", **arrays)

    source_directory = Path(__file__).resolve().parent
    source_paths = (
        source_directory / "purg.py",
        source_directory / "purg_analysis.py",
        source_directory / "exact_reference.py",
        source_directory / "krylov_memory_closure.py",
        source_directory / "hubbard_dimer.py",
    )
    manifest = {
        "files": {
            str(path): {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in source_paths
        },
        "contains_exact_driven_trajectory": False,
        "contains_controller_feedback": False,
        "raw_moment_names": list(RAW_MOMENT_NAMES),
    }
    (output_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phonon-cutoff", type=int, default=16)
    parser.add_argument("--final-time", type=float, default=4.0)
    parser.add_argument("--construction-step", type=float, default=0.0025)
    parser.add_argument("--caps", type=int, nargs="+", default=[32, 64, 96, 128])
    parser.add_argument("--lambda-ep", type=float, default=1.5)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--drive", type=float, default=1.0)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    parameters = DimerParameters(
        lambda_ep=args.lambda_ep,
        gamma=args.gamma,
        drive_amplitude=args.drive,
    )
    settings = PurgConstructionSettings(
        caps=tuple(args.caps),
        final_time=args.final_time,
        construction_step=args.construction_step,
    )
    result = run_purg_construction_gate(
        parameters,
        phonon_cutoff=args.phonon_cutoff,
        settings=settings,
    )
    write_purg_construction_gate_artifact(result, args.output_directory)
    report = {
        "passed": result.passed,
        "selected_cap": result.selected_cap,
        "gate_a_passed": result.gate_a["passed"],
        "gate_b": {
            str(cap): gate["passed"] for cap, gate in result.gate_b.items()
        },
        "gate_c": {
            str(cap): gate["passed"] for cap, gate in result.gate_c.items()
        },
        "output_directory": str(args.output_directory.resolve()),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
