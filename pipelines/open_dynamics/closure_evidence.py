"""Matched closure-versus-coherent evidence from immutable Paper V runs.

The input is a completed ``paper5.stability.electron_phonon_analysis`` run.
This module verifies the recorded artifact hashes, reuses its exact/raw/
corrected trajectories without altering them, and adds the missing matched
five-coordinate Ehrenfest comparator.  Exact arrays are reporting inputs only;
they never enter any autonomous right-hand side after the shared initial lower
moments have been selected.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from itertools import product
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
EVIDENCE_SCHEMA_VERSION = "open_dynamics.closure_evidence.v1"
EVIDENCE_STATUS = "exploratory_local_not_promoted"
_GRAM_TOLERANCE = 1.0e-8
_MATERIAL_ERROR_FRACTION = 0.1
_SOURCE_POINT = (0.5, 0.5, 1.0)


class EvidenceSourceError(RuntimeError):
    """Raised when an immutable source run cannot be trusted or interpreted."""


@dataclass(frozen=True, slots=True)
class ClosureEvidenceResult:
    """Machine-readable summary and matched trajectory arrays."""

    summary: Mapping[str, Any]
    arrays: Mapping[str, NDArray[np.generic]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise EvidenceSourceError(f"{path} must contain a JSON object")
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_npz_atomic(
    path: Path,
    arrays: Mapping[str, NDArray[np.generic]],
) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _verified_source(
    run_directory: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    plan_path = run_directory / "plan.json"
    summary_path = run_directory / "summary.json"
    runtime_path = run_directory / "runtime_manifest.json"
    arrays_path = run_directory / "parameter_grid_trajectories.npz"
    for path in (plan_path, summary_path, runtime_path, arrays_path):
        if not path.is_file():
            raise EvidenceSourceError(f"missing source artifact: {path}")

    runtime = _read_json(runtime_path)
    recorded = runtime.get("artifact_hashes")
    if not isinstance(recorded, dict):
        raise EvidenceSourceError("runtime manifest lacks artifact hashes")
    verified: dict[str, str] = {}
    for path in (plan_path, summary_path, arrays_path):
        expected = recorded.get(path.name)
        if not isinstance(expected, str):
            raise EvidenceSourceError(f"no recorded hash for {path.name}")
        actual = _sha256(path)
        if actual != expected:
            raise EvidenceSourceError(
                f"source hash mismatch for {path.name}: {actual} != {expected}"
            )
        verified[path.name] = actual
    plan = _read_json(plan_path)
    source_summary = _read_json(summary_path)
    if runtime.get("status") != "complete" or source_summary.get("status") != "complete":
        raise EvidenceSourceError("source run must be complete")
    if runtime.get("evidence_status") != EVIDENCE_STATUS:
        raise EvidenceSourceError(
            "source run must remain exploratory_local_not_promoted"
        )
    if "never" not in str(runtime.get("exact_reference_usage", "")).lower():
        raise EvidenceSourceError(
            "source run does not declare exact-reference decision isolation"
        )
    return plan, source_summary, runtime, verified


def _paper5_imports() -> dict[str, object]:
    try:
        from paper5.stability.hubbard_dimer import DimerParameters, ehrenfest_rhs
        from paper5.stability.matrix_reference import (
            closed_scalar_to_matrix_state,
            electron_phonon_moment_matrix,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "paper5" or str(exc.name).startswith("paper5."):
            raise ImportError(
                "install paper_5 or include paper_5/src in PYTHONPATH"
            ) from exc
        raise
    return {
        "DimerParameters": DimerParameters,
        "closed_scalar_to_matrix_state": closed_scalar_to_matrix_state,
        "electron_phonon_moment_matrix": electron_phonon_moment_matrix,
        "ehrenfest_rhs": ehrenfest_rhs,
    }


def _case_key(lambda_ep: float, gamma: float, drive: float) -> str:
    return (
        f"lambda_{lambda_ep:g}__gamma_{gamma:g}__drive_{drive:g}"
    ).replace(".", "p")


def _closed_fields(
    coordinates: FloatArray,
    *,
    closed_scalar_to_matrix_state: object,
    electron_phonon_moment_matrix: object,
) -> tuple[ComplexArray, ComplexArray, FloatArray]:
    states = [
        closed_scalar_to_matrix_state(row)  # type: ignore[operator]
        for row in np.asarray(coordinates, dtype=float)
    ]
    electron = np.asarray(
        [state.electron_density for state in states], dtype=np.complex128
    )
    coherent = np.asarray(
        [state.coherent_phonon for state in states], dtype=np.complex128
    )
    gram = np.asarray(
        [
            np.linalg.eigvalsh(
                electron_phonon_moment_matrix(state)  # type: ignore[operator]
            )[0]
            for state in states
        ],
        dtype=float,
    )
    return electron, coherent, gram


def _coherent_fields(
    *,
    times: FloatArray,
    exact_electron: ComplexArray,
    exact_coherent: ComplexArray,
    parameters: object,
    ehrenfest_rhs: object,
    relative_tolerance: float,
    absolute_tolerance: float,
    maximum_step: float,
) -> tuple[ComplexArray, ComplexArray, int]:
    from scipy.integrate import solve_ivp

    electron_initial = exact_electron[0]
    coherent_initial = exact_coherent[0]
    relative_initial = coherent_initial[0] - coherent_initial[1]
    initial = np.asarray(
        [
            (electron_initial[0, 0] - electron_initial[1, 1]).real,
            electron_initial[0, 1].real,
            electron_initial[0, 1].imag,
            relative_initial.real,
            relative_initial.imag,
        ],
        dtype=float,
    )
    center = complex(np.mean(coherent_initial))
    solution = solve_ivp(
        lambda time, state: ehrenfest_rhs(  # type: ignore[operator]
            time, state, parameters
        ),
        (float(times[0]), float(times[-1])),
        initial,
        method="DOP853",
        t_eval=times,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
        max_step=maximum_step,
    )
    if not solution.success or solution.y.shape[1] != times.size:
        raise RuntimeError(
            f"coherent-only propagation failed: {solution.message}"
        )

    values = np.asarray(solution.y.T, dtype=float)
    electron = np.empty((times.size, 2, 2), dtype=np.complex128)
    coherent = np.empty((times.size, 2), dtype=np.complex128)
    for index, row in enumerate(values):
        delta_n, rho_real, rho_imag, delta_b_real, delta_b_imag = row
        electron[index] = np.asarray(
            [
                [0.5 * (1.0 + delta_n), rho_real + 1j * rho_imag],
                [rho_real - 1j * rho_imag, 0.5 * (1.0 - delta_n)],
            ],
            dtype=np.complex128,
        )
        relative = delta_b_real + 1j * delta_b_imag
        coherent[index] = np.asarray(
            [center + 0.5 * relative, center - 0.5 * relative],
            dtype=np.complex128,
        )
    return electron, coherent, int(solution.nfev)


def _series_metrics(
    reference: NDArray[np.generic],
    candidate: NDArray[np.generic],
) -> tuple[dict[str, float], FloatArray, float]:
    reference_array = np.asarray(reference)
    candidate_array = np.asarray(candidate)
    if reference_array.shape != candidate_array.shape:
        raise ValueError("reference and candidate arrays must match")
    sample_count = reference_array.shape[0]
    errors = np.linalg.norm(
        (candidate_array - reference_array).reshape(sample_count, -1), axis=1
    )
    changes = np.linalg.norm(
        (reference_array - reference_array[:1]).reshape(sample_count, -1),
        axis=1,
    )
    dynamic_rms = float(np.sqrt(np.mean(changes**2)))
    rms_error = float(np.sqrt(np.mean(errors**2)))
    return (
        {
            "maximum_frobenius_error": float(np.max(errors)),
            "final_frobenius_error": float(errors[-1]),
            "rms_frobenius_error": rms_error,
            "exact_dynamic_rms_scale": dynamic_rms,
            "rms_error_over_exact_dynamic_rms": (
                rms_error / max(dynamic_rms, 1.0e-14)
            ),
        },
        np.asarray(errors, dtype=float),
        dynamic_rms,
    )


def _maximum_series_difference(
    first: NDArray[np.generic],
    second: NDArray[np.generic],
) -> float:
    first_array = np.asarray(first)
    second_array = np.asarray(second)
    if first_array.shape != second_array.shape:
        raise ValueError("series shapes must match")
    return float(
        np.max(
            np.linalg.norm(
                (first_array - second_array).reshape(first_array.shape[0], -1),
                axis=1,
            )
        )
    )
def _polarization(electron: ComplexArray) -> FloatArray:
    values = 0.5 * (electron[:, 1, 1] - electron[:, 0, 0])
    if float(np.max(np.abs(values.imag))) > 1.0e-10:
        raise ValueError("polarization contains an unexpected imaginary part")
    return np.asarray(values.real, dtype=float)


def _first_time_below(
    times: FloatArray,
    values: FloatArray,
    threshold: float,
) -> float | None:
    indices = np.flatnonzero(values < threshold)
    return float(times[indices[0]]) if indices.size else None


def _first_material_error_time(
    times: FloatArray,
    errors: FloatArray,
    dynamic_rms: float,
) -> float | None:
    normalized = errors / max(dynamic_rms, 1.0e-14)
    indices = np.flatnonzero(normalized > _MATERIAL_ERROR_FRACTION)
    return float(times[indices[0]]) if indices.size else None


def _spearman(x: list[float], y: list[float]) -> dict[str, float | int | None]:
    from scipy.stats import spearmanr

    if len(x) < 3 or np.ptp(np.asarray(x, dtype=float)) == 0.0:
        return {"sample_count": len(x), "statistic": None, "pvalue": None}
    result = spearmanr(x, y)
    return {
        "sample_count": len(x),
        "statistic": float(result.statistic),
        "pvalue": float(result.pvalue),
    }


def analyze_closure_evidence(
    source_run: str | Path,
) -> ClosureEvidenceResult:
    """Add a matched coherent comparator to a verified completed grid run."""

    run_directory = Path(source_run).resolve()
    plan, source_summary, runtime, verified_hashes = _verified_source(
        run_directory
    )
    imports = _paper5_imports()
    DimerParameters = imports["DimerParameters"]
    closed_to_matrix = imports["closed_scalar_to_matrix_state"]
    joint_gram = imports["electron_phonon_moment_matrix"]
    ehrenfest_rhs = imports["ehrenfest_rhs"]

    grid = plan.get("parameter_grid")
    integration = plan.get("integration")
    baseline = plan.get("baseline_parameters")
    if not all(isinstance(item, dict) for item in (grid, integration, baseline)):
        raise EvidenceSourceError("source plan lacks grid or integration metadata")
    parameter_axes = (
        grid.get("lambda_ep"),
        grid.get("gamma"),
        grid.get("drive_amplitude"),
    )
    if not all(isinstance(axis, list) and axis for axis in parameter_axes):
        raise EvidenceSourceError("source parameter grid must contain three axes")

    relative_tolerance = float(integration["exact_relative_tolerance"])
    absolute_tolerance = float(integration["exact_absolute_tolerance"])
    maximum_step = min(
        float(integration["exact_maximum_step"]),
        float(grid["time_step"]),
    )
    cases: list[dict[str, Any]] = []
    arrays: dict[str, NDArray[np.generic]] = {}
    source_arrays_path = run_directory / "parameter_grid_trajectories.npz"
    with np.load(source_arrays_path, allow_pickle=False) as source_arrays:
        for lambda_ep, gamma, drive in product(*parameter_axes):
            values = (float(lambda_ep), float(gamma), float(drive))
            key = _case_key(*values)
            required_keys = tuple(
                f"{key}__{suffix}"
                for suffix in ("times", "exact", "raw", "corrected")
            )
            missing = [name for name in required_keys if name not in source_arrays]
            if missing:
                raise EvidenceSourceError(
                    f"source trajectory archive lacks {missing}"
                )
            times = np.asarray(source_arrays[f"{key}__times"], dtype=float)
            exact_coordinates = np.asarray(
                source_arrays[f"{key}__exact"], dtype=float
            )
            raw_coordinates = np.asarray(
                source_arrays[f"{key}__raw"], dtype=float
            )
            corrected_coordinates = np.asarray(
                source_arrays[f"{key}__corrected"], dtype=float
            )
            if (
                times.ndim != 1
                or times.size < 2
                or np.any(np.diff(times) <= 0.0)
                or exact_coordinates.shape != raw_coordinates.shape
                or exact_coordinates.shape != corrected_coordinates.shape
                or exact_coordinates.shape != (times.size, 31)
            ):
                raise EvidenceSourceError(f"malformed source case {key}")

            exact_rho, exact_b, exact_gram = _closed_fields(
                exact_coordinates,
                closed_scalar_to_matrix_state=closed_to_matrix,
                electron_phonon_moment_matrix=joint_gram,
            )
            raw_rho, raw_b, raw_gram = _closed_fields(
                raw_coordinates,
                closed_scalar_to_matrix_state=closed_to_matrix,
                electron_phonon_moment_matrix=joint_gram,
            )
            corrected_rho, corrected_b, corrected_gram = _closed_fields(
                corrected_coordinates,
                closed_scalar_to_matrix_state=closed_to_matrix,
                electron_phonon_moment_matrix=joint_gram,
            )
            parameters = DimerParameters(  # type: ignore[operator]
                hopping=float(baseline["hopping"]),
                lambda_ep=values[0],
                gamma=values[1],
                drive_amplitude=values[2],
                pulse_width=float(baseline["pulse_width"]),
            )
            coherent_rho, coherent_b, coherent_nfev = _coherent_fields(
                times=times,
                exact_electron=exact_rho,
                exact_coherent=exact_b,
                parameters=parameters,
                ehrenfest_rhs=ehrenfest_rhs,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
                maximum_step=maximum_step,
            )
            refined_coherent_rho, refined_coherent_b, refined_coherent_nfev = (
                _coherent_fields(
                    times=times,
                    exact_electron=exact_rho,
                    exact_coherent=exact_b,
                    parameters=parameters,
                    ehrenfest_rhs=ehrenfest_rhs,
                    relative_tolerance=relative_tolerance,
                    absolute_tolerance=absolute_tolerance,
                    maximum_step=0.5 * maximum_step,
                )
            )

            method_fields = {
                "coherent_only": (coherent_rho, coherent_b),
                "raw_closure": (raw_rho, raw_b),
                "gram_corrected_closure": (corrected_rho, corrected_b),
            }
            method_metrics: dict[str, Any] = {}
            rho_errors: dict[str, FloatArray] = {}
            rho_scales: dict[str, float] = {}
            for method, (candidate_rho, candidate_b) in method_fields.items():
                rho_metrics, rho_error, rho_scale = _series_metrics(
                    exact_rho, candidate_rho
                )
                b_metrics, _, _ = _series_metrics(exact_b, candidate_b)
                polarization_metrics, _, _ = _series_metrics(
                    _polarization(exact_rho)[:, None],
                    _polarization(candidate_rho)[:, None],
                )
                method_metrics[method] = {
                    "electron_1rdm": rho_metrics,
                    "coherent_phonon_amplitude": b_metrics,
                    "polarization": polarization_metrics,
                }
                rho_errors[method] = rho_error
                rho_scales[method] = rho_scale
            method_metrics["coherent_only"]["function_evaluations"] = (
                coherent_nfev
            )
            method_metrics["coherent_only"]["step_refinement"] = {
                "refined_maximum_step": 0.5 * maximum_step,
                "refined_function_evaluations": refined_coherent_nfev,
                "maximum_electron_1rdm_frobenius_difference": (
                    _maximum_series_difference(
                        coherent_rho, refined_coherent_rho
                    )
                ),
                "maximum_coherent_phonon_frobenius_difference": (
                    _maximum_series_difference(
                        coherent_b, refined_coherent_b
                    )
                ),
            }

            raw_rho_ratio = method_metrics["raw_closure"]["electron_1rdm"][
                "rms_error_over_exact_dynamic_rms"
            ]
            coherent_rho_ratio = method_metrics["coherent_only"][
                "electron_1rdm"
            ]["rms_error_over_exact_dynamic_rms"]
            corrected_rho_ratio = method_metrics["gram_corrected_closure"][
                "electron_1rdm"
            ]["rms_error_over_exact_dynamic_rms"]
            raw_b_ratio = method_metrics["raw_closure"][
                "coherent_phonon_amplitude"
            ]["rms_error_over_exact_dynamic_rms"]
            coherent_b_ratio = method_metrics["coherent_only"][
                "coherent_phonon_amplitude"
            ]["rms_error_over_exact_dynamic_rms"]
            corrected_b_ratio = method_metrics["gram_corrected_closure"][
                "coherent_phonon_amplitude"
            ]["rms_error_over_exact_dynamic_rms"]
            first_gram_loss = _first_time_below(
                times, raw_gram, -_GRAM_TOLERANCE
            )
            first_rho_error = _first_material_error_time(
                times,
                rho_errors["raw_closure"],
                rho_scales["raw_closure"],
            )
            cases.append(
                {
                    "case_id": key,
                    "parameters": {
                        "lambda_ep": values[0],
                        "gamma": values[1],
                        "drive_amplitude": values[2],
                    },
                    "sample_count": int(times.size),
                    "final_time": float(times[-1]),
                    "methods": method_metrics,
                    "comparisons": {
                        "raw_to_coherent_rho_rms_ratio": (
                            raw_rho_ratio / max(coherent_rho_ratio, 1.0e-14)
                        ),
                        "corrected_to_coherent_rho_rms_ratio": (
                            corrected_rho_ratio
                            / max(coherent_rho_ratio, 1.0e-14)
                        ),
                        "raw_to_coherent_B_rms_ratio": (
                            raw_b_ratio / max(coherent_b_ratio, 1.0e-14)
                        ),
                        "corrected_to_coherent_B_rms_ratio": (
                            corrected_b_ratio / max(coherent_b_ratio, 1.0e-14)
                        ),
                    },
                    "representability": {
                        "exact_minimum_joint_gram_eigenvalue": float(
                            np.min(exact_gram)
                        ),
                        "raw_minimum_joint_gram_eigenvalue": float(
                            np.min(raw_gram)
                        ),
                        "corrected_minimum_joint_gram_eigenvalue": float(
                            np.min(corrected_gram)
                        ),
                        "first_raw_joint_gram_below_minus_tolerance": (
                            first_gram_loss
                        ),
                        "first_raw_rho_material_error": first_rho_error,
                        "gram_lead_time_to_rho_error": (
                            first_rho_error - first_gram_loss
                            if first_gram_loss is not None
                            and first_rho_error is not None
                            else None
                        ),
                    },
                }
            )

            arrays.update(
                {
                    f"{key}__time": times,
                    f"{key}__exact_electron_1rdm": exact_rho,
                    f"{key}__raw_electron_1rdm": raw_rho,
                    f"{key}__corrected_electron_1rdm": corrected_rho,
                    f"{key}__coherent_electron_1rdm": coherent_rho,
                    f"{key}__exact_coherent_phonon": exact_b,
                    f"{key}__raw_coherent_phonon": raw_b,
                    f"{key}__corrected_coherent_phonon": corrected_b,
                    f"{key}__coherent_coherent_phonon": coherent_b,
                    f"{key}__exact_joint_gram_minimum": exact_gram,
                    f"{key}__raw_joint_gram_minimum": raw_gram,
                    f"{key}__corrected_joint_gram_minimum": corrected_gram,
                }
            )

    expected_count = int(np.prod([len(axis) for axis in parameter_axes]))
    if len(cases) != expected_count:
        raise EvidenceSourceError(
            f"analyzed {len(cases)} cases, expected {expected_count}"
        )
    source_point = next(
        (
            case
            for case in cases
            if all(
                np.isclose(case["parameters"][name], value)
                for name, value in zip(
                    ("lambda_ep", "gamma", "drive_amplitude"),
                    _SOURCE_POINT,
                    strict=True,
                )
            )
        ),
        None,
    )
    if source_point is None:
        raise EvidenceSourceError("source-anchored comparison point is absent")

    raw_better_rho = sum(
        case["comparisons"]["raw_to_coherent_rho_rms_ratio"] < 1.0
        for case in cases
    )
    corrected_better_rho = sum(
        case["comparisons"]["corrected_to_coherent_rho_rms_ratio"] < 1.0
        for case in cases
    )
    gram_loss_count = sum(
        case["representability"][
            "first_raw_joint_gram_below_minus_tolerance"
        ]
        is not None
        for case in cases
    )
    gram_loss_without_material_error = sum(
        case["representability"][
            "first_raw_joint_gram_below_minus_tolerance"
        ]
        is not None
        and case["representability"]["first_raw_rho_material_error"] is None
        for case in cases
    )
    severity = [
        max(
            0.0,
            -case["representability"][
                "raw_minimum_joint_gram_eigenvalue"
            ],
        )
        for case in cases
    ]
    raw_rho_error = [
        case["methods"]["raw_closure"]["electron_1rdm"][
            "rms_error_over_exact_dynamic_rms"
        ]
        for case in cases
    ]
    raw_to_coherent = [
        case["comparisons"]["raw_to_coherent_rho_rms_ratio"]
        for case in cases
    ]
    associations: dict[str, Any] = {
        "pooled_violation_severity_vs_raw_rho_error": _spearman(
            severity, raw_rho_error
        ),
        "pooled_violation_severity_vs_raw_to_coherent_ratio": _spearman(
            severity, raw_to_coherent
        ),
        "stratified_by_lambda_ep": {},
    }
    for lambda_ep in sorted({case["parameters"]["lambda_ep"] for case in cases}):
        selected = [
            index
            for index, case in enumerate(cases)
            if np.isclose(case["parameters"]["lambda_ep"], lambda_ep)
        ]
        associations["stratified_by_lambda_ep"][f"{lambda_ep:g}"] = {
            "violation_severity_vs_raw_rho_error": _spearman(
                [severity[index] for index in selected],
                [raw_rho_error[index] for index in selected],
            ),
            "violation_severity_vs_raw_to_coherent_ratio": _spearman(
                [severity[index] for index in selected],
                [raw_to_coherent[index] for index in selected],
            ),
        }

    summary: dict[str, Any] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "classification": "diagnostic",
        "evidence_status": EVIDENCE_STATUS,
        "scientific_questions": [
            "Does the independent 31-coordinate closure improve shared "
            "observables over matched coherent-only dynamics?",
            "Does joint-Gram representability loss predict electronic error?",
            "Does Gram-barrier admissibility correction also improve accuracy?",
        ],
        "source_run": str(run_directory),
        "source_run_id": runtime.get("run_id"),
        "source_scientific_question": source_summary.get("scientific_question"),
        "verified_source_artifact_hashes": verified_hashes,
        "matching_contract": {
            "time_grid": "identical stored samples",
            "parameters": "identical hopping, lambda_ep, gamma, drive, pulse width",
            "initialization": (
                "exact contracted initial lower moments; coherent-only receives "
                "only the electronic 1-RDM and coherent phonon amplitude"
            ),
            "exact_reference_access": "offline reporting only",
            "coherent_integrator": "DOP853",
            "coherent_relative_tolerance": relative_tolerance,
            "coherent_absolute_tolerance": absolute_tolerance,
            "coherent_maximum_step": maximum_step,
        },
        "thresholds": {
            "joint_gram_loss": -_GRAM_TOLERANCE,
            "instantaneous_rho_error_over_exact_dynamic_rms": (
                _MATERIAL_ERROR_FRACTION
            ),
        },
        "cases": cases,
        "aggregate": {
            "case_count": len(cases),
            "raw_closure_better_than_coherent_rho_count": raw_better_rho,
            "corrected_closure_better_than_coherent_rho_count": (
                corrected_better_rho
            ),
            "raw_joint_gram_loss_count": gram_loss_count,
            "gram_loss_without_material_rho_error_count": (
                gram_loss_without_material_error
            ),
            "source_point_case_id": source_point["case_id"],
            "source_point": source_point,
            "associations": associations,
            "interpretive_boundary": (
                "The bounded grid can test temporal precedence and matched "
                "method error. It cannot establish causal prediction, "
                "specificity, or collaborator validation."
            ),
        },
    }
    return ClosureEvidenceResult(summary=summary, arrays=arrays)


def _plot_source_point(
    result: ClosureEvidenceResult,
    output_directory: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    case = result.summary["aggregate"]["source_point"]
    key = case["case_id"]
    arrays = result.arrays
    times = np.asarray(arrays[f"{key}__time"], dtype=float)
    figure, axes = plt.subplots(2, 1, figsize=(6.6, 6.0), sharex=True)
    styles = (
        ("exact", "truncated exact", "#222222", "-"),
        ("coherent", "coherent only", "#d4860b", "--"),
        ("raw", "31D closure", "#2367a2", "-"),
        ("corrected", "Gram-corrected", "#7b3f98", ":"),
    )
    for prefix, label, color, linestyle in styles:
        electron = np.asarray(
            arrays[f"{key}__{prefix}_electron_1rdm"], dtype=complex
        )
        axes[0].plot(
            times,
            _polarization(electron),
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
        )
    axes[0].set_ylabel(r"polarization $P(t)$")
    axes[0].legend(frameon=False, ncol=2)
    axes[0].grid(alpha=0.22)

    for prefix, label, color, linestyle in (
        ("exact", "truncated exact", "#222222", "-"),
        ("raw", "31D closure", "#2367a2", "-"),
        ("corrected", "Gram-corrected", "#7b3f98", ":"),
    ):
        axes[1].plot(
            times,
            arrays[f"{key}__{prefix}_joint_gram_minimum"],
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
        )
    axes[1].axhline(-_GRAM_TOLERANCE, color="#8b1a1a", linewidth=0.9)
    axes[1].set_xlabel(r"time $t\,t_{\rm hop}$")
    axes[1].set_ylabel(r"$\lambda_{\min}(\mathcal{G})$")
    axes[1].grid(alpha=0.22)
    axes[1].legend(frameon=False)
    figure.suptitle(
        r"Matched source-parameter point: $\lambda=\gamma=0.5$, $V=1$"
    )
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            output_directory / f"source_point_trajectory.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(figure)


def _plot_grid_diagnostics(
    result: ClosureEvidenceResult,
    output_directory: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = list(result.summary["cases"])
    labels = [
        (
            rf"$\lambda={case['parameters']['lambda_ep']:g}$, "
            rf"$\gamma={case['parameters']['gamma']:g}$, "
            rf"$V={case['parameters']['drive_amplitude']:g}$"
        )
        for case in cases
    ]
    ratios = np.asarray(
        [
            case["comparisons"]["raw_to_coherent_rho_rms_ratio"]
            for case in cases
        ],
        dtype=float,
    )
    colors = [
        "#2a9d8f" if case["parameters"]["lambda_ep"] < 1.0 else "#e76f51"
        for case in cases
    ]
    figure, axis = plt.subplots(figsize=(7.0, 5.1))
    positions = np.arange(len(cases))
    axis.bar(positions, ratios, color=colors, width=0.72)
    axis.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    axis.set_yscale("log")
    axis.set_ylabel("31D / coherent-only electronic 1-RDM RMS error")
    axis.set_xticks(positions, labels, rotation=62, ha="right", fontsize=8)
    axis.grid(axis="y", alpha=0.22)
    axis.set_title("When connected equal-time dynamics improves the coherent limit")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            output_directory / f"grid_method_comparison.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.5, 4.8))
    for lambda_ep, marker, color in (
        (0.5, "o", "#2a9d8f"),
        (1.5, "s", "#e76f51"),
    ):
        selected = [
            case
            for case in cases
            if np.isclose(case["parameters"]["lambda_ep"], lambda_ep)
        ]
        x = [
            -case["representability"]["raw_minimum_joint_gram_eigenvalue"]
            for case in selected
        ]
        y = [
            case["methods"]["raw_closure"]["electron_1rdm"][
                "rms_error_over_exact_dynamic_rms"
            ]
            for case in selected
        ]
        axis.scatter(
            x,
            y,
            marker=marker,
            color=color,
            s=62,
            label=rf"$\lambda={lambda_ep:g}$",
        )
        for x_value, y_value, case in zip(x, y, selected, strict=True):
            axis.annotate(
                rf"$\gamma={case['parameters']['gamma']:g},V={case['parameters']['drive_amplitude']:g}$",
                (x_value, y_value),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )
    all_severities = [
        -case["representability"]["raw_minimum_joint_gram_eigenvalue"]
        for case in cases
    ]
    if min(all_severities) > 0.0:
        axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(r"Gram-violation severity $-\min_t\lambda_{\min}(\mathcal{G})$")
    axis.set_ylabel("31D electronic 1-RDM error / exact dynamic RMS")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    axis.set_title("Gram violation is a necessary-state alarm, not an error scale")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            output_directory / f"gram_severity_vs_error.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(figure)


def write_closure_evidence(
    result: ClosureEvidenceResult,
    output_directory: str | Path,
) -> Path:
    """Write a new non-overwriting exploratory evidence directory."""

    destination = Path(output_directory).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    summary_path = destination / "summary.json"
    arrays_path = destination / "matched_trajectories.npz"
    _write_json_atomic(summary_path, result.summary)
    _write_npz_atomic(arrays_path, result.arrays)
    _plot_source_point(result, destination)
    _plot_grid_diagnostics(result, destination)

    artifacts = [
        path
        for path in destination.iterdir()
        if path.is_file() and path.name != "runtime_manifest.json"
    ]
    runtime = {
        "schema_version": 1,
        "status": "complete",
        "classification": "diagnostic",
        "evidence_status": EVIDENCE_STATUS,
        "source_run": result.summary["source_run"],
        "exact_reference_usage": (
            "stored exact arrays used only for offline metrics and figures; "
            "coherent propagation received shared initial lower moments only"
        ),
        "analyzer_sha256": _sha256(Path(__file__).resolve()),
        "artifact_hashes": {path.name: _sha256(path) for path in artifacts},
    }
    _write_json_atomic(destination / "runtime_manifest.json", runtime)
    return destination


def run_closure_evidence(
    source_run: str | Path,
    output_directory: str | Path,
) -> ClosureEvidenceResult:
    """Analyze one verified source run and persist a new evidence package."""

    result = analyze_closure_evidence(source_run)
    write_closure_evidence(result, output_directory)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    run_closure_evidence(arguments.source_run, arguments.output_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ClosureEvidenceResult",
    "EVIDENCE_SCHEMA_VERSION",
    "EvidenceSourceError",
    "analyze_closure_evidence",
    "run_closure_evidence",
    "write_closure_evidence",
]
