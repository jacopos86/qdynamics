"""Blind rank audit and sealed empirical scorer for frozen PURG ranks.

This module implements the post-pilot PURG decision without reviving the
retired correction/dual certificate ladder.  It has two deliberately separate
stages:

``prepare``
    Load an already frozen rank-128 construction artifact, build exactly 32
    blind residual directions from the rank-128 reduced path, and freeze a
    pre-scorer manifest.  No full-space driven trajectory is generated.

``score``
    Verify the frozen manifest, then compare the full cutoff model with the
    frozen ranks 128 and 160 using independently resolution-checked
    exponential-midpoint and DOP853 propagations.  Rank 160 is audit-only.

The resulting decision is empirical and preparation/protocol specific.  It is
not a theorem-level certificate and does not authorize another rank, threshold
change, or same-reference rescore after a scientific failure.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.sparse import csc_matrix, eye, issparse
from scipy.sparse.linalg._expm_multiply import (  # type: ignore[attr-defined]
    LazyOperatorNormInfo,
    _exact_1_norm,
    _expm_multiply_simple_core,
    _fragment_3_1,
)

from .exact_reference import _build_exact_dimer_model
from .hubbard_dimer import DimerParameters
from .krylov_memory_closure import (
    RAW_MOMENT_NAMES,
    _build_raw_moment_basis_from_model,
    raw_moments_to_closed_coordinates,
    raw_velocity_to_closed_velocity,
)
from .matrix_reference import CLOSED_SCALAR_STATE_NAMES
from .purg import PurgReducedModel

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
Operator = ComplexArray | csc_matrix

_SCHEMA = "paper5.purg.pre_scorer_manifest.v1"
_SCORE_SCHEMA = "paper5.purg.sealed_score.v1"
_CONSUMPTION_RECEIPT = "score_consumption_receipt.json"
_SCORE_FILE_NAMES = frozenset(
    {
        "score_summary.json",
        "score_arrays.npz",
        "rank_128_model.npz",
    }
)
_QUANTITIES = ("output", "derivative")
_STATISTICS = ("rms", "max")
_BLOCK_SLICES: dict[str, slice] = {
    "rho": slice(0, 3),
    "B": slice(3, 7),
    "N": slice(7, 11),
    "A": slice(11, 17),
    "C": slice(17, 31),
}


@dataclass(frozen=True)
class BlockBudget:
    """Discrete RMS and maximum ceilings for one scored block."""

    rms: float
    maximum: float


DERIVATIVE_BUDGETS: dict[str, BlockBudget] = {
    "rho": BlockBudget(1.0e-4, 3.0e-4),
    "B": BlockBudget(2.5e-3, 7.5e-3),
    "N": BlockBudget(2.5e-3, 7.5e-3),
    "A": BlockBudget(2.5e-3, 7.5e-3),
    "C": BlockBudget(2.5e-2, 7.5e-2),
}
OUTPUT_BUDGETS: dict[str, BlockBudget] = {
    "rho": BlockBudget(1.5e-4, 4.5e-4),
    "B": BlockBudget(7.5e-3, 2.25e-2),
    "N": BlockBudget(7.5e-3, 2.25e-2),
    "A": BlockBudget(7.5e-3, 2.25e-2),
    "C": BlockBudget(7.5e-2, 2.25e-1),
}


def _metric_key(quantity: str, block: str, statistic: str) -> str:
    if quantity not in _QUANTITIES:
        raise ValueError(f"unknown quantity {quantity!r}")
    if block not in _BLOCK_SLICES:
        raise ValueError(f"unknown block {block!r}")
    if statistic not in _STATISTICS:
        raise ValueError(f"unknown statistic {statistic!r}")
    return f"{quantity}.{block}.{statistic}"


def _all_metric_keys() -> tuple[str, ...]:
    return tuple(
        _metric_key(quantity, block, statistic)
        for quantity in _QUANTITIES
        for block in _BLOCK_SLICES
        for statistic in _STATISTICS
    )


_METRIC_KEYS = _all_metric_keys()


def _budget_by_key() -> dict[str, float]:
    budgets: dict[str, float] = {}
    for quantity, table in (
        ("output", OUTPUT_BUDGETS),
        ("derivative", DERIVATIVE_BUDGETS),
    ):
        for block, budget in table.items():
            budgets[_metric_key(quantity, block, "rms")] = budget.rms
            budgets[_metric_key(quantity, block, "max")] = budget.maximum
    return budgets


_BUDGET_BY_KEY = _budget_by_key()


@dataclass(frozen=True)
class BlindAuditSettings:
    """Frozen construction settings for the blind rank-160 audit."""

    base_rank: int = 128
    appended_directions: int = 32
    final_time: float = 4.0
    step: float = 0.0025
    exponential_action_tolerance: float = 1.0e-13
    tie_relative_tolerance: float = 1.0e-14
    deflation_relative_tolerance: float = 1.0e-12
    orthogonality_tolerance: float = 1.0e-12
    nesting_tolerance: float = 1.0e-12

    @property
    def audit_rank(self) -> int:
        return self.base_rank + self.appended_directions

    def __post_init__(self) -> None:
        if self.base_rank <= 0 or self.appended_directions <= 0:
            raise ValueError("base_rank and appended_directions must be positive")
        if self.final_time <= 0.0 or self.step <= 0.0:
            raise ValueError("final_time and step must be positive")
        count = int(round(self.final_time / self.step))
        if count <= 0 or not np.isclose(count * self.step, self.final_time):
            raise ValueError("final_time must be an integer multiple of step")
        for name in (
            "exponential_action_tolerance",
            "tie_relative_tolerance",
            "deflation_relative_tolerance",
            "orthogonality_tolerance",
            "nesting_tolerance",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class BlindAuditResult:
    """Successful blind residual append and its complete pivot history."""

    basis: ComplexArray
    appended_basis: ComplexArray
    pivot_indices: tuple[int, ...]
    pivot_times: FloatArray
    pivot_norms: FloatArray
    first_pivot_norm: float
    orthogonality_residual: float
    nesting_residual: float


class BlindAuditConstructionStop(RuntimeError):
    """Raised before scorer access when the fixed blind append cannot pass."""


@dataclass(frozen=True)
class ObservablePath:
    """Analytic centered outputs and derivatives on one sample grid."""

    times: FloatArray
    outputs: FloatArray
    derivatives: FloatArray
    maximum_norm_drift: float
    method: str

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float)
        outputs = np.asarray(self.outputs, dtype=float)
        derivatives = np.asarray(self.derivatives, dtype=float)
        if times.ndim != 1 or times.size < 2:
            raise ValueError("times must be a one-dimensional grid")
        expected = (times.size, len(CLOSED_SCALAR_STATE_NAMES))
        if outputs.shape != expected or derivatives.shape != expected:
            raise ValueError(
                f"output paths must have shape {expected}, got "
                f"{outputs.shape} and {derivatives.shape}"
            )
        if not np.all(np.isfinite(times)):
            raise ValueError("times must be finite")
        if not np.all(np.isfinite(outputs)) or not np.all(
            np.isfinite(derivatives)
        ):
            raise ValueError("output and derivative paths must be finite")
        if not np.isfinite(self.maximum_norm_drift):
            raise ValueError("maximum_norm_drift must be finite")
        if self.maximum_norm_drift < 0.0:
            raise ValueError("maximum_norm_drift must be nonnegative")


@dataclass(frozen=True)
class PropagationFamily:
    """All independent paths required for one numerical resolution audit."""

    fine_primary: ObservablePath
    fine_repeat: ObservablePath
    coarse_primary: ObservablePath
    coarse_repeat: ObservablePath
    dop853: ObservablePath


@dataclass(frozen=True)
class ResolutionEvaluation:
    """Ordered numerical and scientific evaluation for one score attempt."""

    numerical_passed: bool
    numerical_failures: tuple[str, ...]
    model_errors: dict[int, dict[str, float]]
    model_resolution: dict[int, dict[str, float]]
    model_resolution_components: dict[int, dict[str, dict[str, float]]]
    rank_difference: dict[str, float]
    rank_resolution: dict[str, float]
    rank_resolution_components: dict[str, dict[str, float]]
    tolerance_repeat: dict[str, float]
    norm_drifts: dict[str, float]
    scientific_passed: bool | None
    scientific_failures: tuple[str, ...]


@dataclass(frozen=True)
class FrozenScoreConfig:
    """Registered primary and one-fallback numerical score settings."""

    final_time: float = 4.0
    score_step: float = 0.0025
    fine_step: float = 0.0025
    coarse_step: float = 0.005
    primary_exponential_tolerance: float = 1.0e-13
    repeat_exponential_tolerance: float = 1.0e-12
    dop853_relative_tolerance: float = 1.0e-12
    dop853_absolute_tolerance: float = 1.0e-14
    dop853_maximum_step: float = 0.00125
    norm_drift_tolerance: float = 1.0e-11
    numerical_budget_fraction: float = 0.01
    rank_guard_fraction: float = 0.25
    fallback_fine_step: float = 0.00125
    fallback_coarse_step: float = 0.0025
    fallback_dop853_maximum_step: float = 0.000625

    def __post_init__(self) -> None:
        if self.final_time <= 0.0 or self.score_step <= 0.0:
            raise ValueError("final_time and score_step must be positive")
        for name in (
            "fine_step",
            "coarse_step",
            "primary_exponential_tolerance",
            "repeat_exponential_tolerance",
            "dop853_relative_tolerance",
            "dop853_absolute_tolerance",
            "dop853_maximum_step",
            "norm_drift_tolerance",
            "numerical_budget_fraction",
            "rank_guard_fraction",
            "fallback_fine_step",
            "fallback_coarse_step",
            "fallback_dop853_maximum_step",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not self.primary_exponential_tolerance < self.repeat_exponential_tolerance:
            raise ValueError("primary exponential tolerance must be tighter")
        for step in (
            self.score_step,
            self.fine_step,
            self.coarse_step,
            self.fallback_fine_step,
            self.fallback_coarse_step,
        ):
            count = int(round(self.final_time / step))
            if count <= 0 or not np.isclose(count * step, self.final_time):
                raise ValueError("every step must divide final_time exactly")


@dataclass(frozen=True)
class FrozenPreparedSystem:
    """Verified construction data available before the scorer is opened."""

    parameters: DimerParameters
    phonon_cutoff: int
    full_static_hamiltonian: csc_matrix
    full_drive_hamiltonian: csc_matrix
    full_raw_observables: tuple[csc_matrix, ...]
    full_initial_state: ComplexArray
    rank_128_basis: ComplexArray
    rank_160_basis: ComplexArray
    rank_128_model: PurgReducedModel
    rank_160_model: PurgReducedModel
    pivot_history: BlindAuditResult
    manifest: dict[str, Any]
    manifest_sha256: str


def _time_grid(final_time: float, step: float) -> FloatArray:
    count = int(round(final_time / step))
    if count <= 0 or not np.isclose(count * step, final_time, atol=1.0e-13):
        raise ValueError("final_time must be an integer multiple of step")
    return np.linspace(0.0, final_time, count + 1, dtype=float)


def _require_orthonormal_basis(
    basis: ComplexArray,
    *,
    expected_rank: int,
    tolerance: float,
    name: str,
) -> ComplexArray:
    values = np.asarray(basis, dtype=complex)
    if values.ndim != 2 or values.shape[1] != expected_rank:
        raise ValueError(
            f"{name} must be a matrix with {expected_rank} columns, "
            f"got {values.shape}"
        )
    residual = float(
        np.linalg.norm(
            values.conj().T @ values - np.eye(expected_rank),
            ord=2,
        )
    )
    if residual > tolerance:
        raise ValueError(
            f"{name} is not orthonormal: spectral residual {residual:.3e}"
        )
    return values


def _exponential_action(
    generator: Operator,
    vector: ComplexArray,
    *,
    step: float,
    relative_tolerance: float,
) -> ComplexArray:
    """Apply one exponential with an explicit Al-Mohy--Higham tolerance.

    SciPy's public ``expm_multiply`` fixes its internal tolerance at machine
    precision.  The registered score requires two requested tolerances, so the
    manifest freezes this small adapter to SciPy's implementation of Algorithm
    3.2 and records the SciPy version.
    """

    if step <= 0.0 or relative_tolerance <= 0.0:
        raise ValueError("step and relative_tolerance must be positive")
    operator = generator.tocsc() if issparse(generator) else np.asarray(generator)
    state = np.asarray(vector, dtype=complex)
    if operator.shape[0] != operator.shape[1] or state.shape != (operator.shape[0],):
        raise ValueError("generator and vector shapes are incompatible")
    dimension = operator.shape[0]
    trace = complex(operator.diagonal().sum())
    mean = trace / float(dimension)
    identity = (
        eye(dimension, format="csc", dtype=complex)
        if issparse(operator)
        else np.eye(dimension, dtype=complex)
    )
    shifted = operator - mean * identity
    one_norm = float(_exact_1_norm(shifted))
    if step * one_norm == 0.0:
        taylor_degree, scaling = 0, 1
    else:
        norm_info = LazyOperatorNormInfo(
            step * shifted,
            A_1_norm=step * one_norm,
            ell=2,
        )
        taylor_degree, scaling = _fragment_3_1(
            norm_info,
            1,
            relative_tolerance,
            ell=2,
        )
    result = _expm_multiply_simple_core(
        shifted,
        state,
        step,
        mean,
        taylor_degree,
        scaling,
        relative_tolerance,
    )
    return np.asarray(result, dtype=complex)


def propagate_state_midpoint(
    static_hamiltonian: Operator,
    drive_hamiltonian: Operator,
    initial_state: ComplexArray,
    parameters: DimerParameters,
    *,
    final_time: float,
    step: float,
    exponential_action_tolerance: float,
) -> tuple[FloatArray, ComplexArray, float]:
    """Return every midpoint endpoint without renormalizing the state."""

    times = _time_grid(final_time, step)
    state = np.asarray(initial_state, dtype=complex).copy()
    if state.shape != (static_hamiltonian.shape[0],):
        raise ValueError("initial_state has the wrong dimension")
    states = np.empty((times.size, state.size), dtype=complex)
    states[0] = state
    initial_norm = float(np.linalg.norm(state))
    if initial_norm <= 0.0:
        raise ValueError("initial_state must have nonzero norm")
    maximum_drift = 0.0
    for index in range(times.size - 1):
        midpoint = 0.5 * (times[index] + times[index + 1])
        hamiltonian = (
            static_hamiltonian
            + parameters.drive_difference(float(midpoint)) * drive_hamiltonian
        )
        state = _exponential_action(
            -1j * hamiltonian,
            state,
            step=step,
            relative_tolerance=exponential_action_tolerance,
        )
        states[index + 1] = state
        maximum_drift = max(
            maximum_drift,
            abs(float(np.linalg.norm(state)) / initial_norm - 1.0),
        )
    return times, states, maximum_drift


def build_blind_residual_audit_basis(
    base_basis: ComplexArray,
    residual_columns: ComplexArray,
    sample_times: FloatArray,
    *,
    settings: BlindAuditSettings | None = None,
) -> BlindAuditResult:
    """Append the fixed number of twice-reorthogonalized greedy pivots.

    ``residual_columns`` must already include the trapezoidal square-root
    weights.  Pivots are selected from the original time ordering; ties use
    the earliest time and no alternative packet family is attempted.
    """

    resolved = settings or BlindAuditSettings()
    base = _require_orthonormal_basis(
        base_basis,
        expected_rank=resolved.base_rank,
        tolerance=resolved.orthogonality_tolerance,
        name="base_basis",
    )
    candidates = np.asarray(residual_columns, dtype=complex)
    times = np.asarray(sample_times, dtype=float)
    if candidates.ndim != 2 or candidates.shape[0] != base.shape[0]:
        raise ValueError("residual_columns must share the basis row dimension")
    if times.shape != (candidates.shape[1],):
        raise ValueError("sample_times must label every residual column")
    if np.any(np.diff(times) < 0.0):
        raise ValueError("sample_times must be nondecreasing")

    accepted: list[ComplexArray] = []
    pivot_indices: list[int] = []
    pivot_norms: list[float] = []
    first_pivot_norm: float | None = None
    active = np.ones(candidates.shape[1], dtype=bool)

    for _ in range(resolved.appended_directions):
        growing_basis = (
            base if not accepted else np.column_stack((base, *accepted))
        )
        residual = candidates.copy()
        for _ in range(2):
            residual -= growing_basis @ (growing_basis.conj().T @ residual)
        residual[:, ~active] = 0.0
        norms = np.linalg.norm(residual, axis=0)
        norms[~active] = -np.inf
        maximum = float(np.max(norms))
        if not np.isfinite(maximum) or maximum < 0.0:
            raise BlindAuditConstructionStop(
                "fewer than the required blind residual pivots remain"
            )
        tied = np.flatnonzero(
            active
            & (
                np.abs(norms - maximum)
                <= resolved.tie_relative_tolerance * max(maximum, 0.0)
            )
        )
        pivot = int(tied[0])
        selected_norm = float(norms[pivot])
        if first_pivot_norm is None:
            first_pivot_norm = selected_norm
            if first_pivot_norm <= 0.0:
                raise BlindAuditConstructionStop(
                    "the first blind residual pivot has zero norm"
                )
        if selected_norm <= resolved.deflation_relative_tolerance * first_pivot_norm:
            raise BlindAuditConstructionStop(
                "fewer than 32 pivots survived the fixed relative deflation rule"
            )

        direction = residual[:, pivot] / selected_norm
        accepted.append(direction)
        pivot_indices.append(pivot)
        pivot_norms.append(selected_norm)
        active[pivot] = False

    appended = np.column_stack(accepted)
    combined = np.column_stack((base, appended))
    orthogonality = float(
        np.linalg.norm(
            combined.conj().T @ combined - np.eye(combined.shape[1]),
            ord=2,
        )
    )
    nesting = float(
        np.linalg.norm(
            base - combined @ (combined.conj().T @ base),
            ord=2,
        )
    )
    if combined.shape[1] != resolved.audit_rank:
        raise BlindAuditConstructionStop(
            f"requested rank {resolved.audit_rank}, reached {combined.shape[1]}"
        )
    if orthogonality > resolved.orthogonality_tolerance:
        raise BlindAuditConstructionStop(
            "blind audit orthogonality gate failed: "
            f"{orthogonality:.3e} > {resolved.orthogonality_tolerance:.3e}"
        )
    if nesting > resolved.nesting_tolerance:
        raise BlindAuditConstructionStop(
            "blind audit nesting gate failed: "
            f"{nesting:.3e} > {resolved.nesting_tolerance:.3e}"
        )
    assert first_pivot_norm is not None
    return BlindAuditResult(
        basis=combined,
        appended_basis=appended,
        pivot_indices=tuple(pivot_indices),
        pivot_times=times[np.asarray(pivot_indices, dtype=int)],
        pivot_norms=np.asarray(pivot_norms, dtype=float),
        first_pivot_norm=first_pivot_norm,
        orthogonality_residual=orthogonality,
        nesting_residual=nesting,
    )


def build_blind_rank_160(
    *,
    base_basis: ComplexArray,
    base_model: PurgReducedModel,
    full_static_hamiltonian: csc_matrix,
    full_drive_hamiltonian: csc_matrix,
    parameters: DimerParameters,
    settings: BlindAuditSettings | None = None,
) -> BlindAuditResult:
    """Construct W_160 from the W_128 reduced path and no scorer data."""

    resolved = settings or BlindAuditSettings()
    base = _require_orthonormal_basis(
        base_basis,
        expected_rank=resolved.base_rank,
        tolerance=resolved.orthogonality_tolerance,
        name="W_128",
    )
    if base_model.dimension != resolved.base_rank:
        raise ValueError("base_model does not have the frozen base rank")
    times, states, _ = propagate_state_midpoint(
        base_model.static_hamiltonian,
        base_model.drive_hamiltonian,
        base_model.initial_state,
        parameters,
        final_time=resolved.final_time,
        step=resolved.step,
        exponential_action_tolerance=resolved.exponential_action_tolerance,
    )
    residuals = np.empty((base.shape[0], times.size), dtype=complex)
    for index, (time, state) in enumerate(zip(times, states, strict=True)):
        lifted = base @ state
        drive_value = parameters.drive_difference(float(time))
        full_action = (
            full_static_hamiltonian @ lifted
            + drive_value * (full_drive_hamiltonian @ lifted)
        )
        compressed_action = (
            base_model.static_hamiltonian @ state
            + drive_value * (base_model.drive_hamiltonian @ state)
        )
        residuals[:, index] = 1j * (full_action - base @ compressed_action)
    weights = np.ones(times.size, dtype=float)
    weights[[0, -1]] = 0.5
    residuals *= np.sqrt(weights)[None, :]
    return build_blind_residual_audit_basis(
        base,
        residuals,
        times,
        settings=resolved,
    )


def _compress_hermitian(
    basis: ComplexArray,
    operator: csc_matrix,
) -> ComplexArray:
    compressed = np.asarray(basis.conj().T @ (operator @ basis), dtype=complex)
    scale = max(1.0, float(np.linalg.norm(compressed)))
    leakage = float(np.linalg.norm(compressed - compressed.conj().T)) / scale
    if leakage > 1.0e-12:
        raise BlindAuditConstructionStop(
            f"compressed Hermitian leakage {leakage:.3e} exceeds 1e-12"
        )
    return 0.5 * (compressed + compressed.conj().T)


def _model_from_basis(
    *,
    basis: ComplexArray,
    phonon_cutoff: int,
    cap_label: int,
    static_hamiltonian: csc_matrix,
    drive_hamiltonian: csc_matrix,
    raw_observables: Sequence[csc_matrix],
    initial_state: ComplexArray,
) -> PurgReducedModel:
    projected_initial = np.asarray(basis.conj().T @ initial_state, dtype=complex)
    norm = float(np.linalg.norm(projected_initial))
    if norm <= 0.0:
        raise BlindAuditConstructionStop("projected initial state has zero norm")
    return PurgReducedModel(
        phonon_cutoff=phonon_cutoff,
        cap_label=cap_label,
        static_hamiltonian=_compress_hermitian(basis, static_hamiltonian),
        drive_hamiltonian=_compress_hermitian(basis, drive_hamiltonian),
        raw_observables=np.asarray(
            [_compress_hermitian(basis, operator) for operator in raw_observables],
            dtype=complex,
        ),
        initial_state=projected_initial / norm,
    )


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_fingerprint(values: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(values)
    header = _canonical_json_bytes(
        {"dtype": array.dtype.str, "shape": list(array.shape)}
    )
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": _sha256_bytes(header + array.tobytes(order="C")),
    }


def _sparse_fingerprint(operator: csc_matrix) -> dict[str, Any]:
    matrix = operator.tocsc()
    payload = {
        "shape": list(matrix.shape),
        "dtype": matrix.dtype.str,
        "data": _array_fingerprint(matrix.data),
        "indices": _array_fingerprint(matrix.indices),
        "indptr": _array_fingerprint(matrix.indptr),
    }
    return {**payload, "sha256": _sha256_bytes(_canonical_json_bytes(payload))}


def _source_revision(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _load_frozen_construction(
    construction_directory: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    summary_path = construction_directory / "summary.json"
    arrays_path = construction_directory / "arrays.npz"
    manifest_path = construction_directory / "manifest.json"
    for path in (summary_path, arrays_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen construction file: {path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("contains_exact_driven_trajectory") is not False:
        raise ValueError("construction artifact is not exact-trajectory-free")
    if source_manifest.get("contains_controller_feedback") is not False:
        raise ValueError("construction artifact contains controller feedback")
    if int(summary.get("phonon_cutoff", -1)) != 16:
        raise ValueError("the sealed score requires the cutoff-16 construction")
    if tuple(summary.get("settings", {}).get("caps", ())) != (32, 64, 96, 128):
        raise ValueError("construction does not contain the frozen rank ladder")
    if float(summary["settings"]["construction_step"]) != 0.0025:
        raise ValueError("the frozen construction step must be 0.0025")
    if float(summary["settings"]["final_time"]) != 4.0:
        raise ValueError("the frozen construction horizon must be 4.0")

    # A source hash mismatch means the artifact can still be archived, but it
    # cannot be used to generate a new sealed scorer manifest.
    for path_text, record in source_manifest.get("files", {}).items():
        source_path = Path(path_text)
        if not source_path.is_file():
            raise ValueError(f"frozen construction source is missing: {source_path}")
        if _sha256_file(source_path) != record.get("sha256"):
            raise ValueError(f"frozen construction source hash changed: {source_path}")

    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    required = {
        "ground_state",
        "cap_128_basis",
        "cap_128_static_hamiltonian",
        "cap_128_drive_hamiltonian",
        "cap_128_raw_observables",
        "cap_128_initial_state",
        "raw_to_centered_jacobian_at_zero",
        "raw_to_centered_hessian",
    }
    missing = required - arrays.keys()
    if missing:
        raise ValueError(f"frozen construction arrays are missing {sorted(missing)}")
    provenance = {
        "directory": str(construction_directory.resolve()),
        "summary": {
            "sha256": _sha256_file(summary_path),
            "bytes": summary_path.stat().st_size,
        },
        "arrays": {
            "sha256": _sha256_file(arrays_path),
            "bytes": arrays_path.stat().st_size,
        },
        "manifest": {
            "sha256": _sha256_file(manifest_path),
            "bytes": manifest_path.stat().st_size,
        },
    }
    return summary, arrays, provenance


def _verify_frozen_rank_128(
    *,
    arrays: Mapping[str, np.ndarray],
    static_hamiltonian: csc_matrix,
    drive_hamiltonian: csc_matrix,
    raw_observables: Sequence[csc_matrix],
) -> tuple[ComplexArray, PurgReducedModel, ComplexArray]:
    basis = _require_orthonormal_basis(
        np.asarray(arrays["cap_128_basis"], dtype=complex),
        expected_rank=128,
        tolerance=1.0e-12,
        name="frozen W_128",
    )
    initial = np.asarray(arrays["ground_state"], dtype=complex)
    if initial.shape != (basis.shape[0],):
        raise ValueError("frozen ground state and W_128 dimensions disagree")
    if abs(float(np.linalg.norm(initial)) - 1.0) > 1.0e-12:
        raise ValueError("frozen ground state is not normalized")

    compressed_static = _compress_hermitian(basis, static_hamiltonian)
    compressed_drive = _compress_hermitian(basis, drive_hamiltonian)
    compressed_raw = np.asarray(
        [_compress_hermitian(basis, operator) for operator in raw_observables],
        dtype=complex,
    )
    np.testing.assert_array_equal(
        compressed_static,
        np.asarray(arrays["cap_128_static_hamiltonian"], dtype=complex),
    )
    np.testing.assert_array_equal(
        compressed_drive,
        np.asarray(arrays["cap_128_drive_hamiltonian"], dtype=complex),
    )
    np.testing.assert_array_equal(
        compressed_raw,
        np.asarray(arrays["cap_128_raw_observables"], dtype=complex),
    )
    projected = np.asarray(basis.conj().T @ initial, dtype=complex)
    projected /= np.linalg.norm(projected)
    np.testing.assert_allclose(
        projected,
        np.asarray(arrays["cap_128_initial_state"], dtype=complex),
        atol=2.0e-14,
        rtol=2.0e-14,
    )
    model = PurgReducedModel(
        phonon_cutoff=16,
        cap_label=128,
        static_hamiltonian=np.asarray(
            arrays["cap_128_static_hamiltonian"], dtype=complex
        ),
        drive_hamiltonian=np.asarray(
            arrays["cap_128_drive_hamiltonian"], dtype=complex
        ),
        raw_observables=np.asarray(
            arrays["cap_128_raw_observables"], dtype=complex
        ),
        initial_state=np.asarray(arrays["cap_128_initial_state"], dtype=complex),
    )
    return basis, model, initial


def _source_hashes() -> dict[str, dict[str, Any]]:
    directory = Path(__file__).resolve().parent
    paths = (
        Path(__file__).resolve(),
        directory / "purg.py",
        directory / "exact_reference.py",
        directory / "krylov_memory_closure.py",
        directory / "matrix_reference.py",
        directory / "hubbard_dimer.py",
    )
    return {
        str(path): {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
        for path in paths
    }


def _solver_environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OMP_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
        "exponential_action": (
            "frozen adapter to scipy.sparse.linalg._expm_multiply "
            "Algorithm 3.2 with explicit tolerance"
        ),
        "dop853": "scipy.integrate.solve_ivp(method='DOP853')",
    }


def _budgets_json() -> dict[str, dict[str, dict[str, float]]]:
    return {
        "derivative": {
            name: {"rms": value.rms, "max": value.maximum}
            for name, value in DERIVATIVE_BUDGETS.items()
        },
        "output": {
            name: {"rms": value.rms, "max": value.maximum}
            for name, value in OUTPUT_BUDGETS.items()
        },
    }


def _build_pre_scorer_manifest(
    *,
    repo_root: Path,
    parameters: DimerParameters,
    construction_provenance: Mapping[str, Any],
    full_static_hamiltonian: csc_matrix,
    full_drive_hamiltonian: csc_matrix,
    full_raw_observables: Sequence[csc_matrix],
    full_initial_state: ComplexArray,
    rank_128_basis: ComplexArray,
    rank_160_basis: ComplexArray,
    rank_128_model: PurgReducedModel,
    rank_160_model: PurgReducedModel,
    blind_settings: BlindAuditSettings,
    pivot_history: BlindAuditResult,
    score_config: FrozenScoreConfig,
) -> dict[str, Any]:
    score_times = _time_grid(score_config.final_time, score_config.score_step)
    drive_values = np.asarray(
        [parameters.drive_difference(float(time)) for time in score_times],
        dtype=float,
    )
    drive_source = inspect.getsource(DimerParameters.drive_difference).encode("utf-8")
    coordinate_sources = {
        "raw_moments_to_closed_coordinates": _sha256_bytes(
            inspect.getsource(raw_moments_to_closed_coordinates).encode("utf-8")
        ),
        "raw_velocity_to_closed_velocity": _sha256_bytes(
            inspect.getsource(raw_velocity_to_closed_velocity).encode("utf-8")
        ),
    }
    models = {
        "128": {
            "static_hamiltonian": _array_fingerprint(
                rank_128_model.static_hamiltonian
            ),
            "drive_hamiltonian": _array_fingerprint(
                rank_128_model.drive_hamiltonian
            ),
            "raw_observables": _array_fingerprint(rank_128_model.raw_observables),
            "initial_state": _array_fingerprint(rank_128_model.initial_state),
        },
        "160": {
            "static_hamiltonian": _array_fingerprint(
                rank_160_model.static_hamiltonian
            ),
            "drive_hamiltonian": _array_fingerprint(
                rank_160_model.drive_hamiltonian
            ),
            "raw_observables": _array_fingerprint(rank_160_model.raw_observables),
            "initial_state": _array_fingerprint(rank_160_model.initial_state),
        },
    }
    return {
        "schema": _SCHEMA,
        "status": "frozen_pre_scorer",
        "scorer_open_authorized": True,
        "construction_contains_full_space_driven_trajectory": False,
        "retired_acceptance_paths": [
            "rank_192_correction_space",
            "rank_224_online_candidate",
            "rank_256_dual_or_audit_space",
            "correction_interval_acceptance",
            "dual_interval_acceptance",
        ],
        "scope": {
            "claim": (
                "empirical cutoff-16 correlated-preparation short-horizon "
                "rank-128 surrogate"
            ),
            "phonon_cutoff": 16,
            "preparation": "correlated_ground_ket",
            "pulse": "DimerParameters.drive_difference",
            "time_interval": [0.0, 4.0],
            "score_nodes": 1601,
            "rank_160_role": "blind_audit_only_cannot_rescue_rank_128",
            "excluded_claims": [
                "formal_certificate",
                "full_state_norm_bound",
                "off_grid_supremum",
                "cutoff_convergence",
                "long_time_stability",
                "other_preparations_or_drives",
            ],
        },
        "parameters": asdict(parameters),
        "construction_artifact": _jsonable(construction_provenance),
        "construction": {
            "blind_audit_settings": asdict(blind_settings),
            "pivot_indices": list(pivot_history.pivot_indices),
            "pivot_times": pivot_history.pivot_times.tolist(),
            "pivot_norms": pivot_history.pivot_norms.tolist(),
            "first_pivot_norm": pivot_history.first_pivot_norm,
            "orthogonality_residual": pivot_history.orthogonality_residual,
            "nesting_residual": pivot_history.nesting_residual,
        },
        "full_operators": {
            "H0": _sparse_fingerprint(full_static_hamiltonian),
            "HV": _sparse_fingerprint(full_drive_hamiltonian),
            "raw_observables": [
                _sparse_fingerprint(operator) for operator in full_raw_observables
            ],
            "psi0": _array_fingerprint(full_initial_state),
        },
        "bases": {
            "W128": _array_fingerprint(rank_128_basis),
            "W160": _array_fingerprint(rank_160_basis),
        },
        "compressed_models": models,
        "drive": {
            "callable": "paper5.stability.hubbard_dimer.DimerParameters.drive_difference",
            "source_sha256": _sha256_bytes(drive_source),
            "score_grid_values": _array_fingerprint(drive_values),
        },
        "coordinate_contract": {
            "raw_moment_names": list(RAW_MOMENT_NAMES),
            "centered_coordinate_names": list(CLOSED_SCALAR_STATE_NAMES),
            "block_slices": {
                name: [block.start, block.stop]
                for name, block in _BLOCK_SLICES.items()
            },
            "C_packing": {
                "kind": "frozen_14_real_slots",
                "slice": [17, 31],
                "names": list(CLOSED_SCALAR_STATE_NAMES[17:31]),
            },
            "analytic_map_source_hashes": coordinate_sources,
            "finite_difference_derivatives_prohibited": True,
        },
        "score_contract": {
            "config": asdict(score_config),
            "budgets": _budgets_json(),
            "metrics": "discrete_euclidean_block_rms_and_max",
            "numerical_gates_evaluated_before_scientific_gates": True,
            "one_fixed_numerical_fallback_only": True,
            "rank_guard_fraction": score_config.rank_guard_fraction,
            "numerical_budget_fraction": score_config.numerical_budget_fraction,
            "no_renormalization": True,
        },
        "solver_environment": _solver_environment(),
        "source": {
            "git_revision": _source_revision(repo_root),
            "files": _source_hashes(),
        },
        "forbidden_construction_inputs": [
            "full_space_driven_states",
            "full_space_driven_moments",
            "full_space_driven_derivatives",
            "full_space_driven_residuals",
            "scorer_results",
        ],
    }


def _manifest_digest(manifest: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_json_bytes(_jsonable(manifest)))


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen artifact: {path}")
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def prepare_sealed_score(
    construction_directory: Path,
    output_directory: Path,
    *,
    repo_root: Path,
    blind_settings: BlindAuditSettings | None = None,
    score_config: FrozenScoreConfig | None = None,
) -> FrozenPreparedSystem:
    """Build W_160 and atomically freeze all pre-scorer choices."""

    if output_directory.exists():
        raise FileExistsError(
            f"refusing to overwrite prospective scorer directory: {output_directory}"
        )
    resolved_blind = blind_settings or BlindAuditSettings()
    resolved_score = score_config or FrozenScoreConfig()
    if resolved_blind != BlindAuditSettings():
        raise ValueError("production preparation requires the registered blind settings")
    if resolved_score != FrozenScoreConfig():
        raise ValueError("production preparation requires the registered score config")
    summary, arrays, provenance = _load_frozen_construction(
        construction_directory
    )
    parameters = DimerParameters(**summary["parameters"])
    exact_model = _build_exact_dimer_model(parameters, phonon_cutoff=16)
    static_hamiltonian = exact_model.static_hamiltonian.tocsc()
    drive_hamiltonian = exact_model.drive_operator.tocsc()
    raw_basis = _build_raw_moment_basis_from_model(
        exact_model,
        phonon_cutoff=16,
    )
    raw_observables = tuple(operator.tocsc() for operator in raw_basis.observables)
    rank_128_basis, rank_128_model, initial_state = _verify_frozen_rank_128(
        arrays=arrays,
        static_hamiltonian=static_hamiltonian,
        drive_hamiltonian=drive_hamiltonian,
        raw_observables=raw_observables,
    )
    pivot_history = build_blind_rank_160(
        base_basis=rank_128_basis,
        base_model=rank_128_model,
        full_static_hamiltonian=static_hamiltonian,
        full_drive_hamiltonian=drive_hamiltonian,
        parameters=parameters,
        settings=resolved_blind,
    )
    rank_160_basis = pivot_history.basis
    rank_160_model = _model_from_basis(
        basis=rank_160_basis,
        phonon_cutoff=16,
        cap_label=160,
        static_hamiltonian=static_hamiltonian,
        drive_hamiltonian=drive_hamiltonian,
        raw_observables=raw_observables,
        initial_state=initial_state,
    )
    manifest = _build_pre_scorer_manifest(
        repo_root=repo_root,
        parameters=parameters,
        construction_provenance=provenance,
        full_static_hamiltonian=static_hamiltonian,
        full_drive_hamiltonian=drive_hamiltonian,
        full_raw_observables=raw_observables,
        full_initial_state=initial_state,
        rank_128_basis=rank_128_basis,
        rank_160_basis=rank_160_basis,
        rank_128_model=rank_128_model,
        rank_160_model=rank_160_model,
        blind_settings=resolved_blind,
        pivot_history=pivot_history,
        score_config=resolved_score,
    )
    manifest_sha256 = _manifest_digest(manifest)

    output_directory.mkdir(parents=True, exist_ok=False)
    arrays_path = output_directory / "construction_arrays.npz"
    np.savez_compressed(
        arrays_path,
        W128=rank_128_basis,
        W160=rank_160_basis,
        appended_W160=pivot_history.appended_basis,
        pivot_indices=np.asarray(pivot_history.pivot_indices, dtype=np.int64),
        pivot_times=pivot_history.pivot_times,
        pivot_norms=pivot_history.pivot_norms,
        psi0=initial_state,
        H0_128=rank_128_model.static_hamiltonian,
        HV_128=rank_128_model.drive_hamiltonian,
        F_128=rank_128_model.raw_observables,
        c0_128=rank_128_model.initial_state,
        H0_160=rank_160_model.static_hamiltonian,
        HV_160=rank_160_model.drive_hamiltonian,
        F_160=rank_160_model.raw_observables,
        c0_160=rank_160_model.initial_state,
    )
    artifact_record = {
        "file": arrays_path.name,
        "sha256": _sha256_file(arrays_path),
        "bytes": arrays_path.stat().st_size,
    }
    manifest = {**manifest, "construction_arrays_artifact": artifact_record}
    manifest_sha256 = _manifest_digest(manifest)
    manifest_with_digest = {**manifest, "manifest_sha256": manifest_sha256}
    _write_json_exclusive(
        output_directory / "pre_scorer_manifest.json",
        manifest_with_digest,
    )
    _write_json_exclusive(
        output_directory / "construction_summary.json",
        {
            "schema": _SCHEMA,
            "status": "passed_blind_construction_scorer_still_unopened",
            "manifest_sha256": manifest_sha256,
            "W128_sha256": manifest["bases"]["W128"]["sha256"],
            "W160_sha256": manifest["bases"]["W160"]["sha256"],
            "pivot_indices": list(pivot_history.pivot_indices),
            "pivot_times": pivot_history.pivot_times.tolist(),
            "orthogonality_residual": pivot_history.orthogonality_residual,
            "nesting_residual": pivot_history.nesting_residual,
            "full_space_driven_scorer_opened": False,
        },
    )
    return FrozenPreparedSystem(
        parameters=parameters,
        phonon_cutoff=16,
        full_static_hamiltonian=static_hamiltonian,
        full_drive_hamiltonian=drive_hamiltonian,
        full_raw_observables=raw_observables,
        full_initial_state=initial_state,
        rank_128_basis=rank_128_basis,
        rank_160_basis=rank_160_basis,
        rank_128_model=rank_128_model,
        rank_160_model=rank_160_model,
        pivot_history=pivot_history,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
    )


def _verify_fingerprint(
    label: str,
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
) -> None:
    if dict(expected) != dict(actual):
        raise ValueError(f"pre-scorer fingerprint mismatch for {label}")


def load_prepared_score(
    directory: Path,
    *,
    repo_root: Path,
    require_unconsumed: bool = False,
) -> FrozenPreparedSystem:
    """Reload and verify a frozen pre-scorer artifact before reference access."""

    manifest_path = directory / "pre_scorer_manifest.json"
    arrays_path = directory / "construction_arrays.npz"
    if not manifest_path.is_file() or not arrays_path.is_file():
        raise FileNotFoundError("prepared score directory is incomplete")
    consumption_path = directory / _CONSUMPTION_RECEIPT
    if require_unconsumed and consumption_path.exists():
        raise RuntimeError(
            "the frozen pre-scorer manifest has already been consumed; "
            "same-reference rescoring is prohibited"
        )
    if any((directory / name).exists() for name in _SCORE_FILE_NAMES):
        raise ValueError("score outputs must not be stored in the pre-scorer directory")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = manifest.pop("manifest_sha256", None)
    if manifest.get("schema") != _SCHEMA or manifest.get("status") != "frozen_pre_scorer":
        raise ValueError("unknown or non-frozen pre-scorer manifest")
    if digest != _manifest_digest(manifest):
        raise ValueError("pre-scorer manifest digest mismatch")
    if manifest.get("solver_environment") != _solver_environment():
        raise ValueError("solver environment changed after the pre-scorer freeze")
    expected_score_contract = {
        "config": asdict(FrozenScoreConfig()),
        "budgets": _budgets_json(),
        "metrics": "discrete_euclidean_block_rms_and_max",
        "numerical_gates_evaluated_before_scientific_gates": True,
        "one_fixed_numerical_fallback_only": True,
        "rank_guard_fraction": FrozenScoreConfig().rank_guard_fraction,
        "numerical_budget_fraction": (
            FrozenScoreConfig().numerical_budget_fraction
        ),
        "no_renormalization": True,
    }
    if manifest.get("score_contract") != expected_score_contract:
        raise ValueError("frozen score config or budget table changed")
    if manifest.get("construction", {}).get("blind_audit_settings") != asdict(
        BlindAuditSettings()
    ):
        raise ValueError("frozen blind-audit settings changed")
    if _sha256_file(arrays_path) != manifest["construction_arrays_artifact"]["sha256"]:
        raise ValueError("frozen construction_arrays.npz hash mismatch")
    for path_text, record in manifest["source"]["files"].items():
        path = Path(path_text)
        if not path.is_file() or _sha256_file(path) != record["sha256"]:
            raise ValueError(f"pre-scorer source changed: {path}")
    if manifest["source"]["git_revision"] != _source_revision(repo_root):
        raise ValueError("repository revision changed after the pre-scorer freeze")

    parameters = DimerParameters(**manifest["parameters"])
    exact_model = _build_exact_dimer_model(parameters, phonon_cutoff=16)
    static_hamiltonian = exact_model.static_hamiltonian.tocsc()
    drive_hamiltonian = exact_model.drive_operator.tocsc()
    raw_basis = _build_raw_moment_basis_from_model(
        exact_model,
        phonon_cutoff=16,
    )
    raw_observables = tuple(operator.tocsc() for operator in raw_basis.observables)
    _verify_fingerprint(
        "H0", manifest["full_operators"]["H0"], _sparse_fingerprint(static_hamiltonian)
    )
    _verify_fingerprint(
        "HV", manifest["full_operators"]["HV"], _sparse_fingerprint(drive_hamiltonian)
    )
    for index, (expected, operator) in enumerate(
        zip(
            manifest["full_operators"]["raw_observables"],
            raw_observables,
            strict=True,
        )
    ):
        _verify_fingerprint(
            f"raw_observable_{index}", expected, _sparse_fingerprint(operator)
        )

    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    rank_128_basis = np.asarray(arrays["W128"], dtype=complex)
    rank_160_basis = np.asarray(arrays["W160"], dtype=complex)
    initial_state = np.asarray(arrays["psi0"], dtype=complex)
    _verify_fingerprint(
        "W128", manifest["bases"]["W128"], _array_fingerprint(rank_128_basis)
    )
    _verify_fingerprint(
        "W160", manifest["bases"]["W160"], _array_fingerprint(rank_160_basis)
    )
    _verify_fingerprint(
        "psi0", manifest["full_operators"]["psi0"], _array_fingerprint(initial_state)
    )
    rank_128_model = PurgReducedModel(
        phonon_cutoff=16,
        cap_label=128,
        static_hamiltonian=np.asarray(arrays["H0_128"], dtype=complex),
        drive_hamiltonian=np.asarray(arrays["HV_128"], dtype=complex),
        raw_observables=np.asarray(arrays["F_128"], dtype=complex),
        initial_state=np.asarray(arrays["c0_128"], dtype=complex),
    )
    rank_160_model = PurgReducedModel(
        phonon_cutoff=16,
        cap_label=160,
        static_hamiltonian=np.asarray(arrays["H0_160"], dtype=complex),
        drive_hamiltonian=np.asarray(arrays["HV_160"], dtype=complex),
        raw_observables=np.asarray(arrays["F_160"], dtype=complex),
        initial_state=np.asarray(arrays["c0_160"], dtype=complex),
    )
    for rank, model in ((128, rank_128_model), (160, rank_160_model)):
        for name, values in (
            ("static_hamiltonian", model.static_hamiltonian),
            ("drive_hamiltonian", model.drive_hamiltonian),
            ("raw_observables", model.raw_observables),
            ("initial_state", model.initial_state),
        ):
            _verify_fingerprint(
                f"rank_{rank}_{name}",
                manifest["compressed_models"][str(rank)][name],
                _array_fingerprint(values),
            )
    pivot_history = BlindAuditResult(
        basis=rank_160_basis,
        appended_basis=np.asarray(arrays["appended_W160"], dtype=complex),
        pivot_indices=tuple(int(value) for value in arrays["pivot_indices"]),
        pivot_times=np.asarray(arrays["pivot_times"], dtype=float),
        pivot_norms=np.asarray(arrays["pivot_norms"], dtype=float),
        first_pivot_norm=float(manifest["construction"]["first_pivot_norm"]),
        orthogonality_residual=float(
            manifest["construction"]["orthogonality_residual"]
        ),
        nesting_residual=float(manifest["construction"]["nesting_residual"]),
    )
    return FrozenPreparedSystem(
        parameters=parameters,
        phonon_cutoff=16,
        full_static_hamiltonian=static_hamiltonian,
        full_drive_hamiltonian=drive_hamiltonian,
        full_raw_observables=raw_observables,
        full_initial_state=initial_state,
        rank_128_basis=rank_128_basis,
        rank_160_basis=rank_160_basis,
        rank_128_model=rank_128_model,
        rank_160_model=rank_160_model,
        pivot_history=pivot_history,
        manifest=manifest,
        manifest_sha256=str(digest),
    )


def contract_analytic_observable_path(
    *,
    times: FloatArray,
    states: ComplexArray,
    static_hamiltonian: Operator,
    drive_hamiltonian: Operator,
    raw_observables: Sequence[Operator] | ComplexArray,
    parameters: DimerParameters,
    method: str,
    maximum_norm_drift: float | None = None,
) -> ObservablePath:
    """Contract one state path with the complete analytic connected chain rule."""

    sample_times = np.asarray(times, dtype=float)
    vectors = np.asarray(states, dtype=complex)
    if vectors.shape != (sample_times.size, static_hamiltonian.shape[0]):
        raise ValueError("state path shape does not match times and Hamiltonian")
    observables = tuple(raw_observables)
    if len(observables) != len(RAW_MOMENT_NAMES):
        raise ValueError("raw observable count does not match the frozen chart")
    outputs = np.empty((sample_times.size, len(CLOSED_SCALAR_STATE_NAMES)))
    derivatives = np.empty_like(outputs)
    initial_norm = float(np.linalg.norm(vectors[0]))
    if initial_norm <= 0.0:
        raise ValueError("state path begins with a zero vector")
    measured_drift = 0.0

    for index, (time, state) in enumerate(
        zip(sample_times, vectors, strict=True)
    ):
        drive_value = parameters.drive_difference(float(time))
        state_velocity = -1j * (
            static_hamiltonian @ state
            + drive_value * (drive_hamiltonian @ state)
        )
        raw = np.empty(len(observables), dtype=float)
        raw_velocity = np.empty(len(observables), dtype=float)
        for observable_index, observable in enumerate(observables):
            action = observable @ state
            raw[observable_index] = float(np.vdot(state, action).real)
            raw_velocity[observable_index] = float(
                2.0 * np.vdot(state_velocity, action).real
            )
        outputs[index] = raw_moments_to_closed_coordinates(raw)
        derivatives[index] = raw_velocity_to_closed_velocity(raw, raw_velocity)
        measured_drift = max(
            measured_drift,
            abs(float(np.linalg.norm(state)) / initial_norm - 1.0),
        )
    if maximum_norm_drift is not None:
        measured_drift = max(measured_drift, float(maximum_norm_drift))
    return ObservablePath(
        times=sample_times.copy(),
        outputs=outputs,
        derivatives=derivatives,
        maximum_norm_drift=measured_drift,
        method=method,
    )


def propagate_observable_midpoint(
    *,
    static_hamiltonian: Operator,
    drive_hamiltonian: Operator,
    raw_observables: Sequence[Operator] | ComplexArray,
    initial_state: ComplexArray,
    parameters: DimerParameters,
    final_time: float,
    integration_step: float,
    sample_step: float,
    exponential_action_tolerance: float,
) -> ObservablePath:
    """Run exponential midpoint and score analytic observables at fixed nodes."""

    ratio = int(round(sample_step / integration_step))
    if ratio <= 0 or not np.isclose(
        ratio * integration_step,
        sample_step,
        atol=1.0e-13,
    ):
        raise ValueError("sample_step must be an integer multiple of integration_step")
    times, states, norm_drift = propagate_state_midpoint(
        static_hamiltonian,
        drive_hamiltonian,
        initial_state,
        parameters,
        final_time=final_time,
        step=integration_step,
        exponential_action_tolerance=exponential_action_tolerance,
    )
    indices = np.arange(0, times.size, ratio, dtype=int)
    return contract_analytic_observable_path(
        times=times[indices],
        states=states[indices],
        static_hamiltonian=static_hamiltonian,
        drive_hamiltonian=drive_hamiltonian,
        raw_observables=raw_observables,
        parameters=parameters,
        method=(
            f"exponential_midpoint_h={integration_step:.8g}_"
            f"rtol={exponential_action_tolerance:.1e}"
        ),
        maximum_norm_drift=norm_drift,
    )


def propagate_observable_dop853(
    *,
    static_hamiltonian: Operator,
    drive_hamiltonian: Operator,
    raw_observables: Sequence[Operator] | ComplexArray,
    initial_state: ComplexArray,
    parameters: DimerParameters,
    sample_times: FloatArray,
    relative_tolerance: float,
    absolute_tolerance: float,
    maximum_step: float,
) -> ObservablePath:
    """Run the independent DOP853 comparison without state renormalization."""

    times = np.asarray(sample_times, dtype=float)
    if times.ndim != 1 or times.size < 2 or np.any(np.diff(times) <= 0.0):
        raise ValueError("sample_times must be a strictly increasing grid")

    def rhs(time: float, state: ComplexArray) -> ComplexArray:
        return -1j * (
            static_hamiltonian @ state
            + parameters.drive_difference(time) * (drive_hamiltonian @ state)
        )

    solution = solve_ivp(
        rhs,
        (float(times[0]), float(times[-1])),
        np.asarray(initial_state, dtype=complex),
        method="DOP853",
        t_eval=times,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
        max_step=maximum_step,
    )
    if not solution.success or solution.y.shape[1] != times.size:
        raise RuntimeError(f"DOP853 propagation failed: {solution.message}")
    states = np.asarray(solution.y.T, dtype=complex)
    initial_norm = float(np.linalg.norm(states[0]))
    norm_drift = float(
        np.max(np.abs(np.linalg.norm(states, axis=1) / initial_norm - 1.0))
    )
    return contract_analytic_observable_path(
        times=np.asarray(solution.t, dtype=float),
        states=states,
        static_hamiltonian=static_hamiltonian,
        drive_hamiltonian=drive_hamiltonian,
        raw_observables=raw_observables,
        parameters=parameters,
        method=(
            f"DOP853_rtol={relative_tolerance:.1e}_"
            f"atol={absolute_tolerance:.1e}_maxstep={maximum_step:.8g}"
        ),
        maximum_norm_drift=norm_drift,
    )


def build_propagation_family(
    *,
    static_hamiltonian: Operator,
    drive_hamiltonian: Operator,
    raw_observables: Sequence[Operator] | ComplexArray,
    initial_state: ComplexArray,
    parameters: DimerParameters,
    score_config: FrozenScoreConfig,
    fallback: bool,
) -> PropagationFamily:
    """Generate all independently resolved paths for one Hilbert space."""

    fine_step = (
        score_config.fallback_fine_step if fallback else score_config.fine_step
    )
    coarse_step = (
        score_config.fallback_coarse_step
        if fallback
        else score_config.coarse_step
    )
    dop853_maximum_step = (
        score_config.fallback_dop853_maximum_step
        if fallback
        else score_config.dop853_maximum_step
    )
    common = {
        "static_hamiltonian": static_hamiltonian,
        "drive_hamiltonian": drive_hamiltonian,
        "raw_observables": raw_observables,
        "initial_state": initial_state,
        "parameters": parameters,
        "final_time": score_config.final_time,
    }
    fine_primary = propagate_observable_midpoint(
        **common,
        integration_step=fine_step,
        sample_step=score_config.score_step,
        exponential_action_tolerance=(
            score_config.primary_exponential_tolerance
        ),
    )
    fine_repeat = propagate_observable_midpoint(
        **common,
        integration_step=fine_step,
        sample_step=score_config.score_step,
        exponential_action_tolerance=score_config.repeat_exponential_tolerance,
    )
    coarse_primary = propagate_observable_midpoint(
        **common,
        integration_step=coarse_step,
        sample_step=coarse_step,
        exponential_action_tolerance=(
            score_config.primary_exponential_tolerance
        ),
    )
    coarse_repeat = propagate_observable_midpoint(
        **common,
        integration_step=coarse_step,
        sample_step=coarse_step,
        exponential_action_tolerance=score_config.repeat_exponential_tolerance,
    )
    dop853 = propagate_observable_dop853(
        static_hamiltonian=static_hamiltonian,
        drive_hamiltonian=drive_hamiltonian,
        raw_observables=raw_observables,
        initial_state=initial_state,
        parameters=parameters,
        sample_times=_time_grid(score_config.final_time, score_config.score_step),
        relative_tolerance=score_config.dop853_relative_tolerance,
        absolute_tolerance=score_config.dop853_absolute_tolerance,
        maximum_step=dop853_maximum_step,
    )
    return PropagationFamily(
        fine_primary=fine_primary,
        fine_repeat=fine_repeat,
        coarse_primary=coarse_primary,
        coarse_repeat=coarse_repeat,
        dop853=dop853,
    )


def _assert_same_grid(first: ObservablePath, second: ObservablePath) -> None:
    np.testing.assert_allclose(first.times, second.times, atol=1.0e-13, rtol=0.0)


def _path_difference(
    first: ObservablePath,
    second: ObservablePath,
) -> tuple[FloatArray, FloatArray]:
    _assert_same_grid(first, second)
    return first.outputs - second.outputs, first.derivatives - second.derivatives


def _restrict_to_grid(
    fine: ObservablePath,
    coarse: ObservablePath,
) -> ObservablePath:
    indices = np.searchsorted(fine.times, coarse.times)
    if np.any(indices >= fine.times.size):
        raise ValueError("coarse grid lies outside fine grid")
    np.testing.assert_allclose(
        fine.times[indices], coarse.times, atol=1.0e-13, rtol=0.0
    )
    return ObservablePath(
        times=fine.times[indices],
        outputs=fine.outputs[indices],
        derivatives=fine.derivatives[indices],
        maximum_norm_drift=fine.maximum_norm_drift,
        method=f"{fine.method}_restricted_to_{coarse.method}",
    )


def block_path_metrics(
    output_errors: FloatArray,
    derivative_errors: FloatArray,
) -> dict[str, float]:
    """Compute the registered discrete Euclidean block RMS and maximum."""

    outputs = np.asarray(output_errors, dtype=float)
    derivatives = np.asarray(derivative_errors, dtype=float)
    if outputs.ndim != 2 or outputs.shape != derivatives.shape:
        raise ValueError("output and derivative error paths must share a matrix shape")
    if outputs.shape[1] != len(CLOSED_SCALAR_STATE_NAMES):
        raise ValueError("error paths must use the frozen 31-coordinate chart")
    metrics: dict[str, float] = {}
    for quantity, values in (("output", outputs), ("derivative", derivatives)):
        for block, block_slice in _BLOCK_SLICES.items():
            norms = np.linalg.norm(values[:, block_slice], axis=1)
            metrics[_metric_key(quantity, block, "rms")] = float(
                np.sqrt(np.mean(norms**2))
            )
            metrics[_metric_key(quantity, block, "max")] = float(np.max(norms))
    return metrics


def _difference_metrics(
    first: ObservablePath,
    second: ObservablePath,
) -> dict[str, float]:
    output, derivative = _path_difference(first, second)
    return block_path_metrics(output, derivative)


def _sum_metrics(
    first: Mapping[str, float],
    second: Mapping[str, float],
) -> dict[str, float]:
    return {key: float(first[key] + second[key]) for key in _METRIC_KEYS}


def _max_component(
    components: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    return {
        key: max(float(component[key]) for component in components.values())
        for key in _METRIC_KEYS
    }


def _family_resolution_components(
    full: PropagationFamily,
    reduced: PropagationFamily,
) -> dict[str, dict[str, float]]:
    method = _sum_metrics(
        _difference_metrics(full.fine_primary, full.dop853),
        _difference_metrics(reduced.fine_primary, reduced.dop853),
    )
    full_restricted = _restrict_to_grid(full.fine_primary, full.coarse_primary)
    reduced_restricted = _restrict_to_grid(
        reduced.fine_primary, reduced.coarse_primary
    )
    step = _sum_metrics(
        _difference_metrics(full_restricted, full.coarse_primary),
        _difference_metrics(reduced_restricted, reduced.coarse_primary),
    )
    tolerance = _sum_metrics(
        _difference_metrics(full.fine_primary, full.fine_repeat),
        _difference_metrics(reduced.fine_primary, reduced.fine_repeat),
    )
    return {"method": method, "step": step, "tolerance": tolerance}


def _rank_resolution_components(
    rank_160: PropagationFamily,
    rank_128: PropagationFamily,
) -> dict[str, dict[str, float]]:
    method = _sum_metrics(
        _difference_metrics(rank_160.fine_primary, rank_160.dop853),
        _difference_metrics(rank_128.fine_primary, rank_128.dop853),
    )
    rank_160_restricted = _restrict_to_grid(
        rank_160.fine_primary, rank_160.coarse_primary
    )
    rank_128_restricted = _restrict_to_grid(
        rank_128.fine_primary, rank_128.coarse_primary
    )
    step = _sum_metrics(
        _difference_metrics(rank_160_restricted, rank_160.coarse_primary),
        _difference_metrics(rank_128_restricted, rank_128.coarse_primary),
    )
    tolerance = _sum_metrics(
        _difference_metrics(rank_160.fine_primary, rank_160.fine_repeat),
        _difference_metrics(rank_128.fine_primary, rank_128.fine_repeat),
    )
    return {"method": method, "step": step, "tolerance": tolerance}


def _tolerance_repeat_metrics(
    families: Mapping[str, PropagationFamily],
) -> dict[str, float]:
    per_path: list[dict[str, float]] = []
    for family in families.values():
        per_path.append(
            _difference_metrics(family.fine_primary, family.fine_repeat)
        )
        per_path.append(
            _difference_metrics(family.coarse_primary, family.coarse_repeat)
        )
    return {
        key: max(metrics[key] for metrics in per_path) for key in _METRIC_KEYS
    }


def evaluate_resolution_and_science(
    *,
    full: PropagationFamily,
    rank_128: PropagationFamily,
    rank_160: PropagationFamily,
    config: FrozenScoreConfig | None = None,
) -> ResolutionEvaluation:
    """Evaluate numerical gates first and scientific gates only after a pass."""

    resolved = config or FrozenScoreConfig()
    model_errors = {
        128: _difference_metrics(full.fine_primary, rank_128.fine_primary),
        160: _difference_metrics(full.fine_primary, rank_160.fine_primary),
    }
    model_components = {
        128: _family_resolution_components(full, rank_128),
        160: _family_resolution_components(full, rank_160),
    }
    model_resolution = {
        rank: _max_component(components)
        for rank, components in model_components.items()
    }
    rank_difference = _difference_metrics(
        rank_160.fine_primary, rank_128.fine_primary
    )
    rank_components = _rank_resolution_components(rank_160, rank_128)
    rank_resolution = _max_component(rank_components)
    families = {"full": full, "rank_128": rank_128, "rank_160": rank_160}
    tolerance_repeat = _tolerance_repeat_metrics(families)
    norm_drifts = {
        f"{name}.{path_name}": path.maximum_norm_drift
        for name, family in families.items()
        for path_name, path in (
            ("fine_primary", family.fine_primary),
            ("fine_repeat", family.fine_repeat),
            ("coarse_primary", family.coarse_primary),
            ("coarse_repeat", family.coarse_repeat),
            ("dop853", family.dop853),
        )
    }

    numerical_failures: list[str] = []
    for rank in (128, 160):
        for key in _METRIC_KEYS:
            limit = resolved.numerical_budget_fraction * _BUDGET_BY_KEY[key]
            if model_resolution[rank][key] > limit:
                numerical_failures.append(
                    f"rank_{rank}.resolution.{key}="
                    f"{model_resolution[rank][key]:.8g}>{limit:.8g}"
                )
    for key in _METRIC_KEYS:
        limit = resolved.numerical_budget_fraction * _BUDGET_BY_KEY[key]
        if rank_resolution[key] > limit:
            numerical_failures.append(
                f"rank_difference.resolution.{key}="
                f"{rank_resolution[key]:.8g}>{limit:.8g}"
            )
        if tolerance_repeat[key] > limit:
            numerical_failures.append(
                f"tolerance_repeat.{key}={tolerance_repeat[key]:.8g}>{limit:.8g}"
            )
    for name, drift in norm_drifts.items():
        if drift > resolved.norm_drift_tolerance:
            numerical_failures.append(
                f"norm_drift.{name}={drift:.8g}>{resolved.norm_drift_tolerance:.8g}"
            )
    numerical_passed = not numerical_failures

    scientific_failures: list[str] = []
    scientific_passed: bool | None = None
    if numerical_passed:
        for rank in (128, 160):
            for key in _METRIC_KEYS:
                corrected = model_errors[rank][key] + model_resolution[rank][key]
                if corrected > _BUDGET_BY_KEY[key]:
                    scientific_failures.append(
                        f"rank_{rank}.scientific.{key}="
                        f"{corrected:.8g}>{_BUDGET_BY_KEY[key]:.8g}"
                    )
        for key in _METRIC_KEYS:
            corrected = rank_difference[key] + rank_resolution[key]
            limit = resolved.rank_guard_fraction * _BUDGET_BY_KEY[key]
            if corrected > limit:
                scientific_failures.append(
                    f"rank_guard.{key}={corrected:.8g}>{limit:.8g}"
                )
        scientific_passed = not scientific_failures

    return ResolutionEvaluation(
        numerical_passed=numerical_passed,
        numerical_failures=tuple(numerical_failures),
        model_errors=model_errors,
        model_resolution=model_resolution,
        model_resolution_components=model_components,
        rank_difference=rank_difference,
        rank_resolution=rank_resolution,
        rank_resolution_components=rank_components,
        tolerance_repeat=tolerance_repeat,
        norm_drifts=norm_drifts,
        scientific_passed=scientific_passed,
        scientific_failures=tuple(scientific_failures),
    )


def _build_all_families(
    prepared: FrozenPreparedSystem,
    *,
    config: FrozenScoreConfig,
    fallback: bool,
) -> dict[str, PropagationFamily]:
    """Open the scorer and generate the full/rank-128/rank-160 paths."""

    full = build_propagation_family(
        static_hamiltonian=prepared.full_static_hamiltonian,
        drive_hamiltonian=prepared.full_drive_hamiltonian,
        raw_observables=prepared.full_raw_observables,
        initial_state=prepared.full_initial_state,
        parameters=prepared.parameters,
        score_config=config,
        fallback=fallback,
    )
    rank_128 = build_propagation_family(
        static_hamiltonian=prepared.rank_128_model.static_hamiltonian,
        drive_hamiltonian=prepared.rank_128_model.drive_hamiltonian,
        raw_observables=prepared.rank_128_model.raw_observables,
        initial_state=prepared.rank_128_model.initial_state,
        parameters=prepared.parameters,
        score_config=config,
        fallback=fallback,
    )
    rank_160 = build_propagation_family(
        static_hamiltonian=prepared.rank_160_model.static_hamiltonian,
        drive_hamiltonian=prepared.rank_160_model.drive_hamiltonian,
        raw_observables=prepared.rank_160_model.raw_observables,
        initial_state=prepared.rank_160_model.initial_state,
        parameters=prepared.parameters,
        score_config=config,
        fallback=fallback,
    )
    return {"full": full, "rank_128": rank_128, "rank_160": rank_160}


def _evaluation_json(
    evaluation: ResolutionEvaluation,
) -> dict[str, Any]:
    scientific = None
    if evaluation.numerical_passed:
        scientific = {
            "passed": evaluation.scientific_passed,
            "failures": list(evaluation.scientific_failures),
            "model_errors": evaluation.model_errors,
            "rank_difference": evaluation.rank_difference,
        }
    return {
        "numerical": {
            "passed": evaluation.numerical_passed,
            "failures": list(evaluation.numerical_failures),
            "model_resolution": evaluation.model_resolution,
            "model_resolution_components": (
                evaluation.model_resolution_components
            ),
            "rank_resolution": evaluation.rank_resolution,
            "rank_resolution_components": (
                evaluation.rank_resolution_components
            ),
            "tolerance_repeat": evaluation.tolerance_repeat,
            "norm_drifts": evaluation.norm_drifts,
        },
        "scientific": scientific,
    }


def _authoritative_arrays(
    families: Mapping[str, PropagationFamily],
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name, family in families.items():
        for path_name, path in (
            ("fine_primary", family.fine_primary),
            ("fine_repeat", family.fine_repeat),
            ("coarse_primary", family.coarse_primary),
            ("coarse_repeat", family.coarse_repeat),
            ("dop853", family.dop853),
        ):
            prefix = f"{name}_{path_name}"
            arrays[f"{prefix}_times"] = path.times
            arrays[f"{prefix}_outputs"] = path.outputs
            arrays[f"{prefix}_derivatives"] = path.derivatives
    return arrays


def _serialize_rank_128_model(
    path: Path,
    prepared: FrozenPreparedSystem,
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite promoted model: {path}")
    np.savez_compressed(
        path,
        phonon_cutoff=np.asarray([16], dtype=np.int64),
        cap_label=np.asarray([128], dtype=np.int64),
        static_hamiltonian=prepared.rank_128_model.static_hamiltonian,
        drive_hamiltonian=prepared.rank_128_model.drive_hamiltonian,
        raw_observables=prepared.rank_128_model.raw_observables,
        initial_state=prepared.rank_128_model.initial_state,
        pre_scorer_manifest_sha256=np.asarray(
            [prepared.manifest_sha256], dtype="U64"
        ),
        validation_scope=np.asarray(
            [
                "empirical_cutoff16_correlated_ground_pulse_t0_to_4_"
                "score_grid_0p0025"
            ],
            dtype="U96",
        ),
    )


def run_sealed_score(
    prepared_directory: Path,
    output_directory: Path,
    *,
    repo_root: Path,
    config: FrozenScoreConfig | None = None,
) -> dict[str, Any]:
    """Run the ordered scorer, including at most one fixed numerical fallback."""

    if output_directory.exists():
        raise FileExistsError(
            f"refusing to overwrite sealed score directory: {output_directory}"
        )
    resolved = config or FrozenScoreConfig()
    if resolved != FrozenScoreConfig():
        raise ValueError("the sealed production scorer uses only the frozen config")
    prepared = load_prepared_score(
        prepared_directory,
        repo_root=repo_root,
        require_unconsumed=True,
    )
    _write_json_exclusive(
        prepared_directory / _CONSUMPTION_RECEIPT,
        {
            "schema": "paper5.purg.scorer_consumption.v1",
            "status": "full_space_scorer_opened_manifest_consumed",
            "pre_scorer_manifest_sha256": prepared.manifest_sha256,
            "score_output_directory": str(output_directory.resolve()),
            "same_reference_rescore_permitted": False,
            "source_files": _source_hashes(),
            "solver_environment": _solver_environment(),
        },
    )

    initial_families = _build_all_families(
        prepared,
        config=resolved,
        fallback=False,
    )
    initial_evaluation = evaluate_resolution_and_science(
        full=initial_families["full"],
        rank_128=initial_families["rank_128"],
        rank_160=initial_families["rank_160"],
        config=resolved,
    )
    fallback_used = False
    final_families = initial_families
    final_evaluation = initial_evaluation
    if not initial_evaluation.numerical_passed:
        fallback_used = True
        final_families = _build_all_families(
            prepared,
            config=resolved,
            fallback=True,
        )
        final_evaluation = evaluate_resolution_and_science(
            full=final_families["full"],
            rank_128=final_families["rank_128"],
            rank_160=final_families["rank_160"],
            config=resolved,
        )

    if not final_evaluation.numerical_passed:
        status = "indeterminate_numerical_stop"
        passed = False
        permitted_conclusion = (
            "rank-128 PURG remains unvalidated; no closure conclusion follows"
        )
    elif final_evaluation.scientific_passed:
        status = "passed_empirical_rank_128_validation"
        passed = True
        permitted_conclusion = (
            "At cutoff 16 for the frozen correlated preparation, pulse, "
            "0<=t<=4, ranks 128/160, and 1601-node grid, rank-128 PURG "
            "meets the registered empirically resolved output and derivative "
            "budgets."
        )
    else:
        status = "scientific_hard_stop"
        passed = False
        permitted_conclusion = (
            "terminate PURG for this preparation/protocol; any memory closure "
            "requires a separate preregistration and independent holdout"
        )

    output_directory.mkdir(parents=True, exist_ok=False)
    arrays_path = output_directory / "score_arrays.npz"
    np.savez_compressed(arrays_path, **_authoritative_arrays(final_families))
    arrays_record = {
        "sha256": _sha256_file(arrays_path),
        "bytes": arrays_path.stat().st_size,
    }
    model_record: dict[str, Any] | None = None
    if passed:
        model_path = output_directory / "rank_128_model.npz"
        _serialize_rank_128_model(model_path, prepared)
        model_record = {
            "sha256": _sha256_file(model_path),
            "bytes": model_path.stat().st_size,
            "rank": 128,
            "rank_160_promoted": False,
        }

    summary = {
        "schema": _SCORE_SCHEMA,
        "status": status,
        "passed": passed,
        "pre_scorer_manifest_sha256": prepared.manifest_sha256,
        "config": asdict(resolved),
        "fallback_used": fallback_used,
        "fallback_count": int(fallback_used),
        "initial_attempt": _evaluation_json(initial_evaluation),
        "final_attempt": _evaluation_json(final_evaluation),
        "score_arrays": arrays_record,
        "serialized_model": model_record,
        "rank_160_role": "audit_only_not_promoted",
        "permitted_conclusion": permitted_conclusion,
        "prohibited_followup": [
            "same_reference_rescore",
            "basis_rotation_from_scorer",
            "rank_or_threshold_tuning",
            "rank_160_rescue_of_rank_128",
            "correction_or_dual_acceptance",
        ],
    }
    _write_json_exclusive(output_directory / "score_summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[4],
        help="Holstein repository root used for the frozen source revision",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser(
        "prepare",
        help="build blind W160 and freeze the pre-scorer manifest",
    )
    prepare.add_argument("--construction-directory", type=Path, required=True)
    prepare.add_argument("--output-directory", type=Path, required=True)
    score = subparsers.add_parser(
        "score",
        help="open the verified full-space scorer and run the frozen decision",
    )
    score.add_argument("--prepared-directory", type=Path, required=True)
    score.add_argument("--output-directory", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    repo_root = args.repo_root.resolve()
    if args.command == "prepare":
        prepared = prepare_sealed_score(
            args.construction_directory.resolve(),
            args.output_directory.resolve(),
            repo_root=repo_root,
        )
        report = {
            "status": "passed_blind_construction_scorer_still_unopened",
            "manifest_sha256": prepared.manifest_sha256,
            "W128_sha256": prepared.manifest["bases"]["W128"]["sha256"],
            "W160_sha256": prepared.manifest["bases"]["W160"]["sha256"],
            "output_directory": str(args.output_directory.resolve()),
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.command == "score":
        summary = run_sealed_score(
            args.prepared_directory.resolve(),
            args.output_directory.resolve(),
            repo_root=repo_root,
        )
        print(
            json.dumps(
                {
                    "status": summary["status"],
                    "passed": summary["passed"],
                    "fallback_used": summary["fallback_used"],
                    "output_directory": str(args.output_directory.resolve()),
                },
                indent=2,
                sort_keys=True,
            )
        )
        if summary["status"] == "passed_empirical_rank_128_validation":
            return 0
        if summary["status"] == "scientific_hard_stop":
            return 2
        return 3
    raise AssertionError(f"unexpected command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
