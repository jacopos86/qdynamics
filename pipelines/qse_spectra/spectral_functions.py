"""Diagnostic spectral-function postprocessing for the QSE spectra sidecar.

The helpers in this module are intentionally pure: they consume already-computed
QSE transition strengths, eigenvalues, and basis matrix vectors, then produce
JSON-ready diagnostic payloads.  They do not change the QSE solve and must not be
used as realtime/controller decision inputs.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.qse_spectra.core import QSEResult


_KERNELS = {"lorentzian", "gaussian"}
_CUTOFF_ENCODINGS = {"binary", "unary"}
_CONTROLLER_BOUNDARY = {
    "feeds_controller_decisions": False,
    "controller_usable": False,
    "post_run_diagnostic_only": True,
}


@dataclass(frozen=True)
class SpectralGrid:
    """Uniform frequency grid for diagnostic broadened spectra."""

    omega_min: float
    omega_max: float
    num_points: int

    def __post_init__(self) -> None:
        omega_min = _finite_float(self.omega_min, name="omega_min")
        omega_max = _finite_float(self.omega_max, name="omega_max")
        num_points = _strict_int(self.num_points, name="num_points", min_value=2)
        if not omega_min < omega_max:
            raise ValueError("SpectralGrid requires omega_min < omega_max.")
        object.__setattr__(self, "omega_min", omega_min)
        object.__setattr__(self, "omega_max", omega_max)
        object.__setattr__(self, "num_points", num_points)

    def values(self) -> np.ndarray:
        return np.linspace(float(self.omega_min), float(self.omega_max), int(self.num_points), dtype=float)

    def to_manifest(self, *, include_values: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "omega_min": float(self.omega_min),
            "omega_max": float(self.omega_max),
            "num_points": int(self.num_points),
        }
        if include_values:
            out["values"] = [float(x) for x in self.values()]
        return out


@dataclass(frozen=True)
class BroadeningKernelConfig:
    """Continuous unit-area broadening kernel configuration."""

    kernel: str
    eta: float
    normalization: str = "unit_area_continuous"

    def __post_init__(self) -> None:
        kernel = str(self.kernel).strip().lower()
        if kernel not in _KERNELS:
            raise ValueError(f"kernel must be one of {sorted(_KERNELS)!r}.")
        eta = _finite_float(self.eta, name="eta")
        if eta <= 0.0:
            raise ValueError("eta must be positive.")
        normalization = str(self.normalization).strip()
        if normalization != "unit_area_continuous":
            raise ValueError("Only normalization='unit_area_continuous' is supported.")
        object.__setattr__(self, "kernel", kernel)
        object.__setattr__(self, "eta", eta)
        object.__setattr__(self, "normalization", normalization)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "name": str(self.kernel),
            "eta": float(self.eta),
            "normalization": str(self.normalization),
        }


@dataclass(frozen=True)
class SpectralWindow:
    """Closed diagnostic frequency window used for integrated spectral metrics."""

    name: str
    omega_min: float
    omega_max: float

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if name == "":
            raise ValueError("SpectralWindow.name must be non-empty.")
        omega_min = _finite_float(self.omega_min, name=f"{name}.omega_min")
        omega_max = _finite_float(self.omega_max, name=f"{name}.omega_max")
        if not omega_min < omega_max:
            raise ValueError("SpectralWindow requires omega_min < omega_max.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "omega_min", omega_min)
        object.__setattr__(self, "omega_max", omega_max)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "name": str(self.name),
            "omega_min": float(self.omega_min),
            "omega_max": float(self.omega_max),
        }


@dataclass(frozen=True)
class SpectralReference:
    """Optional reference spectrum used only for diagnostic comparisons."""

    observable_name: str
    grid: Sequence[float]
    values: Sequence[float]
    label: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        observable_name = str(self.observable_name).strip()
        if observable_name == "":
            raise ValueError("SpectralReference.observable_name must be non-empty.")
        grid = tuple(_finite_float(x, name="SpectralReference.grid") for x in self.grid)
        values = tuple(_finite_float(x, name="SpectralReference.values") for x in self.values)
        if len(grid) != len(values):
            raise ValueError("SpectralReference grid and values lengths must match.")
        if len(grid) < 2:
            raise ValueError("SpectralReference requires at least two grid points.")
        if any(grid[idx] >= grid[idx + 1] for idx in range(len(grid) - 1)):
            raise ValueError("SpectralReference grid must be strictly increasing.")
        label = None if self.label is None else str(self.label)
        if label is not None and label.strip() == "":
            label = None
        if not isinstance(self.metadata, Mapping):
            raise TypeError("SpectralReference.metadata must be a mapping.")
        object.__setattr__(self, "observable_name", observable_name)
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_manifest(self) -> dict[str, Any]:
        return {
            "observable_name": str(self.observable_name),
            "label": self.label,
            "grid_points": int(len(self.grid)),
            "omega_min": float(self.grid[0]),
            "omega_max": float(self.grid[-1]),
            "metadata": _json_safe_mapping(self.metadata),
        }


@dataclass(frozen=True)
class CutoffBoundaryLayout:
    """Explicit boson cutoff layout for QSE-root boundary diagnostics."""

    num_sites: int
    n_ph_max: int
    boson_encoding: str
    fermion_qubits: int = 0

    def __post_init__(self) -> None:
        num_sites = _strict_int(self.num_sites, name="num_sites", min_value=1)
        n_ph_max = _strict_int(self.n_ph_max, name="n_ph_max", min_value=0)
        boson_encoding = str(self.boson_encoding).strip().lower()
        if boson_encoding not in _CUTOFF_ENCODINGS:
            raise ValueError(f"boson_encoding must be one of {sorted(_CUTOFF_ENCODINGS)!r}.")
        fermion_qubits = _strict_int(self.fermion_qubits, name="fermion_qubits", min_value=0)
        object.__setattr__(self, "num_sites", num_sites)
        object.__setattr__(self, "n_ph_max", n_ph_max)
        object.__setattr__(self, "boson_encoding", boson_encoding)
        object.__setattr__(self, "fermion_qubits", fermion_qubits)

    @property
    def qubits_per_boson_site(self) -> int:
        if self.boson_encoding == "binary":
            return int(self.n_ph_max).bit_length()
        if self.boson_encoding == "unary":
            return int(self.n_ph_max) + 1
        raise ValueError(f"Unsupported boson_encoding {self.boson_encoding!r}.")

    @property
    def total_qubits(self) -> int:
        return int(self.fermion_qubits) + int(self.num_sites) * int(self.qubits_per_boson_site)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "num_sites": int(self.num_sites),
            "n_ph_max": int(self.n_ph_max),
            "boson_encoding": str(self.boson_encoding),
            "fermion_qubits": int(self.fermion_qubits),
            "qubits_per_boson_site": int(self.qubits_per_boson_site),
            "total_qubits": int(self.total_qubits),
            "layout_order": "fermion_qubits_then_site_boson_blocks_low_to_high_qubit_offsets",
        }


def lorentzian_kernel(offsets: np.ndarray | Sequence[float] | float, eta: float) -> np.ndarray:
    """Evaluate ``(eta / pi) / (x**2 + eta**2)``."""

    eta_f = _positive_eta(eta)
    x = np.asarray(offsets, dtype=float)
    _finite_real_array(x, name="lorentzian offsets")
    return np.asarray((eta_f / math.pi) / (x * x + eta_f * eta_f), dtype=float)


def gaussian_kernel(offsets: np.ndarray | Sequence[float] | float, eta: float) -> np.ndarray:
    """Evaluate a unit-area Gaussian with standard deviation ``eta``."""

    eta_f = _positive_eta(eta)
    x = np.asarray(offsets, dtype=float)
    _finite_real_array(x, name="gaussian offsets")
    return np.asarray(
        (1.0 / (eta_f * math.sqrt(2.0 * math.pi))) * np.exp(-0.5 * (x / eta_f) ** 2),
        dtype=float,
    )


def evaluate_broadening_kernel(
    offsets: np.ndarray | Sequence[float] | float,
    config: BroadeningKernelConfig,
) -> np.ndarray:
    if config.kernel == "lorentzian":
        return lorentzian_kernel(offsets, config.eta)
    if config.kernel == "gaussian":
        return gaussian_kernel(offsets, config.eta)
    raise ValueError(f"Unsupported kernel {config.kernel!r}.")


def parse_spectral_window_spec(spec: str, *, index: int = 0) -> SpectralWindow:
    """Parse ``min:max`` or ``name:min:max`` CLI window syntax."""

    parts = [part.strip() for part in str(spec).split(":")]
    if len(parts) == 2:
        name = f"window_{int(index)}"
        omega_min, omega_max = parts
    elif len(parts) == 3:
        name, omega_min, omega_max = parts
        if name == "":
            raise ValueError(f"Invalid spectral window {spec!r}: name must be non-empty.")
    else:
        raise ValueError(f"Invalid spectral window {spec!r}; expected min:max or name:min:max.")
    return SpectralWindow(name=name, omega_min=float(omega_min), omega_max=float(omega_max))


def load_spectral_references_json(path: Path) -> tuple[SpectralReference, ...]:
    """Load optional diagnostic reference spectra from a small JSON file.

    Accepted shapes are intentionally lightweight:
    - ``{"references": [{"observable_name": ..., "grid": [...], "values": [...]}]}``
    - ``{"observable_name": ..., "grid": [...], "values": [...]}``
    - ``{"grid": [...], "observables": {"name": [...]}}``
    """

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    references = _references_from_payload(payload)
    if not references:
        raise ValueError(f"No spectral references found in {path}.")
    return tuple(references)


def build_spectral_function_payload(
    result: QSEResult,
    *,
    grid: SpectralGrid,
    kernel_config: BroadeningKernelConfig,
) -> dict[str, Any]:
    """Build broadened spectral functions from QSE transition strengths."""

    if not result.transition_observables:
        raise ValueError("Spectral functions require at least one QSE transition observable.")
    grid_values = grid.values()
    _finite_real_array(grid_values, name="spectral grid")
    energies = np.asarray(result.eigenvalues, dtype=float).reshape(-1)
    _finite_real_array(energies, name="QSE eigenvalues")
    reference_energy = _finite_float(result.matrices.reference_energy, name="reference_energy")
    omegas = np.asarray(energies - reference_energy, dtype=float)

    observables: list[dict[str, Any]] = []
    for transition in result.transition_observables:
        strengths = np.asarray(transition.transition_strengths, dtype=float).reshape(-1)
        _finite_real_array(strengths, name=f"transition strengths {transition.observable.name}")
        if strengths.shape != energies.shape:
            raise ValueError(
                f"Transition strengths for {transition.observable.name!r} have length {strengths.size}; "
                f"expected {energies.size}."
            )
        if np.any(strengths < -1.0e-12):
            raise ValueError(f"Transition strengths for {transition.observable.name!r} contain negative values.")
        strengths = np.maximum(strengths, 0.0)

        values = np.zeros_like(grid_values, dtype=float)
        roots: list[dict[str, Any]] = []
        for state_index, (energy, omega, strength) in enumerate(zip(energies, omegas, strengths, strict=True)):
            values += float(strength) * evaluate_broadening_kernel(grid_values - float(omega), kernel_config)
            roots.append(
                {
                    "state_index": int(state_index),
                    "energy": float(energy),
                    "omega": float(omega),
                    "transition_strength": float(strength),
                }
            )
        _finite_real_array(values, name=f"spectral values {transition.observable.name}")
        peak_idx = int(np.argmax(values)) if values.size else 0
        observables.append(
            {
                "name": str(transition.observable.name),
                "kind": str(transition.observable.kind),
                "values": [float(x) for x in values],
                "roots": roots,
                "area_trapezoid": _trapz(values, grid_values),
                "peak_omega": float(grid_values[peak_idx]),
                "peak_value": float(values[peak_idx]),
            }
        )

    return {
        "schema_version": "qse_spectral_functions_v1",
        "policy": "diagnostic_only_transition_strength_postprocessing",
        "controller_boundary": dict(_CONTROLLER_BOUNDARY),
        "grid": grid.to_manifest(include_values=True),
        "kernel": kernel_config.to_manifest(),
        "reference_energy": float(reference_energy),
        "observables": observables,
    }


def build_spectral_window_metrics_payload(
    spectral_functions_payload: Mapping[str, Any],
    *,
    windows: Sequence[SpectralWindow],
    references: Sequence[SpectralReference] | None = None,
) -> dict[str, Any]:
    """Compute per-window spectral metrics and optional reference comparisons."""

    window_tuple = tuple(windows)
    if not window_tuple:
        raise ValueError("At least one SpectralWindow is required for spectral-window metrics.")
    grid_block = _mapping(spectral_functions_payload.get("grid"), name="spectral_functions.grid")
    grid_values = np.asarray(_sequence(grid_block.get("values"), name="spectral_functions.grid.values"), dtype=float)
    _validate_strict_grid(grid_values, name="spectral_functions.grid.values")
    reference_by_name = {ref.observable_name: ref for ref in tuple(references or ())}

    out_observables: list[dict[str, Any]] = []
    for obs_index, raw_obs in enumerate(_sequence(spectral_functions_payload.get("observables"), name="spectral_functions.observables")):
        obs = _mapping(raw_obs, name=f"spectral_functions.observables[{obs_index}]")
        name = str(obs.get("name", f"observable_{obs_index}"))
        values = np.asarray(_sequence(obs.get("values"), name=f"spectral_functions.observables[{obs_index}].values"), dtype=float)
        if values.shape != grid_values.shape:
            raise ValueError(f"Spectral values for {name!r} do not match grid length.")
        _finite_real_array(values, name=f"spectral values {name}")
        ref = reference_by_name.get(name)
        metrics = [
            _window_metric_record(grid_values, values, window, reference=ref)
            for window in window_tuple
        ]
        out_observables.append({"name": name, "window_metrics": metrics})

    return {
        "schema_version": "qse_spectral_window_metrics_v1",
        "policy": "diagnostic_only_spectral_window_postprocessing",
        "controller_boundary": dict(_CONTROLLER_BOUNDARY),
        "windows": [window.to_manifest() for window in window_tuple],
        "references": [ref.to_manifest() for ref in tuple(references or ())],
        "observables": out_observables,
    }


def build_spectral_postprocessing_payloads(
    result: QSEResult,
    *,
    grid: SpectralGrid,
    kernel_config: BroadeningKernelConfig,
    windows: Sequence[SpectralWindow] = (),
    references: Sequence[SpectralReference] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Build spectral-function payload plus optional window metrics payload."""

    spectral_payload = build_spectral_function_payload(result, grid=grid, kernel_config=kernel_config)
    window_payload = None
    if tuple(windows):
        window_payload = build_spectral_window_metrics_payload(
            spectral_payload,
            windows=tuple(windows),
            references=tuple(references or ()),
        )
    return spectral_payload, window_payload


def build_cutoff_boundary_diagnostics(
    result: QSEResult,
    *,
    layout: CutoffBoundaryLayout,
) -> dict[str, Any]:
    """Compute diagnostic QSE-root boson cutoff-boundary probabilities.

    QSE roots are reconstructed internally from basis matrix vectors and basis
    coefficients.  Raw reconstructed statevectors are never serialized.
    """

    if int(layout.total_qubits) != int(result.matrices.nq):
        raise ValueError(
            f"Cutoff layout expects {layout.total_qubits} qubits but QSE result has {result.matrices.nq}. "
            "Pass an explicit layout matching the sidecar Hilbert space."
        )
    basis_vectors = tuple(np.asarray(vec, dtype=complex).reshape(-1) for vec in result.matrices.basis_matrix_vectors)
    if len(basis_vectors) != len(result.matrices.basis_elements):
        raise ValueError("Cutoff diagnostics require QSE basis_matrix_vectors for every basis element.")
    if not basis_vectors:
        raise ValueError("Cutoff diagnostics require non-empty QSE basis_matrix_vectors.")
    for idx, vec in enumerate(basis_vectors):
        if vec.size != int(result.matrices.hilbert_dim):
            raise ValueError(f"basis_matrix_vectors[{idx}] dimension does not match matrices.hilbert_dim.")
        _finite_complex_array(vec, name=f"basis_matrix_vectors[{idx}]")
    phi_matrix = np.column_stack(basis_vectors)
    coeffs = np.asarray(result.eigenvectors_basis, dtype=complex)
    if coeffs.ndim != 2:
        raise ValueError("eigenvectors_basis must be a 2D array.")
    if coeffs.shape[0] != phi_matrix.shape[1]:
        raise ValueError("eigenvectors_basis row count must match number of basis matrix vectors.")
    if coeffs.shape[1] != int(np.asarray(result.eigenvalues).reshape(-1).size):
        raise ValueError("eigenvectors_basis column count must match eigenvalue count.")
    _finite_complex_array(coeffs.reshape(-1), name="eigenvectors_basis")

    energies = np.asarray(result.eigenvalues, dtype=float).reshape(-1)
    _finite_real_array(energies, name="QSE eigenvalues")
    reference_energy = _finite_float(result.matrices.reference_energy, name="reference_energy")
    qps = int(layout.qubits_per_boson_site)
    roots: list[dict[str, Any]] = []
    for state_index, energy in enumerate(energies):
        root = phi_matrix @ coeffs[:, state_index]
        _finite_complex_array(root, name=f"reconstructed QSE root {state_index}")
        norm = float(np.linalg.norm(root))
        if not math.isfinite(norm) or norm <= 0.0:
            raise ValueError(f"Reconstructed QSE root {state_index} has non-positive norm {norm}.")
        probabilities = np.abs(root / norm) ** 2
        _finite_real_array(probabilities, name=f"QSE root probabilities {state_index}")

        boundary_by_site: list[float] = []
        legal_by_site: list[float] = []
        illegal_by_site: list[float] = []
        for site in range(int(layout.num_sites)):
            boundary = 0.0
            legal = 0.0
            illegal = 0.0
            for basis_index, prob in enumerate(probabilities):
                code = _local_site_code(int(basis_index), site=site, layout=layout, qubits_per_site=qps)
                is_legal = _is_legal_code(code, layout)
                if is_legal:
                    legal += float(prob)
                else:
                    illegal += float(prob)
                if _is_boundary_code(code, layout):
                    boundary += float(prob)
            boundary_by_site.append(float(boundary))
            legal_by_site.append(float(legal))
            illegal_by_site.append(float(illegal))

        roots.append(
            {
                "state_index": int(state_index),
                "energy": float(energy),
                "omega": float(float(energy) - reference_energy),
                "norm_before_normalization": float(norm),
                "ell_cut": float(sum(boundary_by_site)),
                "boundary_probability_by_site": boundary_by_site,
                "legal_probability_by_site": legal_by_site,
                "illegal_probability_by_site": illegal_by_site,
                "legal_probability_min": float(min(legal_by_site) if legal_by_site else 0.0),
                "illegal_probability_max": float(max(illegal_by_site) if illegal_by_site else 0.0),
            }
        )

    return {
        "schema_version": "qse_cutoff_boundary_diagnostics_v1",
        "policy": "diagnostic_only_qse_root_statevector_postprocessing",
        "controller_boundary": dict(_CONTROLLER_BOUNDARY),
        "layout": layout.to_manifest(),
        "roots": roots,
    }


def _finite_float(value: Any, *, name: str) -> float:
    try:
        out = float(value)
    except Exception as exc:  # pragma: no cover - defensive conversion guard
        raise ValueError(f"{name} must be numeric; got {value!r}.") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite; got {out!r}.")
    return out


def _strict_int(value: Any, *, name: str, min_value: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer; got {value!r}.")
    out = int(value)
    if min_value is not None and out < int(min_value):
        raise ValueError(f"{name} must be >= {int(min_value)}; got {out}.")
    return out


def _positive_eta(eta: float) -> float:
    eta_f = _finite_float(eta, name="eta")
    if eta_f <= 0.0:
        raise ValueError("eta must be positive.")
    return eta_f


def _finite_real_array(values: np.ndarray, *, name: str) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values.")


def _finite_complex_array(values: np.ndarray, *, name: str) -> None:
    if not np.all(np.isfinite(np.real(values))) or not np.all(np.isfinite(np.imag(values))):
        raise ValueError(f"{name} contains non-finite values.")


def _validate_strict_grid(grid: np.ndarray, *, name: str) -> None:
    _finite_real_array(grid, name=name)
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError(f"{name} must be a 1D grid with at least two points.")
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing.")


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    if y.size < 2:
        return 0.0
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])))


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return value


def _sequence(value: Any, *, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be a sequence.")
    return value


def _window_domain(grid: np.ndarray, window: SpectralWindow) -> tuple[np.ndarray, np.ndarray]:
    lo = max(float(window.omega_min), float(grid[0]))
    hi = min(float(window.omega_max), float(grid[-1]))
    if lo > hi:
        return np.asarray([], dtype=float), np.asarray([], dtype=int)
    inside = np.nonzero((grid > lo) & (grid < hi))[0]
    x = np.concatenate(([lo], grid[inside], [hi]))
    x = np.unique(x)
    return np.asarray(x, dtype=float), inside


def _window_metric_record(
    grid: np.ndarray,
    values: np.ndarray,
    window: SpectralWindow,
    *,
    reference: SpectralReference | None,
) -> dict[str, Any]:
    x, _ = _window_domain(grid, window)
    if x.size == 0:
        return {
            "window_name": str(window.name),
            "integrated_weight": 0.0,
            "first_moment": 0.0,
            "centroid": None,
            "peak_omega": None,
            "peak_value": None,
            "points_in_window_count": 0,
            "reference_comparison": None,
        }
    y = np.interp(x, grid, values)
    weight = _trapz(y, x)
    first_moment = _trapz(x * y, x)
    centroid = None if weight <= 0.0 else float(first_moment / weight)
    peak_idx = int(np.argmax(y))
    return {
        "window_name": str(window.name),
        "integrated_weight": float(weight),
        "first_moment": float(first_moment),
        "centroid": centroid,
        "peak_omega": float(x[peak_idx]),
        "peak_value": float(y[peak_idx]),
        "points_in_window_count": int(x.size),
        "reference_comparison": None if reference is None else _reference_comparison(grid, values, window, reference),
    }


def _reference_comparison(
    grid: np.ndarray,
    values: np.ndarray,
    window: SpectralWindow,
    reference: SpectralReference,
) -> dict[str, Any] | None:
    ref_grid = np.asarray(reference.grid, dtype=float)
    ref_values = np.asarray(reference.values, dtype=float)
    lo = max(float(window.omega_min), float(grid[0]), float(ref_grid[0]))
    hi = min(float(window.omega_max), float(grid[-1]), float(ref_grid[-1]))
    if lo > hi:
        return {
            "reference_label": reference.label,
            "overlap_omega_min": None,
            "overlap_omega_max": None,
            "overlap_points_count": 0,
            "l1_error": None,
            "l2_error": None,
            "max_abs_error": None,
            "normalized_l1_error": None,
            "feeds_controller_decisions": False,
        }
    inside = np.nonzero((grid > lo) & (grid < hi))[0]
    x = np.unique(np.concatenate(([lo], grid[inside], [hi]))).astype(float)
    y = np.interp(x, grid, values)
    y_ref = np.interp(x, ref_grid, ref_values)
    diff = y - y_ref
    l1 = _trapz(np.abs(diff), x)
    l2 = math.sqrt(max(0.0, _trapz(diff * diff, x)))
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    ref_area = _trapz(np.abs(y_ref), x)
    return {
        "reference_label": reference.label,
        "overlap_omega_min": float(lo),
        "overlap_omega_max": float(hi),
        "overlap_points_count": int(x.size),
        "l1_error": float(l1),
        "l2_error": float(l2),
        "max_abs_error": max_abs,
        "normalized_l1_error": None if ref_area <= 0.0 else float(l1 / ref_area),
        "feeds_controller_decisions": False,
    }


def _references_from_payload(payload: Any) -> list[SpectralReference]:
    if isinstance(payload, Mapping) and isinstance(payload.get("references"), Sequence):
        return [_reference_from_record(item, index=idx) for idx, item in enumerate(payload["references"])]
    if isinstance(payload, Mapping) and "grid" in payload and isinstance(payload.get("observables"), Mapping):
        grid = payload["grid"]
        refs = []
        for name, values in payload["observables"].items():
            refs.append(
                SpectralReference(
                    observable_name=str(name),
                    grid=grid,
                    values=values,
                    label=payload.get("label"),
                    metadata=payload.get("metadata", {}),
                )
            )
        return refs
    if isinstance(payload, Mapping):
        return [_reference_from_record(payload, index=0)]
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return [_reference_from_record(item, index=idx) for idx, item in enumerate(payload)]
    raise ValueError("Spectral reference JSON must be a mapping or list.")


def _reference_from_record(record: Any, *, index: int) -> SpectralReference:
    if not isinstance(record, Mapping):
        raise ValueError(f"references[{index}] must be a mapping.")
    name = record.get("observable_name", record.get("observable", record.get("name")))
    if name is None:
        raise ValueError(f"references[{index}] is missing observable_name.")
    grid = record.get("grid", record.get("omega", record.get("omegas")))
    values = record.get("values", record.get("spectrum", record.get("spectral_values")))
    if grid is None or values is None:
        raise ValueError(f"references[{index}] requires grid and values.")
    return SpectralReference(
        observable_name=str(name),
        grid=grid,
        values=values,
        label=None if record.get("label") is None else str(record.get("label")),
        metadata=record.get("metadata", {}),
    )


def _local_site_code(index: int, *, site: int, layout: CutoffBoundaryLayout, qubits_per_site: int) -> int:
    if qubits_per_site == 0:
        return 0
    shift = int(layout.fermion_qubits) + int(site) * int(qubits_per_site)
    mask = (1 << int(qubits_per_site)) - 1
    return int((int(index) >> shift) & mask)


def _is_legal_code(code: int, layout: CutoffBoundaryLayout) -> bool:
    if layout.boson_encoding == "binary":
        return 0 <= int(code) <= int(layout.n_ph_max)
    if layout.boson_encoding == "unary":
        legal_codes = {1 << n for n in range(int(layout.n_ph_max) + 1)}
        return int(code) in legal_codes
    return False


def _is_boundary_code(code: int, layout: CutoffBoundaryLayout) -> bool:
    if layout.boson_encoding == "binary":
        return int(code) == int(layout.n_ph_max)
    if layout.boson_encoding == "unary":
        return int(code) == (1 << int(layout.n_ph_max))
    return False


def _json_safe_mapping(metadata: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[str(key)] = value
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            out[str(key)] = list(value)
        elif isinstance(value, Mapping):
            out[str(key)] = _json_safe_mapping(value)
        else:
            out[str(key)] = str(value)
    return out


__all__ = [
    "SpectralGrid",
    "BroadeningKernelConfig",
    "SpectralWindow",
    "SpectralReference",
    "CutoffBoundaryLayout",
    "lorentzian_kernel",
    "gaussian_kernel",
    "evaluate_broadening_kernel",
    "parse_spectral_window_spec",
    "load_spectral_references_json",
    "build_spectral_function_payload",
    "build_spectral_window_metrics_payload",
    "build_spectral_postprocessing_payloads",
    "build_cutoff_boundary_diagnostics",
]
