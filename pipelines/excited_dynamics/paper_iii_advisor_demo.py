#!/usr/bin/env python3
"""Source-locked Hubbard--Holstein QSE excited-state dynamics demonstration.

This is a local, ideal-statevector diagnostic for an advisor discussion.  It
does not claim Paper-III evidence and does not feed exact-reference data into
QSE construction, root selection, or the driven subspace evolution.

The demonstration:

1. extracts a compact static seed from a Paper-I checkpoint tarball;
2. constructs a Q0-projected canonical full-meta-plus-Hamiltonian QSE basis;
3. identifies the lowest orthogonal Ritz root (root zero) as the first-excited
   candidate and validates it against fixed-sector exact diagonalization;
4. drives that same QSE state with a weak staggered-density pulse, comparing
   retained-QSE and exact-sector midpoint propagation from identical initial
   states; and
5. writes JSON, Markdown, and a compact PNG figure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tarfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.qse_spectra.core import (
    QSEBasisVectorPolicy,
    QSEPruningConfig,
    QSEResult,
    compute_qse_spectra,
)
from pipelines.qse_spectra.hh_response_observables import (
    HHResponseLayout,
    build_hh_neutral_response_observable_bundle,
)
from pipelines.qse_spectra.io import (
    basis_elements_from_artifact_source,
    load_polynomial_json,
    load_state_json,
    qse_result_to_manifest,
    write_manifest_json,
)
from src.quantum.drives_time_potential import gaussian_sinusoid_waveform
from src.quantum.hubbard_latex_python_pairs import (
    SPIN_DN,
    SPIN_UP,
    boson_qubits_per_site,
    mode_index,
)
from src.quantum.vqe_latex_python_pairs import hamiltonian_matrix


SCHEMA_VERSION = "paper_iii_hh_advisor_demo_v1"
SOURCE_SEED_SCHEMA_VERSION = "paper_iii_source_locked_static_seed_v1"
DEFAULT_CHECKPOINT_MEMBER = "worker_outputs/checkpoint.json"


class AdvisorDemoError(ValueError):
    """Raised when the advisor demonstration cannot be validated."""


@dataclass(frozen=True)
class AdvisorDemoConfig:
    source_bundle: Path
    output_dir: Path
    checkpoint_member: str = DEFAULT_CHECKPOINT_MEMBER
    initial_root_index: int = 0
    reported_root_count: int = 8
    drive_amplitude: float = 0.20
    drive_omega: float | None = None
    drive_tbar: float = 4.0
    drive_phi: float = 0.0
    t_final: float = 10.0
    num_steps: int = 200
    overlap_relative_cutoff: float = 1.0e-10
    overlap_absolute_cutoff: float = 1.0e-12


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _validate_config(config: AdvisorDemoConfig) -> None:
    if not Path(config.source_bundle).is_file():
        raise AdvisorDemoError(f"Source checkpoint bundle not found: {config.source_bundle}")
    if int(config.initial_root_index) < 0:
        raise AdvisorDemoError("initial_root_index must be non-negative")
    if int(config.reported_root_count) < 1:
        raise AdvisorDemoError("reported_root_count must be positive")
    if not math.isfinite(float(config.drive_amplitude)):
        raise AdvisorDemoError("drive_amplitude must be finite")
    if config.drive_omega is not None and (not math.isfinite(float(config.drive_omega)) or float(config.drive_omega) <= 0.0):
        raise AdvisorDemoError("drive_omega must be finite and positive when supplied")
    if not math.isfinite(float(config.drive_tbar)) or float(config.drive_tbar) <= 0.0:
        raise AdvisorDemoError("drive_tbar must be finite and positive")
    if not math.isfinite(float(config.t_final)) or float(config.t_final) <= 0.0:
        raise AdvisorDemoError("t_final must be finite and positive")
    if int(config.num_steps) < 1:
        raise AdvisorDemoError("num_steps must be positive")


def _extract_checkpoint_fields(bundle: Path, member_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Stream only the seed fields needed from a potentially huge checkpoint."""

    try:
        import ijson
    except ImportError as exc:  # pragma: no cover - the active scientific environment provides ijson.
        raise AdvisorDemoError("Streaming checkpoint extraction requires the installed 'ijson' package") from exc

    targets = (
        "settings",
        "initial_state",
        "ansatz_input_state",
        "adapt_vqe.parameterization",
        "adapt_vqe.operators",
        "adapt_vqe.optimal_point",
        "adapt_vqe.logical_optimal_point",
        "adapt_vqe.energy",
        "adapt_vqe.ansatz_depth",
        "adapt_vqe.num_parameters",
        "adapt_vqe.logical_num_parameters",
        "adapt_vqe.method",
        "adapt_vqe.stop_reason",
        "adapt_vqe.success",
    )
    builders = {target: ijson.common.ObjectBuilder() for target in targets}
    with tarfile.open(bundle, "r:gz") as archive:
        try:
            member = archive.getmember(str(member_name))
        except KeyError as exc:
            raise AdvisorDemoError(f"Checkpoint member {member_name!r} not found in {bundle}") from exc
        handle = archive.extractfile(member)
        if handle is None:
            raise AdvisorDemoError(f"Checkpoint member {member_name!r} is not a readable file")
        for prefix, event, value in ijson.parse(handle, use_float=True):
            for target, builder in builders.items():
                if prefix == target or prefix.startswith(target + "."):
                    builder.event(event, value)

    values = {target: builder.value for target, builder in builders.items()}
    settings = values["settings"]
    initial_state = values["initial_state"]
    parameterization = values["adapt_vqe.parameterization"]
    if not isinstance(settings, Mapping) or not isinstance(initial_state, Mapping):
        raise AdvisorDemoError("Checkpoint is missing settings or initial_state")
    if not isinstance(parameterization, Mapping):
        raise AdvisorDemoError("Checkpoint is missing adapt_vqe.parameterization")

    source_lock = {
        "bundle_path": str(bundle.resolve()),
        "bundle_sha256": _sha256_file(bundle),
        "checkpoint_member": str(member_name),
        "checkpoint_member_size_bytes": int(member.size),
        "extraction_mode": "ijson_stream_selected_fields",
        "extracted_fields": list(targets),
    }
    scientific_seed_payload = {
        "settings": dict(settings),
        "initial_state": dict(initial_state),
        "ansatz_input_state": dict(values["ansatz_input_state"]),
        "adapt_vqe": {
            "parameterization": dict(parameterization),
            "operators": list(values["adapt_vqe.operators"]),
            "optimal_point": list(values["adapt_vqe.optimal_point"]),
            "logical_optimal_point": list(values["adapt_vqe.logical_optimal_point"]),
            "energy": float(values["adapt_vqe.energy"]),
            "ansatz_depth": int(values["adapt_vqe.ansatz_depth"]),
            "num_parameters": int(values["adapt_vqe.num_parameters"]),
            "logical_num_parameters": int(values["adapt_vqe.logical_num_parameters"]),
            "method": str(values["adapt_vqe.method"]),
            "stop_reason": (
                None
                if values["adapt_vqe.stop_reason"] is None
                else str(values["adapt_vqe.stop_reason"])
            ),
            "success": bool(values["adapt_vqe.success"]),
            "num_particles": {"n_up": 1, "n_dn": 1},
        },
    }
    source_lock["scientific_seed_payload_sha256"] = _sha256_json(scientific_seed_payload)
    seed = {
        "schema_version": SOURCE_SEED_SCHEMA_VERSION,
        "generated_utc": _utc_now(),
        "diagnostic_only": True,
        "source_lock": source_lock,
        **scientific_seed_payload,
    }
    source_lock["extracted_seed_payload_sha256"] = _sha256_json(seed)
    return seed, source_lock


def _num_qubits_from_polynomial(polynomial: Any) -> int:
    terms = list(polynomial.return_polynomial())
    if not terms:
        raise AdvisorDemoError("Hamiltonian contains no Pauli terms")
    return int(terms[0].nqubit())


def _require_hh_demo_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    normalized = {
        "problem": str(settings.get("problem", "")).strip().lower(),
        "L": int(settings.get("L")),
        "n_ph_max": int(settings.get("n_ph_max")),
        "boson_encoding": str(settings.get("boson_encoding", "")).strip().lower(),
        "ordering": str(settings.get("ordering", "")).strip().lower(),
        "boundary": str(settings.get("boundary", "")).strip().lower(),
    }
    expected = {
        "problem": "hh",
        "L": 2,
        "n_ph_max": 3,
        "boson_encoding": "binary",
        "ordering": "blocked",
        "boundary": "open",
    }
    if normalized != expected:
        raise AdvisorDemoError(f"This diagnostic is locked to {expected}; resolved {normalized}")
    return normalized


def _half_filled_sector_indices(
    *,
    num_sites: int,
    n_ph_max: int,
    boson_encoding: str,
    ordering: str,
    nq_total: int,
) -> np.ndarray:
    """Return the HH half-filled legal encoded sector in computational order."""

    if str(boson_encoding).strip().lower() != "binary":
        raise AdvisorDemoError("The advisor diagnostic currently supports binary boson encoding only")
    n_sites = int(num_sites)
    qpb = int(boson_qubits_per_site(int(n_ph_max), "binary"))
    expected_nq = 2 * n_sites + n_sites * qpb
    if int(nq_total) != expected_nq:
        raise AdvisorDemoError(f"Expected nq_total={expected_nq}, got {nq_total}")
    n_up = n_sites // 2
    n_dn = n_sites // 2
    up_qubits = [int(mode_index(site, SPIN_UP, indexing=str(ordering), n_sites=n_sites)) for site in range(n_sites)]
    dn_qubits = [int(mode_index(site, SPIN_DN, indexing=str(ordering), n_sites=n_sites)) for site in range(n_sites)]
    indices: list[int] = []
    for index in range(1 << int(nq_total)):
        if sum((index >> qubit) & 1 for qubit in up_qubits) != n_up:
            continue
        if sum((index >> qubit) & 1 for qubit in dn_qubits) != n_dn:
            continue
        legal = True
        for site in range(n_sites):
            base = 2 * n_sites + site * qpb
            occupation = sum(((index >> (base + offset)) & 1) << offset for offset in range(qpb))
            if occupation > int(n_ph_max):
                legal = False
                break
        if legal:
            indices.append(index)
    return np.asarray(indices, dtype=int)


def _normalize(state: np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(state, dtype=complex).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise AdvisorDemoError(f"{name} has non-positive norm")
    return vector / norm


def _expectation(state: np.ndarray, matrix: np.ndarray) -> float:
    vector = _normalize(state, name="expectation state")
    value = complex(np.vdot(vector, np.asarray(matrix, dtype=complex) @ vector))
    if abs(value.imag) > 1.0e-9 * max(1.0, abs(value.real)):
        raise AdvisorDemoError(f"Observable expectation has non-negligible imaginary part {value.imag}")
    return float(value.real)


def _ritz_states(result: QSEResult) -> tuple[np.ndarray, ...]:
    phi = np.column_stack(result.matrices.basis_matrix_vectors)
    return tuple(
        _normalize(phi @ result.eigenvectors_basis[:, root], name=f"QSE root {root}")
        for root in range(int(result.eigenvalues.size))
    )


def _match_roots_by_overlap(
    qse_states: Sequence[np.ndarray],
    exact_vectors_sector: np.ndarray,
    sector_indices: np.ndarray,
    *,
    root_count: int,
) -> dict[int, int]:
    count = min(int(root_count), len(qse_states), max(0, exact_vectors_sector.shape[1] - 1))
    if count <= 0:
        return {}
    candidate_exact = np.arange(1, exact_vectors_sector.shape[1], dtype=int)
    overlaps = np.zeros((count, candidate_exact.size), dtype=float)
    for root in range(count):
        state_sector = _normalize(qse_states[root][sector_indices], name=f"QSE root {root} sector restriction")
        overlaps[root, :] = np.abs(exact_vectors_sector[:, candidate_exact].conj().T @ state_sector) ** 2
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(1.0 - overlaps)
        return {int(row): int(candidate_exact[col]) for row, col in zip(rows, cols)}
    except ImportError:  # pragma: no cover - SciPy is part of the active numerical environment.
        available = set(int(x) for x in candidate_exact)
        mapping: dict[int, int] = {}
        for root in range(count):
            chosen = max(available, key=lambda index: float(overlaps[root, index - 1]))
            mapping[root] = int(chosen)
            available.remove(chosen)
        return mapping


def _boundary_probability(state: np.ndarray, *, num_sites: int, n_ph_max: int, nq_total: int) -> float:
    vector = _normalize(state, name="boundary diagnostic state")
    qpb = int(boson_qubits_per_site(int(n_ph_max), "binary"))
    probability = 0.0
    for index, amplitude in enumerate(vector):
        on_boundary = False
        for site in range(int(num_sites)):
            base = 2 * int(num_sites) + site * qpb
            occupation = sum(((index >> (base + offset)) & 1) << offset for offset in range(qpb))
            on_boundary = on_boundary or occupation == int(n_ph_max)
        if on_boundary:
            probability += float(abs(amplitude) ** 2)
    return float(probability)


def _static_rows(
    *,
    result: QSEResult,
    qse_states: Sequence[np.ndarray],
    exact_energies: np.ndarray,
    exact_vectors: np.ndarray,
    sector_indices: np.ndarray,
    hamiltonian: np.ndarray,
    root_count: int,
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    matches = _match_roots_by_overlap(
        qse_states,
        exact_vectors,
        sector_indices,
        root_count=int(root_count),
    )
    transition_by_name = {
        str(item.observable.name): np.asarray(item.transition_strengths, dtype=float)
        for item in result.transition_observables
    }
    rows: list[dict[str, Any]] = []
    for root in range(min(int(root_count), len(qse_states))):
        state = qse_states[root]
        matched = int(matches[root])
        state_sector = _normalize(state[sector_indices], name=f"QSE root {root} sector state")
        overlaps = np.abs(exact_vectors.conj().T @ state_sector) ** 2
        matched_energy = float(exact_energies[matched])
        cluster = np.flatnonzero(np.abs(exact_energies - matched_energy) <= 1.0e-8)
        qse_energy = float(result.eigenvalues[root])
        residual = float(np.linalg.norm(hamiltonian @ state - qse_energy * state))
        rows.append(
            {
                "root_index": int(root),
                "root_role": "lowest_orthogonal_ritz_root" if root == 0 else "orthogonal_ritz_root",
                "qse_energy": qse_energy,
                "qse_gap_from_ra_reference": float(qse_energy - result.matrices.reference_energy),
                "matched_exact_state_index": matched,
                "matched_exact_energy": matched_energy,
                "matched_exact_gap": float(matched_energy - exact_energies[0]),
                "energy_error": float(qse_energy - matched_energy),
                "gap_error": float(
                    (qse_energy - result.matrices.reference_energy)
                    - (matched_energy - exact_energies[0])
                ),
                "matched_state_fidelity": float(overlaps[matched]),
                "matched_degenerate_subspace_fidelity": float(np.sum(overlaps[cluster])),
                "physical_ritz_residual_norm": residual,
                "generalized_projected_residual_norm": float(result.generalized_residual_norms[root]),
                "transition_strengths_from_ra_reference": {
                    name: float(values[root]) for name, values in transition_by_name.items()
                },
            }
        )
    return rows, matches


def _lowdin_basis(result: QSEResult) -> tuple[np.ndarray, np.ndarray]:
    overlap = np.asarray(result.matrices.overlap, dtype=complex)
    values, vectors = np.linalg.eigh(0.5 * (overlap + overlap.conj().T))
    retained = np.asarray(result.retained_overlap_indices, dtype=int)
    x_map = vectors[:, retained] @ np.diag(1.0 / np.sqrt(np.maximum(values[retained], 0.0)))
    phi = np.column_stack(result.matrices.basis_matrix_vectors)
    physical_basis = phi @ x_map
    # Very ill-conditioned QSE metrics can lose a few digits when Phi @ X is
    # formed explicitly.  Apply the polar/symmetric correction inside the same
    # retained support so driven propagation starts from a genuinely
    # orthonormal physical basis without changing that support.
    gram = 0.5 * (
        physical_basis.conj().T @ physical_basis
        + (physical_basis.conj().T @ physical_basis).conj().T
    )
    gram_values, gram_vectors = np.linalg.eigh(gram)
    if float(np.min(gram_values)) <= 0.0:
        raise AdvisorDemoError("Retained QSE physical-basis Gram matrix is not positive definite")
    correction = gram_vectors @ np.diag(1.0 / np.sqrt(gram_values)) @ gram_vectors.conj().T
    x_map = x_map @ correction
    physical_basis = phi @ x_map
    orthogonality_error = float(
        np.max(
            np.abs(
                physical_basis.conj().T @ physical_basis
                - np.eye(physical_basis.shape[1])
            )
        )
    )
    if orthogonality_error > 1.0e-10:
        raise AdvisorDemoError(f"Retained QSE physical basis is not orthonormal: {orthogonality_error}")
    return x_map, physical_basis


def _midpoint_step(state: np.ndarray, hamiltonian: np.ndarray, dt: float) -> np.ndarray:
    energies, vectors = np.linalg.eigh(0.5 * (hamiltonian + hamiltonian.conj().T))
    phases = np.exp(-1.0j * float(dt) * energies)
    return vectors @ (phases * (vectors.conj().T @ state))


def _driven_trajectory(
    *,
    result: QSEResult,
    initial_root_state: np.ndarray,
    exact_matched_vector: np.ndarray,
    sector_indices: np.ndarray,
    hamiltonian_full: np.ndarray,
    drive_full: np.ndarray,
    phonon_full: np.ndarray,
    drive_amplitude: float,
    drive_omega: float,
    drive_tbar: float,
    drive_phi: float,
    t_final: float,
    num_steps: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    x_map, retained_physical_basis = _lowdin_basis(result)
    phi = np.column_stack(result.matrices.basis_matrix_vectors)
    drive_qse = phi.conj().T @ drive_full @ phi
    h0_orth = x_map.conj().T @ result.matrices.hamiltonian @ x_map
    drive_orth = x_map.conj().T @ drive_qse @ x_map

    y = _normalize(retained_physical_basis.conj().T @ initial_root_state, name="retained QSE initial coordinates")
    exact = _normalize(initial_root_state[sector_indices], name="exact comparator initial state")
    exact_initial = exact.copy()
    h0_sector = hamiltonian_full[np.ix_(sector_indices, sector_indices)]
    drive_sector = drive_full[np.ix_(sector_indices, sector_indices)]
    phonon_sector = phonon_full[np.ix_(sector_indices, sector_indices)]
    dt = float(t_final) / float(num_steps)
    times = np.linspace(0.0, float(t_final), int(num_steps) + 1)
    rows: list[dict[str, Any]] = []

    for step, time_value in enumerate(times):
        qse_full = _normalize(retained_physical_basis @ y, name=f"QSE trajectory state {step}")
        sector_weight = float(np.linalg.norm(qse_full[sector_indices]) ** 2)
        qse_sector = _normalize(qse_full[sector_indices], name=f"QSE trajectory sector state {step}")
        fidelity = float(abs(np.vdot(exact, qse_sector)) ** 2)
        waveform = gaussian_sinusoid_waveform(
            float(time_value),
            A=float(drive_amplitude),
            omega=float(drive_omega),
            tbar=float(drive_tbar),
            phi=float(drive_phi),
        )
        rows.append(
            {
                "step_index": int(step),
                "time": float(time_value),
                "drive_coefficient": float(waveform),
                "qse_exact_state_fidelity": fidelity,
                "qse_sector_weight": sector_weight,
                "qse_norm": float(np.linalg.norm(y)),
                "exact_norm": float(np.linalg.norm(exact)),
                "staggered_density_qse": _expectation(qse_sector, drive_sector),
                "staggered_density_exact": _expectation(exact, drive_sector),
                "staggered_phonon_displacement_qse": _expectation(qse_sector, phonon_sector),
                "staggered_phonon_displacement_exact": _expectation(exact, phonon_sector),
                "static_energy_qse": _expectation(qse_sector, h0_sector),
                "static_energy_exact": _expectation(exact, h0_sector),
                "initial_state_survival_qse": float(abs(np.vdot(exact_initial, qse_sector)) ** 2),
                "initial_state_survival_exact": float(abs(np.vdot(exact_initial, exact)) ** 2),
                "matched_exact_state_population_qse": float(abs(np.vdot(exact_matched_vector, qse_sector)) ** 2),
                "matched_exact_state_population_exact": float(abs(np.vdot(exact_matched_vector, exact)) ** 2),
            }
        )
        if step == int(num_steps):
            continue
        midpoint = float(time_value) + 0.5 * dt
        coefficient = gaussian_sinusoid_waveform(
            midpoint,
            A=float(drive_amplitude),
            omega=float(drive_omega),
            tbar=float(drive_tbar),
            phi=float(drive_phi),
        )
        y = _midpoint_step(y, h0_orth + coefficient * drive_orth, dt)
        exact = _midpoint_step(exact, h0_sector + coefficient * drive_sector, dt)

    fidelities = np.asarray([row["qse_exact_state_fidelity"] for row in rows], dtype=float)
    density_error = np.asarray(
        [row["staggered_density_qse"] - row["staggered_density_exact"] for row in rows],
        dtype=float,
    )
    phonon_error = np.asarray(
        [
            row["staggered_phonon_displacement_qse"]
            - row["staggered_phonon_displacement_exact"]
            for row in rows
        ],
        dtype=float,
    )
    energy_error = np.asarray([row["static_energy_qse"] - row["static_energy_exact"] for row in rows], dtype=float)
    density_exact = np.asarray([row["staggered_density_exact"] for row in rows], dtype=float)
    phonon_exact = np.asarray([row["staggered_phonon_displacement_exact"] for row in rows], dtype=float)
    metrics = {
        "propagation_kind": "fixed_retained_qse_support_driven_hamiltonian",
        "exact_reference_method": "fixed_sector_exponential_midpoint_magnus2_order2",
        "identical_initial_state_for_qse_and_exact": True,
        "exact_reference_used_for_controller_or_drive_selection": False,
        "drive_frequency_source": "selected_qse_gap_from_ra_reference",
        "retained_qse_rank": int(retained_physical_basis.shape[1]),
        "trajectory_rows": int(len(rows)),
        "dt": float(dt),
        "minimum_qse_exact_state_fidelity": float(np.min(fidelities)),
        "final_qse_exact_state_fidelity": float(fidelities[-1]),
        "maximum_staggered_density_abs_error": float(np.max(np.abs(density_error))),
        "maximum_staggered_phonon_abs_error": float(np.max(np.abs(phonon_error))),
        "maximum_static_energy_abs_error": float(np.max(np.abs(energy_error))),
        "exact_staggered_density_peak_to_peak": float(np.ptp(density_exact)),
        "exact_staggered_phonon_peak_to_peak": float(np.ptp(phonon_exact)),
        "maximum_qse_norm_error": float(max(abs(float(row["qse_norm"]) - 1.0) for row in rows)),
        "maximum_exact_norm_error": float(max(abs(float(row["exact_norm"]) - 1.0) for row in rows)),
        "minimum_qse_sector_weight": float(min(float(row["qse_sector_weight"]) for row in rows)),
    }
    return rows, metrics


def _build_plot(result: Mapping[str, Any], output_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    static_rows = list(result["static_excited_states"])
    trajectory = list(result["dynamics"]["trajectory"])
    roots = np.asarray([row["root_index"] for row in static_rows], dtype=int)
    qse_gaps = np.asarray([row["qse_gap_from_ra_reference"] for row in static_rows], dtype=float)
    exact_gaps = np.asarray([row["matched_exact_gap"] for row in static_rows], dtype=float)
    energy_errors = np.abs(np.asarray([row["gap_error"] for row in static_rows], dtype=float))
    residuals = np.asarray([row["physical_ritz_residual_norm"] for row in static_rows], dtype=float)
    times = np.asarray([row["time"] for row in trajectory], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(roots, exact_gaps, "o-", color="#222222", label="exact sector (overlap matched)")
    ax.plot(roots, qse_gaps, "s--", color="#2673b8", label="QSE from RA seed")
    ax.set_xlabel("QSE root index (0 = lowest orthogonal root)")
    ax.set_ylabel("excitation gap")
    ax.set_title("Static excited-state spectrum")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    ax.semilogy(roots, np.maximum(energy_errors, 1.0e-16), "o-", label="|gap error|", color="#2673b8")
    ax.semilogy(roots, np.maximum(residuals, 1.0e-16), "s--", label="physical Ritz residual", color="#c44e52")
    ax.set_xlabel("QSE root index")
    ax.set_ylabel("absolute diagnostic")
    ax.set_title("Root validation")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    ax.plot(times, [row["staggered_density_exact"] for row in trajectory], color="#222222", label="density exact")
    ax.plot(times, [row["staggered_density_qse"] for row in trajectory], "--", color="#2673b8", label="density QSE")
    ax.plot(times, [row["staggered_phonon_displacement_exact"] for row in trajectory], color="#7a3e9d", label="phonon exact")
    ax.plot(times, [row["staggered_phonon_displacement_qse"] for row in trajectory], "--", color="#dd8452", label="phonon QSE")
    ax.set_xlabel("time")
    ax.set_ylabel("expectation value")
    ax.set_title("Driven observables from the QSE excited state")
    ax.legend(frameon=False, fontsize=7, ncol=2)

    ax = axes[1, 1]
    probability_series = [
        np.asarray([row["qse_exact_state_fidelity"] for row in trajectory], dtype=float),
        np.asarray([row["matched_exact_state_population_exact"] for row in trajectory], dtype=float),
        np.asarray([row["initial_state_survival_exact"] for row in trajectory], dtype=float),
    ]
    ax.plot(times, probability_series[0], color="#2673b8", label="QSE/exact fidelity")
    ax.plot(times, probability_series[1], color="#c44e52", label="matched-state population (exact)")
    ax.plot(times, probability_series[2], color="#55a868", label="initial-state survival (exact)")
    lower_probability = max(0.0, min(float(np.min(values)) for values in probability_series) - 0.005)
    ax.set_ylim(lower_probability, 1.0005)
    ax.set_xlabel("time")
    ax.set_ylabel("probability")
    ax.set_title("State-level dynamics diagnostics")
    ax.legend(frameon=False, fontsize=7)

    fig.suptitle("Hubbard--Holstein advisor diagnostic: QSE excited state and driven dynamics", fontsize=12)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def _build_markdown(result: Mapping[str, Any], output_md: Path) -> None:
    selected = result["selected_excited_state"]
    metrics = result["dynamics"]["metrics"]
    seed = result["ra_seed_validation"]
    lines = [
        "# Hubbard--Holstein QSE excited-state advisor demonstration",
        "",
        "> Diagnostic only. This is a modest local proof of concept, not Paper-III evidence.",
        "",
        "## Result",
        "",
        f"- Model: weak--weak HH, L=2, n_ph_max=3, binary bosons, half filling.",
        f"- RA seed same-cutoff energy error: {seed['energy_error']:.3e}.",
        f"- Source checkpoint success flag: {str(seed['source_checkpoint_success_flag']).lower()} (the extracted state is independently revalidated here).",
        f"- Selected state: QSE root {selected['root_index']} (lowest orthogonal Ritz root).",
        f"- Excitation gap: QSE {selected['qse_gap_from_ra_reference']:.9f}; exact {selected['matched_exact_gap']:.9f}.",
        f"- Gap error: {selected['gap_error']:.3e}.",
        f"- Exact-state fidelity: {selected['matched_state_fidelity']:.10f}.",
        f"- Physical Ritz residual: {selected['physical_ritz_residual_norm']:.3e}.",
        "",
        "## Driven dynamics",
        "",
        f"- Pulse: staggered density, A={result['dynamics']['drive']['amplitude']}, omega={result['dynamics']['drive']['omega']:.9f}, tbar={result['dynamics']['drive']['tbar']}.",
        f"- QSE/exact minimum trajectory fidelity: {metrics['minimum_qse_exact_state_fidelity']:.8f}.",
        f"- Maximum staggered-density error: {metrics['maximum_staggered_density_abs_error']:.3e}.",
        f"- Maximum staggered-phonon error: {metrics['maximum_staggered_phonon_abs_error']:.3e}.",
        f"- Exact density peak-to-peak response: {metrics['exact_staggered_density_peak_to_peak']:.3e}.",
        f"- Exact phonon peak-to-peak response: {metrics['exact_staggered_phonon_peak_to_peak']:.3e}.",
        "",
        "The exact trajectory is a diagnostic comparator only. The QSE root, drive frequency, and retained-support evolution do not use exact-reference data for decisions.",
        "",
        "## Scope and next step",
        "",
        "This establishes an excited state and a nontrivial exact-sector driven trajectory initialized from that QSE state. The fixed retained-QSE propagation is an algorithm diagnostic: its high state fidelity but visible observable error shows that drive-support improvement is still needed. This does not yet demonstrate AP-McLachlan propagation or QPU-preparable excited-state promotion. The next experiment is to refit/promote this root and run the same pulse through AP-McLachlan with the exact trajectory retained only as a post-run diagnostic.",
        "",
        "![Advisor diagnostic](advisor_demo.png)",
        "",
    ]
    output_md.write_text("\n".join(lines), encoding="utf-8")


def run_advisor_demo(config: AdvisorDemoConfig) -> dict[str, Any]:
    """Run the source-locked diagnostic and write its advisor artifacts."""

    _validate_config(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_path = output_dir / "source_seed.json"
    qse_path = output_dir / "qse_result.json"
    result_path = output_dir / "advisor_demo_result.json"
    plot_path = output_dir / "advisor_demo.png"
    summary_path = output_dir / "README.md"

    seed, source_lock = _extract_checkpoint_fields(Path(config.source_bundle), str(config.checkpoint_member))
    _write_json(seed_path, seed)
    source_lock["source_seed_path"] = str(seed_path.resolve())
    source_lock["source_seed_sha256"] = _sha256_file(seed_path)

    settings = dict(seed["settings"])
    resolved_model = _require_hh_demo_settings(settings)
    hamiltonian_poly, hamiltonian_provenance = load_polynomial_json(seed_path)
    nq = _num_qubits_from_polynomial(hamiltonian_poly)
    prepared_state, state_provenance = load_state_json(seed_path, expected_nq=nq, state_key="initial_state")
    basis, basis_provenance = basis_elements_from_artifact_source(
        seed_path,
        nq=nq,
        hamiltonian=hamiltonian_poly,
        source="full_meta",
        include_hamiltonian_terms=True,
        canonical_hh_full_meta=True,
    )
    layout = HHResponseLayout(
        num_sites=2,
        n_ph_max=3,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        total_qubits=nq,
        num_particles=(1, 1),
        source_metadata={"source_seed_sha256": source_lock["source_seed_sha256"]},
    )
    observable_bundle = build_hh_neutral_response_observable_bundle(
        layout=layout,
        channels=("nn", "XX"),
        form_factor="staggered",
        prepared_state=prepared_state,
    )
    observable_by_family = {
        str(observable.metadata.get("channel_family")): observable
        for observable in observable_bundle.observables
        if isinstance(observable.metadata, Mapping)
    }
    density_observable = observable_by_family["n"]
    phonon_observable = observable_by_family["X"]

    qse_config = QSEPruningConfig(
        overlap_relative_cutoff=float(config.overlap_relative_cutoff),
        overlap_absolute_cutoff=float(config.overlap_absolute_cutoff),
    )
    basis_policy = QSEBasisVectorPolicy(
        reference_projection="q0",
        basis_vector_normalization="raw_projected",
        sector_projection="identity",
        sector_label="HH L=2 half-filled N_up=1 N_dn=1; binary n_ph_max=3 legal code space",
    )
    qse_result = compute_qse_spectra(
        hamiltonian_poly,
        prepared_state,
        basis,
        config=qse_config,
        basis_vector_policy=basis_policy,
        transition_observables=observable_bundle.observables,
    )
    qse_manifest = qse_result_to_manifest(
        qse_result,
        input_provenance={
            "hamiltonian": hamiltonian_provenance,
            "state": state_provenance,
            "operator_basis": basis_provenance,
            "source_lock": source_lock,
        },
        settings_provenance={
            **asdict(qse_config),
            "run_class": "diagnostic",
            "basis_source": "canonical_hh_full_meta_plus_hamiltonian_terms",
        },
        include_matrices=True,
    )
    qse_manifest["advisor_demo_contract"] = {
        "diagnostic_only": True,
        "paper_facing": False,
        "root_index_zero_role": "lowest_orthogonal_ritz_root_intended_first_excited_candidate",
        "reference_projection": "q0",
        "exact_reference_used_for_qse_construction_or_selection": False,
    }
    write_manifest_json(qse_path, qse_manifest)

    hamiltonian_full = np.asarray(hamiltonian_matrix(hamiltonian_poly), dtype=complex)
    density_full = np.asarray(hamiltonian_matrix(density_observable.polynomial), dtype=complex)
    phonon_full = np.asarray(hamiltonian_matrix(phonon_observable.polynomial), dtype=complex)
    sector_indices = _half_filled_sector_indices(
        num_sites=2,
        n_ph_max=3,
        boson_encoding="binary",
        ordering="blocked",
        nq_total=nq,
    )
    hamiltonian_sector = hamiltonian_full[np.ix_(sector_indices, sector_indices)]
    exact_energies, exact_vectors = np.linalg.eigh(hamiltonian_sector)
    qse_states = _ritz_states(qse_result)
    static_rows, matches = _static_rows(
        result=qse_result,
        qse_states=qse_states,
        exact_energies=exact_energies,
        exact_vectors=exact_vectors,
        sector_indices=sector_indices,
        hamiltonian=hamiltonian_full,
        root_count=int(config.reported_root_count),
    )
    root_index = int(config.initial_root_index)
    if root_index not in matches:
        raise AdvisorDemoError(f"Selected root {root_index} was not overlap matched")
    selected_row = next(row for row in static_rows if int(row["root_index"]) == root_index)
    selected_state = qse_states[root_index]
    matched_exact_index = int(matches[root_index])

    reference_energy = _expectation(prepared_state, hamiltonian_full)
    prepared_sector = _normalize(prepared_state[sector_indices], name="RA seed sector state")
    seed_ground_fidelity = float(abs(np.vdot(exact_vectors[:, 0], prepared_sector)) ** 2)
    drive_omega = (
        float(config.drive_omega)
        if config.drive_omega is not None
        else float(selected_row["qse_gap_from_ra_reference"])
    )
    trajectory, dynamics_metrics = _driven_trajectory(
        result=qse_result,
        initial_root_state=selected_state,
        exact_matched_vector=exact_vectors[:, matched_exact_index],
        sector_indices=sector_indices,
        hamiltonian_full=hamiltonian_full,
        drive_full=density_full,
        phonon_full=phonon_full,
        drive_amplitude=float(config.drive_amplitude),
        drive_omega=float(drive_omega),
        drive_tbar=float(config.drive_tbar),
        drive_phi=float(config.drive_phi),
        t_final=float(config.t_final),
        num_steps=int(config.num_steps),
    )

    result_payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "pipeline": "paper_iii_hh_advisor_demo",
        "generated_utc": _utc_now(),
        "run_class": "diagnostic",
        "diagnostic_only": True,
        "paper_facing": False,
        "uses_qiskit": False,
        "model": {
            **resolved_model,
            "num_qubits": int(nq),
            "half_filled_sector_dimension": int(sector_indices.size),
            "settings": settings,
        },
        "source_lock": source_lock,
        "qse": {
            "result_path": str(qse_path.resolve()),
            "result_sha256": _sha256_file(qse_path),
            "basis_source": "canonical_hh_full_meta_plus_hamiltonian_terms",
            "basis_size": int(len(basis)),
            "retained_rank": int(qse_result.retained_rank),
            "discarded_rank": int(qse_result.discarded_rank),
            "overlap_condition_estimate": qse_result.overlap_condition_estimate,
            "basis_provenance": basis_provenance,
            "basis_vector_policy": asdict(basis_policy),
            "root_zero_semantics": "lowest_orthogonal_ritz_root_first_excited_candidate",
        },
        "ra_seed_validation": {
            "reference_energy": float(reference_energy),
            "exact_ground_energy": float(exact_energies[0]),
            "energy_error": float(reference_energy - exact_energies[0]),
            "exact_ground_state_fidelity": seed_ground_fidelity,
            "source_checkpoint_success_flag": bool(seed["adapt_vqe"]["success"]),
            "source_checkpoint_stop_reason": seed["adapt_vqe"]["stop_reason"],
            "source_checkpoint_ansatz_depth": int(seed["adapt_vqe"]["ansatz_depth"]),
            "validation_role": "independent_same_cutoff_energy_and_state_fidelity_check",
            "sector_weight": float(np.linalg.norm(prepared_state[sector_indices]) ** 2),
            "phonon_cutoff_boundary_probability": _boundary_probability(
                prepared_state,
                num_sites=2,
                n_ph_max=3,
                nq_total=nq,
            ),
        },
        "static_excited_states": static_rows,
        "selected_excited_state": {
            **selected_row,
            "selection_policy": "explicit_root_zero_lowest_orthogonal_qse_root_no_exact_input",
            "phonon_cutoff_boundary_probability": _boundary_probability(
                selected_state,
                num_sites=2,
                n_ph_max=3,
                nq_total=nq,
            ),
            "sector_weight": float(np.linalg.norm(selected_state[sector_indices]) ** 2),
        },
        "dynamics": {
            "drive": {
                "operator": str(density_observable.name),
                "spatial_pattern": "staggered",
                "waveform": "A*sin(omega*t+phi)*exp(-t^2/(2*tbar^2))",
                "amplitude": float(config.drive_amplitude),
                "omega": float(drive_omega),
                "omega_source": "selected_qse_gap_from_ra_reference" if config.drive_omega is None else "explicit_cli",
                "tbar": float(config.drive_tbar),
                "phi": float(config.drive_phi),
                "time_sampling": "midpoint",
            },
            "config": {"t_final": float(config.t_final), "num_steps": int(config.num_steps)},
            "metrics": dynamics_metrics,
            "trajectory": trajectory,
        },
        "decision_data_flow": {
            "qse_seed_source": "paper_i_ra_checkpoint",
            "drive_frequency_source": "selected_qse_gap_from_ra_reference" if config.drive_omega is None else "explicit_cli",
            "exact_reference_used_for_qse_construction": False,
            "exact_reference_used_for_root_selection": False,
            "exact_reference_used_for_drive_selection": False,
            "exact_reference_used_for_propagation_decisions": False,
            "exact_reference_role": "post_run_diagnostic_comparator",
        },
        "limitations": [
            "diagnostic_only_not_paper_iii_evidence",
            "ideal_statevector_backend",
            "fixed_retained_qse_support_not_ap_mclachlan",
            "qse_root_not_yet_refit_or_promoted_to_qpu_preparable_ansatz",
            "single_weak_weak_hh_point_and_single_cutoff",
        ],
    }
    _write_json(result_path, result_payload)
    _build_plot(result_payload, plot_path)
    _build_markdown(result_payload, summary_path)
    result_payload["artifacts"] = {
        "result_json": {"path": str(result_path.resolve())},
        "qse_json": {"path": str(qse_path.resolve()), "sha256": _sha256_file(qse_path)},
        "source_seed_json": {"path": str(seed_path.resolve()), "sha256": _sha256_file(seed_path)},
        "advisor_plot_png": {"path": str(plot_path.resolve()), "sha256": _sha256_file(plot_path)},
        "advisor_summary_md": {"path": str(summary_path.resolve()), "sha256": _sha256_file(summary_path)},
    }
    _write_json(result_path, result_payload)
    return result_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--checkpoint-member", default=DEFAULT_CHECKPOINT_MEMBER)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-root-index", type=int, default=0)
    parser.add_argument("--reported-root-count", type=int, default=8)
    parser.add_argument("--drive-amplitude", type=float, default=0.20)
    parser.add_argument("--drive-omega", type=float, default=None)
    parser.add_argument("--drive-tbar", type=float, default=4.0)
    parser.add_argument("--drive-phi", type=float, default=0.0)
    parser.add_argument("--t-final", type=float, default=10.0)
    parser.add_argument("--num-steps", type=int, default=200)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = AdvisorDemoConfig(
        source_bundle=Path(args.source_bundle),
        checkpoint_member=str(args.checkpoint_member),
        output_dir=Path(args.output_dir),
        initial_root_index=int(args.initial_root_index),
        reported_root_count=int(args.reported_root_count),
        drive_amplitude=float(args.drive_amplitude),
        drive_omega=None if args.drive_omega is None else float(args.drive_omega),
        drive_tbar=float(args.drive_tbar),
        drive_phi=float(args.drive_phi),
        t_final=float(args.t_final),
        num_steps=int(args.num_steps),
    )
    try:
        result = run_advisor_demo(config)
    except AdvisorDemoError as exc:
        print(f"error: {exc}")
        return 2
    selected = result["selected_excited_state"]
    metrics = result["dynamics"]["metrics"]
    print(f"output_dir: {Path(args.output_dir).resolve()}")
    print(f"selected_qse_root: {selected['root_index']}")
    print(f"qse_gap: {selected['qse_gap_from_ra_reference']:.12g}")
    print(f"matched_exact_gap: {selected['matched_exact_gap']:.12g}")
    print(f"matched_state_fidelity: {selected['matched_state_fidelity']:.12g}")
    print(f"physical_ritz_residual_norm: {selected['physical_ritz_residual_norm']:.12g}")
    print(f"minimum_dynamics_fidelity: {metrics['minimum_qse_exact_state_fidelity']:.12g}")
    return 0


__all__ = [
    "AdvisorDemoConfig",
    "AdvisorDemoError",
    "SCHEMA_VERSION",
    "SOURCE_SEED_SCHEMA_VERSION",
    "_half_filled_sector_indices",
    "_midpoint_step",
    "build_parser",
    "main",
    "run_advisor_demo",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
