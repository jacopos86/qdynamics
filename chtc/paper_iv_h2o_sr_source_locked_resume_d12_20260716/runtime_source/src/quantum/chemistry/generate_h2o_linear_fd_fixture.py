from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.quantum.chemistry.molecular_hamiltonian import (
    build_restricted_closed_shell_molecular_hamiltonian,
)
from src.quantum.chemistry.molecular_uccsd import build_molecular_uccsd_pool
from src.quantum.chemistry.psi4_adapter import (
    RestrictedClosedShellMolecularProblem,
    RestrictedClosedShellPsi4Snapshot,
    load_restricted_closed_shell_snapshot_from_psi4,
)
from src.quantum.chemistry.vibronic_h2 import (
    _boson_momentum_operator,
    _clean_real_polynomial,
    _lift_fermion_polynomial,
    pauli_polynomial_to_jsonable,
)
from src.quantum.chemistry.vibronic_h2o_linear_fd import (
    H2O_LINEAR_FD_CUTOFF_BOUNDARY_WEIGHT_TOLERANCE,
    H2O_LINEAR_FD_CUTOFF_ENERGY_TOLERANCE_HARTREE,
    H2O_LINEAR_FD_DERIVATIVE_SOURCE,
    H2O_LINEAR_FD_FAMILY_KEY,
    H2O_LINEAR_FD_FIXTURE_SCHEMA,
    H2O_LINEAR_FD_MODEL_ROLE,
    H2O_UMBRELLA_FAMILY_KEY,
    ActiveSpaceRecord,
    AlignedActiveTensorRecord,
    AlignmentDiagnosticsRecord,
    AlignmentThresholds,
    BosonModeRegister,
    BoundaryWeightRecord,
    CutoffDiagnosticsRecord,
    DerivativeNorms,
    DisplacedGeometryRecord,
    EncodedOperatorBundle,
    EvidenceHooksRecord,
    ExactReferenceRecord,
    ExactStateVectorRecord,
    FirstDerivativeRecord,
    FixtureManifest,
    GeometryRecord,
    NormalModeRecord,
    PhysicalSectorRecord,
    ProductionVibronicH2OFixture,
    RegisterLayout,
    assess_h2o_linear_fd_cutoff_diagnostics,
    fixed_sector_dimension,
    h2o_linear_fd_boundary_weight_for_state,
    production_vibronic_h2o_fixture_to_jsonable,
    validate_production_vibronic_h2o_fixture,
    validate_paper_iv_h2o_linear_fd_evidence_fixture,
)
from src.quantum.hartree_fock_reference_state import hartree_fock_occupied_qubits
from src.quantum.hubbard_latex_python_pairs import boson_operator, boson_qubits_per_site
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.vqe_latex_python_pairs import hamiltonian_matrix


BACKEND_RECORD_SCHEMA = "h2o_linear_fd_backend_record_v1"
GENERATOR_VERSION = "h2o_linear_fd_generator_v2"
AMU_TO_ELECTRON_MASS = 1822.888486209
HARTREE_TO_CM_INV = 219474.63136320


@dataclass(frozen=True)
class DerivativeValidationConfig:
    tier: str
    rel_tol: float
    scalar_abs_tol: float
    tensor_rms_abs_tol: float
    tensor_max_abs_tol: float
    scalar_zero_tol: float
    tensor_rms_zero_tol: float
    tensor_max_zero_tol: float


def canonical_json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_jsonable(payload: Any) -> str:
    return hashlib.sha256(canonical_json_dumps(payload).encode("utf-8")).hexdigest()


def _as_mapping(payload: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object.")
    return payload


def _array(payload: Any, *, label: str, ndim: int | None = None) -> np.ndarray:
    arr = np.asarray(payload, dtype=float)
    if ndim is not None and arr.ndim != int(ndim):
        raise ValueError(f"{label} must have ndim={int(ndim)}; got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} contains non-finite values.")
    return arr


def _tuple_int(raw: Any, *, label: str) -> tuple[int, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"{label} must be a sequence.")
    return tuple(int(v) for v in raw)


def _parse_int_tuple(text: str | None, *, label: str) -> tuple[int, ...] | None:
    if text in {None, ""}:
        return None
    values = tuple(int(part.strip()) for part in str(text).split(",") if part.strip())
    if not values:
        raise ValueError(f"{label} must contain at least one integer.")
    return values


def _write_json(path: Path, payload: Mapping[str, Any], *, force: bool) -> None:
    if path.exists() and not bool(force):
        raise FileExistsError(f"Refusing to overwrite existing JSON without --force: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _default_h2o_geometry_spec_angstrom() -> str:
    bond = 0.9572
    angle = math.radians(104.52)
    z = bond * math.cos(angle / 2.0)
    y = bond * math.sin(angle / 2.0)
    return "\n".join(
        (
            "O 0.000000000000 0.000000000000 0.000000000000",
            f"H 0.000000000000 {y:.12f} {z:.12f}",
            f"H 0.000000000000 {-y:.12f} {z:.12f}",
        )
    )


def _read_text_or_default(path: Path | None, *, default: str) -> str:
    if path is None:
        return str(default)
    return Path(path).read_text(encoding="utf-8")


def _geometry_spec_from_bohr(symbols: Sequence[str], coordinates_bohr: np.ndarray) -> str:
    coords = np.asarray(coordinates_bohr, dtype=float)
    if coords.shape != (len(tuple(symbols)), 3):
        raise ValueError("geometry coordinate shape does not match symbols.")
    return "\n".join(
        f"{str(symbol)} {float(x):.14f} {float(y):.14f} {float(z):.14f}"
        for symbol, (x, y, z) in zip(symbols, coords)
    )


def _normalized_geometry_block(
    *,
    geometry_spec: str,
    charge: int,
    multiplicity: int,
    units: str,
) -> str:
    unit_key = str(units).strip().lower()
    if unit_key not in {"angstrom", "bohr"}:
        raise ValueError("units must be 'angstrom' or 'bohr'.")
    geom = str(geometry_spec).strip()
    if not geom:
        raise ValueError("geometry_spec must be non-empty.")
    return (
        f"{int(charge)} {int(multiplicity)}\n"
        f"{geom}\n"
        "symmetry c1\n"
        f"units {unit_key}\n"
        "no_reorient\n"
        "no_com\n"
    )


def _matrix_to_ndarray(obj: Any) -> np.ndarray:
    if hasattr(obj, "np"):
        return np.asarray(obj.np, dtype=float)
    if hasattr(obj, "to_array"):
        return np.asarray(obj.to_array(), dtype=float)
    return np.asarray(obj, dtype=float)


def _psi4_gradient_diagnostics(
    psi4_module: Any,
    *,
    method: str,
    molecule: Any,
) -> dict[str, Any]:
    try:
        gradient_raw = psi4_module.gradient(str(method), molecule=molecule)
        gradient = _matrix_to_ndarray(gradient_raw)
    except Exception as exc:  # pragma: no cover - optional backend diagnostics
        return {
            "available": False,
            "error": str(exc),
        }
    return {
        "available": True,
        "units": "hartree_per_bohr",
        "gradient_hartree_per_bohr": np.asarray(gradient, dtype=float).tolist(),
        "fro_norm_hartree_per_bohr": float(np.linalg.norm(gradient)),
        "max_abs_hartree_per_bohr": float(np.max(np.abs(gradient))) if gradient.size else 0.0,
    }


def _molecule_arrays_from_psi4_molecule(molecule: Any) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    if hasattr(molecule, "update_geometry"):
        molecule.update_geometry()
    if hasattr(molecule, "to_arrays"):
        arrays = molecule.to_arrays()
        coordinates_bohr = np.asarray(arrays[0], dtype=float)
        masses_amu = np.asarray(arrays[1], dtype=float)
        raw_symbols = arrays[2]
        symbols = tuple(
            str(symbol.decode("utf-8") if isinstance(symbol, bytes) else symbol)
            for symbol in raw_symbols
        )
        return symbols, coordinates_bohr, masses_amu * AMU_TO_ELECTRON_MASS
    n_atom = int(molecule.natom())
    symbols = tuple(str(molecule.symbol(i)) for i in range(n_atom))
    coordinates_bohr = np.asarray(
        [[float(molecule.x(i)), float(molecule.y(i)), float(molecule.z(i))] for i in range(n_atom)],
        dtype=float,
    )
    masses_me = np.asarray([float(molecule.mass(i)) * AMU_TO_ELECTRON_MASS for i in range(n_atom)], dtype=float)
    return symbols, coordinates_bohr, masses_me


def _validate_h2o_symbols(symbols: Sequence[str]) -> None:
    normalized = tuple(str(symbol).strip().upper() for symbol in symbols)
    if len(normalized) != 3 or sorted(normalized) != ["H", "H", "O"]:
        raise ValueError(f"expected an H2O geometry with one O and two H atoms; got {tuple(symbols)!r}.")


def _displace_coordinates_along_mass_weighted_mode(
    coordinates_bohr: np.ndarray,
    *,
    mode_vector: np.ndarray,
    masses_me: np.ndarray,
    q_displacement_au: float,
) -> np.ndarray:
    coords = np.asarray(coordinates_bohr, dtype=float)
    mode = np.asarray(mode_vector, dtype=float)
    masses = np.asarray(masses_me, dtype=float).reshape(-1)
    if coords.shape != mode.shape or coords.shape != (len(masses), 3):
        raise ValueError("mass-weighted displacement shapes disagree.")
    if np.any(masses <= 0.0):
        raise ValueError("masses must be positive for mass-weighted displacement.")
    return coords + float(q_displacement_au) * mode / np.sqrt(masses)[:, None]


def _mass_weighted_displacement_norm(
    displacement_bohr: np.ndarray,
    *,
    masses_me: np.ndarray,
) -> float:
    disp = np.asarray(displacement_bohr, dtype=float)
    masses = np.asarray(masses_me, dtype=float).reshape(-1)
    if disp.shape != (len(masses), 3):
        raise ValueError("mass-weighted norm displacement shape disagrees with masses.")
    return float(math.sqrt(float(np.sum((masses[:, None] * disp * disp)))))


def _h2o_atom_indices(symbols: Sequence[str]) -> tuple[int, int, int]:
    normalized = tuple(str(symbol).strip().upper() for symbol in symbols)
    oxygen = [idx for idx, symbol in enumerate(normalized) if symbol == "O"]
    hydrogens = [idx for idx, symbol in enumerate(normalized) if symbol == "H"]
    if len(oxygen) != 1 or len(hydrogens) != 2:
        raise ValueError(f"expected one O and two H atoms for H2O internal coordinates; got {tuple(symbols)!r}.")
    return int(oxygen[0]), int(hydrogens[0]), int(hydrogens[1])


def _h2o_internal_coordinates(
    symbols: Sequence[str],
    coordinates_bohr: np.ndarray,
) -> tuple[float, float, float]:
    coords = np.asarray(coordinates_bohr, dtype=float)
    o_idx, h1_idx, h2_idx = _h2o_atom_indices(symbols)
    if coords.shape != (len(tuple(symbols)), 3):
        raise ValueError("H2O internal coordinate shape disagrees with symbols.")
    r1_vec = coords[int(h1_idx)] - coords[int(o_idx)]
    r2_vec = coords[int(h2_idx)] - coords[int(o_idx)]
    r1 = float(np.linalg.norm(r1_vec))
    r2 = float(np.linalg.norm(r2_vec))
    if r1 <= 0.0 or r2 <= 0.0:
        raise ValueError("H2O bond length is zero.")
    cos_angle = float(np.dot(r1_vec, r2_vec) / (r1 * r2))
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return r1, r2, float(math.acos(cos_angle))


def _h2o_mode_character(
    *,
    symbols: Sequence[str],
    coordinates_bohr: np.ndarray,
    mode_vector: np.ndarray,
    masses_me: np.ndarray,
    q_probe_au: float = 1.0,
) -> dict[str, Any]:
    coords = np.asarray(coordinates_bohr, dtype=float)
    mode = np.asarray(mode_vector, dtype=float)
    masses = np.asarray(masses_me, dtype=float).reshape(-1)
    if float(q_probe_au) <= 0.0:
        raise ValueError("q_probe_au must be positive.")
    plus = _displace_coordinates_along_mass_weighted_mode(
        coords,
        mode_vector=mode,
        masses_me=masses,
        q_displacement_au=float(q_probe_au),
    )
    minus = _displace_coordinates_along_mass_weighted_mode(
        coords,
        mode_vector=mode,
        masses_me=masses,
        q_displacement_au=-float(q_probe_au),
    )
    r1_plus, r2_plus, angle_plus = _h2o_internal_coordinates(symbols, plus)
    r1_minus, r2_minus, angle_minus = _h2o_internal_coordinates(symbols, minus)
    dr1_dq = (float(r1_plus) - float(r1_minus)) / (2.0 * float(q_probe_au))
    dr2_dq = (float(r2_plus) - float(r2_minus)) / (2.0 * float(q_probe_au))
    dtheta_dq = (float(angle_plus) - float(angle_minus)) / (2.0 * float(q_probe_au))
    sym_score = abs(float(dr1_dq) + float(dr2_dq))
    asym_score = abs(float(dr1_dq) - float(dr2_dq))
    bend_score = abs(float(dtheta_dq))
    scores = {
        "bend": float(bend_score),
        "symmetric_stretch": float(sym_score),
        "antisymmetric_stretch": float(asym_score),
    }
    dominant = max(scores, key=scores.get)
    return {
        "q_probe_au": float(q_probe_au),
        "delta_r1_bohr_per_q": float(dr1_dq),
        "delta_r2_bohr_per_q": float(dr2_dq),
        "delta_angle_rad_per_q": float(dtheta_dq),
        "symmetric_stretch_score": float(sym_score),
        "antisymmetric_stretch_score": float(asym_score),
        "bend_score": float(bend_score),
        "dominant_label": str(dominant),
        "orientation_indicators": {
            "bend": float(dtheta_dq),
            "symmetric_stretch": float(dr1_dq + dr2_dq),
            "antisymmetric_stretch": float(dr1_dq - dr2_dq),
        },
    }


def _translation_rotation_basis(
    *,
    coordinates_bohr: np.ndarray,
    masses_me: np.ndarray,
) -> np.ndarray:
    coords = np.asarray(coordinates_bohr, dtype=float)
    masses = np.asarray(masses_me, dtype=float).reshape(-1)
    if coords.shape != (len(masses), 3):
        raise ValueError("translation/rotation basis shapes disagree.")
    sqrt_m = np.sqrt(masses)
    total_mass = float(np.sum(masses))
    center_of_mass = np.sum(masses[:, None] * coords, axis=0) / total_mass
    shifted = coords - center_of_mass[None, :]
    raw: list[np.ndarray] = []
    for axis in range(3):
        vec = np.zeros_like(coords)
        vec[:, axis] = sqrt_m
        raw.append(vec.reshape(-1))
    unit_axes = np.eye(3)
    for axis in range(3):
        cross = np.cross(unit_axes[int(axis)], shifted)
        raw.append((sqrt_m[:, None] * cross).reshape(-1))
    basis: list[np.ndarray] = []
    for vec in raw:
        work = np.asarray(vec, dtype=float).copy()
        for prev in basis:
            work -= float(np.dot(prev, work)) * prev
        norm = float(np.linalg.norm(work))
        if norm > 1.0e-12:
            basis.append(work / norm)
    if not basis:
        return np.zeros((3 * len(masses), 0), dtype=float)
    return np.column_stack(basis)


def _translation_rotation_overlap(
    mode_vector: np.ndarray,
    *,
    coordinates_bohr: np.ndarray,
    masses_me: np.ndarray,
) -> float:
    mode = np.asarray(mode_vector, dtype=float).reshape(-1)
    basis = _translation_rotation_basis(
        coordinates_bohr=np.asarray(coordinates_bohr, dtype=float),
        masses_me=np.asarray(masses_me, dtype=float),
    )
    if basis.size == 0:
        return 0.0
    return float(np.linalg.norm(basis.T @ mode))


def _assign_h2o_mode_labels_by_character(rows: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    labels = ("bend", "symmetric_stretch", "antisymmetric_stretch")
    if len(rows) != 3:
        return tuple(dict(row) for row in rows)
    best_perm: tuple[str, ...] | None = None
    best_score = -math.inf
    for perm in itertools.permutations(labels):
        score = 0.0
        for row, label in zip(rows, perm):
            character = dict(row.get("mode_character", {}))
            score += float(character.get(f"{label}_score", 0.0))
        if score > best_score:
            best_score = float(score)
            best_perm = tuple(str(v) for v in perm)
    assert best_perm is not None
    oriented: list[dict[str, Any]] = []
    for row, label in zip(rows, best_perm):
        out = dict(row)
        character = dict(out.get("mode_character", {}))
        indicators = dict(character.get("orientation_indicators", {}))
        vector = np.asarray(out["mass_weighted_eigenvector"], dtype=float)
        if float(indicators.get(str(label), 0.0)) < 0.0:
            vector = -vector
            if "mode_character_for_negative_vector" in out:
                character = dict(out["mode_character_for_negative_vector"])
        out["label"] = str(label)
        out["mass_weighted_eigenvector"] = vector
        out["mode_character"] = character
        out["label_source"] = "internal_coordinate_character_v1"
        oriented.append(out)
    order = {label: idx for idx, label in enumerate(labels)}
    oriented.sort(key=lambda row: order[str(row["label"])])
    for idx, row in enumerate(oriented):
        row["mode_index"] = int(idx)
    return tuple(oriented)


def _mass_weighted_vibrational_modes_from_hessian(
    hessian_cartesian_bohr: np.ndarray,
    *,
    masses_me: np.ndarray,
    symbols: Sequence[str] | None = None,
    coordinates_bohr: np.ndarray | None = None,
    n_vibrational_modes: int = 3,
) -> tuple[dict[str, Any], ...]:
    hess = np.asarray(hessian_cartesian_bohr, dtype=float)
    masses = np.asarray(masses_me, dtype=float).reshape(-1)
    n_atoms = len(masses)
    expected = (3 * n_atoms, 3 * n_atoms)
    if hess.shape != expected:
        raise ValueError(f"hessian shape must be {expected}; got {hess.shape}.")
    if np.any(masses <= 0.0):
        raise ValueError("masses must be positive.")
    masses_by_coord = np.repeat(masses, 3)
    mass_weighted = hess / np.sqrt(np.outer(masses_by_coord, masses_by_coord))
    mass_weighted = 0.5 * (mass_weighted + mass_weighted.T)
    evals, evecs = np.linalg.eigh(mass_weighted)
    positive = [idx for idx, value in enumerate(evals) if float(value) > 0.0]
    if len(positive) < int(n_vibrational_modes):
        raise ValueError(
            f"not enough positive Hessian eigenvalues for {int(n_vibrational_modes)} vibrational modes."
        )
    selected = positive[-int(n_vibrational_modes):]
    selected = sorted(selected, key=lambda idx: float(evals[int(idx)]))
    rows: list[dict[str, Any]] = []
    for mode_index, eig_index in enumerate(selected):
        omega = math.sqrt(float(evals[int(eig_index)]))
        vector = np.asarray(evecs[:, int(eig_index)], dtype=float).reshape(n_atoms, 3)
        max_pos = np.unravel_index(int(np.argmax(np.abs(vector))), vector.shape)
        if float(vector[max_pos]) < 0.0:
            vector = -vector
        norm = float(np.linalg.norm(vector))
        if norm <= 0.0 or not np.isfinite(norm):
            raise ValueError("normal-mode vector has invalid norm.")
        vector = vector / norm
        row: dict[str, Any] = {
            "mode_index": int(mode_index),
            "label": ("bend", "symmetric_stretch", "antisymmetric_stretch")[int(mode_index)],
            "frequency_hartree": float(omega),
            "frequency_cm1": float(omega * HARTREE_TO_CM_INV),
            "mass_weighted_eigenvector": vector,
            "raw_hessian_eigenvalue": float(evals[int(eig_index)]),
            "raw_hessian_eigenvalue_index": int(eig_index),
            "positive_hessian_eigenvalue_count": int(len(positive)),
            "label_source": "ascending_frequency_h2o_bend_symmetric_antisymmetric_v1",
        }
        if symbols is not None and coordinates_bohr is not None:
            coords = np.asarray(coordinates_bohr, dtype=float)
            row["mode_character"] = _h2o_mode_character(
                symbols=tuple(str(v) for v in symbols),
                coordinates_bohr=coords,
                mode_vector=vector,
                masses_me=masses,
            )
            row["mode_character_for_negative_vector"] = _h2o_mode_character(
                symbols=tuple(str(v) for v in symbols),
                coordinates_bohr=coords,
                mode_vector=-vector,
                masses_me=masses,
            )
            row["trans_rot_overlap"] = _translation_rotation_overlap(
                vector,
                coordinates_bohr=coords,
                masses_me=masses,
            )
        rows.append(row)
    if symbols is not None and coordinates_bohr is not None and int(n_vibrational_modes) == 3:
        return _assign_h2o_mode_labels_by_character(rows)
    return tuple(rows)


def _eri_slice(tensor: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    idx = np.asarray(tuple(int(v) for v in indices), dtype=int)
    return np.asarray(tensor, dtype=float)[np.ix_(idx, idx, idx, idx)]


def _freeze_core_active_tensors(
    problem: RestrictedClosedShellMolecularProblem,
    *,
    active_indices: Sequence[int],
    frozen_core_indices: Sequence[int],
) -> tuple[float, np.ndarray, np.ndarray]:
    active = tuple(int(v) for v in active_indices)
    frozen = tuple(int(v) for v in frozen_core_indices)
    h = np.asarray(problem.one_body_integrals_mo, dtype=float)
    g = np.asarray(problem.two_body_integrals_mo, dtype=float)
    scalar = float(problem.nuclear_repulsion_energy)
    for i in frozen:
        scalar += 2.0 * float(h[i, i])
    for i in frozen:
        for j in frozen:
            scalar += 2.0 * float(g[i, i, j, j]) - float(g[i, j, j, i])
    h_eff = np.asarray(h[np.ix_(active, active)], dtype=float).copy()
    for p_out, p in enumerate(active):
        for q_out, q in enumerate(active):
            correction = 0.0
            for i in frozen:
                correction += 2.0 * float(g[p, q, i, i]) - float(g[p, i, i, q])
            h_eff[p_out, q_out] += correction
    g_active = _eri_slice(g, active)
    return float(scalar), h_eff, g_active


def _select_active_space(
    problem: RestrictedClosedShellMolecularProblem,
    *,
    policy: str,
) -> tuple[str, tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, int]]:
    key = str(policy).strip().lower()
    n_spatial = int(problem.n_spatial_orbitals)
    n_alpha = int(problem.n_alpha)
    n_beta = int(problem.n_beta)
    if key in {"full", "all"}:
        active = tuple(range(n_spatial))
        return "full_mo_space", (), active, (), (n_alpha, n_beta)
    if key in {"valence_8e_6o", "cas_8e_6o", "cas_8e_6o_valence"}:
        if n_spatial < 7 or n_alpha < 5 or n_beta < 5:
            raise ValueError("valence_8e_6o active space requires at least 7 spatial orbitals and 10 electrons.")
        frozen = (0,)
        active = tuple(range(1, 7))
        external = tuple(range(7, n_spatial))
        return "cas_8e_6o_valence", frozen, active, external, (n_alpha - 1, n_beta - 1)
    if key in {"frontier_2e_2o", "active2_frontier", "smoke_2e_2o"}:
        homo = n_alpha - 1
        lumo = n_alpha
        if homo < 0 or lumo >= n_spatial:
            raise ValueError("frontier_2e_2o active space requires at least one occupied and one virtual orbital.")
        frozen = tuple(range(homo))
        active = (homo, lumo)
        external = tuple(idx for idx in range(n_spatial) if idx not in set(frozen).union(active))
        return "frontier_2e_2o_center_aligned_psi4_smoke", frozen, active, external, (1, 1)
    raise ValueError(f"unsupported H2O active-space policy: {policy!r}")


def _active_tensor_payload_from_problem(
    problem: RestrictedClosedShellMolecularProblem,
    *,
    active_indices: Sequence[int],
    frozen_core_indices: Sequence[int],
) -> dict[str, Any]:
    scalar, h_active, g_active = _freeze_core_active_tensors(
        problem,
        active_indices=active_indices,
        frozen_core_indices=frozen_core_indices,
    )
    return {
        "scalar_energy_hartree": float(scalar),
        "one_body_integrals": np.asarray(h_active, dtype=float).tolist(),
        "two_body_integrals": np.asarray(g_active, dtype=float).tolist(),
    }


def _align_snapshot_to_center_with_diagnostics(
    snapshot: RestrictedClosedShellPsi4Snapshot,
    *,
    center_snapshot: RestrictedClosedShellPsi4Snapshot,
    active_indices: Sequence[int],
    displacement_id: str,
    alignment_id: str,
) -> tuple[RestrictedClosedShellMolecularProblem, dict[str, Any]]:
    try:
        import psi4
    except Exception as exc:  # pragma: no cover - optional backend
        raise ImportError("Psi4 is required for center-MO overlap alignment.") from exc
    if int(snapshot.problem.n_spatial_orbitals) != int(center_snapshot.problem.n_spatial_orbitals):
        raise ValueError("Snapshot orbital counts do not match center geometry.")

    mints = psi4.core.MintsHelper(center_snapshot.basis_set)
    s_cross = _matrix_to_ndarray(mints.ao_overlap(center_snapshot.basis_set, snapshot.basis_set))
    overlap_mo = (
        np.asarray(center_snapshot.coeff_alpha_mo, dtype=float).T
        @ s_cross
        @ np.asarray(snapshot.coeff_alpha_mo, dtype=float)
    )
    u, singular_values_full, vh = np.linalg.svd(overlap_mo, full_matrices=False)
    rotation = np.asarray(vh.T @ u.T, dtype=float)

    h_old = np.asarray(snapshot.problem.one_body_integrals_mo, dtype=float)
    eri_old = np.asarray(snapshot.problem.two_body_integrals_mo, dtype=float)
    h_aligned = rotation.T @ h_old @ rotation
    eri_aligned = np.einsum("ap,bq,cr,ds,abcd->pqrs", rotation, rotation, rotation, rotation, eri_old, optimize=True)

    active = tuple(int(v) for v in active_indices)
    all_indices = tuple(range(int(snapshot.problem.n_spatial_orbitals)))
    external = tuple(idx for idx in all_indices if idx not in set(active))
    active_overlap = overlap_mo[np.ix_(active, active)]
    active_singular = np.linalg.svd(active_overlap, compute_uv=False)
    active_to_external = None
    external_to_active = None
    if external:
        active_to_external = float(np.linalg.norm(overlap_mo[np.ix_(active, external)]))
        external_to_active = float(np.linalg.norm(overlap_mo[np.ix_(external, active)]))
    active_rotation = rotation[np.ix_(active, active)]
    active_rotation_residual = float(np.linalg.norm(active_rotation.T @ active_rotation - np.eye(len(active))))
    alignment_payload = {
        "alignment_id": str(alignment_id),
        "singular_values": [float(v) for v in active_singular],
        "min_singular_value": float(np.min(active_singular)),
        "alignment_residual_fro": float(np.linalg.norm(overlap_mo @ rotation - np.eye(overlap_mo.shape[0]))),
        "active_to_external_leakage_fro": active_to_external,
        "external_to_active_leakage_fro": external_to_active,
        "active_rotation": active_rotation.tolist() if active_rotation_residual <= 1.0e-6 else None,
        "full_singular_values": [float(v) for v in singular_values_full],
        "displacement_id": str(displacement_id),
        "passed": True,
    }
    aligned_problem = RestrictedClosedShellMolecularProblem(
        geometry_spec=str(snapshot.problem.geometry_spec),
        basis=str(snapshot.problem.basis),
        charge=int(snapshot.problem.charge),
        multiplicity=int(snapshot.problem.multiplicity),
        reference=str(snapshot.problem.reference),
        n_spatial_orbitals=int(snapshot.problem.n_spatial_orbitals),
        n_alpha=int(snapshot.problem.n_alpha),
        n_beta=int(snapshot.problem.n_beta),
        hf_energy=float(snapshot.problem.hf_energy),
        nuclear_repulsion_energy=float(snapshot.problem.nuclear_repulsion_energy),
        one_body_integrals_mo=np.asarray(h_aligned, dtype=float),
        two_body_integrals_mo=np.asarray(eri_aligned, dtype=float),
    )
    return aligned_problem, alignment_payload


def _normalize_mode_label(label: str) -> str:
    key = str(label).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "bend": "bend",
        "bending": "bend",
        "sym": "symmetric_stretch",
        "symmetric": "symmetric_stretch",
        "sym_stretch": "symmetric_stretch",
        "symmetric_stretch": "symmetric_stretch",
        "asym": "antisymmetric_stretch",
        "asymmetric": "antisymmetric_stretch",
        "antisymmetric": "antisymmetric_stretch",
        "asym_stretch": "antisymmetric_stretch",
        "asymmetric_stretch": "antisymmetric_stretch",
        "antisymmetric_stretch": "antisymmetric_stretch",
    }
    return aliases.get(key, key)


def _relative_delta(primary: float, alt: float, *, floor: float = 1.0e-14) -> float:
    return float(abs(float(primary) - float(alt)) / max(abs(float(primary)), abs(float(alt)), float(floor)))


def _derivative_validation_config(
    *,
    tier: str,
    max_derivative_drift: float,
) -> DerivativeValidationConfig:
    key = str(tier).strip().lower().replace("-", "_")
    if key in {"prod", "production"}:
        base = DerivativeValidationConfig(
            tier="production",
            rel_tol=1.0e-5,
            scalar_abs_tol=1.0e-9,
            tensor_rms_abs_tol=1.0e-10,
            tensor_max_abs_tol=1.0e-9,
            scalar_zero_tol=1.0e-9,
            tensor_rms_zero_tol=1.0e-10,
            tensor_max_zero_tol=1.0e-9,
        )
    elif key == "smoke":
        base = DerivativeValidationConfig(
            tier="smoke",
            rel_tol=1.0e-4,
            scalar_abs_tol=1.0e-8,
            tensor_rms_abs_tol=1.0e-9,
            tensor_max_abs_tol=1.0e-8,
            scalar_zero_tol=1.0e-8,
            tensor_rms_zero_tol=1.0e-9,
            tensor_max_zero_tol=1.0e-8,
        )
    elif key in {"tight", "tight_diagnostic"}:
        base = DerivativeValidationConfig(
            tier="tight",
            rel_tol=1.0e-6,
            scalar_abs_tol=1.0e-10,
            tensor_rms_abs_tol=1.0e-10,
            tensor_max_abs_tol=1.0e-10,
            scalar_zero_tol=1.0e-10,
            tensor_rms_zero_tol=1.0e-10,
            tensor_max_zero_tol=1.0e-10,
        )
    else:
        raise ValueError(f"unsupported derivative validation tier: {tier!r}.")
    rel_tol = float(max_derivative_drift)
    if rel_tol <= 0.0 or not math.isfinite(rel_tol):
        rel_tol = float(base.rel_tol)
    return DerivativeValidationConfig(
        tier=base.tier,
        rel_tol=rel_tol,
        scalar_abs_tol=float(base.scalar_abs_tol),
        tensor_rms_abs_tol=float(base.tensor_rms_abs_tol),
        tensor_max_abs_tol=float(base.tensor_max_abs_tol),
        scalar_zero_tol=float(base.scalar_zero_tol),
        tensor_rms_zero_tol=float(base.tensor_rms_zero_tol),
        tensor_max_zero_tol=float(base.tensor_max_zero_tol),
    )


def _scaled_residual(delta: float, tolerance: float) -> float:
    tol = max(float(tolerance), 1.0e-300)
    return float(delta) / tol


def _scalar_component_diagnostics(
    *,
    name: str,
    primary: float,
    alt: float,
    config: DerivativeValidationConfig,
) -> dict[str, Any]:
    primary_abs = abs(float(primary))
    alt_abs = abs(float(alt))
    delta_abs = abs(float(primary) - float(alt))
    scale = max(primary_abs, alt_abs)
    active = scale > float(config.scalar_zero_tol)
    tolerance = (
        float(config.scalar_abs_tol) + float(config.rel_tol) * scale
        if active
        else float(config.scalar_abs_tol)
    )
    passed = delta_abs <= tolerance
    return {
        "component": str(name),
        "kind": "scalar",
        "classification": "active" if active else "suppressed",
        "passed": bool(passed),
        "primary_abs": float(primary_abs),
        "alt_abs": float(alt_abs),
        "delta_abs": float(delta_abs),
        "scale": float(scale),
        "zero_tol": float(config.scalar_zero_tol),
        "abs_tol": float(config.scalar_abs_tol),
        "rel_tol": float(config.rel_tol),
        "tolerance": float(tolerance),
        "scaled_residual": _scaled_residual(delta_abs, tolerance),
    }


def _tensor_component_diagnostics(
    *,
    name: str,
    primary: np.ndarray,
    alt: np.ndarray,
    config: DerivativeValidationConfig,
) -> dict[str, Any]:
    lhs = np.asarray(primary, dtype=float)
    rhs = np.asarray(alt, dtype=float)
    if lhs.shape != rhs.shape:
        raise ValueError(f"{name} derivative shape mismatch: {lhs.shape} != {rhs.shape}.")
    delta = lhs - rhs
    size = max(int(lhs.size), 1)
    primary_fro = float(np.linalg.norm(lhs))
    alt_fro = float(np.linalg.norm(rhs))
    delta_fro = float(np.linalg.norm(delta))
    primary_rms = primary_fro / math.sqrt(size)
    alt_rms = alt_fro / math.sqrt(size)
    delta_rms = delta_fro / math.sqrt(size)
    primary_max = float(np.max(np.abs(lhs))) if lhs.size else 0.0
    alt_max = float(np.max(np.abs(rhs))) if rhs.size else 0.0
    delta_max = float(np.max(np.abs(delta))) if delta.size else 0.0
    rms_scale = max(primary_rms, alt_rms)
    max_scale = max(primary_max, alt_max)
    active = (
        rms_scale > float(config.tensor_rms_zero_tol)
        or max_scale > float(config.tensor_max_zero_tol)
    )
    rms_tolerance = (
        float(config.tensor_rms_abs_tol) + float(config.rel_tol) * rms_scale
        if active
        else float(config.tensor_rms_abs_tol)
    )
    max_tolerance = (
        float(config.tensor_max_abs_tol) + float(config.rel_tol) * max_scale
        if active
        else float(config.tensor_max_abs_tol)
    )
    denom = primary_fro * alt_fro
    direction_cosine = None
    if denom > 1.0e-300:
        direction_cosine = float(np.vdot(lhs.reshape(-1), rhs.reshape(-1)).real / denom)
    rms_passed = delta_rms <= rms_tolerance
    max_passed = delta_max <= max_tolerance
    return {
        "component": str(name),
        "kind": "tensor",
        "classification": "active" if active else "suppressed",
        "passed": bool(rms_passed and max_passed),
        "shape": [int(v) for v in lhs.shape],
        "primary_fro": float(primary_fro),
        "alt_fro": float(alt_fro),
        "delta_fro": float(delta_fro),
        "primary_rms": float(primary_rms),
        "alt_rms": float(alt_rms),
        "delta_rms": float(delta_rms),
        "primary_max": float(primary_max),
        "alt_max": float(alt_max),
        "delta_max": float(delta_max),
        "rms_scale": float(rms_scale),
        "max_scale": float(max_scale),
        "rms_zero_tol": float(config.tensor_rms_zero_tol),
        "max_zero_tol": float(config.tensor_max_zero_tol),
        "rms_abs_tol": float(config.tensor_rms_abs_tol),
        "max_abs_tol": float(config.tensor_max_abs_tol),
        "rel_tol": float(config.rel_tol),
        "rms_tolerance": float(rms_tolerance),
        "max_tolerance": float(max_tolerance),
        "rms_scaled_residual": _scaled_residual(delta_rms, rms_tolerance),
        "max_scaled_residual": _scaled_residual(delta_max, max_tolerance),
        "direction_cosine": direction_cosine,
    }


def _validate_derivative_pair(
    *,
    scalar_primary: float,
    scalar_alt: float,
    one_body_primary: np.ndarray,
    one_body_alt: np.ndarray,
    two_body_primary: np.ndarray,
    two_body_alt: np.ndarray,
    config: DerivativeValidationConfig,
) -> dict[str, Any]:
    scalar_diag = _scalar_component_diagnostics(
        name="scalar",
        primary=float(scalar_primary),
        alt=float(scalar_alt),
        config=config,
    )
    one_diag = _tensor_component_diagnostics(
        name="one_body",
        primary=np.asarray(one_body_primary, dtype=float),
        alt=np.asarray(one_body_alt, dtype=float),
        config=config,
    )
    two_diag = _tensor_component_diagnostics(
        name="two_body",
        primary=np.asarray(two_body_primary, dtype=float),
        alt=np.asarray(two_body_alt, dtype=float),
        config=config,
    )
    components = (scalar_diag, one_diag, two_diag)
    passed = all(bool(row["passed"]) for row in components)
    any_active = any(str(row["classification"]) == "active" for row in components)
    legacy_relative_drift = max(
        _relative_delta(float(scalar_primary), float(scalar_alt)),
        _relative_delta(
            float(np.linalg.norm(one_body_primary)),
            float(np.linalg.norm(one_body_alt)),
        ),
        _relative_delta(
            float(np.linalg.norm(two_body_primary)),
            float(np.linalg.norm(two_body_alt)),
        ),
    )
    scaled_residuals = [
        float(scalar_diag["scaled_residual"]),
        float(one_diag["rms_scaled_residual"]),
        float(one_diag["max_scaled_residual"]),
        float(two_diag["rms_scaled_residual"]),
        float(two_diag["max_scaled_residual"]),
    ]
    return {
        "tier": str(config.tier),
        "rel_tol": float(config.rel_tol),
        "passed": bool(passed),
        "classification": (
            "failed"
            if not passed
            else "active"
            if any_active
            else "numerically_suppressed"
        ),
        "legacy_relative_drift": float(legacy_relative_drift),
        "max_scaled_residual": float(max(scaled_residuals)),
        "components": {
            "scalar": scalar_diag,
            "one_body": one_diag,
            "two_body": two_diag,
        },
    }


def _eri_symmetry_residual(eri: np.ndarray) -> float:
    arr = np.asarray(eri, dtype=float)
    residuals = [
        np.linalg.norm(arr - np.swapaxes(arr, 0, 1)),
        np.linalg.norm(arr - np.swapaxes(arr, 2, 3)),
        np.linalg.norm(arr - np.transpose(arr, (2, 3, 0, 1))),
    ]
    return float(max(residuals))


def _active_problem_from_tensors(
    *,
    n_spatial_orbitals: int,
    num_particles: tuple[int, int],
    scalar_energy_hartree: float,
    one_body_integrals: np.ndarray,
    two_body_integrals: np.ndarray,
    basis: str,
    charge: int,
    multiplicity: int,
    reference: str,
) -> RestrictedClosedShellMolecularProblem:
    return RestrictedClosedShellMolecularProblem(
        geometry_spec="active-space tensor record",
        basis=str(basis),
        charge=int(charge),
        multiplicity=int(multiplicity),
        reference=str(reference),
        n_spatial_orbitals=int(n_spatial_orbitals),
        n_alpha=int(num_particles[0]),
        n_beta=int(num_particles[1]),
        hf_energy=float(scalar_energy_hartree),
        nuclear_repulsion_energy=float(scalar_energy_hartree),
        one_body_integrals_mo=np.asarray(one_body_integrals, dtype=float),
        two_body_integrals_mo=np.asarray(two_body_integrals, dtype=float),
    )


def _build_register_layout(
    *,
    n_spatial_orbitals: int,
    mode_labels: Sequence[str],
    mode_cutoffs: Sequence[int],
) -> RegisterLayout:
    n_fermion = 2 * int(n_spatial_orbitals)
    start = int(n_fermion)
    blocks: list[BosonModeRegister] = []
    for idx, (label, cutoff) in enumerate(zip(mode_labels, mode_cutoffs)):
        qpb = boson_qubits_per_site(int(cutoff), encoding="binary")
        blocks.append(
            BosonModeRegister(
                mode_index=int(idx),
                mode_label=str(label),
                qubit_start=int(start),
                n_qubits=int(qpb),
                n_ph_max=int(cutoff),
                encoding="binary",
            )
        )
        start += int(qpb)
    return RegisterLayout(
        n_fermion_qubits=int(n_fermion),
        fermion_qubits=tuple(range(int(n_fermion))),
        boson_modes=tuple(blocks),
        spin_orbital_ordering="blocked",
    )


def _reference_state(layout: RegisterLayout, *, n_spatial_orbitals: int, num_particles: tuple[int, int]) -> np.ndarray:
    occupied = hartree_fock_occupied_qubits(
        int(n_spatial_orbitals),
        tuple(int(v) for v in num_particles),
        indexing="blocked",
    )
    idx = 0
    for q in occupied:
        idx |= 1 << int(q)
    state = np.zeros(1 << int(layout.n_total_qubits), dtype=complex)
    state[int(idx)] = 1.0
    return state


def _physical_basis_indices(
    layout: RegisterLayout,
    *,
    n_spatial_orbitals: int,
    num_particles: tuple[int, int],
) -> list[int]:
    n_spatial = int(n_spatial_orbitals)
    n_alpha, n_beta = (int(num_particles[0]), int(num_particles[1]))
    basis: list[int] = []
    alpha_patterns: list[int] = []
    beta_patterns: list[int] = []
    for occ in itertools.combinations(range(n_spatial), n_alpha):
        bits = 0
        for q in occ:
            bits |= 1 << int(q)
        alpha_patterns.append(bits)
    for occ in itertools.combinations(range(n_spatial), n_beta):
        bits = 0
        for q in occ:
            bits |= 1 << (n_spatial + int(q))
        beta_patterns.append(bits)
    boson_ranges = [range(int(block.n_ph_max) + 1) for block in layout.boson_modes]
    for a_bits in alpha_patterns:
        for b_bits in beta_patterns:
            fermion_bits = int(a_bits | b_bits)
            for occupations in itertools.product(*boson_ranges):
                index = int(fermion_bits)
                for occupation, block in zip(occupations, layout.boson_modes):
                    index |= int(occupation) << int(block.qubit_start)
                basis.append(index)
    return basis


def _solve_sector_dense(
    h_poly: PauliPolynomial,
    *,
    layout: RegisterLayout,
    n_spatial_orbitals: int,
    num_particles: tuple[int, int],
    dense_full_dim_cap: int,
    return_state: bool,
    n_low_energies: int,
) -> tuple[float, tuple[float, ...], np.ndarray | None]:
    full_dim = 1 << int(layout.n_total_qubits)
    if full_dim > int(dense_full_dim_cap):
        raise ValueError(
            "Dense H2O linear-FD exact solve refused: "
            f"full dimension {full_dim} exceeds cap {dense_full_dim_cap}."
        )
    basis = _physical_basis_indices(
        layout,
        n_spatial_orbitals=int(n_spatial_orbitals),
        num_particles=tuple(int(v) for v in num_particles),
    )
    if not basis:
        raise ValueError("Physical sector basis is empty.")
    h_full = hamiltonian_matrix(h_poly)
    sub = h_full[np.ix_(basis, basis)]
    evals, evecs = np.linalg.eigh(np.asarray(sub, dtype=complex))
    order = np.argsort(np.real(evals))
    low = tuple(float(np.real(evals[int(i)])) for i in order[: max(1, int(n_low_energies))])
    ground_energy = float(low[0])
    if not bool(return_state):
        return ground_energy, low, None
    sector_vec = np.asarray(evecs[:, int(order[0])], dtype=complex).reshape(-1)
    state = np.zeros(full_dim, dtype=complex)
    for local_idx, basis_idx in enumerate(basis):
        state[int(basis_idx)] = complex(sector_vec[int(local_idx)])
    norm = float(np.vdot(state, state).real)
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError("Exact sector state has invalid norm.")
    state /= math.sqrt(norm)
    return ground_energy, low, state


def _apply_pauli_word_to_basis_index(pauli: str, index: int) -> tuple[int, complex]:
    word = str(pauli).strip().lower().replace("i", "e")
    nq = len(word)
    out = int(index)
    phase = 1.0 + 0.0j
    for q in range(nq):
        op = word[nq - 1 - q]
        bit = (int(index) >> q) & 1
        if op == "e":
            continue
        if op == "z":
            if bit:
                phase = -phase
            continue
        if op == "x":
            out ^= 1 << q
            continue
        if op == "y":
            out ^= 1 << q
            phase *= 1j if bit == 0 else -1j
            continue
        raise ValueError(f"invalid Pauli symbol {op!r} in {pauli!r}.")
    return int(out), complex(phase)


def _sector_sparse_matrix(
    h_poly: PauliPolynomial,
    *,
    layout: RegisterLayout,
    n_spatial_orbitals: int,
    num_particles: tuple[int, int],
    coeff_tol: float = 1.0e-12,
) -> tuple[Any, list[int]]:
    try:
        from scipy.sparse import coo_matrix  # type: ignore
    except Exception as exc:  # pragma: no cover - scipy is a project dependency
        raise ImportError("scipy is required for sparse H2O sector exact reference.") from exc
    basis = _physical_basis_indices(
        layout,
        n_spatial_orbitals=int(n_spatial_orbitals),
        num_particles=tuple(int(v) for v in num_particles),
    )
    if not basis:
        raise ValueError("Physical sector basis is empty.")
    row_by_full_index = {int(full_idx): int(row_idx) for row_idx, full_idx in enumerate(basis)}
    rows: list[int] = []
    cols: list[int] = []
    data: list[complex] = []
    for term in h_poly.return_polynomial():
        coeff = complex(term.p_coeff)
        if abs(coeff) <= float(coeff_tol):
            continue
        word = str(term.pw2strng())
        for col, full_index in enumerate(basis):
            mapped, phase = _apply_pauli_word_to_basis_index(word, int(full_index))
            row = row_by_full_index.get(int(mapped))
            if row is None:
                continue
            rows.append(int(row))
            cols.append(int(col))
            data.append(complex(coeff) * complex(phase))
    dim = len(basis)
    matrix = coo_matrix((data, (rows, cols)), shape=(dim, dim), dtype=complex).tocsr()
    matrix.sum_duplicates()
    return matrix, basis


def build_fixed_particle_sector_sparse_matrix(
    h_poly: PauliPolynomial,
    *,
    layout: RegisterLayout,
    n_spatial_orbitals: int,
    num_particles: tuple[int, int],
    coeff_tol: float = 1.0e-12,
) -> tuple[Any, list[int]]:
    """Public sparse-sector constructor shared by H2O control exports."""

    return _sector_sparse_matrix(
        h_poly,
        layout=layout,
        n_spatial_orbitals=int(n_spatial_orbitals),
        num_particles=tuple(int(value) for value in num_particles),
        coeff_tol=float(coeff_tol),
    )


def _solve_sector_sparse(
    h_poly: PauliPolynomial,
    *,
    layout: RegisterLayout,
    n_spatial_orbitals: int,
    num_particles: tuple[int, int],
    return_state: bool,
    n_low_energies: int,
    solver_tolerance: float = 1.0e-10,
) -> tuple[float, tuple[float, ...], np.ndarray | None]:
    try:
        from scipy.sparse.linalg import eigsh  # type: ignore
    except Exception as exc:  # pragma: no cover - scipy is a project dependency
        raise ImportError("scipy is required for sparse H2O sector exact reference.") from exc
    matrix, basis = _sector_sparse_matrix(
        h_poly,
        layout=layout,
        n_spatial_orbitals=int(n_spatial_orbitals),
        num_particles=tuple(int(v) for v in num_particles),
    )
    dim = int(matrix.shape[0])
    requested = max(1, int(n_low_energies))
    if dim <= requested + 1:
        dense = matrix.toarray()
        evals, evecs = np.linalg.eigh(np.asarray(dense, dtype=complex))
        order = np.argsort(np.real(evals))
        low = tuple(float(np.real(evals[int(i)])) for i in order[:requested])
        sector_vec = np.asarray(evecs[:, int(order[0])], dtype=complex).reshape(-1)
    else:
        k = min(requested, dim - 1)
        evals, evecs = eigsh(matrix, k=int(k), which="SA", tol=float(solver_tolerance))
        order = np.argsort(np.real(evals))
        low = tuple(float(np.real(evals[int(i)])) for i in order)
        sector_vec = np.asarray(evecs[:, int(order[0])], dtype=complex).reshape(-1)
    ground_energy = float(low[0])
    if not bool(return_state):
        return ground_energy, low, None
    full_dim = 1 << int(layout.n_total_qubits)
    state = np.zeros(full_dim, dtype=complex)
    for local_idx, basis_idx in enumerate(basis):
        state[int(basis_idx)] = complex(sector_vec[int(local_idx)])
    norm = float(np.vdot(state, state).real)
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError("Sparse exact sector state has invalid norm.")
    state /= math.sqrt(norm)
    return ground_energy, low, state


def _exact_state_record(state: np.ndarray | None, *, total_qubits: int) -> ExactStateVectorRecord | None:
    if state is None:
        return None
    arr = np.asarray(state, dtype=complex).reshape(-1)
    norm = float(np.vdot(arr, arr).real)
    amplitudes: dict[str, dict[str, float]] = {}
    for idx, amp in enumerate(arr):
        value = complex(amp)
        if abs(value) <= 1.0e-14:
            continue
        amplitudes[format(int(idx), f"0{int(total_qubits)}b")] = {
            "re": float(value.real),
            "im": float(value.imag),
        }
    return ExactStateVectorRecord(
        available=True,
        representation="sparse_full_register_qn_to_q0",
        n_qubits=int(total_qubits),
        norm=float(norm),
        amplitudes_qn_to_q0=amplitudes,
    )


def _build_encoded_operators(
    *,
    n_spatial_orbitals: int,
    num_particles: tuple[int, int],
    basis: str,
    charge: int,
    multiplicity: int,
    reference: str,
    active_scalar: float,
    active_one_body: np.ndarray,
    active_two_body: np.ndarray,
    derivatives: Sequence[FirstDerivativeRecord],
    layout: RegisterLayout,
    frequencies_hartree: Sequence[float],
    operator_cleanup_tol: float = 1.0e-12,
) -> tuple[EncodedOperatorBundle, PauliPolynomial, tuple[PauliPolynomial, ...], tuple[PauliPolynomial, ...], tuple[PauliPolynomial, ...]]:
    electronic_problem = _active_problem_from_tensors(
        n_spatial_orbitals=int(n_spatial_orbitals),
        num_particles=tuple(num_particles),
        scalar_energy_hartree=float(active_scalar),
        one_body_integrals=np.asarray(active_one_body, dtype=float),
        two_body_integrals=np.asarray(active_two_body, dtype=float),
        basis=str(basis),
        charge=int(charge),
        multiplicity=int(multiplicity),
        reference=str(reference),
    )
    h_electronic = _clean_real_polynomial(
        build_restricted_closed_shell_molecular_hamiltonian(electronic_problem),
        tol=float(operator_cleanup_tol),
    )
    boson_qubits = int(layout.n_boson_qubits)
    h_vibronic = _lift_fermion_polynomial(h_electronic, boson_qubits=boson_qubits)
    q_ops: list[PauliPolynomial] = []
    p_ops: list[PauliPolynomial] = []
    n_ops: list[PauliPolynomial] = []
    d_polys: list[PauliPolynomial] = []
    for idx, (derivative, block, omega) in enumerate(zip(derivatives, layout.boson_modes, frequencies_hartree)):
        d_problem = _active_problem_from_tensors(
            n_spatial_orbitals=int(n_spatial_orbitals),
            num_particles=tuple(num_particles),
            scalar_energy_hartree=float(derivative.scalar_derivative_hartree_per_q),
            one_body_integrals=np.asarray(derivative.one_body_derivative, dtype=float),
            two_body_integrals=np.asarray(derivative.two_body_derivative, dtype=float),
            basis=str(basis),
            charge=int(charge),
            multiplicity=int(multiplicity),
            reference=str(reference),
        )
        d_poly = _clean_real_polynomial(
            build_restricted_closed_shell_molecular_hamiltonian(d_problem),
            tol=float(operator_cleanup_tol),
        )
        d_polys.append(d_poly)
        lifted_d = _lift_fermion_polynomial(d_poly, boson_qubits=boson_qubits)
        q_op = _clean_real_polynomial(
            (1.0 / math.sqrt(2.0 * float(omega)))
            * boson_operator(
                "JW",
                int(layout.n_total_qubits),
                block.qubits,
                which="x",
                n_ph_max=int(block.n_ph_max),
                encoding="binary",
            ),
            tol=float(operator_cleanup_tol),
        )
        p_op = _clean_real_polynomial(
            math.sqrt(float(omega) / 2.0)
            * _boson_momentum_operator(
                nq_total=int(layout.n_total_qubits),
                boson_qubits=block.qubits,
                n_ph_max=int(block.n_ph_max),
                boson_encoding="binary",
            ),
            tol=float(operator_cleanup_tol),
        )
        n_op = _clean_real_polynomial(
            boson_operator(
                "JW",
                int(layout.n_total_qubits),
                block.qubits,
                which="n",
                n_ph_max=int(block.n_ph_max),
                encoding="binary",
            ),
            tol=float(operator_cleanup_tol),
        )
        q_ops.append(q_op)
        p_ops.append(p_op)
        n_ops.append(n_op)
        h_vibronic = h_vibronic + (float(omega) * (n_op + 0.5)) + (lifted_d * q_op)
    h_vibronic = _clean_real_polynomial(h_vibronic, tol=float(operator_cleanup_tol))
    encoded = EncodedOperatorBundle(
        h_electronic=pauli_polynomial_to_jsonable(h_electronic),
        dH_dQ_by_mode=tuple(pauli_polynomial_to_jsonable(poly) for poly in d_polys),
        h_vibronic=pauli_polynomial_to_jsonable(h_vibronic),
        q_by_mode=tuple(pauli_polynomial_to_jsonable(poly) for poly in q_ops),
        p_by_mode=tuple(pauli_polynomial_to_jsonable(poly) for poly in p_ops),
        n_by_mode=tuple(pauli_polynomial_to_jsonable(poly) for poly in n_ops),
    )
    return encoded, h_vibronic, tuple(q_ops), tuple(p_ops), tuple(d_polys)


def _build_pool_rows(
    *,
    n_spatial_orbitals: int,
    num_particles: tuple[int, int],
    layout: RegisterLayout,
    p_ops: Sequence[PauliPolynomial],
    derivative_polys: Sequence[PauliPolynomial],
) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    boson_qubits = int(layout.n_boson_qubits)
    for term in build_molecular_uccsd_pool(
        n_spatial_orbitals=int(n_spatial_orbitals),
        num_particles=tuple(int(v) for v in num_particles),
        ordering="blocked",
    ):
        rows.append(
            {
                "label": f"el::{term.label}",
                "polynomial": pauli_polynomial_to_jsonable(
                    _lift_fermion_polynomial(term.polynomial, boson_qubits=boson_qubits)
                ),
                "execution_mode": str(getattr(term, "execution_mode", "termwise_product")),
                "generator_family": "electronic_uccsd",
            }
        )
    for block, p_op, derivative_poly in zip(layout.boson_modes, p_ops, derivative_polys):
        lifted_derivative = _lift_fermion_polynomial(derivative_poly, boson_qubits=boson_qubits)
        rows.append(
            {
                "label": f"boson::{block.mode_label}::p",
                "polynomial": pauli_polynomial_to_jsonable(p_op),
                "execution_mode": "termwise_product",
                "generator_family": "single_mode_momentum",
            }
        )
        rows.append(
            {
                "label": f"coupled::{block.mode_label}::dH_dQ_times_p",
                "polynomial": pauli_polynomial_to_jsonable(_clean_real_polynomial(lifted_derivative * p_op)),
                "execution_mode": "grouped_exact",
                "generator_family": "linear_vibronic_derivative_momentum",
            }
        )
    if not rows:
        raise ValueError("Generated H2O linear-FD production pool is empty.")
    return tuple(rows)


def _tensor_record_from_json(obj: Mapping[str, Any], *, label: str) -> tuple[float, np.ndarray, np.ndarray]:
    scalar = float(obj.get("scalar_energy_hartree", 0.0))
    h = _array(obj["one_body_integrals"], label=f"{label}.one_body_integrals", ndim=2)
    g = _array(obj["two_body_integrals"], label=f"{label}.two_body_integrals", ndim=4)
    return scalar, h, g


def _alignment_diagnostic(
    *,
    alignment_id: str,
    center_snapshot_id: str,
    displaced_snapshot_id: str,
    displacement_id: str,
    h: np.ndarray,
    g: np.ndarray,
    rotation: np.ndarray | None,
    payload: Mapping[str, Any],
) -> AlignmentDiagnosticsRecord:
    singular_values = np.asarray(payload.get("singular_values", [1.0]), dtype=float)
    thresholds = AlignmentThresholds()
    rotation_residual = 0.0
    if rotation is not None:
        r = np.asarray(rotation, dtype=float)
        rotation_residual = float(np.linalg.norm(r.T @ r - np.eye(r.shape[1])))
    hermiticity = float(np.linalg.norm(np.asarray(h, dtype=float) - np.asarray(h, dtype=float).T))
    eri_residual = _eri_symmetry_residual(np.asarray(g, dtype=float))
    alignment_residual = float(payload.get("alignment_residual_fro", 0.0))
    leakage_a = None if payload.get("active_to_external_leakage_fro") is None else float(payload["active_to_external_leakage_fro"])
    leakage_b = None if payload.get("external_to_active_leakage_fro") is None else float(payload["external_to_active_leakage_fro"])
    min_sv = float(payload.get("min_singular_value", np.min(singular_values)))
    passed = (
        min_sv >= float(thresholds.min_active_singular_value)
        and alignment_residual <= float(thresholds.max_active_residual_fro)
        and hermiticity <= float(thresholds.max_hermiticity_residual)
        and eri_residual <= float(thresholds.max_eri_symmetry_residual)
        and rotation_residual <= float(thresholds.max_active_residual_fro)
        and (leakage_a is None or leakage_a <= float(thresholds.max_active_to_external_leakage_fro))
        and (leakage_b is None or leakage_b <= float(thresholds.max_active_to_external_leakage_fro))
    )
    if "passed" in payload:
        passed = bool(payload["passed"]) and bool(passed)
    return AlignmentDiagnosticsRecord(
        alignment_id=str(alignment_id),
        center_snapshot_id=str(center_snapshot_id),
        displaced_snapshot_id=str(displaced_snapshot_id),
        displacement_id=str(displacement_id),
        block="active",
        singular_values=singular_values,
        min_singular_value=float(min_sv),
        alignment_residual_fro=float(alignment_residual),
        active_to_external_leakage_fro=leakage_a,
        external_to_active_leakage_fro=leakage_b,
        hermiticity_residual=float(hermiticity),
        eri_symmetry_residual=float(eri_residual),
        rotation_orthogonality_residual=float(rotation_residual),
        thresholds=thresholds,
        passed=bool(passed),
        warnings=tuple(str(v) for v in payload.get("warnings", ())),
    )


def build_h2o_linear_fd_fixture_from_record(
    record: Mapping[str, Any],
    *,
    mode_cutoffs: Sequence[int],
    reference_cutoffs: Sequence[int] | None = None,
    dense_full_dim_cap: int = 8192,
    embed_exact_state: bool = True,
    require_reference_cutoff: bool = True,
    max_derivative_drift: float = 1.0e-6,
    derivative_validation_tier: str = "production",
    exact_reference_policy: str = "auto_sector",
    operator_cleanup_tol: float = 1.0e-12,
    cutoff_energy_tolerance_hartree: float = H2O_LINEAR_FD_CUTOFF_ENERGY_TOLERANCE_HARTREE,
    cutoff_boundary_weight_tolerance: float = H2O_LINEAR_FD_CUTOFF_BOUNDARY_WEIGHT_TOLERANCE,
) -> ProductionVibronicH2OFixture:
    obj = _as_mapping(record, label="H2O linear-FD backend record")
    if str(obj.get("schema")) != BACKEND_RECORD_SCHEMA:
        raise ValueError(f"Unsupported H2O linear-FD backend record schema: {obj.get('schema')!r}")
    backend = dict(_as_mapping(obj.get("backend", {}), label="backend"))
    system = dict(_as_mapping(obj.get("system", {}), label="system"))
    geometry_payload = _as_mapping(obj["geometry"], label="geometry")
    active_payload = _as_mapping(obj["active_space"], label="active_space")
    modes_payload = list(obj.get("normal_modes", ()))
    aligned_payload = list(obj.get("aligned_tensors", ()))
    if len(modes_payload) != 3:
        raise ValueError("H2O linear-FD backend record must contain exactly three normal modes.")
    mode_cutoffs_t = tuple(int(v) for v in mode_cutoffs)
    if len(mode_cutoffs_t) != 3:
        raise ValueError("mode_cutoffs must contain exactly three entries.")
    if reference_cutoffs is not None:
        reference_cutoffs_t: tuple[int, ...] | None = tuple(int(v) for v in reference_cutoffs)
    elif bool(require_reference_cutoff):
        reference_cutoffs_t = tuple(int(v) + 1 for v in mode_cutoffs_t)
    else:
        reference_cutoffs_t = None
    if reference_cutoffs_t is not None and len(reference_cutoffs_t) != 3:
        raise ValueError("reference_cutoffs must contain exactly three entries.")
    exact_policy = str(exact_reference_policy).strip().lower().replace("-", "_")
    if exact_policy not in {
        "auto_sector",
        "dense_required",
        "candidate_without_exact",
        "sparse_sector_eigsh",
    }:
        raise ValueError(f"unsupported exact_reference_policy: {exact_reference_policy!r}.")
    derivative_validation = _derivative_validation_config(
        tier=str(derivative_validation_tier),
        max_derivative_drift=float(max_derivative_drift),
    )

    mode_records: list[NormalModeRecord] = []
    for idx, raw_mode in enumerate(modes_payload):
        mode = _as_mapping(raw_mode, label=f"normal_modes[{idx}]")
        label = _normalize_mode_label(str(mode["label"]))
        q_step = float(mode["q_step_au"])
        if q_step <= 0.0:
            raise ValueError(f"mode {label}: q_step_au must be positive.")
        q_step_alt = float(mode.get("q_step_alt_au", q_step / 2.0))
        if q_step_alt <= 0.0:
            raise ValueError(f"mode {label}: q_step_alt_au must be positive.")
        freq = float(mode["frequency_hartree"])
        if freq <= 0.0 or not math.isfinite(freq):
            raise ValueError(f"mode {label}: frequency_hartree must be positive and finite.")
        mode_records.append(
            NormalModeRecord(
                mode_index=int(idx),
                label=label,
                frequency_hartree=float(freq),
                frequency_cm1=None if mode.get("frequency_cm1") is None else float(mode["frequency_cm1"]),
                mass_weighted_eigenvector=_array(
                    mode["mass_weighted_eigenvector"],
                    label=f"normal_modes[{idx}].mass_weighted_eigenvector",
                    ndim=2,
                ),
                q_step_au=float(q_step),
                q_step_alt_au=float(q_step_alt),
            )
        )
    mode_labels = tuple(row.label for row in mode_records)
    if tuple(sorted(mode_labels)) != tuple(sorted(("bend", "symmetric_stretch", "antisymmetric_stretch"))):
        raise ValueError("H2O linear-FD record must contain bend, symmetric_stretch, and antisymmetric_stretch modes.")

    n_spatial = int(active_payload["n_spatial_orbitals"])
    num_particles = _tuple_int(active_payload["num_particles"], label="active_space.num_particles")
    if len(num_particles) != 2:
        raise ValueError("active_space.num_particles must contain (n_alpha, n_beta).")
    active_scalar, active_h, active_g = _tensor_record_from_json(active_payload, label="active_space")
    if active_h.shape != (n_spatial, n_spatial) or active_g.shape != (n_spatial, n_spatial, n_spatial, n_spatial):
        raise ValueError("active-space tensor shapes do not match n_spatial_orbitals.")
    layout = _build_register_layout(
        n_spatial_orbitals=n_spatial,
        mode_labels=mode_labels,
        mode_cutoffs=mode_cutoffs_t,
    )

    aligned_by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
    for raw_aligned in aligned_payload:
        row = _as_mapping(raw_aligned, label="aligned_tensors[]")
        key = (str(row["mode_label"]), int(row["sign"]))
        step_kind = str(row.get("step_kind", "primary")).strip().lower()
        aligned_by_key[(f"{step_kind}:{key[0]}", key[1])] = row

    displacements: list[DisplacedGeometryRecord] = []
    aligned_records: list[AlignedActiveTensorRecord] = []
    alignment_records: list[AlignmentDiagnosticsRecord] = []
    derivatives: list[FirstDerivativeRecord] = []
    frequencies = tuple(float(row.frequency_hartree) for row in mode_records)
    basis = str(backend.get("basis", "unknown"))
    reference = str(backend.get("reference", "rhf"))
    charge = int(system.get("charge", 0))
    multiplicity = int(system.get("multiplicity", 1))
    center_snapshot_id = str(obj.get("center_snapshot_id", "center"))
    center_geometry_id = str(geometry_payload.get("geometry_id", "h2o_center"))

    for mode in mode_records:
        primary_plus = aligned_by_key.get((f"primary:{mode.label}", 1))
        primary_minus = aligned_by_key.get((f"primary:{mode.label}", -1))
        alt_plus = aligned_by_key.get((f"alt:{mode.label}", 1))
        alt_minus = aligned_by_key.get((f"alt:{mode.label}", -1))
        if primary_plus is None or primary_minus is None:
            raise ValueError(f"mode {mode.label}: primary plus/minus aligned tensors are required.")
        if alt_plus is None or alt_minus is None:
            raise ValueError(f"mode {mode.label}: alt plus/minus aligned tensors are required for drift diagnostics.")

        def _append_tensor(row: Mapping[str, Any]) -> tuple[str, float, np.ndarray, np.ndarray]:
            tensor_id = str(row["aligned_tensor_id"])
            displacement_id = str(row["displacement_id"])
            step_kind = str(row.get("step_kind", "primary")).strip().lower()
            q_disp = float(row.get("q_displacement_au", mode.q_step_au if step_kind == "primary" else mode.q_step_alt_au))
            sign = int(row["sign"])
            scalar, h, g = _tensor_record_from_json(row, label=tensor_id)
            alignment_payload = _as_mapping(row.get("alignment", {}), label=f"{tensor_id}.alignment")
            rotation = None if alignment_payload.get("active_rotation") is None else _array(
                alignment_payload["active_rotation"],
                label=f"{tensor_id}.alignment.active_rotation",
                ndim=2,
            )
            alignment_id = str(alignment_payload.get("alignment_id", f"align_{tensor_id}"))
            snapshot_id = str(row.get("source_snapshot_id", f"snap_{displacement_id}"))
            geometry_id = str(row.get("geometry_id", f"h2o_{displacement_id}"))
            coordinates = _array(
                row.get("coordinates_bohr", geometry_payload["coordinates_bohr"]),
                label=f"{tensor_id}.coordinates_bohr",
                ndim=2,
            )
            displacements.append(
                DisplacedGeometryRecord(
                    displacement_id=displacement_id,
                    purpose="first_derivative" if step_kind == "primary" else "finite_difference_drift",
                    mode_indices=(int(mode.mode_index),),
                    signs=(int(sign),),
                    q_displacements_au=(float(q_disp),),
                    geometry_id=geometry_id,
                    snapshot_id=snapshot_id,
                    coordinates_bohr=coordinates,
                )
            )
            aligned_records.append(
                AlignedActiveTensorRecord(
                    aligned_tensor_id=tensor_id,
                    source_snapshot_id=snapshot_id,
                    displacement_id=displacement_id,
                    scalar_energy_hartree=float(scalar),
                    one_body_integrals=h,
                    two_body_integrals=g,
                    alignment_id=alignment_id,
                )
            )
            alignment_records.append(
                _alignment_diagnostic(
                    alignment_id=alignment_id,
                    center_snapshot_id=center_snapshot_id,
                    displaced_snapshot_id=snapshot_id,
                    displacement_id=displacement_id,
                    h=h,
                    g=g,
                    rotation=rotation,
                    payload=alignment_payload,
                )
            )
            return tensor_id, scalar, h, g

        plus_id, plus_scalar, plus_h, plus_g = _append_tensor(primary_plus)
        minus_id, minus_scalar, minus_h, minus_g = _append_tensor(primary_minus)
        _alt_plus_id, alt_plus_scalar, alt_plus_h, alt_plus_g = _append_tensor(alt_plus)
        _alt_minus_id, alt_minus_scalar, alt_minus_h, alt_minus_g = _append_tensor(alt_minus)
        q_step = float(mode.q_step_au)
        q_step_alt = float(mode.q_step_alt_au)
        scalar_deriv = (float(plus_scalar) - float(minus_scalar)) / (2.0 * q_step)
        h_deriv = (np.asarray(plus_h, dtype=float) - np.asarray(minus_h, dtype=float)) / (2.0 * q_step)
        g_deriv = (np.asarray(plus_g, dtype=float) - np.asarray(minus_g, dtype=float)) / (2.0 * q_step)
        alt_scalar_deriv = (float(alt_plus_scalar) - float(alt_minus_scalar)) / (2.0 * q_step_alt)
        alt_h_deriv = (np.asarray(alt_plus_h, dtype=float) - np.asarray(alt_minus_h, dtype=float)) / (2.0 * q_step_alt)
        alt_g_deriv = (np.asarray(alt_plus_g, dtype=float) - np.asarray(alt_minus_g, dtype=float)) / (2.0 * q_step_alt)
        drift_diagnostics = _validate_derivative_pair(
            scalar_primary=float(scalar_deriv),
            scalar_alt=float(alt_scalar_deriv),
            one_body_primary=h_deriv,
            one_body_alt=alt_h_deriv,
            two_body_primary=g_deriv,
            two_body_alt=alt_g_deriv,
            config=derivative_validation,
        )
        if not bool(drift_diagnostics["passed"]):
            raise ValueError(
                f"mode {mode.label}: finite-difference drift validation failed "
                f"(classification={drift_diagnostics['classification']}, "
                f"max_scaled_residual={float(drift_diagnostics['max_scaled_residual']):.3e}, "
                f"legacy_relative_drift={float(drift_diagnostics['legacy_relative_drift']):.3e})."
            )
        derivatives.append(
            FirstDerivativeRecord(
                derivative_id=f"d_{mode.label}",
                mode_index=int(mode.mode_index),
                mode_label=str(mode.label),
                q_step_au=float(q_step),
                plus_aligned_tensor_id=plus_id,
                minus_aligned_tensor_id=minus_id,
                scalar_derivative_hartree_per_q=float(scalar_deriv),
                one_body_derivative=h_deriv,
                two_body_derivative=g_deriv,
                scalar_derivative_included=True,
                scalar_derivative_convention="nuclear_repulsion_plus_closed_shell_frozen_core_scalar",
                derivative_source=H2O_LINEAR_FD_DERIVATIVE_SOURCE,
                norms=DerivativeNorms(
                    scalar_abs=float(abs(scalar_deriv)),
                    one_body_fro=float(np.linalg.norm(h_deriv)),
                    two_body_fro=float(np.linalg.norm(g_deriv)),
                ),
                finite_difference_drift=float(drift_diagnostics["max_scaled_residual"]),
                finite_difference_diagnostics=drift_diagnostics,
                active_equilibrium_force=None,
                passed=True,
            )
        )

    encoded, h_vibronic, _q_ops, p_ops, derivative_polys = _build_encoded_operators(
        n_spatial_orbitals=int(n_spatial),
        num_particles=tuple(int(v) for v in num_particles),
        basis=str(basis),
        charge=int(charge),
        multiplicity=int(multiplicity),
        reference=str(reference),
        active_scalar=float(active_scalar),
        active_one_body=active_h,
        active_two_body=active_g,
        derivatives=tuple(derivatives),
        layout=layout,
        frequencies_hartree=frequencies,
        operator_cleanup_tol=float(operator_cleanup_tol),
    )
    psi_ref = _reference_state(layout, n_spatial_orbitals=int(n_spatial), num_particles=tuple(num_particles))
    sector_dimension = fixed_sector_dimension(
        n_spatial_orbitals=int(n_spatial),
        num_particles=tuple(num_particles),
        mode_cutoffs=mode_cutoffs_t,
    )
    full_qubit_dimension = 1 << int(layout.n_total_qubits)
    cutoff_diag: CutoffDiagnosticsRecord | None = None
    if exact_policy == "candidate_without_exact":
        exact_reference = ExactReferenceRecord(
            available=False,
            method="not_computed",
            sector_dimension=int(sector_dimension),
            full_qubit_dimension=int(full_qubit_dimension),
            ground_energy_hartree=None,
            low_energies_hartree=(),
            boundary_weight=None,
            ground_state=None,
            reason_unavailable=(
                "candidate_without_exact_reference_policy: dense exact reference was intentionally skipped; "
                "run sparse sector reference and cutoff diagnostics before accuracy claims."
            ),
        )
    else:
        use_sparse_work_solver = bool(
            exact_policy == "sparse_sector_eigsh"
            or (
                exact_policy == "auto_sector"
                and int(full_qubit_dimension) > int(dense_full_dim_cap)
            )
        )
        if use_sparse_work_solver:
            ground_energy, low_energies, ground_state = _solve_sector_sparse(
                h_vibronic,
                layout=layout,
                n_spatial_orbitals=int(n_spatial),
                num_particles=tuple(int(v) for v in num_particles),
                return_state=True,
                n_low_energies=4,
            )
            exact_method: str = "sparse_sector_eigsh"
            solver_tolerance: float | None = 1.0e-10
        else:
            ground_energy, low_energies, ground_state = _solve_sector_dense(
                h_vibronic,
                layout=layout,
                n_spatial_orbitals=int(n_spatial),
                num_particles=tuple(int(v) for v in num_particles),
                dense_full_dim_cap=int(dense_full_dim_cap),
                return_state=bool(embed_exact_state),
                n_low_energies=4,
            )
            exact_method = "dense_sector_eigh"
            solver_tolerance = None
        exact_boundary = h2o_linear_fd_boundary_weight_for_state(
            ground_state if ground_state is not None else psi_ref,
            layout=layout,
            state_source="exact_ground_state" if ground_state is not None else "reference_state",
        )
        exact_reference = ExactReferenceRecord(
            available=True,
            method=exact_method,  # type: ignore[arg-type]
            sector_dimension=int(sector_dimension),
            full_qubit_dimension=int(full_qubit_dimension),
            ground_energy_hartree=float(ground_energy),
            low_energies_hartree=tuple(low_energies),
            boundary_weight=exact_boundary,
            ground_state=_exact_state_record(
                ground_state if bool(embed_exact_state) else None,
                total_qubits=int(layout.n_total_qubits),
            ),
            solver_tolerance=solver_tolerance,
        )
        if reference_cutoffs_t is not None:
            ref_layout = _build_register_layout(
                n_spatial_orbitals=int(n_spatial),
                mode_labels=mode_labels,
                mode_cutoffs=reference_cutoffs_t,
            )
            ref_encoded, ref_h_vibronic, *_ = _build_encoded_operators(
                n_spatial_orbitals=int(n_spatial),
                num_particles=tuple(int(v) for v in num_particles),
                basis=str(basis),
                charge=int(charge),
                multiplicity=int(multiplicity),
                reference=str(reference),
                active_scalar=float(active_scalar),
                active_one_body=active_h,
                active_two_body=active_g,
                derivatives=tuple(derivatives),
                layout=ref_layout,
                frequencies_hartree=frequencies,
                operator_cleanup_tol=float(operator_cleanup_tol),
            )
            _ = ref_encoded
            use_sparse_reference_solver = bool(
                exact_policy == "sparse_sector_eigsh"
                or (
                    exact_policy == "auto_sector"
                    and (1 << int(ref_layout.n_total_qubits)) > int(dense_full_dim_cap)
                )
            )
            if use_sparse_reference_solver:
                ref_energy, _ref_lows, _ref_state = _solve_sector_sparse(
                    ref_h_vibronic,
                    layout=ref_layout,
                    n_spatial_orbitals=int(n_spatial),
                    num_particles=tuple(num_particles),
                    return_state=False,
                    n_low_energies=1,
                )
                cutoff_policy = "sparse_same_model_reference_cutoff_v1"
            else:
                ref_energy, _ref_lows, _ref_state = _solve_sector_dense(
                    ref_h_vibronic,
                    layout=ref_layout,
                    n_spatial_orbitals=int(n_spatial),
                    num_particles=tuple(num_particles),
                    dense_full_dim_cap=int(dense_full_dim_cap),
                    return_state=False,
                    n_low_energies=1,
                )
                cutoff_policy = "dense_same_model_reference_cutoff_v1"
            delta_energy = float(ground_energy) - float(ref_energy)
            cutoff_assessment = assess_h2o_linear_fd_cutoff_diagnostics(
                delta_energy_hartree=delta_energy,
                work_boundary_weight=exact_boundary,
                energy_tolerance_hartree=float(cutoff_energy_tolerance_hartree),
                boundary_weight_tolerance=float(cutoff_boundary_weight_tolerance),
            )
            cutoff_diag = CutoffDiagnosticsRecord(
                work_cutoffs=mode_cutoffs_t,
                reference_cutoffs=reference_cutoffs_t,
                work_ground_energy_hartree=float(ground_energy),
                reference_ground_energy_hartree=float(ref_energy),
                delta_energy_hartree=float(delta_energy),
                work_boundary_weight=exact_boundary,
                passed=bool(cutoff_assessment["passed"]),
                policy=str(cutoff_policy),
                energy_tolerance_hartree=float(cutoff_assessment["energy_tolerance_hartree"]),
                boundary_weight_tolerance=float(cutoff_assessment["boundary_weight_tolerance"]),
                energy_passed=cutoff_assessment["energy_passed"],
                boundary_passed=cutoff_assessment["boundary_passed"],
            )
        elif bool(require_reference_cutoff):
            raise ValueError("reference_cutoffs are required when require_reference_cutoff=True.")
        else:
            cutoff_assessment = assess_h2o_linear_fd_cutoff_diagnostics(
                delta_energy_hartree=None,
                work_boundary_weight=exact_boundary,
                energy_tolerance_hartree=float(cutoff_energy_tolerance_hartree),
                boundary_weight_tolerance=float(cutoff_boundary_weight_tolerance),
            )
            cutoff_diag = CutoffDiagnosticsRecord(
                work_cutoffs=mode_cutoffs_t,
                reference_cutoffs=None,
                work_ground_energy_hartree=float(ground_energy),
                reference_ground_energy_hartree=None,
                delta_energy_hartree=None,
                work_boundary_weight=exact_boundary,
                passed=bool(cutoff_assessment["passed"]),
                policy="dense_same_cutoff_only_no_reference_cutoff_v1",
                energy_tolerance_hartree=float(cutoff_assessment["energy_tolerance_hartree"]),
                boundary_weight_tolerance=float(cutoff_assessment["boundary_weight_tolerance"]),
                energy_passed=cutoff_assessment["energy_passed"],
                boundary_passed=cutoff_assessment["boundary_passed"],
            )

    pool = _build_pool_rows(
        n_spatial_orbitals=int(n_spatial),
        num_particles=tuple(int(v) for v in num_particles),
        layout=layout,
        p_ops=p_ops,
        derivative_polys=derivative_polys,
    )

    active_kind = str(active_payload.get("active_space_kind", "cas_8e_6o_valence"))
    report_summary = dict(obj.get("report_summary", {}))
    report_summary["exact_reference_policy"] = str(exact_policy)
    report_summary["operator_cleanup_tol"] = float(operator_cleanup_tol)
    report_summary["vibrational_cutoff_converged"] = (
        None if cutoff_diag is None else bool(cutoff_diag.passed)
    )
    report_summary["cutoff_energy_tolerance_hartree"] = float(
        cutoff_energy_tolerance_hartree
    )
    report_summary["cutoff_boundary_weight_tolerance"] = float(
        cutoff_boundary_weight_tolerance
    )
    if exact_policy == "candidate_without_exact":
        report_summary["candidate_without_exact_reference"] = True
        report_summary["candidate_without_exact_reference_reason"] = exact_reference.reason_unavailable
    if not (
        int(n_spatial) == 6
        and tuple(int(v) for v in num_particles) == (4, 4)
        and 2 * int(n_spatial) == 12
    ):
        report_summary.setdefault("paper_iv_active_space_variant", active_kind)

    fixture = ProductionVibronicH2OFixture(
        manifest=FixtureManifest(
            schema=H2O_LINEAR_FD_FIXTURE_SCHEMA,
            schema_version=1,
            family_key=H2O_LINEAR_FD_FAMILY_KEY,
            molecule_family_key=H2O_UMBRELLA_FAMILY_KEY,
            model_role=H2O_LINEAR_FD_MODEL_ROLE,
            production_status=(
                "production_candidate"
                if exact_policy == "candidate_without_exact"
                else "production_validated"
            ),
            derivative_source=H2O_LINEAR_FD_DERIVATIVE_SOURCE,
            created_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            generator_version=GENERATOR_VERSION,
            repository_commit=None,
            provenance_hashes={"backend_record_sha256": sha256_jsonable(record)},
        ),
        geometry=GeometryRecord(
            geometry_id=center_geometry_id,
            molecule="H2O",
            symbols=tuple(str(v) for v in geometry_payload["symbols"]),
            coordinates_bohr=_array(geometry_payload["coordinates_bohr"], label="geometry.coordinates_bohr", ndim=2),
            masses_me=_array(geometry_payload["masses_me"], label="geometry.masses_me", ndim=1),
            charge=int(charge),
            multiplicity=int(multiplicity),
            method=str(backend.get("method", "unknown")),
            basis=str(basis),
            reference=str(reference).upper(),
            optimized=bool(system.get("optimized", True)),
            provenance=dict(geometry_payload.get("provenance", {})),
        ),
        normal_modes=tuple(mode_records),
        displacements=tuple(displacements),
        active_space=ActiveSpaceRecord(
            active_space_kind=active_kind,
            frozen_core_indices=_tuple_int(active_payload.get("frozen_core_indices", ()), label="active_space.frozen_core_indices"),
            active_indices_center=_tuple_int(active_payload.get("active_indices_center", range(n_spatial)), label="active_space.active_indices_center"),
            external_indices=_tuple_int(active_payload.get("external_indices", ()), label="active_space.external_indices"),
            n_spatial_orbitals=int(n_spatial),
            num_particles=tuple(int(v) for v in num_particles),
            scalar_energy_hartree=float(active_scalar),
            one_body_integrals=active_h,
            two_body_integrals=active_g,
        ),
        aligned_tensors=tuple(aligned_records),
        alignment_diagnostics=tuple(alignment_records),
        first_derivatives=tuple(derivatives),
        layout=layout,
        physical_sector=PhysicalSectorRecord(
            n_alpha=int(num_particles[0]),
            n_beta=int(num_particles[1]),
            n_ph_max_by_mode=mode_cutoffs_t,
            mode_labels=mode_labels,
        ),
        encoded_operators=encoded,
        reference_state=psi_ref,
        exact_reference=exact_reference,
        cutoff_diagnostics=cutoff_diag,
        evidence_hooks=EvidenceHooksRecord(
            static_ground_state_ready=True,
            exact_reference_ready=bool(exact_reference.available),
            dynamics_hooks_ready=False,
            qse_hooks_ready=False,
            qse_probe_families=("Q_mu", "P_mu"),
        ),
        pool=pool,
        report_summary=report_summary,
        provenance={
            "backend": backend,
            "backend_record_schema": BACKEND_RECORD_SCHEMA,
            "backend_record_sha256": sha256_jsonable(record),
            "generator": f"src.quantum.chemistry.generate_h2o_linear_fd_fixture:{GENERATOR_VERSION}",
        },
    )
    if exact_policy == "candidate_without_exact":
        validate_production_vibronic_h2o_fixture(
            fixture,
            require_exact_policy=True,
            require_production_validated=False,
        )
    else:
        validate_paper_iv_h2o_linear_fd_evidence_fixture(
            fixture,
            require_exact_state=bool(embed_exact_state),
            require_reference_cutoff=bool(require_reference_cutoff),
            require_cutoff_converged=False,
        )
    return fixture


def reencode_h2o_linear_fd_fixture(
    source_fixture: ProductionVibronicH2OFixture,
    *,
    mode_cutoffs: Sequence[int] | None = None,
    reference_cutoffs: Sequence[int] | None = None,
    dense_full_dim_cap: int = 8192,
    embed_exact_state: bool = True,
    exact_reference_policy: str = "auto_sector",
    operator_cleanup_tol: float = 1.0e-12,
    cutoff_energy_tolerance_hartree: float = H2O_LINEAR_FD_CUTOFF_ENERGY_TOLERANCE_HARTREE,
    cutoff_boundary_weight_tolerance: float = H2O_LINEAR_FD_CUTOFF_BOUNDARY_WEIGHT_TOLERANCE,
) -> ProductionVibronicH2OFixture:
    """Rebuild encoded operators and references from retained validated tensors."""

    validate_production_vibronic_h2o_fixture(
        source_fixture,
        require_exact_policy=False,
        require_production_validated=False,
    )
    exact_policy = str(exact_reference_policy).strip().lower().replace("-", "_")
    if exact_policy not in {"auto_sector", "dense_required", "sparse_sector_eigsh"}:
        raise ValueError(
            "re-encoding requires auto_sector, dense_required, or sparse_sector_eigsh "
            f"exact-reference policy; got {exact_reference_policy!r}."
        )

    active = source_fixture.active_space
    geometry = source_fixture.geometry
    modes = tuple(source_fixture.normal_modes)
    mode_labels = tuple(str(mode.label) for mode in modes)
    work_cutoffs = (
        tuple(int(block.n_ph_max) for block in source_fixture.layout.boson_modes)
        if mode_cutoffs is None
        else tuple(int(value) for value in mode_cutoffs)
    )
    if len(work_cutoffs) != len(modes):
        raise ValueError("mode_cutoffs must contain one entry per retained normal mode.")
    if reference_cutoffs is None:
        source_cutoff = source_fixture.cutoff_diagnostics
        if source_cutoff is not None and source_cutoff.reference_cutoffs is not None:
            reference_cutoffs_t = tuple(int(value) for value in source_cutoff.reference_cutoffs)
        else:
            reference_cutoffs_t = tuple(int(value) + 1 for value in work_cutoffs)
    else:
        reference_cutoffs_t = tuple(int(value) for value in reference_cutoffs)
    if len(reference_cutoffs_t) != len(modes):
        raise ValueError("reference_cutoffs must contain one entry per retained normal mode.")

    layout = _build_register_layout(
        n_spatial_orbitals=int(active.n_spatial_orbitals),
        mode_labels=mode_labels,
        mode_cutoffs=work_cutoffs,
    )
    frequencies = tuple(float(mode.frequency_hartree) for mode in modes)
    encoded, h_vibronic, _q_ops, p_ops, derivative_polys = _build_encoded_operators(
        n_spatial_orbitals=int(active.n_spatial_orbitals),
        num_particles=tuple(int(value) for value in active.num_particles),
        basis=str(geometry.basis),
        charge=int(geometry.charge),
        multiplicity=int(geometry.multiplicity),
        reference=str(geometry.reference),
        active_scalar=float(active.scalar_energy_hartree),
        active_one_body=np.asarray(active.one_body_integrals, dtype=float),
        active_two_body=np.asarray(active.two_body_integrals, dtype=float),
        derivatives=tuple(source_fixture.first_derivatives),
        layout=layout,
        frequencies_hartree=frequencies,
        operator_cleanup_tol=float(operator_cleanup_tol),
    )

    def _solve(
        polynomial: PauliPolynomial,
        solve_layout: RegisterLayout,
        *,
        return_state: bool,
        n_low_energies: int,
    ) -> tuple[float, tuple[float, ...], np.ndarray | None, str, float | None]:
        full_dimension = 1 << int(solve_layout.n_total_qubits)
        use_sparse = bool(
            exact_policy == "sparse_sector_eigsh"
            or (
                exact_policy == "auto_sector"
                and int(full_dimension) > int(dense_full_dim_cap)
            )
        )
        if use_sparse:
            energy, lows, state = _solve_sector_sparse(
                polynomial,
                layout=solve_layout,
                n_spatial_orbitals=int(active.n_spatial_orbitals),
                num_particles=tuple(int(value) for value in active.num_particles),
                return_state=bool(return_state),
                n_low_energies=int(n_low_energies),
            )
            return energy, lows, state, "sparse_sector_eigsh", 1.0e-10
        energy, lows, state = _solve_sector_dense(
            polynomial,
            layout=solve_layout,
            n_spatial_orbitals=int(active.n_spatial_orbitals),
            num_particles=tuple(int(value) for value in active.num_particles),
            dense_full_dim_cap=int(dense_full_dim_cap),
            return_state=bool(return_state),
            n_low_energies=int(n_low_energies),
        )
        return energy, lows, state, "dense_sector_eigh", None

    ground_energy, low_energies, ground_state, exact_method, solver_tolerance = _solve(
        h_vibronic,
        layout,
        return_state=True,
        n_low_energies=4,
    )
    exact_boundary = h2o_linear_fd_boundary_weight_for_state(
        ground_state,
        layout=layout,
        state_source="exact_ground_state",
    )
    exact_reference = ExactReferenceRecord(
        available=True,
        method=exact_method,  # type: ignore[arg-type]
        sector_dimension=fixed_sector_dimension(
            n_spatial_orbitals=int(active.n_spatial_orbitals),
            num_particles=tuple(int(value) for value in active.num_particles),
            mode_cutoffs=work_cutoffs,
        ),
        full_qubit_dimension=1 << int(layout.n_total_qubits),
        ground_energy_hartree=float(ground_energy),
        low_energies_hartree=tuple(float(value) for value in low_energies),
        boundary_weight=exact_boundary,
        ground_state=_exact_state_record(
            ground_state if bool(embed_exact_state) else None,
            total_qubits=int(layout.n_total_qubits),
        ),
        solver_tolerance=solver_tolerance,
    )

    reference_layout = _build_register_layout(
        n_spatial_orbitals=int(active.n_spatial_orbitals),
        mode_labels=mode_labels,
        mode_cutoffs=reference_cutoffs_t,
    )
    _reference_encoded, reference_hamiltonian, *_ = _build_encoded_operators(
        n_spatial_orbitals=int(active.n_spatial_orbitals),
        num_particles=tuple(int(value) for value in active.num_particles),
        basis=str(geometry.basis),
        charge=int(geometry.charge),
        multiplicity=int(geometry.multiplicity),
        reference=str(geometry.reference),
        active_scalar=float(active.scalar_energy_hartree),
        active_one_body=np.asarray(active.one_body_integrals, dtype=float),
        active_two_body=np.asarray(active.two_body_integrals, dtype=float),
        derivatives=tuple(source_fixture.first_derivatives),
        layout=reference_layout,
        frequencies_hartree=frequencies,
        operator_cleanup_tol=float(operator_cleanup_tol),
    )
    reference_energy, _reference_lows, _reference_state_unused, reference_method, _ = _solve(
        reference_hamiltonian,
        reference_layout,
        return_state=False,
        n_low_energies=1,
    )
    delta_energy = float(ground_energy) - float(reference_energy)
    cutoff_assessment = assess_h2o_linear_fd_cutoff_diagnostics(
        delta_energy_hartree=delta_energy,
        work_boundary_weight=exact_boundary,
        energy_tolerance_hartree=float(cutoff_energy_tolerance_hartree),
        boundary_weight_tolerance=float(cutoff_boundary_weight_tolerance),
    )
    cutoff_diagnostics = CutoffDiagnosticsRecord(
        work_cutoffs=work_cutoffs,
        reference_cutoffs=reference_cutoffs_t,
        work_ground_energy_hartree=float(ground_energy),
        reference_ground_energy_hartree=float(reference_energy),
        delta_energy_hartree=float(delta_energy),
        work_boundary_weight=exact_boundary,
        passed=bool(cutoff_assessment["passed"]),
        policy=(
            "sparse_same_model_reference_cutoff_v2"
            if reference_method == "sparse_sector_eigsh"
            else "dense_same_model_reference_cutoff_v2"
        ),
        energy_tolerance_hartree=float(cutoff_assessment["energy_tolerance_hartree"]),
        boundary_weight_tolerance=float(cutoff_assessment["boundary_weight_tolerance"]),
        energy_passed=cutoff_assessment["energy_passed"],
        boundary_passed=cutoff_assessment["boundary_passed"],
    )
    pool = _build_pool_rows(
        n_spatial_orbitals=int(active.n_spatial_orbitals),
        num_particles=tuple(int(value) for value in active.num_particles),
        layout=layout,
        p_ops=p_ops,
        derivative_polys=derivative_polys,
    )
    psi_ref = _reference_state(
        layout,
        n_spatial_orbitals=int(active.n_spatial_orbitals),
        num_particles=tuple(int(value) for value in active.num_particles),
    )
    source_payload = production_vibronic_h2o_fixture_to_jsonable(source_fixture)
    source_hash = sha256_jsonable(source_payload)
    provenance_hashes = dict(source_fixture.manifest.provenance_hashes)
    provenance_hashes["source_fixture_sha256"] = source_hash
    report_summary = dict(source_fixture.report_summary)
    report_summary.update(
        {
            "exact_reference_policy": exact_policy,
            "operator_cleanup_tol": float(operator_cleanup_tol),
            "vibrational_cutoff_converged": bool(cutoff_diagnostics.passed),
            "cutoff_energy_tolerance_hartree": float(cutoff_energy_tolerance_hartree),
            "cutoff_boundary_weight_tolerance": float(cutoff_boundary_weight_tolerance),
            "reencoded_from_retained_tensor_evidence": True,
        }
    )
    rebuilt = replace(
        source_fixture,
        manifest=replace(
            source_fixture.manifest,
            created_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            generator_version=f"{GENERATOR_VERSION}_reencode_v2",
            provenance_hashes=provenance_hashes,
        ),
        layout=layout,
        physical_sector=PhysicalSectorRecord(
            n_alpha=int(active.num_particles[0]),
            n_beta=int(active.num_particles[1]),
            n_ph_max_by_mode=work_cutoffs,
            mode_labels=mode_labels,
        ),
        encoded_operators=encoded,
        reference_state=psi_ref,
        exact_reference=exact_reference,
        cutoff_diagnostics=cutoff_diagnostics,
        evidence_hooks=replace(
            source_fixture.evidence_hooks,
            static_ground_state_ready=True,
            exact_reference_ready=True,
        ),
        pool=pool,
        report_summary=report_summary,
        provenance={
            **dict(source_fixture.provenance),
            "source_fixture_sha256": source_hash,
            "reencoding_generator": (
                "src.quantum.chemistry.generate_h2o_linear_fd_fixture:"
                f"{GENERATOR_VERSION}_reencode_v2"
            ),
        },
    )
    validate_paper_iv_h2o_linear_fd_evidence_fixture(
        rebuilt,
        require_exact_state=bool(embed_exact_state),
        require_reference_cutoff=True,
        require_cutoff_converged=False,
    )
    return rebuilt


def build_h2o_linear_fd_backend_record_with_psi4(
    *,
    initial_geometry_spec: str | None = None,
    initial_units: str = "angstrom",
    optimize_geometry: bool = True,
    method: str = "scf",
    basis: str = "sto-3g",
    charge: int = 0,
    multiplicity: int = 1,
    reference: str = "rhf",
    scf_type: str = "pk",
    active_space: str = "valence_8e_6o",
    q_step_au: float = 0.1,
    q_step_alt_au: float = 0.05,
    memory: str | None = None,
    output_file: str | None = None,
    options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        import psi4  # type: ignore
    except Exception as exc:  # pragma: no cover - optional backend
        raise ImportError("Psi4 is required for --backend psi4 H2O linear-FD generation.") from exc

    if float(q_step_au) <= 0.0 or float(q_step_alt_au) <= 0.0:
        raise ValueError("q_step_au and q_step_alt_au must be positive.")
    reference_key = str(reference).strip().lower()
    if reference_key != "rhf":
        raise ValueError("H2O linear-FD Psi4 backend currently supports reference='rhf' only.")
    if int(multiplicity) != 1:
        raise ValueError("H2O linear-FD Psi4 backend currently supports multiplicity=1 only.")

    if output_file not in {None, ""}:
        psi4.core.set_output_file(str(output_file), False)
    if memory not in {None, ""}:
        psi4.set_memory(str(memory))
    reserved_option_keys = {"basis", "reference", "scf_type", "e_convergence", "d_convergence"}
    user_options = {str(k): v for k, v in dict(options or {}).items()}
    overlap = reserved_option_keys.intersection(user_options)
    if overlap:
        blocked = ", ".join(sorted(overlap))
        raise ValueError(f"options may not override reserved Psi4 keys: {blocked}")
    psi4.set_options(
        {
            "basis": str(basis),
            "reference": str(reference_key),
            "scf_type": str(scf_type),
            "e_convergence": 1.0e-10,
            "d_convergence": 1.0e-10,
            **user_options,
        }
    )

    geometry_spec = str(initial_geometry_spec or _default_h2o_geometry_spec_angstrom()).strip()
    geom_block = _normalized_geometry_block(
        geometry_spec=geometry_spec,
        charge=int(charge),
        multiplicity=int(multiplicity),
        units=str(initial_units),
    )
    molecule = psi4.geometry(geom_block)
    molecule.update_geometry()
    if bool(optimize_geometry):
        psi4.optimize(str(method), molecule=molecule)
        molecule.update_geometry()
    symbols, center_coordinates_bohr, masses_me = _molecule_arrays_from_psi4_molecule(molecule)
    _validate_h2o_symbols(symbols)

    center_gradient_diagnostics = _psi4_gradient_diagnostics(
        psi4,
        method=str(method),
        molecule=molecule,
    )
    hessian_raw = psi4.hessian(str(method), molecule=molecule)
    hessian = _matrix_to_ndarray(hessian_raw)
    mode_rows = _mass_weighted_vibrational_modes_from_hessian(
        hessian,
        masses_me=masses_me,
        symbols=symbols,
        coordinates_bohr=center_coordinates_bohr,
        n_vibrational_modes=3,
    )

    center_geometry_spec_bohr = _geometry_spec_from_bohr(symbols, center_coordinates_bohr)
    center_snapshot = load_restricted_closed_shell_snapshot_from_psi4(
        geometry_spec=center_geometry_spec_bohr,
        basis=str(basis),
        charge=int(charge),
        multiplicity=int(multiplicity),
        units="bohr",
        reference=str(reference_key),
        scf_type=str(scf_type),
        memory=memory,
        output_file=output_file,
        options=user_options,
    )
    active_kind, frozen_core, active_indices, external_indices, active_particles = _select_active_space(
        center_snapshot.problem,
        policy=str(active_space),
    )
    active_payload = {
        "active_space_kind": str(active_kind),
        "frozen_core_indices": [int(v) for v in frozen_core],
        "active_indices_center": [int(v) for v in active_indices],
        "external_indices": [int(v) for v in external_indices],
        "n_spatial_orbitals": int(len(active_indices)),
        "num_particles": [int(active_particles[0]), int(active_particles[1])],
        **_active_tensor_payload_from_problem(
            center_snapshot.problem,
            active_indices=active_indices,
            frozen_core_indices=frozen_core,
        ),
    }

    aligned_tensors: list[dict[str, Any]] = []
    for mode in mode_rows:
        mode_index = int(mode["mode_index"])
        label = str(mode["label"])
        mode_vector = np.asarray(mode["mass_weighted_eigenvector"], dtype=float)
        for step_kind, q_step in (("primary", float(q_step_au)), ("alt", float(q_step_alt_au))):
            for sign in (1, -1):
                displacement_id = f"{step_kind}_{label}_{'plus' if sign > 0 else 'minus'}"
                q_displacement = float(sign) * float(q_step)
                displaced_coordinates = _displace_coordinates_along_mass_weighted_mode(
                    center_coordinates_bohr,
                    mode_vector=mode_vector,
                    masses_me=masses_me,
                    q_displacement_au=q_displacement,
                )
                displacement_delta = np.asarray(displaced_coordinates, dtype=float) - np.asarray(
                    center_coordinates_bohr,
                    dtype=float,
                )
                displaced_geometry_spec = _geometry_spec_from_bohr(symbols, displaced_coordinates)
                snapshot = load_restricted_closed_shell_snapshot_from_psi4(
                    geometry_spec=displaced_geometry_spec,
                    basis=str(basis),
                    charge=int(charge),
                    multiplicity=int(multiplicity),
                    units="bohr",
                    reference=str(reference_key),
                    scf_type=str(scf_type),
                    memory=memory,
                    output_file=output_file,
                    options=user_options,
                )
                alignment_id = f"align_{displacement_id}"
                aligned_problem, alignment_payload = _align_snapshot_to_center_with_diagnostics(
                    snapshot,
                    center_snapshot=center_snapshot,
                    active_indices=active_indices,
                    displacement_id=displacement_id,
                    alignment_id=alignment_id,
                )
                row: dict[str, Any] = {
                    "aligned_tensor_id": f"aligned_{displacement_id}",
                    "source_snapshot_id": f"snap_{displacement_id}",
                    "displacement_id": displacement_id,
                    "mode_index": int(mode_index),
                    "mode_label": str(label),
                    "step_kind": str(step_kind),
                    "sign": int(sign),
                    "q_displacement_au": float(q_step),
                    "signed_q_displacement_au": float(q_displacement),
                    "mass_weighted_displacement_norm_au": _mass_weighted_displacement_norm(
                        displacement_delta,
                        masses_me=masses_me,
                    ),
                    "max_cartesian_displacement_bohr": float(np.max(np.abs(displacement_delta))),
                    "geometry_id": f"h2o_{displacement_id}",
                    "coordinates_bohr": np.asarray(displaced_coordinates, dtype=float).tolist(),
                    "alignment": alignment_payload,
                }
                row.update(
                    _active_tensor_payload_from_problem(
                        aligned_problem,
                        active_indices=active_indices,
                        frozen_core_indices=frozen_core,
                    )
                )
                aligned_tensors.append(row)

    normal_modes = [
        {
            "mode_index": int(mode["mode_index"]),
            "label": str(mode["label"]),
            "frequency_hartree": float(mode["frequency_hartree"]),
            "frequency_cm1": float(mode["frequency_cm1"]),
            "mass_weighted_eigenvector": np.asarray(mode["mass_weighted_eigenvector"], dtype=float).tolist(),
            "q_step_au": float(q_step_au),
            "q_step_alt_au": float(q_step_alt_au),
            "raw_hessian_eigenvalue": float(mode["raw_hessian_eigenvalue"]),
            "raw_hessian_eigenvalue_index": int(mode.get("raw_hessian_eigenvalue_index", mode["mode_index"])),
            "positive_hessian_eigenvalue_count": int(mode.get("positive_hessian_eigenvalue_count", 3)),
            "label_source": str(mode.get("label_source", "ascending_frequency_h2o_bend_symmetric_antisymmetric_v1")),
            "mode_character": dict(mode.get("mode_character", {})),
            "trans_rot_overlap": None if mode.get("trans_rot_overlap") is None else float(mode["trans_rot_overlap"]),
            "primary_mass_weighted_displacement_norm_au": float(q_step_au),
            "alt_mass_weighted_displacement_norm_au": float(q_step_alt_au),
            "primary_max_cartesian_displacement_bohr": float(
                np.max(
                    np.abs(
                        float(q_step_au)
                        * np.asarray(mode["mass_weighted_eigenvector"], dtype=float)
                        / np.sqrt(np.asarray(masses_me, dtype=float))[:, None]
                    )
                )
            ),
            "alt_max_cartesian_displacement_bohr": float(
                np.max(
                    np.abs(
                        float(q_step_alt_au)
                        * np.asarray(mode["mass_weighted_eigenvector"], dtype=float)
                        / np.sqrt(np.asarray(masses_me, dtype=float))[:, None]
                    )
                )
            ),
        }
        for mode in mode_rows
    ]
    report_summary: dict[str, Any] = {
        "psi4_backend_record_ready": True,
        "normal_mode_label_policy": str(
            normal_modes[0].get("label_source", "ascending_frequency_h2o_bend_symmetric_antisymmetric_v1")
            if normal_modes
            else "unknown"
        ),
        "active_space_policy": str(active_space),
        "active_space_kind": str(active_kind),
        "center_gradient_diagnostics": center_gradient_diagnostics,
        "normal_mode_diagnostics": [
            {
                "mode_index": int(mode["mode_index"]),
                "label": str(mode["label"]),
                "frequency_hartree": float(mode["frequency_hartree"]),
                "frequency_cm1": float(mode["frequency_cm1"]),
                "raw_hessian_eigenvalue": float(mode["raw_hessian_eigenvalue"]),
                "label_source": str(mode.get("label_source", "unknown")),
                "mode_character": dict(mode.get("mode_character", {})),
                "trans_rot_overlap": mode.get("trans_rot_overlap"),
                "primary_max_cartesian_displacement_bohr": mode["primary_max_cartesian_displacement_bohr"],
                "alt_max_cartesian_displacement_bohr": mode["alt_max_cartesian_displacement_bohr"],
            }
            for mode in normal_modes
        ],
    }
    if str(active_kind) != "cas_8e_6o_valence":
        report_summary["paper_iv_active_space_variant"] = str(active_kind)

    return {
        "schema": BACKEND_RECORD_SCHEMA,
        "backend": {
            "name": "psi4",
            "method": str(method),
            "basis": str(basis),
            "reference": str(reference_key),
            "scf_type": str(scf_type),
            "version": str(getattr(psi4, "__version__", "unknown")),
        },
        "system": {
            "molecule": "H2O",
            "charge": int(charge),
            "multiplicity": int(multiplicity),
            "optimized": bool(optimize_geometry),
        },
        "center_snapshot_id": "snap_center",
        "geometry": {
            "geometry_id": "h2o_center",
            "symbols": [str(v) for v in symbols],
            "coordinates_bohr": np.asarray(center_coordinates_bohr, dtype=float).tolist(),
            "masses_me": np.asarray(masses_me, dtype=float).tolist(),
            "gradient_diagnostics": center_gradient_diagnostics,
            "provenance": {
                "coordinate_units": "bohr",
                "mass_units": "electron_mass",
                "gradient_units": "hartree_per_bohr",
            },
        },
        "active_space": active_payload,
        "normal_modes": normal_modes,
        "aligned_tensors": aligned_tensors,
        "report_summary": report_summary,
        "provenance": {
            "generator": f"src.quantum.chemistry.generate_h2o_linear_fd_fixture:{GENERATOR_VERSION}",
            "initial_units": str(initial_units),
            "initial_geometry_spec": geometry_spec,
            "q_step_au": float(q_step_au),
            "q_step_alt_au": float(q_step_alt_au),
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate strict molecular_vibronic_h2o_linear_fd production fixture JSON."
    )
    parser.add_argument("--backend", choices=("record", "psi4"), default="record")
    parser.add_argument("--input-record-json", type=Path, default=None)
    parser.add_argument("--output-record-json", type=Path, default=None)
    parser.add_argument("--record-only", action="store_true")
    parser.add_argument("--output-fixture-json", type=Path, default=None)
    parser.add_argument("--mode-cutoffs", type=str, default="2,2,2")
    parser.add_argument("--reference-mode-cutoffs", type=str, default=None)
    parser.add_argument("--dense-full-dim-cap", type=int, default=8192)
    parser.add_argument("--embed-exact-state", action="store_true")
    parser.add_argument("--no-reference-cutoff", action="store_true")
    parser.add_argument("--max-derivative-drift", type=float, default=1.0e-6)
    parser.add_argument(
        "--derivative-validation-tier",
        choices=("smoke", "production", "tight"),
        default="production",
    )
    parser.add_argument(
        "--exact-reference-policy",
        choices=("auto_sector", "dense_required", "candidate_without_exact", "sparse_sector_eigsh"),
        default="auto_sector",
    )
    parser.add_argument("--operator-cleanup-tol", type=float, default=1.0e-12)
    parser.add_argument(
        "--cutoff-energy-tolerance-hartree",
        type=float,
        default=H2O_LINEAR_FD_CUTOFF_ENERGY_TOLERANCE_HARTREE,
    )
    parser.add_argument(
        "--cutoff-boundary-weight-tolerance",
        type=float,
        default=H2O_LINEAR_FD_CUTOFF_BOUNDARY_WEIGHT_TOLERANCE,
    )
    parser.add_argument("--psi4-initial-geometry-file", type=Path, default=None)
    parser.add_argument("--psi4-units", choices=("angstrom", "bohr"), default="angstrom")
    parser.add_argument("--psi4-no-optimize", action="store_true")
    parser.add_argument("--psi4-method", type=str, default="scf")
    parser.add_argument("--psi4-basis", type=str, default="sto-3g")
    parser.add_argument("--psi4-reference", type=str, default="rhf")
    parser.add_argument("--psi4-scf-type", type=str, default="pk")
    parser.add_argument(
        "--psi4-active-space",
        choices=("valence_8e_6o", "frontier_2e_2o", "full"),
        default="valence_8e_6o",
    )
    parser.add_argument("--psi4-q-step", type=float, default=0.1)
    parser.add_argument("--psi4-q-step-alt", type=float, default=0.05)
    parser.add_argument("--psi4-memory", type=str, default=None)
    parser.add_argument("--psi4-output-file", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    mode_cutoffs = _parse_int_tuple(str(args.mode_cutoffs), label="mode-cutoffs")
    assert mode_cutoffs is not None
    reference_cutoffs = _parse_int_tuple(args.reference_mode_cutoffs, label="reference-mode-cutoffs")
    if bool(args.no_reference_cutoff):
        reference_cutoffs = None
    if str(args.backend) == "record":
        if args.input_record_json is None:
            raise ValueError("--input-record-json is required when --backend record.")
        record = json.loads(Path(args.input_record_json).read_text(encoding="utf-8"))
    else:
        record = build_h2o_linear_fd_backend_record_with_psi4(
            initial_geometry_spec=_read_text_or_default(
                args.psi4_initial_geometry_file,
                default=_default_h2o_geometry_spec_angstrom(),
            ),
            initial_units=str(args.psi4_units),
            optimize_geometry=not bool(args.psi4_no_optimize),
            method=str(args.psi4_method),
            basis=str(args.psi4_basis),
            reference=str(args.psi4_reference),
            scf_type=str(args.psi4_scf_type),
            active_space=str(args.psi4_active_space),
            q_step_au=float(args.psi4_q_step),
            q_step_alt_au=float(args.psi4_q_step_alt),
            memory=args.psi4_memory,
            output_file=args.psi4_output_file,
        )
    if args.output_record_json is not None:
        _write_json(Path(args.output_record_json), record, force=bool(args.force))
        print(f"Wrote H2O linear-FD backend record: {Path(args.output_record_json)}")
        print(f"Backend record SHA256: {sha256_jsonable(record)}")
    if bool(args.record_only):
        if args.output_record_json is None:
            raise ValueError("--record-only requires --output-record-json.")
        return
    if args.output_fixture_json is None:
        raise ValueError("--output-fixture-json is required unless --record-only is set.")
    fixture = build_h2o_linear_fd_fixture_from_record(
        record,
        mode_cutoffs=mode_cutoffs,
        reference_cutoffs=reference_cutoffs,
        dense_full_dim_cap=int(args.dense_full_dim_cap),
        embed_exact_state=bool(args.embed_exact_state),
        require_reference_cutoff=not bool(args.no_reference_cutoff),
        max_derivative_drift=float(args.max_derivative_drift),
        derivative_validation_tier=str(args.derivative_validation_tier),
        exact_reference_policy=str(args.exact_reference_policy),
        operator_cleanup_tol=float(args.operator_cleanup_tol),
        cutoff_energy_tolerance_hartree=float(args.cutoff_energy_tolerance_hartree),
        cutoff_boundary_weight_tolerance=float(args.cutoff_boundary_weight_tolerance),
    )
    payload = production_vibronic_h2o_fixture_to_jsonable(fixture)
    _write_json(Path(args.output_fixture_json), payload, force=bool(args.force))
    print(f"Wrote H2O linear-FD fixture: {Path(args.output_fixture_json)}")
    print(f"Fixture SHA256: {sha256_jsonable(payload)}")


if __name__ == "__main__":
    main()
