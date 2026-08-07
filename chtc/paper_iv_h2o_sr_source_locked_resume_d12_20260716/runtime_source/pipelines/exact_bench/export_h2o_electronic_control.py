"""Export the electronic-only H2O control from a production LVC fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.quantum.chemistry.molecular_hamiltonian import (
    build_restricted_closed_shell_molecular_hamiltonian,
)
from src.quantum.chemistry.generate_h2o_linear_fd_fixture import (
    build_fixed_particle_sector_sparse_matrix,
)
from src.quantum.chemistry.molecular_uccsd import build_molecular_uccsd_pool
from src.quantum.chemistry.psi4_adapter import RestrictedClosedShellMolecularProblem
from src.quantum.chemistry.vibronic_h2o_linear_fd import (
    RegisterLayout,
    load_cached_production_vibronic_h2o_linear_fd_fixture,
)


ELECTRONIC_CONTROL_SCHEMA = "paper_iv_h2o_electronic_control_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _closed_shell_energy(
    scalar: float,
    one_body: np.ndarray,
    two_body: np.ndarray,
    *,
    n_occupied: int,
) -> float:
    occupied = range(int(n_occupied))
    h = np.asarray(one_body, dtype=float)
    g = np.asarray(two_body, dtype=float)
    return float(
        float(scalar)
        + 2.0 * sum(float(h[i, i]) for i in occupied)
        + sum(
            2.0 * float(g[i, i, j, j]) - float(g[i, j, j, i])
            for i in occupied
            for j in occupied
        )
    )


def _geometry_spec(symbols: Sequence[str], coordinates_bohr: np.ndarray) -> str:
    rows = [
        f"{symbol} {float(x):.16g} {float(y):.16g} {float(z):.16g}"
        for symbol, (x, y, z) in zip(symbols, np.asarray(coordinates_bohr, dtype=float))
    ]
    return "\n".join([*rows, "units bohr"])


def _polynomial_coefficients(polynomial: Any) -> dict[str, complex]:
    coefficients: dict[str, complex] = {}
    for term in polynomial.return_polynomial():
        word = str(term.pw2strng()).lower()
        coefficients[word] = coefficients.get(word, 0.0 + 0.0j) + complex(
            term.p_coeff
        )
    return {word: value for word, value in coefficients.items() if value != 0.0}


def _polynomial_delta(left: Any, right: Any) -> dict[str, float | int]:
    left_coefficients = _polynomial_coefficients(left)
    right_coefficients = _polynomial_coefficients(right)
    words = set(left_coefficients) | set(right_coefficients)
    deltas = [
        abs(left_coefficients.get(word, 0.0) - right_coefficients.get(word, 0.0))
        for word in words
    ]
    return {
        "left_term_count": int(len(left_coefficients)),
        "right_term_count": int(len(right_coefficients)),
        "coefficient_delta_max_abs": float(max(deltas, default=0.0)),
        "coefficient_delta_l1": float(sum(deltas)),
    }


def _pool_polynomial_signature(term: Any, *, trim_left_identities: int = 0) -> tuple[Any, ...]:
    rows: list[tuple[str, float, float]] = []
    for pauli in term.polynomial.return_polynomial():
        word = str(pauli.pw2strng()).lower()
        if int(trim_left_identities) > 0:
            prefix = word[: int(trim_left_identities)]
            if prefix != "e" * int(trim_left_identities):
                raise ValueError(
                    f"Electronic source generator {term.label!r} acts on a boson register."
                )
            word = word[int(trim_left_identities) :]
        coefficient = complex(pauli.p_coeff)
        rows.append((word, float(coefficient.real), float(coefficient.imag)))
    return tuple(sorted(rows))


def _pool_signature_sha256(signatures: Sequence[tuple[Any, ...]]) -> str:
    payload = json.dumps(signatures, separators=(",", ":"), sort_keys=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _exact_ground_energy_sparse_sector(
    hamiltonian: Any,
    *,
    n_spatial_orbitals: int,
    num_particles: tuple[int, int],
) -> tuple[float, int]:
    n_fermion_qubits = 2 * int(n_spatial_orbitals)
    layout = RegisterLayout(
        n_fermion_qubits=n_fermion_qubits,
        fermion_qubits=tuple(range(n_fermion_qubits)),
        boson_modes=(),
        spin_orbital_ordering="blocked",
    )
    matrix, basis = build_fixed_particle_sector_sparse_matrix(
        hamiltonian,
        layout=layout,
        n_spatial_orbitals=int(n_spatial_orbitals),
        num_particles=tuple(int(value) for value in num_particles),
        coeff_tol=0.0,
    )
    eigenvalues = np.linalg.eigvalsh(np.asarray(matrix.toarray(), dtype=complex))
    return float(np.min(np.real(eigenvalues))), int(len(basis))


def export_h2o_electronic_control(
    fixture_path: str | Path,
) -> dict[str, Any]:
    source_path = Path(fixture_path)
    cached = load_cached_production_vibronic_h2o_linear_fd_fixture(source_path)
    fixture = cached.fixture
    source_model = cached.model
    active = fixture.active_space
    num_particles = tuple(int(value) for value in active.num_particles)
    n_spatial = int(active.n_spatial_orbitals)
    hf_energy = _closed_shell_energy(
        float(active.scalar_energy_hartree),
        np.asarray(active.one_body_integrals, dtype=float),
        np.asarray(active.two_body_integrals, dtype=float),
        n_occupied=int(num_particles[0]),
    )
    problem = RestrictedClosedShellMolecularProblem(
        geometry_spec=_geometry_spec(
            tuple(str(symbol) for symbol in fixture.geometry.symbols),
            np.asarray(fixture.geometry.coordinates_bohr, dtype=float),
        ),
        basis=str(fixture.geometry.basis),
        charge=int(fixture.geometry.charge),
        multiplicity=int(fixture.geometry.multiplicity),
        reference=str(fixture.geometry.reference).lower(),
        n_spatial_orbitals=n_spatial,
        n_alpha=int(num_particles[0]),
        n_beta=int(num_particles[1]),
        hf_energy=float(hf_energy),
        nuclear_repulsion_energy=float(active.scalar_energy_hartree),
        one_body_integrals_mo=np.asarray(active.one_body_integrals, dtype=float),
        two_body_integrals_mo=np.asarray(active.two_body_integrals, dtype=float),
    )
    rebuilt_hamiltonian = build_restricted_closed_shell_molecular_hamiltonian(
        problem,
        ordering="blocked",
    )
    hamiltonian_parity = _polynomial_delta(
        source_model.h_electronic,
        rebuilt_hamiltonian,
    )
    if float(hamiltonian_parity["coefficient_delta_max_abs"]) > 1.0e-12:
        raise ValueError(
            "Electronic-only Hamiltonian does not reproduce the source fixture "
            f"electronic block: {hamiltonian_parity!r}."
        )

    control_pool = build_molecular_uccsd_pool(
        n_spatial_orbitals=n_spatial,
        num_particles=num_particles,
        ordering="blocked",
    )
    source_electronic_pool = tuple(
        term for term in source_model.pool if str(term.label).startswith("el::")
    )
    boson_qubits = int(source_model.n_boson_qubits)
    source_signatures = sorted(
        _pool_polynomial_signature(term, trim_left_identities=boson_qubits)
        for term in source_electronic_pool
    )
    control_signatures = sorted(_pool_polynomial_signature(term) for term in control_pool)
    if source_signatures != control_signatures:
        raise ValueError(
            "Electronic-only UCCSD pool does not reproduce the source fixture "
            "electronic generator set."
        )

    exact_energy, sector_dimension = _exact_ground_energy_sparse_sector(
        rebuilt_hamiltonian,
        n_spatial_orbitals=n_spatial,
        num_particles=num_particles,
    )
    expected_sector_dimension = int(
        math.comb(n_spatial, num_particles[0])
        * math.comb(n_spatial, num_particles[1])
    )
    if sector_dimension != expected_sector_dimension:
        raise ValueError(
            "Electronic control sector dimension mismatch: "
            f"built {sector_dimension}, expected {expected_sector_dimension}."
        )
    source_hash = _sha256_file(source_path)
    return {
        "schema": ELECTRONIC_CONTROL_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_role": "paper_iv_h2o_electronic_only_control",
        "problem": problem.to_jsonable(),
        "match": {
            "problem": "molecular_restricted_closed_shell",
            "L": n_spatial,
            "n_ph_max": 0,
            "boson_encoding": "binary",
            "ordering": "blocked",
            "boundary": "open",
            "include_zero_point": False,
        },
        "exact_energy": float(exact_energy),
        "exact_reference": {
            "ground_energy_hartree": float(exact_energy),
            "method": "exact_diagonalization_fixed_Nalpha_Nbeta_sector",
            "sector_dimension": sector_dimension,
            "num_particles": list(num_particles),
        },
        "hamiltonian_contract": {
            "n_fermion_qubits": int(2 * n_spatial),
            "n_boson_qubits": 0,
            "scalar_energy_semantics": "nuclear_repulsion_plus_frozen_core",
            "hamiltonian_pauli_term_count": int(
                len(_polynomial_coefficients(rebuilt_hamiltonian))
            ),
            "source_electronic_block_parity": hamiltonian_parity,
        },
        "pool_contract": {
            "pool_key": "uccsd",
            "generator_count": int(len(control_pool)),
            "single_count": int(
                sum(str(term.label).startswith("uccsd_sing(") for term in control_pool)
            ),
            "double_count": int(
                sum(str(term.label).startswith("uccsd_dbl(") for term in control_pool)
            ),
            "matches_source_fixture_electronic_pool": True,
            "ordered_polynomial_signature_sha256": _pool_signature_sha256(
                [_pool_polynomial_signature(term) for term in control_pool]
            ),
        },
        "source": {
            "fixture_path": str(source_path),
            "fixture_sha256": source_hash,
            "fixture_schema": str(fixture.manifest.schema),
            "fixture_generator_version": str(fixture.manifest.generator_version),
            "backend": dict(fixture.provenance.get("backend", {})),
            "active_space_kind": str(active.active_space_kind),
            "active_indices_center": list(active.active_indices_center),
            "frozen_core_indices": list(active.frozen_core_indices),
        },
    }


def write_h2o_electronic_control(
    fixture_path: str | Path,
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    payload = export_h2o_electronic_control(fixture_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    output = write_h2o_electronic_control(args.fixture, args.output)
    payload: Mapping[str, Any] = json.loads(output.read_text(encoding="utf-8"))
    print(f"wrote {output}")
    print(f"exact_energy={float(payload['exact_energy']):.15f}")
    print(f"pool_generators={int(payload['pool_contract']['generator_count'])}")


if __name__ == "__main__":
    main()
