#!/usr/bin/env python3
"""Reproduce the electronic and harmonic H2O values in an external benchmark.

This audit compares two RHF/STO-3G geometries with the same PySCF backend:
the optimized geometry retained by the Paper-IV fixture and a user-specified
fixed geometry.  It is diagnostic only; it does not alter the production
fixture or validate electron-phonon derivative tensors.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np


AUDIT_SCHEMA = "h2o_cross_code_reference_audit_v1"
AMU_TO_ELECTRON_MASS = 1822.888486209

EXTERNAL_BENCHMARK = {
    "source": "colleague_supplied_cross_code_slide_20260714",
    "settings_disclosed_in_slide": False,
    "rows": {
        "pyscf_qcdynamics": {
            "scf_energy_hartree": -74.9630231385,
            "frequencies_cm1": [2043.1, 4488.1, 4790.3],
        },
        "qchem_ab_initio": {
            "scf_energy_hartree": -74.96302314,
            "frequencies_cm1": [2043.29, 4488.45, 4790.73],
        },
        "orca_ab_initio": {
            "scf_energy_hartree": -74.96302313902787,
            "frequencies_cm1": [2043.08, 4488.04, 4790.28],
        },
    },
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array(payload: Any) -> np.ndarray:
    if isinstance(payload, dict) and "data" in payload:
        data = np.asarray(payload["data"], dtype=float)
        shape = tuple(int(value) for value in payload.get("shape", data.shape))
        return data.reshape(shape)
    return np.asarray(payload, dtype=float)


def _fixed_h2o_geometry_angstrom(
    *,
    bond_length_angstrom: float,
    bond_angle_degrees: float,
) -> tuple[tuple[str, tuple[float, float, float]], ...]:
    half_angle = math.radians(float(bond_angle_degrees) / 2.0)
    y = float(bond_length_angstrom) * math.sin(half_angle)
    z = float(bond_length_angstrom) * math.cos(half_angle)
    return (
        ("O", (0.0, 0.0, 0.0)),
        ("H", (0.0, y, z)),
        ("H", (0.0, -y, z)),
    )


def _closed_shell_energy_from_active_tensors(active_space: dict[str, Any]) -> float:
    scalar = float(active_space["scalar_energy_hartree"])
    one_body = _array(active_space["one_body_integrals"])
    two_body = _array(active_space["two_body_integrals"])
    n_occupied = int(active_space["num_particles"][0])
    occupied = range(n_occupied)
    return float(
        scalar
        + 2.0 * sum(float(one_body[i, i]) for i in occupied)
        + sum(
            2.0 * float(two_body[i, i, j, j])
            - float(two_body[i, j, j, i])
            for i in occupied
            for j in occupied
        )
    )


def _run_pyscf_rhf_hessian(
    *,
    atoms: Sequence[tuple[str, Sequence[float]]],
    units: str,
    basis: str,
    masses_amu: np.ndarray,
    convergence_tolerance: float,
) -> dict[str, Any]:
    try:
        import pyscf
        from pyscf import gto, scf
        from pyscf.hessian import thermo
    except Exception as exc:  # pragma: no cover - optional audit dependency
        raise ImportError(
            "PySCF is required for this audit. Install it in an isolated environment."
        ) from exc

    molecule = gto.M(
        atom=[(str(symbol), tuple(float(value) for value in xyz)) for symbol, xyz in atoms],
        unit=str(units),
        basis=str(basis),
        charge=0,
        spin=0,
        symmetry=False,
        verbose=0,
    )
    mean_field = scf.RHF(molecule)
    mean_field.conv_tol = float(convergence_tolerance)
    energy = float(mean_field.kernel())
    if not mean_field.converged:
        raise RuntimeError("PySCF RHF did not converge.")
    hessian = np.asarray(mean_field.Hessian().kernel(), dtype=float)
    harmonic = thermo.harmonic_analysis(
        molecule,
        hessian,
        exclude_trans=True,
        exclude_rot=True,
        mass=np.asarray(masses_amu, dtype=float),
    )
    frequencies = np.real_if_close(harmonic["freq_wavenumber"])
    if np.iscomplexobj(frequencies):
        raise ValueError("The projected H2O Hessian contains imaginary frequencies.")
    return {
        "backend": "pyscf",
        "backend_version": str(pyscf.__version__),
        "basis": str(basis),
        "reference": "RHF",
        "scf_converged": bool(mean_field.converged),
        "scf_convergence_tolerance": float(convergence_tolerance),
        "scf_energy_hartree": energy,
        "frequencies_cm1": [float(value) for value in frequencies],
        "mass_convention": "fixture_isotopic_masses",
        "masses_amu": [float(value) for value in masses_amu],
        "frequency_analysis": "projected_translational_rotational_harmonic_hessian",
    }


def _comparison(
    observed: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    observed_frequencies = np.asarray(observed["frequencies_cm1"], dtype=float)
    reference_frequencies = np.asarray(reference["frequencies_cm1"], dtype=float)
    frequency_delta = observed_frequencies - reference_frequencies
    energy_delta = float(observed["scf_energy_hartree"]) - float(
        reference["scf_energy_hartree"]
    )
    return {
        "energy_delta_hartree": energy_delta,
        "energy_abs_delta_hartree": abs(energy_delta),
        "frequency_deltas_cm1": [float(value) for value in frequency_delta],
        "frequency_max_abs_delta_cm1": float(np.max(np.abs(frequency_delta))),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-json", type=Path, required=True)
    parser.add_argument("--benchmark-image", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--external-bond-length-angstrom", type=float, default=0.9578)
    parser.add_argument("--external-bond-angle-degrees", type=float, default=104.49)
    parser.add_argument("--scf-convergence-tolerance", type=float, default=1.0e-12)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    fixture_path = Path(args.fixture_json).resolve()
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    geometry = fixture["geometry"]
    symbols = tuple(str(value) for value in geometry["symbols"])
    coordinates_bohr = _array(geometry["coordinates_bohr"])
    masses_me = _array(geometry["masses_me"])
    masses_amu = masses_me / AMU_TO_ELECTRON_MASS
    basis = str(geometry["basis"])

    fixture_result = _run_pyscf_rhf_hessian(
        atoms=tuple(zip(symbols, coordinates_bohr)),
        units="Bohr",
        basis=basis,
        masses_amu=masses_amu,
        convergence_tolerance=float(args.scf_convergence_tolerance),
    )
    retained_reference = {
        "scf_energy_hartree": _closed_shell_energy_from_active_tensors(
            fixture["active_space"]
        ),
        "frequencies_cm1": [
            float(mode["frequency_cm1"]) for mode in fixture["normal_modes"]
        ],
    }

    external_geometry = _fixed_h2o_geometry_angstrom(
        bond_length_angstrom=float(args.external_bond_length_angstrom),
        bond_angle_degrees=float(args.external_bond_angle_degrees),
    )
    external_result = _run_pyscf_rhf_hessian(
        atoms=external_geometry,
        units="Angstrom",
        basis=basis,
        masses_amu=masses_amu,
        convergence_tolerance=float(args.scf_convergence_tolerance),
    )
    external_pyscf = EXTERNAL_BENCHMARK["rows"]["pyscf_qcdynamics"]

    image_path = None if args.benchmark_image is None else Path(args.benchmark_image).resolve()
    payload = {
        "schema": AUDIT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "scope": (
            "RHF/STO-3G energy and projected harmonic frequencies only; "
            "electron-phonon derivative tensors are not compared"
        ),
        "fixture": {
            "path": str(fixture_path),
            "sha256": _sha256_file(fixture_path),
            "optimized": bool(geometry.get("optimized", False)),
            "coordinates_bohr": coordinates_bohr.tolist(),
            "retained_reference": retained_reference,
            "pyscf_reproduction": fixture_result,
            "comparison": _comparison(fixture_result, retained_reference),
        },
        "external_benchmark": {
            **EXTERNAL_BENCHMARK,
            "image_path": None if image_path is None else str(image_path),
            "image_sha256": (
                None if image_path is None else _sha256_file(image_path)
            ),
        },
        "matched_fixed_geometry_test": {
            "bond_length_angstrom": float(args.external_bond_length_angstrom),
            "bond_angle_degrees": float(args.external_bond_angle_degrees),
            "geometry_angstrom": [
                {"symbol": symbol, "xyz": [float(value) for value in xyz]}
                for symbol, xyz in external_geometry
            ],
            "pyscf_reproduction": external_result,
            "comparison_to_external_pyscf_row": _comparison(
                external_result,
                external_pyscf,
            ),
        },
        "diagnosis": {
            "primary_difference": "molecular_geometry",
            "production_fixture_geometry_role": "RHF/STO-3G optimized geometry",
            "external_recovery_geometry_role": "fixed near-experimental geometry",
            "production_fixture_reproduced_by_independent_backend": bool(
                _comparison(fixture_result, retained_reference)[
                    "energy_abs_delta_hartree"
                ]
                <= 1.0e-10
                and _comparison(fixture_result, retained_reference)[
                    "frequency_max_abs_delta_cm1"
                ]
                <= 1.0e-3
            ),
            "external_slide_recovered_under_fixed_geometry": bool(
                _comparison(external_result, external_pyscf)[
                    "energy_abs_delta_hartree"
                ]
                <= 1.0e-6
                and _comparison(external_result, external_pyscf)[
                    "frequency_max_abs_delta_cm1"
                ]
                <= 1.0
            ),
            "remaining_validation": (
                "Match the colleague's exact geometry, masses, basis metadata, and "
                "frequency conventions, then compare scalar, one-body, and two-body "
                "normal-coordinate derivative tensors after orbital-frame alignment."
            ),
        },
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fixture_delta = payload["fixture"]["comparison"]
    external_delta = payload["matched_fixed_geometry_test"][
        "comparison_to_external_pyscf_row"
    ]
    print(f"Wrote H2O cross-code audit: {output_path}")
    print(
        "Optimized fixture PySCF reproduction: "
        f"|dE|={fixture_delta['energy_abs_delta_hartree']:.3e} Ha, "
        f"max|dnu|={fixture_delta['frequency_max_abs_delta_cm1']:.3e} cm^-1"
    )
    print(
        "Fixed-geometry external recovery: "
        f"|dE|={external_delta['energy_abs_delta_hartree']:.3e} Ha, "
        f"max|dnu|={external_delta['frequency_max_abs_delta_cm1']:.3e} cm^-1"
    )


if __name__ == "__main__":
    main()
