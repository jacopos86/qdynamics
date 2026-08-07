from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Prototype Psi4-backed closed-shell molecular ADAPT-VQE runner. "
            "Electronic-only, chemistry-folder-local, using the chemistry-local ADAPT core."
        )
    )
    p.add_argument("--geometry", type=str, default=None, help="Geometry spec. Use ';' to separate atoms on the CLI.")
    p.add_argument("--geometry-file", type=str, default=None, help="Path to a file containing the Psi4 geometry body.")
    p.add_argument("--basis", type=str, default="sto-3g")
    p.add_argument("--charge", type=int, default=0)
    p.add_argument("--multiplicity", type=int, default=1)
    p.add_argument("--units", choices=["angstrom", "bohr"], default="angstrom")
    p.add_argument("--psi4-memory", type=str, default=None)
    p.add_argument("--psi4-output", type=str, default=None)
    p.add_argument("--adapt-max-depth", type=int, default=8)
    p.add_argument("--adapt-eps-grad", type=float, default=1e-8)
    p.add_argument("--adapt-eps-energy", type=float, default=1e-10)
    p.add_argument("--adapt-maxiter", type=int, default=400)
    p.add_argument("--adapt-inner-optimizer", choices=["COBYLA", "POWELL", "SPSA"], default="COBYLA")
    p.add_argument("--adapt-finite-angle", type=float, default=0.05)
    p.add_argument("--adapt-finite-angle-min-improvement", type=float, default=0.0)
    p.add_argument("--exact-max-qubits", type=int, default=12)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output-json", type=str, default=None)
    return p.parse_args(argv)


def _resolve_geometry_spec(args: argparse.Namespace) -> str:
    if args.geometry not in {None, ""} and args.geometry_file not in {None, ""}:
        raise SystemExit("Provide only one of --geometry or --geometry-file.")
    if args.geometry_file not in {None, ""}:
        return Path(str(args.geometry_file)).read_text(encoding="utf-8").strip()
    if args.geometry in {None, ""}:
        raise SystemExit("You must provide either --geometry or --geometry-file.")
    return str(args.geometry).replace(";", "\n").strip()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    try:
        from src.quantum.chemistry.molecular_adapt_core import run_local_molecular_adapt_vqe
        from src.quantum.chemistry.molecular_hamiltonian import build_restricted_closed_shell_molecular_hamiltonian
        from src.quantum.chemistry.psi4_adapter import load_restricted_closed_shell_problem_from_psi4
        from src.quantum.vqe_latex_python_pairs import exact_ground_energy_sector
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc

    geometry_spec = _resolve_geometry_spec(args)

    try:
        problem = load_restricted_closed_shell_problem_from_psi4(
            geometry_spec=geometry_spec,
            basis=str(args.basis),
            charge=int(args.charge),
            multiplicity=int(args.multiplicity),
            units=str(args.units),
            memory=args.psi4_memory,
            output_file=args.psi4_output,
        )
    except ImportError as exc:
        raise SystemExit("Psi4 is not installed in this environment. Install psi4, then rerun.") from exc

    h_poly = build_restricted_closed_shell_molecular_hamiltonian(problem, ordering="blocked")
    n_qubits = int(problem.n_spin_orbitals)
    exact_max_qubits = int(args.exact_max_qubits)
    if exact_max_qubits < 0:
        raise SystemExit("--exact-max-qubits must be >= 0.")

    exact_energy: float | None = None
    exact_reference_mode = "skipped_qubit_cutoff"
    if n_qubits <= exact_max_qubits:
        exact_energy = float(
            exact_ground_energy_sector(
                h_poly,
                num_sites=int(problem.n_spatial_orbitals),
                num_particles=tuple(problem.num_particles),
                indexing="blocked",
            )
        )
        exact_reference_mode = "sector_exact_dense"

    adapt_result = run_local_molecular_adapt_vqe(
        h_poly=h_poly,
        n_spatial_orbitals=int(problem.n_spatial_orbitals),
        num_particles=tuple(problem.num_particles),
        exact_gs_energy=exact_energy,
        max_depth=int(args.adapt_max_depth),
        eps_grad=float(args.adapt_eps_grad),
        eps_energy=float(args.adapt_eps_energy),
        maxiter=int(args.adapt_maxiter),
        optimizer=str(args.adapt_inner_optimizer),
        seed=int(args.seed),
        ordering="blocked",
        finite_angle=float(args.adapt_finite_angle),
        finite_angle_min_improvement=float(args.adapt_finite_angle_min_improvement),
    )

    chemistry_dir = Path(__file__).resolve().parent
    default_stem = f"molecule_adapt_nso{int(problem.n_spin_orbitals)}_{str(args.basis).replace('/', '_')}"
    output_json = Path(args.output_json) if args.output_json not in {None, ""} else (chemistry_dir / (default_stem + ".json"))
    output_json.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "prototype": "psi4_molecule_adapt_v1_local_core",
        "chemistry": {
            **problem.to_jsonable(),
            "units": str(args.units),
            "hamiltonian_pauli_terms": int(h_poly.count_number_terms()),
            "exact_reference_mode": str(exact_reference_mode),
            "exact_ground_energy_sector": exact_energy,
            "exact_max_qubits": int(exact_max_qubits),
        },
        "adapt_settings": {
            "adapt_pool": "molecular_uccsd",
            "adapt_max_depth": int(args.adapt_max_depth),
            "adapt_eps_grad": float(args.adapt_eps_grad),
            "adapt_eps_energy": float(args.adapt_eps_energy),
            "adapt_maxiter": int(args.adapt_maxiter),
            "adapt_inner_optimizer": str(args.adapt_inner_optimizer),
            "adapt_finite_angle": float(args.adapt_finite_angle),
            "adapt_finite_angle_min_improvement": float(args.adapt_finite_angle_min_improvement),
            "seed": int(args.seed),
            "ordering": "blocked",
        },
        "adapt_vqe": dict(adapt_result.payload),
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote JSON: {output_json}")


if __name__ == "__main__":
    main()
