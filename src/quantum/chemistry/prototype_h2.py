from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Prototype Psi4-backed H2 ADAPT-VQE runner. "
            "This path is isolated from the Hubbard / HH CLIs and uses a chemistry-local ADAPT core."
        )
    )
    p.add_argument("--bond-length", type=float, default=0.7414)
    p.add_argument("--basis", type=str, default="sto-3g")
    p.add_argument("--charge", type=int, default=0)
    p.add_argument("--multiplicity", type=int, default=1)
    p.add_argument("--psi4-memory", type=str, default=None)
    p.add_argument("--psi4-output", type=str, default=None)
    p.add_argument("--adapt-max-depth", type=int, default=8)
    p.add_argument("--adapt-eps-grad", type=float, default=1e-8)
    p.add_argument("--adapt-eps-energy", type=float, default=1e-10)
    p.add_argument("--adapt-maxiter", type=int, default=400)
    p.add_argument("--adapt-inner-optimizer", choices=["COBYLA", "POWELL", "SPSA"], default="COBYLA")
    p.add_argument("--adapt-finite-angle", type=float, default=0.05)
    p.add_argument("--adapt-finite-angle-min-improvement", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output-json", type=str, default=None)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    try:
        from src.quantum.chemistry.molecular_adapt_core import run_local_molecular_adapt_vqe
        from src.quantum.chemistry.molecular_hamiltonian import (
            build_restricted_closed_shell_molecular_hamiltonian,
        )
        from src.quantum.chemistry.psi4_adapter import build_h2_problem_from_psi4
        from src.quantum.vqe_latex_python_pairs import exact_ground_energy_sector
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc

    try:
        problem = build_h2_problem_from_psi4(
            bond_length_angstrom=float(args.bond_length),
            basis=str(args.basis),
            charge=int(args.charge),
            multiplicity=int(args.multiplicity),
            memory=args.psi4_memory,
            output_file=args.psi4_output,
        )
    except ImportError as exc:
        raise SystemExit(
            "Psi4 is not installed in this environment. Install psi4, then rerun the prototype."
        ) from exc

    if int(problem.n_spatial_orbitals) != 2 or tuple(problem.num_particles) != (1, 1):
        raise SystemExit(
            "Current prototype runner is intentionally narrow: it only supports H2 cases that map "
            "to 2 spatial orbitals and (n_alpha, n_beta) = (1, 1), such as H2/STO-3G."
        )

    h_poly = build_restricted_closed_shell_molecular_hamiltonian(problem, ordering="blocked")
    exact_energy = exact_ground_energy_sector(
        h_poly,
        num_sites=int(problem.n_spatial_orbitals),
        num_particles=tuple(problem.num_particles),
        indexing="blocked",
    )

    adapt_result = run_local_molecular_adapt_vqe(
        h_poly=h_poly,
        n_spatial_orbitals=int(problem.n_spatial_orbitals),
        num_particles=tuple(problem.num_particles),
        exact_gs_energy=float(exact_energy),
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
    default_name = f"h2_adapt_b{float(args.bond_length):.4f}_{str(args.basis).replace('/', '_')}.json"
    output_json = Path(args.output_json) if args.output_json not in {None, ""} else (chemistry_dir / default_name)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "prototype": "psi4_h2_adapt_v2_local_core",
        "chemistry": {
            "system": "H2",
            "geometry_spec": str(problem.geometry_spec),
            "bond_length_angstrom": float(args.bond_length),
            "basis": str(problem.basis),
            "charge": int(problem.charge),
            "multiplicity": int(problem.multiplicity),
            "reference": str(problem.reference),
            "n_spatial_orbitals": int(problem.n_spatial_orbitals),
            "n_spin_orbitals": int(problem.n_spin_orbitals),
            "num_particles": {"n_alpha": int(problem.n_alpha), "n_beta": int(problem.n_beta)},
            "hf_energy": float(problem.hf_energy),
            "nuclear_repulsion_energy": float(problem.nuclear_repulsion_energy),
            "hamiltonian_pauli_terms": int(h_poly.count_number_terms()),
            "exact_ground_energy_sector": float(exact_energy),
        },
        "adapt_settings": {
            "adapt_pool": "uccsd",
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
