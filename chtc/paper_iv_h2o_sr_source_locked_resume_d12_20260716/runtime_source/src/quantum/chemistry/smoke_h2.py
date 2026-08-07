from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


_MATH_SMOKE_GATE = r"|E_{\\mathrm{ADAPT}} - E_{0}| \\le \\varepsilon"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Chemistry-local H2 smoke check for the Psi4-backed molecular ADAPT prototype."
    )
    p.add_argument("--bond-length", type=float, default=0.7414)
    p.add_argument("--basis", type=str, default="sto-3g")
    p.add_argument("--tolerance", type=float, default=1e-8)
    p.add_argument("--psi4-output", type=str, default=None)
    p.add_argument("--adapt-max-depth", type=int, default=8)
    p.add_argument("--adapt-maxiter", type=int, default=400)
    p.add_argument("--adapt-inner-optimizer", choices=["COBYLA", "POWELL", "SPSA"], default="COBYLA")
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
            output_file=args.psi4_output,
        )
    except ImportError as exc:
        raise SystemExit("Psi4 is not installed in this environment. Install psi4, then rerun.") from exc

    h_poly = build_restricted_closed_shell_molecular_hamiltonian(problem, ordering="blocked")
    exact_energy = float(
        exact_ground_energy_sector(
            h_poly,
            num_sites=int(problem.n_spatial_orbitals),
            num_particles=tuple(problem.num_particles),
            indexing="blocked",
        )
    )
    adapt_result = run_local_molecular_adapt_vqe(
        h_poly=h_poly,
        n_spatial_orbitals=int(problem.n_spatial_orbitals),
        num_particles=tuple(problem.num_particles),
        exact_gs_energy=exact_energy,
        max_depth=int(args.adapt_max_depth),
        eps_grad=1e-8,
        eps_energy=1e-10,
        maxiter=int(args.adapt_maxiter),
        optimizer=str(args.adapt_inner_optimizer),
        seed=7,
        ordering="blocked",
    )

    abs_delta_e = adapt_result.payload.get("abs_delta_e")
    if abs_delta_e is None:
        raise SystemExit("Smoke check expected an exact reference energy, but none was produced.")
    abs_delta_e_value = float(abs_delta_e)
    tolerance = float(args.tolerance)
    ok = abs_delta_e_value <= tolerance

    chemistry_dir = Path(__file__).resolve().parent
    output_json = Path(args.output_json) if args.output_json not in {None, ""} else (chemistry_dir / "smoke_h2_result.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "check": "chemistry_local_h2_smoke",
        "ok": bool(ok),
        "tolerance": tolerance,
        "bond_length_angstrom": float(args.bond_length),
        "basis": str(args.basis),
        "exact_ground_energy_sector": exact_energy,
        "adapt_vqe": dict(adapt_result.payload),
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote JSON: {output_json}")
    print(
        "H2 smoke: "
        f"ok={ok} abs_delta_e={abs_delta_e_value:.3e} tol={tolerance:.3e} "
        f"depth={adapt_result.payload.get('ansatz_depth')}"
    )
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
