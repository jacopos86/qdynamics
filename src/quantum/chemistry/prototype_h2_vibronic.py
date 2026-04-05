from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Prototype vibronic H2 ADAPT-VQE runner. Uses existing repo boson encoding, "
            "but keeps all new code in the chemistry folder."
        )
    )
    p.add_argument("--bond-length", type=float, default=0.7414)
    p.add_argument("--bond-step", type=float, default=0.01, help="Central finite-difference step in angstrom.")
    p.add_argument("--basis", type=str, default="sto-3g")
    p.add_argument("--n-ph-max", type=int, default=3)
    p.add_argument("--boson-encoding", choices=["binary", "unary"], default="binary")
    p.add_argument("--coupling-scale", type=float, default=1.0)
    p.add_argument("--adapt-max-depth", type=int, default=8)
    p.add_argument("--adapt-eps-grad", type=float, default=1e-8)
    p.add_argument("--adapt-eps-energy", type=float, default=1e-8)
    p.add_argument("--adapt-maxiter", type=int, default=400)
    p.add_argument("--adapt-inner-optimizer", choices=["COBYLA", "POWELL", "SPSA"], default="COBYLA")
    p.add_argument("--adapt-finite-angle", type=float, default=0.05)
    p.add_argument("--adapt-finite-angle-min-improvement", type=float, default=0.0)
    p.add_argument("--exact-max-qubits", type=int, default=12)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output-json", type=str, default=None)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    try:
        from src.quantum.chemistry.molecular_adapt_core import run_pipeline_local_adapt_vqe_with_pool
        from src.quantum.chemistry.vibronic_h2 import (
            build_vibronic_h2_model,
            exact_ground_energy_physical_sector,
            hf_reference_energy,
        )
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc

    model = build_vibronic_h2_model(
        bond_length_angstrom=float(args.bond_length),
        bond_step_angstrom=float(args.bond_step),
        basis=str(args.basis),
        n_ph_max=int(args.n_ph_max),
        boson_encoding=str(args.boson_encoding),
        coupling_scale=float(args.coupling_scale),
        ordering="blocked",
    )

    exact_energy: float | None = None
    exact_reference_mode = "skipped_qubit_cutoff"
    if int(model.n_total_qubits) <= int(args.exact_max_qubits):
        exact_energy = float(
            exact_ground_energy_physical_sector(
                model.h_vibronic,
                n_spatial_orbitals=2,
                num_particles=(1, 1),
                n_ph_max=int(model.n_ph_max),
                boson_encoding=str(model.boson_encoding),
            )
        )
        exact_reference_mode = "dense_exact_physical_sector"

    hf_energy = hf_reference_energy(model.h_vibronic, model.psi_ref)
    adapt_result = run_pipeline_local_adapt_vqe_with_pool(
        h_poly=model.h_vibronic,
        psi_ref=model.psi_ref,
        pool=model.pool,
        exact_gs_energy=exact_energy,
        max_depth=int(args.adapt_max_depth),
        eps_grad=float(args.adapt_eps_grad),
        eps_energy=float(args.adapt_eps_energy),
        maxiter=int(args.adapt_maxiter),
        optimizer=str(args.adapt_inner_optimizer),
        seed=int(args.seed),
        pool_type="vibronic_h2_local_pool",
        metadata={
            "num_particles": {"n_up": 1, "n_dn": 1},
            "n_total_qubits": int(model.n_total_qubits),
            "n_boson_qubits": int(model.n_boson_qubits),
        },
        parameterization_mode="logical_shared",
        finite_angle=float(args.adapt_finite_angle),
        finite_angle_min_improvement=float(args.adapt_finite_angle_min_improvement),
        reopt_policy="windowed",
        window_size=3,
        window_topk=0,
        full_refit_every=0,
    )

    chemistry_dir = Path(__file__).resolve().parent
    default_name = (
        f"h2_vibronic_b{float(args.bond_length):.4f}_{str(args.basis).replace('/', '_')}"
        f"_nph{int(args.n_ph_max)}_{str(args.boson_encoding)}.json"
    )
    output_json = Path(args.output_json) if args.output_json not in {None, ""} else (chemistry_dir / default_name)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "prototype": "psi4_h2_vibronic_adapt_v1_local_core",
        "model": {
            **model.to_jsonable(),
            "coupling_scale": float(args.coupling_scale),
            "hf_reference_energy": float(hf_energy),
            "exact_reference_mode": str(exact_reference_mode),
            "exact_ground_energy": exact_energy,
        },
        "adapt_settings": {
            "adapt_max_depth": int(args.adapt_max_depth),
            "adapt_eps_grad": float(args.adapt_eps_grad),
            "adapt_eps_energy": float(args.adapt_eps_energy),
            "adapt_maxiter": int(args.adapt_maxiter),
            "adapt_inner_optimizer": str(args.adapt_inner_optimizer),
            "adapt_finite_angle": float(args.adapt_finite_angle),
            "adapt_finite_angle_min_improvement": float(args.adapt_finite_angle_min_improvement),
            "seed": int(args.seed),
        },
        "adapt_vqe": dict(adapt_result.payload),
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote JSON: {output_json}")


if __name__ == "__main__":
    main()
