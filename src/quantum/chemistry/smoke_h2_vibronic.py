from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chemistry-local smoke check for vibronic H2.")
    p.add_argument("--bond-length", type=float, default=0.7414)
    p.add_argument("--bond-step", type=float, default=0.01)
    p.add_argument("--basis", type=str, default="sto-3g")
    p.add_argument("--n-ph-max", type=int, default=3)
    p.add_argument("--boson-encoding", choices=["binary", "unary"], default="binary")
    p.add_argument("--coupling-scale", type=float, default=1.0)
    p.add_argument("--tolerance", type=float, default=5e-5)
    p.add_argument("--output-json", type=str, default=None)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    try:
        from src.quantum.chemistry.molecular_adapt_core import run_pipeline_local_adapt_vqe_with_pool
        from src.quantum.chemistry.vibronic_h2 import build_vibronic_h2_model, exact_ground_energy_physical_sector, hf_reference_energy
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
    exact_energy = float(
        exact_ground_energy_physical_sector(
            model.h_vibronic,
            n_spatial_orbitals=2,
            num_particles=(1, 1),
            n_ph_max=int(model.n_ph_max),
            boson_encoding=str(model.boson_encoding),
        )
    )
    hf_energy = float(hf_reference_energy(model.h_vibronic, model.psi_ref))
    adapt_result = run_pipeline_local_adapt_vqe_with_pool(
        h_poly=model.h_vibronic,
        psi_ref=model.psi_ref,
        pool=model.pool,
        exact_gs_energy=float(exact_energy),
        max_depth=8,
        eps_grad=1e-8,
        eps_energy=1e-8,
        maxiter=400,
        optimizer="COBYLA",
        seed=7,
        pool_type="vibronic_h2_local_pool",
        metadata={"num_particles": {"n_up": 1, "n_dn": 1}},
        parameterization_mode="logical_shared",
        reopt_policy="windowed",
        window_size=3,
        window_topk=0,
        full_refit_every=0,
    )

    adapt_energy = float(adapt_result.payload["energy"])
    abs_delta_e = float(adapt_result.payload["abs_delta_e"])
    improved_vs_hf = adapt_energy < hf_energy - 1e-8
    ok = bool(abs_delta_e <= float(args.tolerance) and improved_vs_hf)

    chemistry_dir = Path(__file__).resolve().parent
    output_json = Path(args.output_json) if args.output_json not in {None, ""} else (chemistry_dir / "smoke_h2_vibronic_result.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "check": "chemistry_local_h2_vibronic_smoke",
        "ok": bool(ok),
        "tolerance": float(args.tolerance),
        "hf_reference_energy": float(hf_energy),
        "exact_ground_energy": float(exact_energy),
        "adapt_vqe": dict(adapt_result.payload),
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote JSON: {output_json}")
    print(
        "H2 vibronic smoke: "
        f"ok={ok} abs_delta_e={abs_delta_e:.3e} tol={float(args.tolerance):.3e} "
        f"hf_drop={hf_energy - adapt_energy:.3e} depth={adapt_result.payload.get('ansatz_depth')}"
    )
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
