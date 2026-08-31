"""Run the sealed low-noise h10 least-negative Phase-III diagnostic."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

OUTPUT_DIR = (
    REPO_ROOT
    / "output/local_runs/ra_adapt_l3_noise_least_negative_20260824"
    / "low_h10_smoke"
)
EXACT_ENERGY = -2.1414935642714075
SCIENCE_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "OMP_DYNAMIC": "FALSE",
    "MKL_DYNAMIC": "FALSE",
    "STATIC_ADAPT_ALLOCATED_CPUS": "1",
    "QISKIT_NUM_PROCS": "1",
    "QISKIT_PARALLEL": "FALSE",
    "RAYON_NUM_THREADS": "1",
    "CUDA_VISIBLE_DEVICES": "",
    "STATIC_ADAPT_CANDIDATE_RECORD_CACHE": "off",
    "STATIC_ADAPT_HH_POOL_CACHE": "off",
    "STATIC_ADAPT_L3_ROUTE_POOL_CACHE": "off",
}


def main() -> int:
    for key, value in SCIENCE_ENVIRONMENT.items():
        os.environ[key] = value

    from pipelines.static_adapt.ra_adapt import run_ra_adapt
    from pipelines.static_adapt.ra_adapt.appendix_demos import (
        build_ra_adapt_appendix_noise_request,
        build_ra_adapt_appendix_pure_hubbard_problem,
        ra_adapt_appendix_application_source_contract,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    problem = build_ra_adapt_appendix_pure_hubbard_problem(num_sites=3)
    request = build_ra_adapt_appendix_noise_request(
        noise_level="low",
        maximum_controller_rounds=10,
        num_sites=3,
        adaptive_phase_shortlists=True,
        phase0_gradient_screen=True,
        adaptive_frontier_ratio=0.9,
        output_dir=OUTPUT_DIR / "observations",
    )
    source = ra_adapt_appendix_application_source_contract(problem, request)
    print(
        "[least-negative-smoke] low h10; "
        f"pool={source['singleton_inventory']['count']}",
        flush=True,
    )
    result = run_ra_adapt(problem, request)
    evidence = dict(result.scientific_receipts["accepted_run_evidence"])
    rounds = [
        {
            "controller_round": int(row["controller_round"]),
            "accepted_energy": float(row["accepted_energy"]),
            "same_cutoff_absolute_error": abs(
                float(row["accepted_energy"]) - EXACT_ENERGY
            ),
            "active_ansatz_depth": int(row["active_ansatz_depth"]),
        }
        for row in evidence["accepted_rounds"]
    ]
    payload = {
        "schema": "paper_i_l3_noise_least_negative_smoke_v1",
        "run_class": "diagnostic",
        "noise_level": "low",
        "horizon": 10,
        "exact_same_cutoff_energy": EXACT_ENERGY,
        "protocol": result.protocol.to_dict(),
        "rounds": rounds,
        "scientific_receipts": dict(result.scientific_receipts),
    }
    (OUTPUT_DIR / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str)
    )
    (OUTPUT_DIR / "source.json").write_text(
        json.dumps(source, indent=2, sort_keys=True, default=str)
    )
    print(
        "[least-negative-smoke] DONE "
        f"rounds={len(rounds)} terminal_error="
        f"{rounds[-1]['same_cutoff_absolute_error']:.6e}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
