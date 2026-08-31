"""Run one sealed Paper-I L=3 fixed-width noise cell with Phase 0 active."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

CAMPAIGN_ID = "ra_adapt_l3_noise_fixed_phase0_20260824"
EXACT_ENERGY = -2.1414935642714075
HORIZON = 20
NOISE_LEVELS = ("low", "high", "extreme")
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


def _json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _source_receipt() -> dict[str, Any]:
    diff = _git("diff", "--binary")
    return {
        "git_head": _git("rev-parse", "HEAD").strip(),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD").strip(),
        "dirty_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
        "dirty_diff_bytes": len(diff.encode()),
    }


def _assert_protocol(protocol: dict[str, Any]) -> None:
    adapter = protocol["request"]["adapter"]
    settings = protocol["route_contract"]["execution_settings"]
    invariants = protocol["route_contract"]["semantic_invariants"]
    assert adapter["phase0_gradient_screen"] is True
    assert adapter["adaptive_phase_shortlists"] is False
    assert settings["ra_phase0_gradient_shortlist_size"] == 24
    assert settings["phase1_shortlist_size"] == 24
    assert settings["phase2_shortlist_size"] == 12
    assert "ra_phase123_shortlist_policy" not in settings
    assert invariants["phase0_active"] is True
    assert invariants["phase0_score"] == "standard_adapt_absolute_gradient_v1"


def run(level: str) -> int:
    for key, value in SCIENCE_ENVIRONMENT.items():
        os.environ[key] = value

    from pipelines.static_adapt import engine_support
    from pipelines.static_adapt.ra_adapt import run_ra_adapt
    from pipelines.static_adapt.ra_adapt import append as append_module
    from pipelines.static_adapt.ra_adapt.appendix_demos import (
        build_ra_adapt_appendix_noise_request,
        build_ra_adapt_appendix_pure_hubbard_problem,
        ra_adapt_appendix_application_source_contract,
    )

    assert engine_support.RA_ACCEPTED_REFIT_POWELL_XTOL == 1.0e-5
    assert engine_support.RA_ACCEPTED_REFIT_POWELL_FTOL == 1.0e-12
    assert append_module.AAVQE_COMPARATOR_POWELL_XTOL == 1.0e-5
    assert append_module.AAVQE_COMPARATOR_POWELL_FTOL == 1.0e-12

    output_dir = (
        REPO_ROOT / "output/local_runs" / CAMPAIGN_ID / f"{level}_h{HORIZON}"
    )
    result_path = output_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(f"Refusing to overwrite completed result: {result_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    problem = build_ra_adapt_appendix_pure_hubbard_problem(num_sites=3)
    request = build_ra_adapt_appendix_noise_request(
        noise_level=level,
        maximum_controller_rounds=HORIZON,
        num_sites=3,
        adaptive_phase_shortlists=False,
        phase0_gradient_screen=True,
        output_dir=output_dir / "observations",
    )
    assert request.adapter.phase0_gradient_screen is True
    assert request.adapter.adaptive_phase_shortlists is False

    source = ra_adapt_appendix_application_source_contract(problem, request)
    launch = {
        "schema": "paper_i_l3_noise_fixed_phase0_launch_v1",
        "campaign_id": CAMPAIGN_ID,
        "noise_level": level,
        "horizon": HORIZON,
        "phase0": {"active": True, "cap": 24, "pool_size": 24},
        "fixed_phase_widths": [24, 12, 1],
        "adaptive_phase_shortlists": False,
        "powell": {"xtol": 1.0e-5, "ftol": 1.0e-12, "maxiter": 200},
        "science_environment": SCIENCE_ENVIRONMENT,
        "source": _source_receipt(),
    }
    _json(output_dir / "launch_profile.json", launch)
    _json(output_dir / "application_source.json", source)
    print(
        f"[fixed-phase0] START level={level} h={HORIZON} widths=24/12/1",
        flush=True,
    )

    result = run_ra_adapt(problem, request)
    protocol = result.protocol.to_dict()
    _assert_protocol(protocol)
    receipts = dict(result.scientific_receipts)
    evidence = dict(receipts["accepted_run_evidence"])
    rounds = [
        {
            "controller_round": int(row["controller_round"]),
            "accepted_energy": float(row["accepted_energy"]),
            "same_cutoff_absolute_error": abs(
                float(row["accepted_energy"]) - EXACT_ENERGY
            ),
            "active_ansatz_depth": int(row["active_ansatz_depth"]),
            "compile": dict(row["compile"]),
        }
        for row in evidence["accepted_rounds"]
    ]
    if len(rounds) != HORIZON:
        raise AssertionError(f"Expected {HORIZON} accepted rounds, got {len(rounds)}")

    payload = {
        "schema": "paper_i_l3_noise_fixed_phase0_result_v1",
        "campaign_id": CAMPAIGN_ID,
        "noise_level": level,
        "horizon": HORIZON,
        "exact_same_cutoff_energy": EXACT_ENERGY,
        "launch_profile": launch,
        "application_source": source,
        "protocol": protocol,
        "rounds": rounds,
        "scientific_receipts": receipts,
    }
    _json(result_path, payload)
    print(
        f"[fixed-phase0] DONE level={level} rounds={len(rounds)} "
        f"terminal_error={rounds[-1]['same_cutoff_absolute_error']:.6e}",
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("level", choices=NOISE_LEVELS)
    args = parser.parse_args()
    return run(str(args.level))


if __name__ == "__main__":
    raise SystemExit(main())
