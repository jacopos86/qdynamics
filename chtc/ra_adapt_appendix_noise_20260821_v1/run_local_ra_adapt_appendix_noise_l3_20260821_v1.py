"""Fixed four-cell L=3 noise campaign: RA-append at three levels, AAVQE at one.

Runs the cells and writes signed artifacts, so the appendix noise figure is
citable on the same footing as the optional-control figures.  Takes no
arguments: the cell allowlist, horizon and noise levels are sealed here.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

CAMPAIGN_ID = "ra_adapt_appendix_noise_l3_20260821_v1"
RUNTIME_RELATIVE_PATH = f"output/local_runs/{CAMPAIGN_ID}"
REPORT_RELATIVE_PATH = "chtc/ra_adapt_appendix_noise_20260821_v1/results"
HORIZON = 20
NUM_SITES = 3
U_OVER_T = 1.25
NOISE_LEVELS = ("low", "high", "extreme")
# Comparator at every level, not one: a single matched level would
# leave the comparison open to the charge of being chosen after the
# fact.
COMPARATOR_LEVELS = NOISE_LEVELS
EXACT_ENERGY = -2.1414935642714075


# Determinism and single-threading, matching the optional-control campaign.
# The record cache in particular must stay off: a cache hit would skip a
# value-noise draw and silently make the trajectory less noisy than declared.
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


def configure_science_environment() -> None:
    for key, value in SCIENCE_ENVIRONMENT.items():
        os.environ[key] = value


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def canonical_sha256(value: Mapping[str, Any], *, field: str = "sha256") -> str:
    payload = {k: v for k, v in dict(value).items() if k != field}
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def signed(payload: dict[str, Any]) -> dict[str, Any]:
    return {**payload, "sha256": canonical_sha256(payload)}


def source_facts() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
            check=True,
        ).stdout.strip()

    if git("status", "--porcelain"):
        raise ValueError("Campaign source must be clean before launch.")
    return signed(
        {
            "schema": "ra_adapt_appendix_noise_source_facts_v1",
            "head": git("rev-parse", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        }
    )


def build_campaign_plan() -> dict[str, Any]:
    cells = [
        {
            "cell_id": f"l3_noise_{level}_h{HORIZON}",
            "arm": "baseline",
            "route": "ra_append",
            "noise_level": level,
            "num_sites": NUM_SITES,
            "u_over_t": U_OVER_T,
            "horizon": HORIZON,
            "execution_entrypoint": "run_ra_adapt",
        }
        for level in NOISE_LEVELS
    ] + [
        {
            "cell_id": f"l3_aavqe_noise_{level}_h{HORIZON}",
            "arm": "aavqe",
            "route": "append_adapt",
            "noise_level": level,
            "num_sites": NUM_SITES,
            "u_over_t": U_OVER_T,
            "horizon": HORIZON,
            "execution_entrypoint": "run_append_adapt",
        }
        for level in COMPARATOR_LEVELS
    ]
    return signed(
        {
            "schema": "ra_adapt_appendix_noise_campaign_plan_v1",
            "campaign_id": CAMPAIGN_ID,
            "exact_same_cutoff_energy": EXACT_ENERGY,
            "generator_representation": "macro_generator_v1",
            "inner_optimizer": "powell",
            "insertion": "append_only",
            "warm_start": "off",
            "cells": cells,
        }
    )


def _write_signed(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, default=str))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_ra_cell(cell: Mapping[str, Any], cell_dir: Path) -> dict[str, Any]:
    from pipelines.static_adapt.ra_adapt import run_ra_adapt
    from pipelines.static_adapt.ra_adapt.appendix_demos import (
        build_ra_adapt_appendix_noise_request,
        build_ra_adapt_appendix_pure_hubbard_problem,
        ra_adapt_appendix_application_source_contract,
    )

    problem = build_ra_adapt_appendix_pure_hubbard_problem(
        num_sites=int(cell["num_sites"])
    )
    request = build_ra_adapt_appendix_noise_request(
        noise_level=str(cell["noise_level"]),
        maximum_controller_rounds=int(cell["horizon"]),
        num_sites=int(cell["num_sites"]),
        output_dir=cell_dir / "observations",
    )
    source = ra_adapt_appendix_application_source_contract(problem, request)
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
            "compile": dict(row["compile"]),
        }
        for row in evidence["accepted_rounds"]
    ]
    _write_signed(cell_dir / "application_source.json", source)
    _write_signed(cell_dir / "accepted_run_evidence.json", evidence)
    _write_signed(
        cell_dir / "controller_replay_evidence.json",
        dict(result.scientific_receipts["controller_replay_evidence"]),
    )
    return {"rounds": rounds, "protocol_sha256": result.protocol.sha256}


def _run_aavqe_cell(cell: Mapping[str, Any], cell_dir: Path) -> dict[str, Any]:
    from pipelines.static_adapt.ra_adapt import run_append_adapt
    from pipelines.static_adapt.ra_adapt.appendix_demos import (
        RAAdaptAppendixPureHubbardCandidateAdapter,
        build_ra_adapt_appendix_pure_hubbard_problem,
        ra_adapt_appendix_generator_representation,
    )
    from pipelines.static_adapt.ra_adapt.contracts import AppendAdaptRequest
    from pipelines.static_adapt.sr_snake.contracts import (
        SRExecutionPolicy,
        SRStopPolicy,
    )

    problem = build_ra_adapt_appendix_pure_hubbard_problem(
        num_sites=int(cell["num_sites"])
    )
    request = AppendAdaptRequest(
        adapter=RAAdaptAppendixPureHubbardCandidateAdapter(
            candidate_representation_id=(
                ra_adapt_appendix_generator_representation(
                    int(cell["num_sites"])
                )
            ),
            noise_level_id=str(cell["noise_level"]),
        ),
        execution=SRExecutionPolicy(
            stop=SRStopPolicy(maximum_controller_rounds=int(cell["horizon"]))
        ),
    )
    result = run_append_adapt(problem, request)
    payload = result.to_dict()
    body = payload["result_payload"]
    rounds = [
        {
            "controller_round": int(row["controller_round"]),
            "accepted_energy": float(row["energy_after"]),
            "same_cutoff_absolute_error": abs(
                float(row["energy_after"]) - EXACT_ENERGY
            ),
            "active_ansatz_depth": len(
                row["active_prefix_checkpoint"]["accepted_operator_labels"]
            ),
            "accepted_operator_labels": list(
                row["active_prefix_checkpoint"]["accepted_operator_labels"]
            ),
            "cumulative_s_alg": int(
                row["active_prefix_checkpoint"]["estimator_prefix"][
                    "cumulative_executed_queries"
                ]["S_alg"]
            ),
        }
        for row in body["history"]
    ]
    _write_signed(
        cell_dir / "application_source.json",
        signed(
            {
                "schema": "ra_adapt_appendix_noise_comparator_source_v1",
                "algorithm_id": str(body["algorithm_id"]),
                "candidate_representation": str(body["candidate_representation"]),
                "selector_identity": str(payload["selector_identity"]),
                "accepted_refit_scope": str(body["accepted_refit_scope"]),
                "noise_level_id": str(cell["noise_level"]),
                "horizon": int(cell["horizon"]),
            }
        ),
    )
    _write_signed(cell_dir / "append_result.json", payload)
    return {"rounds": rounds, "protocol_sha256": str(body["protocol_sha256"])}



def _completed_cell(
    cell_dir: Path, cell: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Reuse a cell only when its receipt still matches what is on disk.

    Adding a cell to a campaign should not force every finished cell to be
    recomputed, but a stale or half-written cell must never be reused, so the
    summary must be for this exact cell spec and every file it lists must
    still hash to the digest the receipt recorded.
    """

    summary_path = cell_dir / "cell_summary.json"
    receipt_path = cell_dir / "artifact_receipt.json"
    if not summary_path.exists() or not receipt_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text())
        receipt = json.loads(receipt_path.read_text())
    except json.JSONDecodeError:
        return None
    if (
        summary.get("sha256") != canonical_sha256(summary)
        or receipt.get("sha256") != canonical_sha256(receipt)
        or dict(summary.get("cell") or {}) != dict(cell)
        or not summary.get("accepted_rounds")
    ):
        return None
    for row in receipt.get("files", []):
        path = cell_dir / str(row["path"])
        if (
            path.name == "artifact_receipt.json"
            or not path.exists()
            or _file_sha256(path) != str(row["sha256"])
        ):
            if path.name != "artifact_receipt.json":
                return None
    return {
        "cell": dict(cell),
        "cell_summary_sha256": str(summary["sha256"]),
        "artifact_receipt_sha256": str(receipt["sha256"]),
        "terminal_error": float(
            summary["accepted_rounds"][-1]["same_cutoff_absolute_error"]
        ),
        "completed_controller_rounds": int(
            summary["completed_controller_rounds"]
        ),
    }


def run_fixed_campaign() -> int:
    configure_science_environment()
    source = source_facts()
    plan = build_campaign_plan()
    runtime_root = REPO_ROOT / RUNTIME_RELATIVE_PATH
    report_root = REPO_ROOT / REPORT_RELATIVE_PATH
    runtime_root.mkdir(parents=True, exist_ok=True)
    _write_signed(runtime_root / "source_facts.json", source)
    _write_signed(runtime_root / "campaign_plan.json", plan)

    curves: dict[str, Any] = {}
    for cell in plan["cells"]:
        cell_id = str(cell["cell_id"])
        cell_dir = runtime_root / "cells" / cell_id
        completed = _completed_cell(cell_dir, cell)
        if completed is not None:
            curves[cell_id] = completed
            print(
                f"[{CAMPAIGN_ID}] {cell_id} already complete, "
                f"floor {completed['terminal_error']:.3e}",
                flush=True,
            )
            continue
        if cell_dir.exists():
            shutil.rmtree(cell_dir)
        cell_dir.mkdir(parents=True)
        print(f"[{CAMPAIGN_ID}] {cell_id} ...", flush=True)
        executed = (
            _run_ra_cell(cell, cell_dir)
            if cell["execution_entrypoint"] == "run_ra_adapt"
            else _run_aavqe_cell(cell, cell_dir)
        )
        summary = signed(
            {
                "schema": "ra_adapt_appendix_noise_cell_summary_v1",
                "campaign_id": CAMPAIGN_ID,
                "cell_id": cell_id,
                "cell": dict(cell),
                "exact_same_cutoff_energy": EXACT_ENERGY,
                "protocol_sha256": executed["protocol_sha256"],
                "completed_controller_rounds": len(executed["rounds"]),
                "accepted_rounds": executed["rounds"],
            }
        )
        _write_signed(cell_dir / "cell_summary.json", summary)
        receipt = signed(
            {
                "schema": "ra_adapt_appendix_noise_artifact_receipt_v1",
                "campaign_id": CAMPAIGN_ID,
                "cell_id": cell_id,
                "files": [
                    {
                        "path": str(path.relative_to(cell_dir)),
                        "sha256": _file_sha256(path),
                        "size_bytes": path.stat().st_size,
                    }
                    for path in sorted(cell_dir.rglob("*"))
                    if path.is_file()
                ],
            }
        )
        _write_signed(cell_dir / "artifact_receipt.json", receipt)
        curves[cell_id] = {
            "cell": dict(cell),
            "cell_summary_sha256": summary["sha256"],
            "artifact_receipt_sha256": receipt["sha256"],
            "terminal_error": executed["rounds"][-1][
                "same_cutoff_absolute_error"
            ],
            "completed_controller_rounds": len(executed["rounds"]),
        }
        print(
            f"[{CAMPAIGN_ID}] {cell_id} done, "
            f"floor {curves[cell_id]['terminal_error']:.3e}",
            flush=True,
        )

    manifest = signed(
        {
            "schema": "ra_adapt_appendix_noise_campaign_manifest_v1",
            "campaign_id": CAMPAIGN_ID,
            "source_facts": source,
            "campaign_plan_sha256": plan["sha256"],
            "runtime_relative_path": RUNTIME_RELATIVE_PATH,
            "cells": curves,
        }
    )
    _write_signed(runtime_root / "campaign_manifest.json", manifest)
    report_root.mkdir(parents=True, exist_ok=True)
    _write_signed(report_root / "campaign_manifest.json", manifest)
    print(f"[{CAMPAIGN_ID}] manifest sha256 {manifest['sha256']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    if list(sys.argv[1:] if argv is None else argv):
        print("This campaign takes no arguments.", file=sys.stderr)
        return 2
    return run_fixed_campaign()


if __name__ == "__main__":
    raise SystemExit(main())
