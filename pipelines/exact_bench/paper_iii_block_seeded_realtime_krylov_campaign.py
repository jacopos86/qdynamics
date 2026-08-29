#!/usr/bin/env python3
"""Acquire the locked block-seeded real-time-Krylov comparator matrix.

The block size, source ordering, time step, overlap cutoff, and dimension caps
are fixed before any exact excited-state audit.  Exact roots and eigenvectors
are supplied only after each online fixed-cap trace terminates.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.exact_bench.paper_iii_seeded_realtime_krylov_campaign import (
    ERROR_TARGETS,
    TARGET_ROOTS,
    _all_case_ids,
    _append_window_audit,
    _git_revision,
    _load_case,
    _problem_receipt,
    _sha256,
    _static_spectral_audit,
)
from pipelines.qse_spectra.block_seeded_realtime_krylov import (
    BLOCK_SEEDED_REALTIME_KRYLOV_POLICY_ID,
    BlockSeededRealtimeKrylovConfig,
    audit_block_krylov_energies,
    run_block_seeded_realtime_krylov,
)


SCHEMA_VERSION = "paper_iii_block_seeded_realtime_krylov_matrix_v1"
CAMPAIGN_ID = "paper_iii_block_seeded_realtime_krylov_matrix_20260829_v1"
DEFAULT_OUTPUT = REPO_ROOT / "output/diagnostics" / CAMPAIGN_ID / "matrix.json"


@dataclass(frozen=True)
class _OnlineBlockProblemView:
    """Online inputs required for deterministic block construction."""

    num_qubits: int
    hamiltonian: Any
    ground: np.ndarray
    basis: tuple[Any, ...]
    resources: tuple[Mapping[str, float], ...]
    n_ph_max: int | None = None
    key: Any = None


def _online_view(problem: Any) -> _OnlineBlockProblemView:
    return _OnlineBlockProblemView(
        num_qubits=int(problem.num_qubits),
        hamiltonian=problem.hamiltonian,
        ground=np.asarray(problem.ground, dtype=complex),
        basis=tuple(problem.basis),
        resources=tuple(dict(row) for row in problem.resources),
        n_ph_max=(
            None if not hasattr(problem, "n_ph_max") else int(problem.n_ph_max)
        ),
        key=getattr(problem, "key", None),
    )


def _threshold_crossings(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    crossings: dict[str, Any] = {}
    for target in ERROR_TARGETS:
        crossing = next(
            (
                row
                for row in rows
                if row["max_root_abs_error"] is not None
                and float(row["max_root_abs_error"]) <= float(target)
            ),
            None,
        )
        if crossing is None:
            crossings[f"{target:.0e}"] = {"status": "NOT_REACHED_WITHIN_CAP"}
            continue
        resources = dict(crossing["resources"])
        crossings[f"{target:.0e}"] = {
            "status": "REACHED",
            "k": int(crossing["k"]),
            "time_shell_count": int(crossing["time_shell_count"]),
            "block_size": int(crossing["block_size"]),
            "retained_rank": int(crossing["retained_rank"]),
            "K": float(crossing["K"]),
            "max_root_abs_error": float(crossing["max_root_abs_error"]),
            "per_root_abs_error": list(crossing["per_root_abs_error"]),
            "matched_root_energies": list(crossing["matched_root_energies"]),
            "rth_root_energy": float(crossing["rth_root_energy"]),
            "r_to_r_plus_1_gap": (
                None
                if crossing["r_to_r_plus_1_gap"] is None
                else float(crossing["r_to_r_plus_1_gap"])
            ),
            "graph_span": dict(resources["graph_span"]),
            "qiskit_compiled": dict(resources["qiskit_compiled"]),
            "estimator": dict(crossing["estimator"]),
        }
    return crossings


def _run_case(case_id: str, *, include_qiskit: bool) -> dict[str, Any]:
    family, problem, k_max = _load_case(case_id)
    started = time.monotonic()
    run = run_block_seeded_realtime_krylov(
        _online_view(problem),
        config=BlockSeededRealtimeKrylovConfig(
            k_max=int(k_max),
            target_root_count=TARGET_ROOTS,
            block_size=TARGET_ROOTS,
            include_qiskit=bool(include_qiskit),
        ),
    )
    exact_references = tuple(float(value) for value in problem.references)
    energy_trace = _append_window_audit(
        audit_block_krylov_energies(run, exact_references), exact_references
    )
    terminal = energy_trace[-1]
    return {
        "case_id": str(case_id),
        "model_family": family,
        "problem_receipt": _problem_receipt(problem, family),
        "method_id": "block_seeded_realtime_krylov",
        "policy_id": BLOCK_SEEDED_REALTIME_KRYLOV_POLICY_ID,
        "policy_receipt": dict(run.policy_receipt),
        "source_receipt": dict(run.source_receipt),
        "resource_receipt": dict(run.resource_receipt),
        "reference_boundary": {
            "exact_excited_roots_available_to_online_route": False,
            "exact_roots_supplied_after_route_termination": True,
            "exact_eigenvectors_supplied_after_route_termination": True,
        },
        "target_root_count": TARGET_ROOTS,
        "reference_roots": [
            float(value) for value in exact_references[:TARGET_ROOTS]
        ],
        "reference_r_plus_1_root": float(exact_references[TARGET_ROOTS]),
        "trace": energy_trace,
        "threshold_crossings": _threshold_crossings(energy_trace),
        "terminal": {
            "status": "CAP_REACHED",
            "k": int(terminal["k"]),
            "time_shell_count": int(terminal["time_shell_count"]),
            "block_size": int(terminal["block_size"]),
            "retained_rank": int(terminal["retained_rank"]),
            "K": float(terminal["K"]),
            "matched_root_energies": list(terminal["matched_root_energies"]),
            "per_root_abs_error": list(terminal["per_root_abs_error"]),
            "max_root_abs_error": float(terminal["max_root_abs_error"]),
            "rth_root_energy": float(terminal["rth_root_energy"]),
            "r_to_r_plus_1_gap": (
                None
                if terminal["r_to_r_plus_1_gap"] is None
                else float(terminal["r_to_r_plus_1_gap"])
            ),
            "graph_span": dict(terminal["resources"]["graph_span"]),
            "qiskit_compiled": dict(
                terminal["resources"]["qiskit_compiled"]
            ),
            "estimator": dict(terminal["estimator"]),
        },
        "static_spectral_audit": _static_spectral_audit(
            problem, family, run, energy_trace
        ),
        "elapsed_seconds": float(time.monotonic() - started),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--cases",
        default=None,
        help="Comma-separated case IDs; default is the full 16-case matrix.",
    )
    parser.add_argument(
        "--include-qiskit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compile block-source actions and the repeated H step on FakeMarrakesh.",
    )
    args = parser.parse_args(argv)

    declared = _all_case_ids()
    cases = (
        declared
        if args.cases is None
        else tuple(
            token.strip() for token in str(args.cases).split(",") if token.strip()
        )
    )
    unknown = sorted(set(cases) - set(declared))
    if unknown:
        raise ValueError(f"undeclared cases: {unknown!r}.")

    started = time.monotonic()
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "evidence_class": "candidate",
        "promotion_decision": "UNRESOLVED_USER_DECISION",
        "git_revision": _git_revision(),
        "route_contract": {
            "method_id": "block_seeded_realtime_krylov",
            "policy_id": BLOCK_SEEDED_REALTIME_KRYLOV_POLICY_ID,
            "same_policy_across_model_families": True,
            "target_root_count": TARGET_ROOTS,
            "block_size": TARGET_ROOTS,
            "block_initialization": (
                "lowest_diagonal_target_sector_computational_probes_first_R_"
                "independent_directions"
            ),
            "matched_error_thresholds": list(ERROR_TARGETS),
            "online_exact_reference_use": False,
            "exact_audit_stage": "strictly_post_run",
            "root_matching": "minimum_total_abs_energy_hungarian_one_to_one",
            "K": "aggregate_graph_span_n2q_including_repeated_source_actions",
            "qiskit_cost": (
                "aggregate_FakeMarrakesh_source_actions_and_hamiltonian_steps"
            ),
            "estimator_queries": (
                "all_independent_S_and_H_pencil_elements_no_qwc_reuse"
            ),
            "prepared_ground_state_cost_included": False,
        },
        "cases": {},
    }
    for case_id in cases:
        print(f"== {case_id}", flush=True)
        payload["cases"][case_id] = _run_case(
            case_id, include_qiskit=bool(args.include_qiskit)
        )
        crossing = payload["cases"][case_id]["threshold_crossings"]["1e-06"]
        print(
            f"   1e-6={crossing['status']} "
            f"k={crossing.get('k')} K={crossing.get('K')}",
            flush=True,
        )
    payload["elapsed_seconds"] = float(time.monotonic() - started)

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    artifact_label = (
        str(output.relative_to(REPO_ROOT))
        if output.is_relative_to(REPO_ROOT)
        else str(output)
    )
    receipt = {
        "schema_version": f"{SCHEMA_VERSION}_receipt",
        "campaign_id": CAMPAIGN_ID,
        "artifact": artifact_label,
        "sha256": _sha256(output),
        "case_count": len(payload["cases"]),
        "all_1e_minus_6_reached": all(
            case["threshold_crossings"]["1e-06"]["status"] == "REACHED"
            for case in payload["cases"].values()
        ),
    }
    receipt_path = output.with_name("receipt.json")
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
