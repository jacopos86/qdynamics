"""Run the matched L=3 graph-span selector diagnostic.

This is deliberately a diagnostic, not a promoted manuscript route.  It holds
the manuscript-cost baseline fixed and changes only the selector-cost scope
from all-phase transpiled Qiskit marginals to the shared Marrakesh graph-span
oracle.  Final circuit resources are still compiled by ``run_arm`` under the
sealed FakeMarrakesh contract.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import pipelines.static_adapt.ra_adapt.engine as ra_engine
from pipelines.static_adapt.adapt_pipeline import (
    BACKEND_COMPILE_SCOPE_SHARED_ALL_PHASES_V1,
)

import run_arm


def _graph_span_route_contract(
    *args: object,
    **kwargs: object,
) -> tuple[str, str, dict[str, object], str]:
    """Replace only the appendix selector-cost realization declaration."""

    request_profile, route_profile, contract, _digest = (
        _ORIGINAL_REPAIRED_ROUTE_CONTRACT(*args, **kwargs)
    )
    updated = copy.deepcopy(contract)
    execution = dict(updated["execution_settings"])
    execution["phase3_backend_cost_scope"] = (
        BACKEND_COMPILE_SCOPE_SHARED_ALL_PHASES_V1
    )
    execution["phase3_hardware_cost_normalization_mode"] = (
        "family_robust_symmetric_arctan_v1"
    )
    execution.pop("ra_marginal_qiskit_cost_all_phases", None)
    updated["execution_settings"] = execution
    lineage = dict(updated.get("lineage_authority", {}))
    changes = list(lineage.get("only_intended_scientific_changes", []))
    changes.append(
        "appendix_selector_cost_qiskit_marginal_to_marrakesh_graph_span_v1"
    )
    lineage["only_intended_scientific_changes"] = changes
    lineage["scientific_result_anchor_claimed"] = False
    updated["lineage_authority"] = lineage
    normalized = json.loads(json.dumps(updated, sort_keys=True))
    return (
        request_profile,
        route_profile,
        normalized,
        ra_engine._route_sha256(normalized),
    )


_ORIGINAL_REPAIRED_ROUTE_CONTRACT = ra_engine._repaired_route_contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--horizon", type=int, default=12)
    args = parser.parse_args()

    output = args.output_root / f"proxy_adaptive_whitened_k{args.horizon}"
    output.mkdir(parents=True, exist_ok=False)

    # The local route wrapper changes only realization of the selector cost;
    # it does not alter the pool, shortlists, response model, refit chart, or
    # final resource compiler.
    ra_engine._repaired_route_contract = _graph_span_route_contract

    started = time.time()
    problem = run_arm.build_ra_adapt_appendix_pure_hubbard_problem(num_sites=3)
    result = run_arm._run_ra(
        problem,
        "baseline",
        output,
        args.horizon,
    )
    payload = result["payload"]
    summary = {
        **result["summary"],
        "diagnostic_status": "non_claim_facing_matched_cost_source_ablation",
        "selector_cost_realization": "marrakesh_graph_span_v1",
        "cost_policy": "marrakesh_graph_span_family_robust_arctan_v1",
        "reported_resource_compiler": (
            "FakeMarrakesh_optimization_level_1_seed_7"
        ),
        "only_intended_runtime_change": (
            "selector_cost_scope_qiskit_marginals_to_shared_graph_span"
        ),
        "wall_seconds": time.time() - started,
    }
    result_path = output / "result.json"
    summary_path = output / "summary.json"
    result_path.write_text(json.dumps(payload, default=str), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["result_sha256"] = hashlib.sha256(result_path.read_bytes()).hexdigest()
    summary["summary_sha256_before_digest_fields"] = hashlib.sha256(
        summary_path.read_bytes()
    ).hexdigest()
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
