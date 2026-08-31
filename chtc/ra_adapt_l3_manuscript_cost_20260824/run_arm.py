"""Run one manuscript-faithful L=3 Paper-I arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import fields, replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import pipelines.static_adapt.engine_support as engine_support
import pipelines.static_adapt.ra_adapt.append as append_module
import pipelines.static_adapt.ra_adapt.contracts as contracts_module
import pipelines.static_adapt.ra_adapt.engine as engine_module
from pipelines.static_adapt.ra_adapt import bundles as bundle_module
from pipelines.hardcoded.adapt_circuit_execution import (
    build_structural_ansatz_circuit,
)
from pipelines.qiskit_backend_tools import (
    compile_circuit_for_backend,
    compiled_gate_stats,
    resolve_backend_targets,
    safe_circuit_depth,
)
from pipelines.static_adapt.ra_adapt.appendix_demos import (
    RAAdaptAppendixPureHubbardCandidateAdapter,
    _observation_policy,
    build_ra_adapt_appendix_pure_hubbard_problem,
    build_ra_adapt_appendix_request,
    ra_adapt_appendix_generator_representation,
)
from pipelines.static_adapt.ra_adapt.contracts import (
    ACTIVE_GRADIENT_MEASURED,
    ACTIVE_GRADIENT_STATIONARY,
    AppendAdaptRequest,
    RA_ADAPT_PROTOCOL_SCHEMA_V1,
    RESOURCE_WEIGHTING_ALL_PHASE,
    ResolvedRAAdaptProtocol,
    _attach_validated_bundle_protocol_authority,
    canonical_sha256,
)
from pipelines.static_adapt.ra_adapt.engine import (
    RA_ADAPT_LEGACY_ALGORITHM_ID,
    RA_ADAPT_LEGACY_ORDINARY_BUNDLE_ID,
    _legacy_ordinary_bundle_digest,
    _repaired_route_contract,
    build_resolved_ra_protocol,
    run_ra_adapt,
)
from pipelines.static_adapt.sr_snake.contracts import (
    FreshStart,
    MetricPruning,
    SRExecutionPolicy,
    SRStopPolicy,
)


EXACT_ENERGY = -2.1414935642714075
XTOL = 1.0e-5
FTOL = 1.0e-12


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _materialize_stationary_joint_protocol(
    problem: object,
    request: object,
    output: Path,
    horizon: int,
) -> ResolvedRAAdaptProtocol:
    """Bind the stationary full-joint diagnostic to its exact sources."""

    source_paths = {
        "runner": Path(__file__).resolve(),
        "engine": Path(engine_module.__file__).resolve(),
        "contracts": Path(contracts_module.__file__).resolve(),
    }
    source_inventory = {
        name: {"path": str(path), "sha256": _sha256_file(path)}
        for name, path in source_paths.items()
    }
    source_locks_payload = {
        "schema": "paper_i_l3_stationary_joint_source_locks_v1",
        "sources": source_inventory,
    }
    source_locks_sha256 = canonical_sha256(source_locks_payload)
    source_lock_id = "paper_i_l3_stationary_joint_exact_sources_v1"
    source_lock_refs = {
        "source_locks_manifest_sha256": source_locks_sha256,
        "cell_source_lock_id": source_lock_id,
        "implementation_source_inventory_sha256": canonical_sha256(
            source_inventory
        ),
    }
    cell = bundle_module.BundleCellSpec(
        cell_id=f"paper_i_l3_stationary_joint_k{int(horizon)}",
        stage="diagnostic",
        regime_id="pure_hubbard_l3_u1p25",
        nph=0,
        route_id="ra_appendix_pure_hubbard_stationary_joint",
        algorithm_id=RA_ADAPT_LEGACY_ALGORITHM_ID,
        selector_family="ra_adapt",
        candidate_representation=str(
            request.adapter.candidate_representation_id
        ),
        horizon=int(horizon),
        source_lock_id=source_lock_id,
    )
    bundle_manifest_payload = {
        "schema": "paper_i_l3_stationary_joint_bundle_v1",
        "cell": cell.to_dict(),
        "active_gradient_policy": ACTIVE_GRADIENT_STATIONARY,
        "phase3_candidate_gain_policy": "joint_total_gain_v1",
        "resource_weighting_scope": RESOURCE_WEIGHTING_ALL_PHASE,
        "source_locks_sha256": source_locks_sha256,
    }
    bundle_id = "paper_i_l3_stationary_joint_diagnostic_v1"
    bundle_manifest_sha256 = canonical_sha256(bundle_manifest_payload)
    receipt = contracts_module.bundle_protocol_materialization_receipt(
        bundle_id=bundle_id,
        bundle_manifest_sha256=bundle_manifest_sha256,
        source_locks_sha256=source_locks_sha256,
        source_lock_refs=source_lock_refs,
        cell_id=cell.cell_id,
        source_lock_id=source_lock_id,
        protocol_schema=RA_ADAPT_PROTOCOL_SCHEMA_V1,
        algorithm_id=RA_ADAPT_LEGACY_ALGORITHM_ID,
        candidate_representation=str(
            request.adapter.candidate_representation_id
        ),
        selector_identity=contracts_module.RA_STAGED_SELECTOR_ID,
        active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
        resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
    )
    resolved = build_resolved_ra_protocol(problem, request)
    protocol_payload = {
        item.name: getattr(resolved, item.name)
        for item in fields(ResolvedRAAdaptProtocol)
        if item.name != "sha256" and item.metadata.get("canonical", True)
    }
    lineage = dict(resolved.lineage_authority)
    lineage.pop("algorithm_semantics", None)
    (_request_profile, _route_profile, route_contract, route_digest) = (
        _repaired_route_contract(
            request,
            active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
            resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
            algorithm_id=RA_ADAPT_LEGACY_ALGORITHM_ID,
            problem=problem,
            sheet_version=1,
        )
    )
    protocol_payload.update(
        {
            "schema": RA_ADAPT_PROTOCOL_SCHEMA_V1,
            "algorithm_id": RA_ADAPT_LEGACY_ALGORITHM_ID,
            "active_gradient_policy": ACTIVE_GRADIENT_STATIONARY,
            "bundle_id": bundle_id,
            "bundle_manifest_sha256": bundle_manifest_sha256,
            "source_locks": source_lock_refs,
            "lineage_authority": lineage,
            "route_contract": {
                **route_contract,
                "sha256": route_digest,
            },
            "bundle_materialization": receipt,
        }
    )
    protocol = ResolvedRAAdaptProtocol(
        **protocol_payload,
        sha256=canonical_sha256(
            {
                key: value
                for key, value in protocol_payload.items()
                if value is not None
            }
        ),
    )
    bound_authority = contracts_module._mint_bundle_protocol_materialization_authority(
        receipt,
        source_lock_refs=source_lock_refs,
        protocol_sha256=protocol.sha256,
    )
    materialization = {
        **bundle_manifest_payload,
        "bundle_id": bundle_id,
        "bundle_manifest_sha256": bundle_manifest_sha256,
        "source_locks": source_locks_payload,
        "source_lock_refs": source_lock_refs,
        "protocol_sha256": protocol.sha256,
    }
    (output / "stationary_materialization.json").write_text(
        json.dumps(materialization, indent=2)
    )
    return _attach_validated_bundle_protocol_authority(
        protocol,
        bound_authority,
    )


def _compiled_tuple(problem: object, labels: list[str]) -> tuple[int, int, int]:
    adapter = RAAdaptAppendixPureHubbardCandidateAdapter(
        candidate_representation_id=ra_adapt_appendix_generator_representation(3)
    )
    pool = {
        str(candidate.label): candidate
        for candidate in adapter.executable_pool(problem).candidates
    }
    reference = np.asarray(problem.reference_state.build_state(), dtype=complex)
    num_qubits = int(round(math.log2(reference.size)))
    _logical, circuit = build_structural_ansatz_circuit(
        [pool[label].term for label in labels],
        nq=num_qubits,
        ref_state=reference,
        structure_theta_value=1.0,
    )
    targets, _audit = resolve_backend_targets(
        requested_names=("FakeMarrakesh",),
        preferred_fake_backends=("FakeMarrakesh",),
        allow_preferred_fallback=False,
        fallback_mode="single",
        allow_runtime_lookup=False,
    )
    compiled = compile_circuit_for_backend(
        circuit,
        targets[0].backend_obj,
        seed_transpiler=7,
        optimization_level=1,
        initial_layout=None,
    )["compiled"]
    stats = compiled_gate_stats(compiled)
    return (
        int(stats["compiled_count_2q"]),
        int(stats["compiled_depth_2q"]),
        int(safe_circuit_depth(compiled)),
    )


def _run_avqe(problem: object, output: Path, horizon: int) -> dict[str, object]:
    append_module.AAVQE_COMPARATOR_POWELL_XTOL = XTOL
    append_module.AAVQE_COMPARATOR_POWELL_FTOL = FTOL
    request = AppendAdaptRequest(
        adapter=RAAdaptAppendixPureHubbardCandidateAdapter(
            candidate_representation_id=ra_adapt_appendix_generator_representation(3)
        ),
        execution=SRExecutionPolicy(
            stop=SRStopPolicy(maximum_controller_rounds=horizon),
            resume=FreshStart(),
        ),
        observation=_observation_policy(output, horizon=horizon),
    )
    payload = append_module.run_append_adapt(problem, request).to_dict()
    result = payload["result_payload"]
    resources = result["compiled_resources"]
    return {
        "payload": payload,
        "summary": {
            "arm": "avqe",
            "horizon": horizon,
            "S_alg": int(result["estimator_accounting"]["S_alg"]),
            "N2q": int(resources["compiled_count_2q_total"]),
            "D2q": int(resources["compiled_depth_2q_total"]),
            "Dc": int(resources["compiled_depth_total"]),
        },
    }


def _run_ra(
    problem: object,
    arm: str,
    output: Path,
    horizon: int,
    *,
    prune_theta: float | None = None,
    prune_protect: int | None = None,
    prune_max_regression: float | None = None,
    prune_retained_gain_ratio: float | None = None,
    prune_max_predicted_loss_per_cost: float | None = None,
    accepted_refit_whitened: bool = True,
    full_joint_gain: bool = False,
    stationarity_on: bool = False,
) -> dict[str, object]:
    engine_support.RA_ACCEPTED_REFIT_POWELL_XTOL = XTOL
    engine_support.RA_ACCEPTED_REFIT_POWELL_FTOL = FTOL
    request = build_ra_adapt_appendix_request(
        arm=arm,
        accepted_refit_whitened=bool(accepted_refit_whitened),
        adaptive_phase_shortlists=True,
        phase0_gradient_screen=True,
        adaptive_frontier_ratio=0.9,
        num_sites=3,
        maximum_controller_rounds=horizon,
        output_dir=output,
    )
    if any(
        value is not None
        for value in (
            prune_theta,
            prune_protect,
            prune_max_regression,
            prune_retained_gain_ratio,
            prune_max_predicted_loss_per_cost,
        )
    ):
        if not isinstance(request.method.pruning, MetricPruning):
            raise ValueError("Prune overrides require a metric-ablation arm.")
        pruning = replace(
            request.method.pruning,
            small_theta_abs=(
                request.method.pruning.small_theta_abs
                if prune_theta is None
                else float(prune_theta)
            ),
            protect_steps=(
                request.method.pruning.protect_steps
                if prune_protect is None
                else int(prune_protect)
            ),
            max_regression=(
                request.method.pruning.max_regression
                if prune_max_regression is None
                else float(prune_max_regression)
            ),
            retained_gain_ratio=(
                request.method.pruning.retained_gain_ratio
                if prune_retained_gain_ratio is None
                else float(prune_retained_gain_ratio)
            ),
            max_predicted_loss_per_cost=(
                request.method.pruning.max_predicted_loss_per_cost
                if prune_max_predicted_loss_per_cost is None
                else float(prune_max_predicted_loss_per_cost)
            ),
        )
        request = replace(
            request,
            method=replace(request.method, pruning=pruning),
        )
    run_request: object = request
    if full_joint_gain:
        # B9 and the historical Paper-I selector score the full coupled
        # candidate--active quadratic response.  Resolve the otherwise
        # identical request through the preserved v1 authority, whose
        # Phase-III scorer defaults to ``joint_total_gain_v1``.  Rebuilding
        # the authenticated protocol here avoids changing the canonical v2
        # facade or rewriting result metadata after execution.
        if stationarity_on:
            run_request = _materialize_stationary_joint_protocol(
                problem,
                request,
                output,
                horizon,
            )
        else:
            resolved = build_resolved_ra_protocol(problem, request)
            protocol_payload = {
                item.name: getattr(resolved, item.name)
                for item in fields(ResolvedRAAdaptProtocol)
                if item.name != "sha256"
                and item.metadata.get("canonical", True)
            }
            lineage = dict(resolved.lineage_authority)
            lineage.pop("algorithm_semantics", None)
            (
                _request_profile,
                _route_profile,
                route_contract,
                route_digest,
            ) = _repaired_route_contract(
                request,
                active_gradient_policy=ACTIVE_GRADIENT_MEASURED,
                resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
                algorithm_id=RA_ADAPT_LEGACY_ALGORITHM_ID,
                problem=problem,
                sheet_version=1,
            )
            protocol_payload.update(
                {
                    "schema": RA_ADAPT_PROTOCOL_SCHEMA_V1,
                    "algorithm_id": RA_ADAPT_LEGACY_ALGORITHM_ID,
                    "bundle_id": RA_ADAPT_LEGACY_ORDINARY_BUNDLE_ID,
                    "bundle_manifest_sha256": (
                        _legacy_ordinary_bundle_digest()
                    ),
                    "lineage_authority": lineage,
                    "route_contract": {
                        **route_contract,
                        "sha256": route_digest,
                    },
                }
            )
            digest_payload = {
                key: value
                for key, value in protocol_payload.items()
                if value is not None
            }
            run_request = ResolvedRAAdaptProtocol(
                **protocol_payload,
                sha256=canonical_sha256(digest_payload),
            )
    payload = run_ra_adapt(problem, run_request).to_dict()
    trajectory = payload["run"]["accepted_trajectory"]
    ledger = json.loads(next(output.rglob("estimator_ledger.json")).read_text())[
        "accounting"
    ]
    pruning = request.method.pruning
    if isinstance(pruning, MetricPruning):
        # Rejected delete--refit siblings are real executed work even though
        # they are not part of the surviving lineage.
        occurrence_scope = ledger.get("all_branch_search_work") or ledger
        unique_scope = ledger.get("all_branch_unique_primitive_diagnostic") or {}
    else:
        occurrence_scope = (
            ledger.get("winning_lineage")
            or ledger.get("all_branch_search_work")
            or ledger
        )
        unique_scope = (
            ledger.get("winning_lineage_unique_primitive_diagnostic")
            or ledger.get("all_branch_unique_primitive_diagnostic")
            or {}
        )
    s_alg_raw = int(occurrence_scope.get("S_alg", ledger["S_alg"]))
    s_alg = int(
        occurrence_scope.get("N_H_outer", 0)
        + occurrence_scope.get("N_H_refit", 0)
        + unique_scope.get("N_grad", occurrence_scope.get("N_grad", 0))
        + unique_scope.get("N_metric", occurrence_scope.get("N_metric", 0))
    )
    final = trajectory[-1]
    labels = [str(label) for label in final["operators"]]
    n2q, d2q, dc = _compiled_tuple(problem, labels)
    energy = float(final.get("accepted_energy", final.get("energy")))
    prune_events = []
    for transition in payload["run"].get("accepted_transitions", []):
        event = transition.get("pruning")
        if not isinstance(event, dict) or not event.get("trial_executed", False):
            continue
        prune_events.append(
            {
                "round": int(transition["controller_round"]),
                "status": str(event.get("status", "")),
                "accepted": bool(event.get("accepted", False)),
                "reason": str(event.get("reason", "")),
                "deleted_label": event.get("deleted_label"),
                "trial_s_alg": int(event.get("trial_s_alg", 0) or 0),
                "measured_energy_before": event.get("measured_energy_before"),
                "measured_energy_after": event.get("measured_energy_after"),
            }
        )
    return {
        "payload": payload,
        "summary": {
            "arm": arm,
            "horizon": horizon,
            "accepted_rounds": len(trajectory),
            "error": abs(energy - EXACT_ENERGY),
            "S_alg": s_alg,
            "S_alg_raw": s_alg_raw,
            "S_alg_corrected": s_alg,
            "S_alg_removed_staged_reuse": s_alg_raw - s_alg,
            "N2q": n2q,
            "D2q": d2q,
            "Dc": dc,
            "accepted_refit_chart": (
                "fubini_study_whitened"
                if accepted_refit_whitened
                else "native_with_metric_informed_directions"
            ),
            "phase0_cap": 24,
            "adaptive_frontier_ratio": 0.9,
            "cost_policy": "compiled_marginal_median_mad_signed_arctan_v1",
            "phase3_candidate_gain_policy": (
                "joint_total_gain_v1"
                if full_joint_gain
                else "joint_minus_active_only_supported_trust_v1"
            ),
            "active_gradient_policy": (
                ACTIVE_GRADIENT_STATIONARY
                if stationarity_on
                else ACTIVE_GRADIENT_MEASURED
            ),
            "prune_controls": (
                {
                    "small_theta_abs": pruning.small_theta_abs,
                    "protect_steps": pruning.protect_steps,
                    "max_regression": pruning.max_regression,
                    "retained_gain_ratio": pruning.retained_gain_ratio,
                    "max_predicted_loss_per_cost": (
                        pruning.max_predicted_loss_per_cost
                    ),
                }
                if isinstance(pruning, MetricPruning)
                else None
            ),
            "prune_events": prune_events,
            "first_prune_trial_round": (
                prune_events[0]["round"] if prune_events else None
            ),
            "first_accepted_deletion_round": next(
                (
                    event["round"]
                    for event in prune_events
                    if event["accepted"]
                ),
                None,
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("arm")
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--variant-label")
    parser.add_argument("--prune-theta", type=float)
    parser.add_argument("--prune-protect", type=int)
    parser.add_argument("--prune-max-regression", type=float)
    parser.add_argument("--prune-retained-gain-ratio", type=float)
    parser.add_argument(
        "--prune-max-predicted-loss-per-cost", type=float
    )
    parser.add_argument("--native-refit", action="store_true")
    parser.add_argument("--full-joint-gain", action="store_true")
    parser.add_argument("--stationarity-on", action="store_true")
    args = parser.parse_args()
    if args.stationarity_on and not args.full_joint_gain:
        parser.error("--stationarity-on requires --full-joint-gain")
    if args.stationarity_on and args.arm == "avqe":
        parser.error("--stationarity-on applies only to RA Phase III")

    label = args.variant_label or args.arm
    output = args.output_root / f"{label}_k{args.horizon}"
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    problem = build_ra_adapt_appendix_pure_hubbard_problem(num_sites=3)
    result = (
        _run_avqe(problem, output, args.horizon)
        if args.arm == "avqe"
        else _run_ra(
            problem,
            args.arm,
            output,
            args.horizon,
            prune_theta=args.prune_theta,
            prune_protect=args.prune_protect,
            prune_max_regression=args.prune_max_regression,
            prune_retained_gain_ratio=args.prune_retained_gain_ratio,
            prune_max_predicted_loss_per_cost=(
                args.prune_max_predicted_loss_per_cost
            ),
            accepted_refit_whitened=not args.native_refit,
            full_joint_gain=bool(args.full_joint_gain),
            stationarity_on=bool(args.stationarity_on),
        )
    )
    (output / "result.json").write_text(
        json.dumps(result["payload"], default=str)
    )
    summary = {**result["summary"], "wall_seconds": time.time() - started}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
