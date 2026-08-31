"""Run one exploratory matched pure-Hubbard L=4 route arm.

This wrapper extends the source-locked L=3 diagnostic runner to the half-filled
open L=4 chain without changing the library's published L=2/L=3 application
surface.  It derives and records an in-process problem/pool lock before running
the unchanged AVQE or RA implementation.  Results are exploratory until L=4 is
promoted to a first-class named application and rerun from that public route.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import inspect
import json
import os
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
L3_RUNNER = REPO / "chtc/ra_adapt_l3_manuscript_cost_20260824/run_arm.py"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pipelines.contracts.problem import ProblemRequest
from pipelines.scaffold.hh_continuation_generators import (
    serialize_polynomial_terms_exyz,
)
from pipelines.static_adapt.builders.problem_registry import (
    resolve_problem_context,
)
from pipelines.static_adapt.ra_adapt.contracts import (
    CANDIDATE_REPRESENTATION_MACRO,
    canonical_sha256,
)
from pipelines.static_adapt.sr_snake.contracts import ResolvedProblemReceipt
import pipelines.static_adapt.ra_adapt.appendix_demos as demos
import pipelines.static_adapt.ra_adapt.accepted_run_evidence as accepted_evidence
import pipelines.static_adapt.ra_adapt.engine as ra_engine
import pipelines.static_adapt.ra_adapt.pools as pools
from pipelines.static_adapt.adapt_pipeline import (
    BACKEND_COMPILE_SCOPE_SHARED_ALL_PHASES_V1,
)
from pipelines.static_adapt.sr_snake.contracts import AcceptedStateResume


NUM_SITES = 4
SECTOR = (2, 2)
EXACT_ENERGY = -3.3827681604170525


def _load_l3_runner():
    spec = importlib.util.spec_from_file_location("paper_i_l3_run_arm", L3_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {L3_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require_l4(problem: object) -> None:
    request = problem.request
    particles = tuple(int(value) for value in problem.sector.num_particles)
    if (
        str(problem.family_key).strip().lower() != "hubbard"
        or str(request.problem_key).strip().lower() != "hubbard"
        or int(request.num_sites) != NUM_SITES
        or float(request.t) != 1.0
        or float(request.u) != 1.25
        or float(request.dv) != 0.0
        or float(request.v_nn) != 0.0
        or float(request.t_prime) != 0.0
        or float(request.g_ep) != 0.0
        or float(request.omega0) != 0.0
        or int(request.n_ph_max) != 0
        or str(request.ordering).strip().lower() != "blocked"
        or str(request.boundary).strip().lower() != "open"
        or particles != SECTOR
        or int(problem.layout.total_qubits) != 8
        or int(problem.layout.fermion_qubits) != 8
        or int(problem.layout.boson_qubits) != 0
        or "full_meta" not in set(problem.admissible_pool_keys)
    ):
        raise ValueError("Exploratory L=4 Hubbard problem drifted from its contract.")


def _build_problem():
    problem = resolve_problem_context(
        ProblemRequest(
            problem_key="hubbard",
            num_sites=NUM_SITES,
            t=1.0,
            u=1.25,
            dv=0.0,
            omega0=0.0,
            g_ep=0.0,
            n_ph_max=0,
            boson_encoding="binary",
            ordering="blocked",
            boundary="open",
            include_zero_point=False,
            v_nn=0.0,
            t_prime=0.0,
            n_fermions=NUM_SITES,
        )
    )
    _require_l4(problem)
    observed = float(problem.exact_target.resolve_energy())
    if abs(observed - EXACT_ENERGY) > 5.0e-13:
        raise RuntimeError(f"L=4 exact energy drifted: {observed} != {EXACT_ENERGY}")
    return problem


def _install_exploratory_lock(problem: object) -> dict[str, object]:
    # Patch only this process.  The public application remains L=2/L=3.
    pools.require_ra_adapt_appendix_pure_hubbard_problem = _require_l4
    demos.require_ra_adapt_appendix_pure_hubbard_problem = _require_l4
    # L=4 needs a longer diagnostic horizon than the published L=3 control.
    # This remains process-local and does not widen the public application.
    demos.RA_ADAPT_APPENDIX_MAXIMUM_HORIZON = 50
    demos.RA_ADAPT_APPENDIX_SITE_GENERATOR_REPRESENTATION[NUM_SITES] = (
        CANDIDATE_REPRESENTATION_MACRO
    )

    parent = pools.build_ra_adapt_appendix_pure_hubbard_parent_template_inventory(
        problem,
        representation_id=CANDIDATE_REPRESENTATION_MACRO,
    )
    executable = pools.build_ra_adapt_appendix_pure_hubbard_sector_guarded_macro_pool(
        problem
    )
    receipt = ResolvedProblemReceipt.from_problem(problem)
    lock = {
        "problem_request_sha256": receipt.problem_request_sha256,
        "hamiltonian_terms_sha256": canonical_sha256(
            serialize_polynomial_terms_exyz(problem.hamiltonian)
        ),
        "exact_energy": EXACT_ENERGY,
        "sector_num_particles": SECTOR,
        "parent_count": parent.receipt.count,
        "parent_labels_sha256": parent.receipt.ordered_labels_sha256,
        "parent_pool_sha256": parent.receipt.ordered_pool_sha256,
        "macro_count": executable.receipt.count,
        "macro_labels_sha256": executable.receipt.ordered_labels_sha256,
        "macro_pool_sha256": executable.receipt.ordered_pool_sha256,
    }
    demos.RA_ADAPT_APPENDIX_PURE_HUBBARD_LOCKS[NUM_SITES] = lock
    return lock


def _install_l4_runner_overrides(
    run_arm: object,
    *,
    resume: AcceptedStateResume | None = None,
) -> None:
    original_request_builder = run_arm.build_ra_adapt_appendix_request
    original_append_problem_guard = run_arm.append_module._require_paper_i_problem
    original_named_appendix_check = (
        run_arm.append_module._is_named_appendix_pure_hubbard_l3_application
    )

    def l4_request_builder(*args: object, **kwargs: object):
        kwargs["num_sites"] = NUM_SITES
        if resume is not None:
            kwargs["resume"] = resume
        return original_request_builder(*args, **kwargs)

    run_arm.build_ra_adapt_appendix_request = l4_request_builder
    run_arm.ra_adapt_appendix_generator_representation = (
        lambda _sites: CANDIDATE_REPRESENTATION_MACRO
    )
    def l4_append_problem_guard(problem: object, request: object | None = None) -> None:
        if int(problem.request.num_sites) == NUM_SITES:
            _require_l4(problem)
            return
        original_append_problem_guard(problem, request)

    def l4_named_appendix_check(problem: object, request: object) -> bool:
        if int(problem.request.num_sites) == NUM_SITES:
            if not isinstance(
                request.adapter,
                demos.RAAdaptAppendixPureHubbardCandidateAdapter,
            ):
                return False
            demos._require_appendix_problem_lock(problem)
            return True
        return bool(original_named_appendix_check(problem, request))

    run_arm.append_module._require_paper_i_problem = l4_append_problem_guard
    run_arm.append_module._is_named_appendix_pure_hubbard_l3_application = (
        l4_named_appendix_check
    )
    # The engine's first guard enumerates the currently public lattice sizes
    # before it validates the named application.  Widen only that enumeration
    # in this process; all subsequent named-application and source-lock checks
    # remain active.
    source = inspect.getsource(ra_engine.run_ra_adapt)
    old = "and int(problem.request.num_sites) in {2, 3}"
    new = "and int(problem.request.num_sites) in {2, 3, 4}"
    if source.count(old) != 1:
        raise RuntimeError("Could not locate the L=2/L=3 engine admission guard.")
    namespace = dict(ra_engine.__dict__)
    exec(compile(source.replace(old, new), str(ra_engine.__file__), "exec"), namespace)
    run_arm.run_ra_adapt = namespace["run_ra_adapt"]

    # Preserve the stationary bundle's source-locked construction while
    # giving the exploratory cell its true lattice-size identity.
    stationary_source = inspect.getsource(
        run_arm._materialize_stationary_joint_protocol
    )
    stationary_source = stationary_source.replace("l3", "l4").replace("L3", "L4")
    stationary_namespace = dict(run_arm.__dict__)
    stationary_namespace["__file__"] = __file__
    exec(
        compile(stationary_source, __file__, "exec"),
        stationary_namespace,
    )
    run_arm._materialize_stationary_joint_protocol = stationary_namespace[
        "_materialize_stationary_joint_protocol"
    ]

    original_route_contract = ra_engine._repaired_route_contract

    def l4_route_contract(*args: object, **kwargs: object):
        request_profile, route_profile, contract, _digest = original_route_contract(
            *args, **kwargs
        )
        updated = copy.deepcopy(contract)
        lineage = dict(updated.get("lineage_authority", {}))
        changes = list(lineage.get("only_intended_scientific_changes", []))
        changes.append("exploratory_l4_half_filled_hubbard_extension_v1")
        lineage["only_intended_scientific_changes"] = list(dict.fromkeys(changes))
        lineage["scientific_result_anchor_claimed"] = False
        updated["lineage_authority"] = lineage
        normalized = json.loads(json.dumps(updated, sort_keys=True))
        return (
            request_profile,
            route_profile,
            normalized,
            ra_engine._route_sha256(normalized),
        )

    ra_engine._repaired_route_contract = l4_route_contract
    run_arm._repaired_route_contract = l4_route_contract
    run_arm.run_ra_adapt.__globals__["_repaired_route_contract"] = (
        l4_route_contract
    )
    run_arm._materialize_stationary_joint_protocol.__globals__[
        "_repaired_route_contract"
    ] = l4_route_contract
    run_arm.EXACT_ENERGY = EXACT_ENERGY


def _install_graph_span_selector(run_arm: object) -> None:
    original = ra_engine._repaired_route_contract

    def graph_span_contract(*args: object, **kwargs: object):
        request_profile, route_profile, contract, _digest = original(
            *args, **kwargs
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
        changes.extend(
            [
                "exploratory_l4_half_filled_hubbard_extension_v1",
                "selector_cost_qiskit_marginal_to_marrakesh_graph_span_v1",
            ]
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

    ra_engine._repaired_route_contract = graph_span_contract
    run_arm._repaired_route_contract = graph_span_contract
    run_arm.run_ra_adapt.__globals__["_repaired_route_contract"] = (
        graph_span_contract
    )
    run_arm._materialize_stationary_joint_protocol.__globals__[
        "_repaired_route_contract"
    ] = graph_span_contract


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _install_route_debug_probe() -> None:
    original = accepted_evidence._project_payload

    def debug_project_payload(*args: object, **kwargs: object):
        protocol = kwargs.get("protocol")
        run = kwargs.get("run")
        protocol_route = getattr(protocol, "route_contract", None)
        invariants = (
            protocol_route.get("semantic_invariants")
            if isinstance(protocol_route, dict)
            else None
        )
        print(
            "L4_ROUTE_DEBUG "
            + json.dumps(
                {
                    "application_lane": (
                        invariants.get("application_lane")
                        if isinstance(invariants, dict)
                        else None
                    ),
                    "protocol_route_sha256": (
                        protocol_route.get("sha256")
                        if isinstance(protocol_route, dict)
                        else None
                    ),
                    "run_route_sha256": str(
                        getattr(getattr(run, "route", None), "contract_sha256", "")
                    ),
                    "allowed_lanes": sorted(accepted_evidence._APPENDIX_APPLICATION_LANES),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return original(*args, **kwargs)

    accepted_evidence._project_payload = debug_project_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "arm",
        choices=(
            "avqe",
            "baseline",
            "batching",
            "beam_escalated",
            "metric_ablation_gated",
        ),
    )
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--selector-cost", choices=("qiskit", "proxy"), default="qiskit")
    parser.add_argument("--refit-chart", choices=("native", "whitened"), default="native")
    parser.add_argument("--active-residual", choices=("measured", "stationary"), default="stationary")
    parser.add_argument("--powell-xtol", type=float, default=1.0e-5)
    parser.add_argument("--powell-ftol", type=float, default=1.0e-12)
    parser.add_argument("--prune-theta", type=float)
    parser.add_argument("--prune-protect", type=int)
    parser.add_argument("--prune-max-regression", type=float)
    parser.add_argument("--prune-retained-gain-ratio", type=float)
    parser.add_argument("--prune-max-predicted-loss-per-cost", type=float)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--resume-sha256")
    parser.add_argument("--variant-label")
    args = parser.parse_args()
    if args.arm == "avqe" and (
        args.selector_cost != "qiskit"
        or args.refit_chart != "native"
        or args.active_residual != "stationary"
    ):
        parser.error("AVQE has no RA selector-cost, whitening, or residual axes.")
    if (args.resume_checkpoint is None) != (args.resume_sha256 is None):
        parser.error("--resume-checkpoint and --resume-sha256 must be supplied together.")
    if args.arm == "avqe" and args.resume_checkpoint is not None:
        parser.error("This exploratory wrapper supports authenticated resume only for RA.")

    label = args.variant_label or (
        "avqe"
        if args.arm == "avqe"
        else f"ra_{args.selector_cost}_{args.refit_chart}_{args.active_residual}"
    )
    output = args.output_root / f"{label}_k{args.horizon}"
    output.mkdir(parents=True, exist_ok=False)

    problem = _build_problem()
    lock = _install_exploratory_lock(problem)
    run_arm = _load_l3_runner()
    run_arm.XTOL = float(args.powell_xtol)
    run_arm.FTOL = float(args.powell_ftol)
    resume = (
        None
        if args.resume_checkpoint is None
        else AcceptedStateResume(
            checkpoint_path=args.resume_checkpoint.resolve(),
            checkpoint_sha256=str(args.resume_sha256),
        )
    )
    _install_l4_runner_overrides(run_arm, resume=resume)
    if args.selector_cost == "proxy":
        _install_graph_span_selector(run_arm)
    if os.environ.get("PAPER_I_L4_DEBUG_ROUTE") == "1":
        _install_route_debug_probe()

    started = time.time()
    if args.arm == "avqe":
        result = run_arm._run_avqe(problem, output, args.horizon)
    else:
        result = run_arm._run_ra(
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
            accepted_refit_whitened=args.refit_chart == "whitened",
            full_joint_gain=True,
            stationarity_on=args.active_residual == "stationary",
        )

    result_path = output / "result.json"
    _write_json(result_path, result["payload"])
    summary = {
        **result["summary"],
        "schema": "paper_i_exploratory_l4_route_matrix_summary_v1",
        "evidence_status": "exploratory_requires_first_class_l4_promotion",
        "num_sites": NUM_SITES,
        "sector_num_particles": list(SECTOR),
        "exact_energy": EXACT_ENERGY,
        "selector_cost": None if args.arm == "avqe" else args.selector_cost,
        "cost_policy": (
            None
            if args.arm == "avqe"
            else "marrakesh_graph_span_family_robust_arctan_v1"
            if args.selector_cost == "proxy"
            else "compiled_marginal_median_mad_signed_arctan_v1"
        ),
        "refit_chart_shorthand": (
            None if args.arm == "avqe" else args.refit_chart
        ),
        "refit_parameterization": (
            None
            if args.arm == "avqe"
            else "native_coordinates_with_fs_whitened_powell_directions_v1"
            if args.refit_chart == "native"
            else "fixed_fs_whitened_affine_coordinate_chart_v1"
        ),
        "active_residual": None if args.arm == "avqe" else args.active_residual,
        "powell_tolerances": {
            "xtol": float(args.powell_xtol),
            "ftol": float(args.powell_ftol),
        },
        "resume": (
            None
            if resume is None
            else {
                "checkpoint_path": str(resume.checkpoint_path),
                "checkpoint_sha256": resume.checkpoint_sha256,
            }
        ),
        "l4_problem_pool_lock": lock,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "l3_runner_source": str(L3_RUNNER),
        "l3_runner_sha256": hashlib.sha256(L3_RUNNER.read_bytes()).hexdigest(),
        "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "wall_seconds": time.time() - started,
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
