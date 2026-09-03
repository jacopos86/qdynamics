"""Checkpointed local launcher for the production H2O RA-ADAPT lane."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

from pipelines.contracts.problem import ProblemRequest
from pipelines.static_adapt.builders.problem_registry import (
    resolve_problem_context,
)
from pipelines.static_adapt.ra_adapt.adapters import (
    H2OLinearFDSectorCompletePauliBlockCandidateAdapter,
    H2OLinearFDSymmetryCompleteCandidateAdapter,
)
from pipelines.static_adapt.ra_adapt.contracts import (
    CANDIDATE_REPRESENTATION_MACRO,
    CANDIDATE_REPRESENTATION_SINGLE_PAULI,
    RAAdaptOperationalControls,
    RAAdaptRequest,
)
from pipelines.static_adapt.ra_adapt.engine import (
    build_resolved_ra_protocol,
    run_ra_adapt,
)
from pipelines.static_adapt.sr_snake.contracts import (
    AcceptedStateResume,
    AppendOnlyInsertion,
    BeamOff,
    CheckpointObservation,
    EstimatorLedgerObservation,
    ExactEDSourceReceipt,
    ExactEDStop,
    FreshStart,
    MARRAKESH_GRAPH_SPAN_COMPILE_COST_POLICY_V1,
    MarrakeshGraphSpanCompileCost,
    PlateauCommutationInsertion,
    PruningOff,
    SRExecutionPolicy,
    SRMethodPolicy,
    SRObservationPolicy,
    SRStopPolicy,
    SingletonAdmission,
)
from src.quantum.chemistry.vibronic_h2o_linear_fd import (
    load_cached_production_vibronic_h2o_linear_fd_fixture,
)


H2O_INSERTION_PLATEAU = "plateau_commutation"
H2O_INSERTION_APPEND_ONLY = "append_only"
_H2O_INSERTION_POLICIES = {
    H2O_INSERTION_PLATEAU: PlateauCommutationInsertion,
    H2O_INSERTION_APPEND_ONLY: AppendOnlyInsertion,
}


def _insertion_policy(name: str):
    try:
        return _H2O_INSERTION_POLICIES[str(name)]()
    except KeyError:
        raise ValueError(
            f"Unknown H2O insertion policy {name!r}; expected one of "
            f"{sorted(_H2O_INSERTION_POLICIES)}."
        ) from None


H2O_COST_MODE_PROXY = "proxy"
H2O_COST_MODE_MARRAKESH_GRAPH_SPAN = (
    MARRAKESH_GRAPH_SPAN_COMPILE_COST_POLICY_V1
)
_H2O_COST_MODES = (
    H2O_COST_MODE_PROXY,
    H2O_COST_MODE_MARRAKESH_GRAPH_SPAN,
)


def _compile_cost_policy(cost_mode: str):
    """Map the CLI cost mode onto the typed selector compile-cost policy.

    ``proxy`` is the H2O route contract's own structural proxy and maps to
    no policy at all (``compile_cost=None``), so the default request and its
    protocol digest are unchanged.  ``marrakesh_graph_span_v1`` opts into
    the Paper-I analytic FakeMarrakesh graph-span cost on the fixed 19-qubit
    embedding and yields a distinct route profile and digest.
    """

    mode = str(cost_mode).strip()
    if mode == H2O_COST_MODE_PROXY:
        return None
    if mode == H2O_COST_MODE_MARRAKESH_GRAPH_SPAN:
        return MarrakeshGraphSpanCompileCost()
    raise ValueError(
        f"Unknown H2O cost mode {cost_mode!r}; expected one of "
        f"{list(_H2O_COST_MODES)}."
    )


DEFAULT_GRADIENT_TOLERANCE = 5.0e-7
DEFAULT_CHEMICAL_ACCURACY_HARTREE = 1.6e-3
H2O_CANDIDATE_MODE_SECTOR_COMPLETE_PAULI_BLOCK = (
    "sector_complete_pauli_block_v1"
)


def _candidate_adapter(candidate_representation: str) -> Any:
    representation = str(candidate_representation).strip()
    if representation == H2O_CANDIDATE_MODE_SECTOR_COMPLETE_PAULI_BLOCK:
        return H2OLinearFDSectorCompletePauliBlockCandidateAdapter()
    if representation == CANDIDATE_REPRESENTATION_SINGLE_PAULI:
        raise ValueError(
            "Raw H2O single-Pauli children are not sector-complete. Use "
            f"{H2O_CANDIDATE_MODE_SECTOR_COMPLETE_PAULI_BLOCK!r}."
        )
    if representation == CANDIDATE_REPRESENTATION_MACRO:
        return H2OLinearFDSymmetryCompleteCandidateAdapter()
    raise ValueError(
        "Unsupported H2O candidate representation "
        f"{candidate_representation!r}."
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _problem(fixture: Path) -> Any:
    cached = load_cached_production_vibronic_h2o_linear_fd_fixture(fixture)
    model = cached.model
    return resolve_problem_context(
        ProblemRequest(
            problem_key="molecular_vibronic_h2o_linear_fd",
            num_sites=int(model.n_spatial_orbitals),
            t=1.0,
            u=0.0,
            dv=0.0,
            omega0=1.0,
            g_ep=1.0,
            n_ph_max=max(int(value) for value in model.mode_cutoffs),
            boson_encoding="binary",
            ordering="blocked",
            boundary="open",
            include_zero_point=True,
            molecular_vibronic_h2o_linear_fd_fixture_json=str(fixture),
        )
    )


def build_h2o_ra_request(
    *,
    problem: Any,
    output_dir: Path,
    maximum_controller_rounds: int,
    gradient_tolerance: float,
    exact_absolute_tolerance: float,
    exact_confirmation_controller_rounds: int = 0,
    candidate_representation: str = (
        H2O_CANDIDATE_MODE_SECTOR_COMPLETE_PAULI_BLOCK
    ),
    insertion: str = H2O_INSERTION_PLATEAU,
    checkpoint_history_tail: int = 100,
    estimator_ledger: bool = True,
    cost_mode: str = H2O_COST_MODE_PROXY,
) -> RAAdaptRequest:
    exact_energy = float(problem.exact_target.resolve_energy())
    exact_source = ExactEDSourceReceipt.from_problem(
        problem,
        source_id="paper_iv_h2o_linear_fd_same_cutoff_sector_ed_v1",
    )
    return RAAdaptRequest(
        adapter=_candidate_adapter(candidate_representation),
        method=SRMethodPolicy(
            admission=SingletonAdmission(),
            insertion=_insertion_policy(insertion),
            pruning=PruningOff(),
            beam=BeamOff(),
            compile_cost=_compile_cost_policy(cost_mode),
        ),
        execution=SRExecutionPolicy(
            stop=SRStopPolicy(
                maximum_controller_rounds=maximum_controller_rounds,
                exact_ed_target=ExactEDStop(
                    energy=exact_energy,
                    absolute_tolerance=exact_absolute_tolerance,
                    source=exact_source,
                    confirmation_controller_rounds=(
                        exact_confirmation_controller_rounds
                    ),
                ),
                gradient_tolerance=gradient_tolerance,
            )
        ),
        observation=SRObservationPolicy(
            checkpoint=CheckpointObservation(
                path=output_dir / "current.json",
                every_controller_rounds=1,
                keep_history_tail=int(checkpoint_history_tail),
            ),
            estimator_ledger=(
                EstimatorLedgerObservation(
                    path=output_dir / "estimator_call_ledger.json"
                )
                if estimator_ledger
                else None
            ),
        ),
    )


def _engine_wall_budget_seconds(
    max_wall_seconds: float | None,
    *,
    startup_seconds: float,
) -> float | None:
    """Translate the CLI wall budget into the engine's run-entry budget.

    The CLI budget is measured from ``main()`` entry.  Fixture loading,
    protocol resolution, and manifest writing happen before the engine's
    own clock starts at ``run_ra_adapt`` entry, so that startup time is
    deducted here (floored at zero: the engine still completes the round
    in flight, so a zero budget means "stop after the first round").
    """

    if max_wall_seconds is None:
        return None
    return max(0.0, float(max_wall_seconds) - float(startup_seconds))


def main() -> None:
    main_entry_monotonic = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-controller-rounds", type=int, default=50)
    parser.add_argument(
        "--gradient-tolerance",
        type=float,
        default=DEFAULT_GRADIENT_TOLERANCE,
    )
    parser.add_argument(
        "--exact-absolute-tolerance",
        type=float,
        default=DEFAULT_CHEMICAL_ACCURACY_HARTREE,
    )
    parser.add_argument(
        "--exact-confirmation-controller-rounds",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--candidate-representation",
        choices=(
            H2O_CANDIDATE_MODE_SECTOR_COMPLETE_PAULI_BLOCK,
            CANDIDATE_REPRESENTATION_SINGLE_PAULI,
            CANDIDATE_REPRESENTATION_MACRO,
        ),
        default=H2O_CANDIDATE_MODE_SECTOR_COMPLETE_PAULI_BLOCK,
    )
    parser.add_argument(
        "--insertion",
        choices=tuple(_H2O_INSERTION_POLICIES),
        default=H2O_INSERTION_PLATEAU,
        help="Insertion policy; the H2O route contract declares append_only.",
    )
    parser.add_argument(
        "--checkpoint-history-tail",
        type=int,
        default=100,
        help=(
            "Controller rounds of checkpoint history retained. 1 keeps "
            "resumability while dropping accumulated past records."
        ),
    )
    parser.add_argument(
        "--no-estimator-ledger",
        action="store_true",
        help="Disable estimator-call ledger accounting.",
    )
    parser.add_argument(
        "--cost-mode",
        choices=_H2O_COST_MODES,
        default=H2O_COST_MODE_PROXY,
        help=(
            "Selector compile-cost source. 'proxy' (default) is the H2O "
            "route contract's structural proxy. 'marrakesh_graph_span_v1' "
            "opts into the Paper-I analytic FakeMarrakesh graph-span cost "
            "on the fixed 19-qubit embedding (no transpile); it is a "
            "distinct route profile and protocol digest."
        ),
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help=(
            "Resume from a prior segment's accepted-state checkpoint "
            "(a current.json written by an earlier run). Its digest-matched "
            "estimator-ledger sidecar must sit beside it. Point "
            "--output-dir at a fresh directory: the resumed run writes its "
            "own current.json and would otherwise overwrite the source."
        ),
    )
    parser.add_argument(
        "--max-wall-seconds",
        type=float,
        default=None,
        help=(
            "Operational wall-clock budget for this segment, measured from "
            "CLI entry (interpreter start-up and imports excluded). It is "
            "checked only at controller-round boundaries, after each "
            "completed round has been checkpointed, so the round in flight "
            "always finishes and the segment makes at least one round of "
            "progress; when the budget is spent the controller stops with "
            "reason wall_clock_budget_v1, writes result.json and "
            "summary.json, and exits 0 so the outputs transfer. Size it as "
            "(pool wall) - (longest round) - margin. Default: no budget. "
            "The resolved protocol and protocol_sha256 are unaffected."
        ),
    )
    args = parser.parse_args()
    if args.max_wall_seconds is not None and not (
        math.isfinite(args.max_wall_seconds) and args.max_wall_seconds >= 0.0
    ):
        parser.error(
            "--max-wall-seconds must be a nonnegative finite number."
        )

    fixture = args.fixture.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not fixture.is_file():
        raise FileNotFoundError(f"H2O fixture does not exist: {fixture}")
    resume_checkpoint: Path | None = None
    if args.resume_checkpoint is not None:
        resume_checkpoint = args.resume_checkpoint.expanduser().resolve()
        if not resume_checkpoint.is_file():
            raise FileNotFoundError(
                f"Resume checkpoint does not exist: {resume_checkpoint}"
            )
        if resume_checkpoint.parent == output_dir:
            raise ValueError(
                "Refusing to resume from a checkpoint inside --output-dir: "
                f"{resume_checkpoint}. The resumed run rewrites current.json "
                "in that directory every controller round and would destroy "
                "the state it is resuming from. Stage the checkpoint (with "
                "its estimator-ledger sidecar) elsewhere, or write to a new "
                "output directory."
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite completed H2O RA result: {result_path}"
        )
    checkpoint_path = output_dir / "current.json"
    if resume_checkpoint is None and checkpoint_path.exists():
        raise FileExistsError(
            "Refusing to fresh-start over an existing accepted-state "
            f"checkpoint: {checkpoint_path}. A fresh run rewrites it at its "
            "own round 1, destroying prior progress. Pass "
            "--resume-checkpoint to continue that run, or choose a new "
            "--output-dir."
        )

    problem = _problem(fixture)
    request = build_h2o_ra_request(
        problem=problem,
        output_dir=output_dir,
        maximum_controller_rounds=args.maximum_controller_rounds,
        gradient_tolerance=args.gradient_tolerance,
        exact_absolute_tolerance=args.exact_absolute_tolerance,
        exact_confirmation_controller_rounds=(
            args.exact_confirmation_controller_rounds
        ),
        candidate_representation=args.candidate_representation,
        insertion=args.insertion,
        checkpoint_history_tail=args.checkpoint_history_tail,
        estimator_ledger=not args.no_estimator_ledger,
        cost_mode=args.cost_mode,
    )
    protocol = build_resolved_ra_protocol(problem, request)
    resolved_cost_mode = str(
        protocol.route_contract["execution_settings"][
            "phase3_backend_cost_mode"
        ]
    )
    _write_json(output_dir / "protocol.json", protocol.to_dict())
    startup_seconds = time.monotonic() - main_entry_monotonic
    engine_wall_budget_seconds = _engine_wall_budget_seconds(
        args.max_wall_seconds,
        startup_seconds=startup_seconds,
    )
    _write_json(
        output_dir / "launch_manifest.json",
        {
            "schema": "paper_iv_h2o_ra_plateau_local_launch_v3",
            "fixture": str(fixture),
            "fixture_sha256": _sha256(fixture),
            "protocol_sha256": protocol.sha256,
            "route_profile": protocol.route_contract["route_profile"],
            "route_contract_sha256": protocol.route_contract["sha256"],
            "maximum_controller_rounds": args.maximum_controller_rounds,
            "max_wall_seconds": args.max_wall_seconds,
            "max_wall_seconds_engine_budget": engine_wall_budget_seconds,
            "max_wall_seconds_startup_deducted": startup_seconds,
            "max_wall_seconds_reference": (
                "cli_main_entry; checked only at controller-round "
                "boundaries after each completed round"
            ),
            "gradient_tolerance": args.gradient_tolerance,
            "exact_absolute_tolerance": args.exact_absolute_tolerance,
            "exact_confirmation_controller_rounds": (
                args.exact_confirmation_controller_rounds
            ),
            "candidate_representation": protocol.candidate_representation,
            "candidate_mode": args.candidate_representation,
            "candidate_adapter_id": request.adapter.adapter_id,
            "cost_mode": args.cost_mode,
            "phase3_backend_cost_mode": resolved_cost_mode,
            "fresh_start": resume_checkpoint is None,
            "resume_checkpoint": (
                None if resume_checkpoint is None else str(resume_checkpoint)
            ),
            "resume_checkpoint_sha256": (
                None
                if resume_checkpoint is None
                else _sha256(resume_checkpoint)
            ),
        },
    )
    print(
        json.dumps(
            {
                "event": "paper_iv_h2o_ra_launch",
                "protocol_sha256": protocol.sha256,
                "route_profile": protocol.route_contract["route_profile"],
                "maximum_controller_rounds": args.maximum_controller_rounds,
                "max_wall_seconds": args.max_wall_seconds,
                "max_wall_seconds_engine_budget": (
                    engine_wall_budget_seconds
                ),
                "gradient_tolerance": args.gradient_tolerance,
                "exact_absolute_tolerance": args.exact_absolute_tolerance,
                "exact_confirmation_controller_rounds": (
                    args.exact_confirmation_controller_rounds
                ),
                "candidate_representation": (
                    protocol.candidate_representation
                ),
                "candidate_mode": args.candidate_representation,
                "candidate_adapter_id": request.adapter.adapter_id,
                "cost_mode": args.cost_mode,
                "phase3_backend_cost_mode": resolved_cost_mode,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    # Operational controls never enter the resolved protocol; they are
    # built only when a segment mechanic (resume, wall budget) is requested,
    # so an unadorned launch still calls the facade exactly as before.
    operational_controls: RAAdaptOperationalControls | None = None
    if resume_checkpoint is not None or engine_wall_budget_seconds is not None:
        operational_controls = RAAdaptOperationalControls(
            maximum_controller_rounds=args.maximum_controller_rounds,
            resume=(
                FreshStart()
                if resume_checkpoint is None
                else AcceptedStateResume(
                    checkpoint_path=resume_checkpoint,
                    checkpoint_sha256=_sha256(resume_checkpoint),
                )
            ),
            observation=request.observation,
            maximum_wall_seconds=engine_wall_budget_seconds,
        )
    if operational_controls is None:
        result = run_ra_adapt(problem, protocol)
    else:
        result = run_ra_adapt(
            problem,
            protocol,
            operational_controls=operational_controls,
        )
    _write_json(result_path, result.to_dict())
    _write_json(
        output_dir / "summary.json",
        {
            "schema": "paper_iv_h2o_ra_plateau_local_summary_v3",
            "protocol_sha256": protocol.sha256,
            "candidate_representation": protocol.candidate_representation,
            "candidate_adapter_id": request.adapter.adapter_id,
            "completed_controller_rounds": (
                result.run.stop.completed_controller_rounds
            ),
            "accepted_operator_count": result.run.stop.accepted_operator_count,
            "stop_reason": result.run.stop.primary_reason,
            "stop_fired_reasons": list(result.run.stop.fired_reasons),
            "max_wall_seconds": args.max_wall_seconds,
            "wall_clock_budget_seconds": (
                result.run.stop.wall_clock_budget_seconds
            ),
            "wall_clock_elapsed_seconds": (
                result.run.stop.wall_clock_elapsed_seconds
            ),
            "energy_hartree": result.run.final_state.energy,
            "exact_energy_hartree": (
                result.run.canonical_reporting.exact_same_cutoff_energy
            ),
            "absolute_error_hartree": abs(
                result.run.final_state.energy
                - result.run.canonical_reporting.exact_same_cutoff_energy
            ),
            "chemical_accuracy_reached": bool(
                abs(
                    result.run.final_state.energy
                    - result.run.canonical_reporting.exact_same_cutoff_energy
                )
                <= args.exact_absolute_tolerance
            ),
            "exact_first_hit_controller_round": (
                result.run.stop.exact_first_hit_controller_round
            ),
            "exact_confirmation_controller_rounds": (
                result.run.stop.exact_confirmation_controller_rounds
            ),
        },
    )
    print((output_dir / "summary.json").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
