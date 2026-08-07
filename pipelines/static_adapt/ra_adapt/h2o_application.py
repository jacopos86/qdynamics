"""Checkpointed local launcher for the production H2O RA-ADAPT lane."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
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
    RAAdaptRequest,
)
from pipelines.static_adapt.ra_adapt.engine import (
    build_resolved_ra_protocol,
    run_ra_adapt,
)
from pipelines.static_adapt.sr_snake.contracts import (
    BeamOff,
    CheckpointObservation,
    EstimatorLedgerObservation,
    ExactEDSourceReceipt,
    ExactEDStop,
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
            insertion=PlateauCommutationInsertion(),
            pruning=PruningOff(),
            beam=BeamOff(),
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
                keep_history_tail=100,
            ),
            estimator_ledger=EstimatorLedgerObservation(
                path=output_dir / "estimator_call_ledger.json"
            ),
        ),
    )


def main() -> None:
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
    args = parser.parse_args()

    fixture = args.fixture.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not fixture.is_file():
        raise FileNotFoundError(f"H2O fixture does not exist: {fixture}")
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite completed H2O RA result: {result_path}"
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
    )
    protocol = build_resolved_ra_protocol(problem, request)
    _write_json(output_dir / "protocol.json", protocol.to_dict())
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
            "gradient_tolerance": args.gradient_tolerance,
            "exact_absolute_tolerance": args.exact_absolute_tolerance,
            "exact_confirmation_controller_rounds": (
                args.exact_confirmation_controller_rounds
            ),
            "candidate_representation": protocol.candidate_representation,
            "candidate_mode": args.candidate_representation,
            "candidate_adapter_id": request.adapter.adapter_id,
            "fresh_start": True,
        },
    )
    print(
        json.dumps(
            {
                "event": "paper_iv_h2o_ra_launch",
                "protocol_sha256": protocol.sha256,
                "route_profile": protocol.route_contract["route_profile"],
                "maximum_controller_rounds": args.maximum_controller_rounds,
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
            },
            sort_keys=True,
        ),
        flush=True,
    )
    result = run_ra_adapt(problem, protocol)
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
