"""Command line for the prospective multi-coherent holdout workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .hubbard_dimer import DimerParameters
from .multi_coherent_holdout import (
    MultiCoherentHoldoutSettings,
    _sha256,
    freeze_blind_multi_coherent_inputs,
    run_frozen_multi_coherent_model_trajectory,
    seal_frozen_multi_coherent_model_batch,
    seal_frozen_multi_coherent_model_cost,
)
from .multi_coherent_propagation import _load_gate_initial_parameters
from .multi_coherent_sealed_score import (
    run_exact_holdout_cost_once,
    score_frozen_multi_coherent_holdout,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--gate-directory", type=Path, required=True)
    prepare.add_argument("--development-directory", type=Path, required=True)
    prepare.add_argument("--output-directory", type=Path, required=True)

    run_model = commands.add_parser("run-model")
    run_model.add_argument("--prepared-directory", type=Path, required=True)
    run_model.add_argument("--output-directory", type=Path, required=True)
    run_model.add_argument(
        "--member",
        choices=("central", "plus", "minus"),
        required=True,
    )
    run_model.add_argument(
        "--resolution",
        choices=("coarse", "fine"),
        required=True,
    )

    seal = commands.add_parser("seal-model")
    seal.add_argument("--prepared-directory", type=Path, required=True)
    seal.add_argument("--batch-directory", type=Path, required=True)

    seal_cost = commands.add_parser("seal-model-cost")
    seal_cost.add_argument("--prepared-directory", type=Path, required=True)
    seal_cost.add_argument("--batch-directory", type=Path, required=True)

    score = commands.add_parser("score")
    score.add_argument("--prepared-directory", type=Path, required=True)
    score.add_argument("--batch-directory", type=Path, required=True)
    score.add_argument("--output-directory", type=Path, required=True)

    exact_cost = commands.add_parser("exact-cost-once")
    exact_cost.add_argument("--prepared-directory", type=Path, required=True)
    exact_cost.add_argument("--batch-directory", type=Path, required=True)
    exact_cost.add_argument("--output-directory", type=Path, required=True)
    exact_cost.add_argument(
        "--method",
        choices=("dop853", "midpoint"),
        required=True,
    )
    exact_cost.add_argument("--refined", action="store_true")
    return parser


def _prepare(args: argparse.Namespace) -> dict:
    settings = MultiCoherentHoldoutSettings()
    parameters = DimerParameters(
        lambda_ep=1.5,
        gamma=0.5,
        drive_amplitude=1.0,
    )
    initial_parameters, gate_hashes = _load_gate_initial_parameters(
        args.gate_directory,
        packet_count=settings.initial_packets_per_electronic_branch,
    )
    development_summary_path = args.development_directory / "summary.json"
    development_arrays_path = (
        args.development_directory / "segmented_horizon.npz"
    )
    development_manifest_path = (
        args.development_directory / "runtime_manifest.json"
    )
    if not all(
        path.is_file()
        for path in (
            development_summary_path,
            development_arrays_path,
            development_manifest_path,
        )
    ):
        raise FileNotFoundError("development trajectory artifacts are incomplete")
    summary = json.loads(
        development_summary_path.read_text(encoding="utf-8")
    )
    if summary.get("status") != "complete":
        raise ValueError("development trajectory is incomplete")
    expected = {
        "phonon_cutoff": settings.phonon_cutoff,
        "packet_count": settings.initial_packets_per_electronic_branch,
        "maximum_packet_count": settings.maximum_packets_per_electronic_branch,
        "tangent_regularization": settings.tangent_regularization,
        "relative_damping": settings.relative_damping,
        "target_final_time": settings.score_interval[1],
        "output_sample_step": settings.output_sample_step,
    }
    for name, value in expected.items():
        if summary["parameters"].get(name) != value:
            raise ValueError(f"development setting mismatch: {name}")
    if summary["parameters"].get("drive_protocol") is not None:
        raise ValueError("development scales must come from the single-pulse path")
    with np.load(development_arrays_path) as arrays:
        times = np.asarray(arrays["times"], dtype=float)
        exact_closed = np.asarray(
            arrays["exact_closed_coordinates"],
            dtype=float,
        )
    input_hashes = {
        **{f"gate:{name}": digest for name, digest in gate_hashes.items()},
        "development_summary": _sha256(development_summary_path),
        "development_arrays": _sha256(development_arrays_path),
        "development_runtime_manifest": _sha256(development_manifest_path),
    }
    return freeze_blind_multi_coherent_inputs(
        args.output_directory,
        initial_parameters=initial_parameters,
        development_times=times,
        development_closed_coordinates=exact_closed,
        settings=settings,
        parameters=parameters,
        input_hashes=input_hashes,
    )


def main() -> int:
    args = _parser().parse_args()
    if args.command == "prepare":
        result = _prepare(args)
    elif args.command == "run-model":
        result = run_frozen_multi_coherent_model_trajectory(
            args.prepared_directory,
            args.output_directory,
            member=args.member,
            resolution=args.resolution,
        )
    elif args.command == "seal-model-cost":
        result = seal_frozen_multi_coherent_model_cost(
            args.prepared_directory,
            args.batch_directory,
        )
    elif args.command == "seal-model":
        result = seal_frozen_multi_coherent_model_batch(
            args.prepared_directory,
            args.batch_directory,
        )
    elif args.command == "score":
        result = score_frozen_multi_coherent_holdout(
            args.prepared_directory,
            args.batch_directory,
            args.output_directory,
        )
    else:
        result = run_exact_holdout_cost_once(
            args.prepared_directory,
            args.batch_directory,
            args.output_directory,
            method=args.method,
            refined=args.refined,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
