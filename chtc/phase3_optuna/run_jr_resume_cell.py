#!/usr/bin/env python3
"""Resume one immutable JR-SNAKE cell through the original validated runner."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.exact_bench import paper_i_hh_route_a_optuna as route_a_optuna


SCHEMA = "paper_i_hh_jr_checkpoint_resume_adapter_v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _replace_flag_value(argv: list[str], flag: str, value: int) -> None:
    try:
        flag_index = argv.index(flag)
    except ValueError as exc:
        raise ValueError(f"JR argv is missing required flag {flag!r}.") from exc
    value_index = flag_index + 1
    if value_index >= len(argv):
        raise ValueError(f"JR argv flag {flag!r} has no value.")
    argv[value_index] = str(int(value))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume-scaffold-json", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gradient-workers", type=int, default=4)
    parser.add_argument("--beam-parent-workers", type=int, default=1)
    parser.add_argument("--runtime-split-child-workers", type=int, default=0)
    parser.add_argument("--joint-pair-workers", type=int, default=4)
    parser.add_argument("--target-controller-round", type=int, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cell_manifest = args.cell_manifest.resolve()
    resume_scaffold = args.resume_scaffold_json.resolve()
    if not cell_manifest.is_file():
        raise FileNotFoundError(cell_manifest)
    if not resume_scaffold.is_file():
        raise FileNotFoundError(resume_scaffold)

    cell = json.loads(cell_manifest.read_text(encoding="utf-8"))
    if cell.get("execution_profile") != route_a_optuna.JR_L10_ROLLBACK_FREE_R50_EXECUTION_PROFILE:
        raise ValueError("Resume adapter accepts only the locked JR-L10 profile.")

    resume_payload = json.loads(resume_scaffold.read_text(encoding="utf-8"))
    resume_adapt = resume_payload.get("adapt_vqe", {})
    resume_controller_round = int(resume_adapt.get("history_count", 0))
    target_controller_round = (
        None
        if args.target_controller_round is None
        else int(args.target_controller_round)
    )
    if (
        target_controller_round is not None
        and target_controller_round < resume_controller_round
    ):
        raise ValueError(
            "target controller round precedes the resume checkpoint: "
            f"{target_controller_round} < {resume_controller_round}"
        )
    additional_controller_rounds = (
        None
        if target_controller_round is None
        else int(target_controller_round) - int(resume_controller_round)
    )
    output_dir = args.output_root.resolve() / str(cell["output_relative_dir"])
    adapter_manifest = {
        "schema": SCHEMA,
        "cell_manifest": str(cell_manifest),
        "cell_manifest_sha256": _sha256(cell_manifest),
        "execution_profile": cell["execution_profile"],
        "policy_sha256": cell["policy_sha256"],
        "scientific_contract_hash": cell["scientific_contract_hash"],
        "regime": cell["regime"],
        "resume_scaffold_json": str(resume_scaffold),
        "resume_scaffold_sha256": _sha256(resume_scaffold),
        "resume_controller_round": resume_controller_round,
        "resume_ansatz_depth": resume_adapt.get("ansatz_depth"),
        "target_max_depth": cell["max_depth"],
        "target_controller_round": target_controller_round,
        "requested_additional_controller_rounds": additional_controller_rounds,
        "scientific_settings_changed": (
            []
            if additional_controller_rounds is None
            else [
                "segment_local_max_controller_rounds:"
                f"{int(cell['max_depth'])}->{int(additional_controller_rounds)}"
            ]
        ),
        "operational_changes": {
            "structural_resume": True,
            "requested_memory_mb": 49152,
        },
    }
    _write_json(output_dir / "resume_adapter_manifest.json", adapter_manifest)

    original_builder = route_a_optuna.build_jr_l10_rollback_free_r50_pareto_argv

    def build_with_resume(*builder_args: Any, **builder_kwargs: Any) -> list[str]:
        argv = list(original_builder(*builder_args, **builder_kwargs))
        if "--resume-scaffold-json" in argv:
            raise ValueError("JR argv already contains a resume scaffold.")
        argv.extend(["--resume-scaffold-json", str(resume_scaffold)])
        if additional_controller_rounds is not None:
            _replace_flag_value(
                argv,
                "--max-depth",
                int(additional_controller_rounds),
            )
        return argv

    route_a_optuna.build_jr_l10_rollback_free_r50_pareto_argv = build_with_resume
    original_validator = route_a_optuna.validate_locked_pareto_plan

    def validate_resume_segment(
        plan: dict[str, Any],
        *validator_args: Any,
        **validator_kwargs: Any,
    ) -> dict[str, Any]:
        if additional_controller_rounds is not None:
            validator_kwargs["max_depth"] = int(additional_controller_rounds)
        return original_validator(plan, *validator_args, **validator_kwargs)

    route_a_optuna.validate_locked_pareto_plan = validate_resume_segment
    result = route_a_optuna.run_cell(
        cell_manifest=cell_manifest,
        output_root=args.output_root.resolve(),
        dry_run=bool(args.dry_run),
        gradient_workers=int(args.gradient_workers),
        beam_parent_workers=int(args.beam_parent_workers),
        runtime_split_child_workers=int(args.runtime_split_child_workers),
        joint_pair_workers=int(args.joint_pair_workers),
    )
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "status": "planned" if args.dry_run else "complete",
                "regime": cell["regime"],
                "resume_controller_round": adapter_manifest["resume_controller_round"],
                "target_max_depth": cell["max_depth"],
                "target_controller_round": target_controller_round,
                "result_status": result.get("status"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
