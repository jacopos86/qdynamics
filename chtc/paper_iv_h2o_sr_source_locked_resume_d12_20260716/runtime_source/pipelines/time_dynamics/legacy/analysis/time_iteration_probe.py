from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from pipelines.time_dynamics.runners.hh_from_adapt_artifact import (
    _to_jsonable,
    build_controller_bundle_from_args,
    build_parser as build_realtime_parser,
)


def _parse_int_tuple(raw: str | None) -> tuple[int, ...]:
    if raw is None:
        return ()
    text = str(raw).strip()
    if not text:
        return ()
    return tuple(int(chunk.strip()) for chunk in text.split(",") if chunk.strip())


"Built Math: probe(time iterations) = ReplayPrefix(artifact, cfg) with optional forced stay actions at selected time iterations."
def run_probe_from_args(args: argparse.Namespace) -> dict[str, Any]:
    output_json = Path(args.output_json).expanduser().resolve()
    probe_checkpoints = _parse_int_tuple(getattr(args, "probe_checkpoints", None))
    force_stay_checkpoints = _parse_int_tuple(getattr(args, "force_stay_checkpoints", None))
    if not probe_checkpoints:
        raise ValueError("--probe-checkpoints must be non-empty")
    reference_payload = None
    if getattr(args, "reference_output_json", None) not in {None, ""}:
        reference_path = Path(str(args.reference_output_json)).expanduser().resolve()
        reference_payload = json.loads(reference_path.read_text(encoding="utf-8"))
    scenario_specs: list[tuple[str, tuple[int, ...]]] = [("actual_prefix", ())]
    if force_stay_checkpoints:
        forced_tag = "_".join(str(x) for x in force_stay_checkpoints)
        scenario_specs.append((f"force_stay_{forced_tag}", tuple(force_stay_checkpoints)))

    scenarios: list[dict[str, Any]] = []
    first_bundle: dict[str, Any] | None = None
    for scenario_name, forced_ids in scenario_specs:
        bundle = build_controller_bundle_from_args(args)
        if first_bundle is None:
            first_bundle = dict(bundle)
        controller = bundle["controller"]
        probe_payload = controller.debug_probe_exact_v1(
            probe_checkpoints=probe_checkpoints,
            force_stay_checkpoints=forced_ids,
            candidate_rank_limit=int(args.candidate_rank_limit),
            baseline_variant_limit=int(args.baseline_variant_limit),
            reference_payload=reference_payload,
        )
        scenarios.append(
            {
                "scenario_name": str(scenario_name),
                "force_stay_checkpoints": [int(x) for x in forced_ids],
                **_to_jsonable(probe_payload),
            }
        )

    if first_bundle is None:
        raise RuntimeError("failed to build controller bundle for probe")
    payload = {
        "run_tag": str(args.run_tag),
        "artifact_json": str(Path(args.artifact_json).expanduser().resolve()),
        "output_json": str(output_json),
        "reference_output_json": (
            None
            if getattr(args, "reference_output_json", None) in {None, ""}
            else str(Path(str(args.reference_output_json)).expanduser().resolve())
        ),
        "probe_checkpoints": [int(x) for x in probe_checkpoints],
        "requested_force_stay_checkpoints": [int(x) for x in force_stay_checkpoints],
        "candidate_rank_limit": int(args.candidate_rank_limit),
        "baseline_variant_limit": int(args.baseline_variant_limit),
        "controller_config": _to_jsonable(first_bundle["cfg"]),
        "drive_config": _to_jsonable(first_bundle["drive_config"]),
        "oracle_config": _to_jsonable(first_bundle["oracle_config"]),
        "scenarios": scenarios,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = build_realtime_parser()
    parser.description = (
        "Legacy exact_v1 AP-McLachlan time-iteration probe. "
        "Old checkpoint-named flags are retained as compatibility aliases."
    )
    parser.add_argument("--probe-checkpoints", required=True)
    parser.add_argument("--force-stay-checkpoints", default="")
    parser.add_argument("--candidate-rank-limit", type=int, default=4)
    parser.add_argument("--baseline-variant-limit", type=int, default=8)
    parser.add_argument("--reference-output-json", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_probe_from_args(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
