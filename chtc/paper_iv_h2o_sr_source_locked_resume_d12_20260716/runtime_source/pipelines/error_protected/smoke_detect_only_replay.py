#!/usr/bin/env python3
"""Opt-in smoke runner for the ADAPT detect-only sidecar.

This is not a production route and is not intended for broad CI.  It exercises
the milestone-one sidecar on a small known HH artifact with a fake backend.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from pipelines.error_protected.adapt_detect_only_replay import run_detect_only_replay
from pipelines.error_protected.contracts import DetectionReplayInput, ErrorDetectionConfig


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = REPO_ROOT / "artifacts/json/campaign_A6_L2_backend_proxy_baseline.json"
DEFAULT_OUTPUT = Path("/tmp/hh_detect_only_sidecar_smoke.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-json", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--backend-name", type=str, default="FakeGuadalupeV2")
    parser.add_argument("--shots", type=int, default=32)
    parser.add_argument("--oracle-repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-accepted-shots", type=int, default=1)
    parser.set_defaults(strict=True)
    parser.add_argument("--strict", dest="strict", action="store_true")
    parser.add_argument("--no-strict", dest="strict", action="store_false")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    request = DetectionReplayInput(
        artifact_json=Path(args.artifact_json),
        output_json=Path(args.output_json),
        noise_mode="backend_scheduled",
        execution_surface="raw_measurement_v1",
        backend_name=str(args.backend_name),
        use_fake_backend=True,
        shots=int(args.shots),
        oracle_repeats=int(args.oracle_repeats),
        oracle_aggregate="mean",
        seed=int(args.seed),
        raw_grouping_mode="qwc_basis_cover_reuse",
        detection=ErrorDetectionConfig(
            mode="sector_audit",
            strict=bool(args.strict),
            min_accepted_shots=int(args.min_accepted_shots),
        ),
    )
    summary = run_detect_only_replay(request)
    estimates = summary.get("estimates", {})
    energy_status = estimates.get("energy_raw", {}).get("status")
    sector_status = estimates.get("sector_audit", {}).get("status")
    failure = summary.get("failure")
    print(f"output_json={Path(args.output_json)}")
    print(f"schema_version={summary.get('schema_version')}")
    print(f"energy_raw.status={energy_status}")
    print(f"sector_audit.status={sector_status}")
    print(f"failure={failure}")
    return 0 if failure is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
