#!/usr/bin/env python3
"""Run one sealed threshold package locally as explicitly diagnostic evidence."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType


EXECUTION_ID = (
    "phase3_on_plateau_r50__weak_weak__nph3__ra_singleton_plateau"
)


def _load_runner(package_dir: Path) -> ModuleType:
    runner_path = package_dir / "run_cell.py"
    sys.path.insert(0, package_dir.as_posix())
    spec = importlib.util.spec_from_file_location(
        "paper_i_threshold_local_diagnostic_runner", runner_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load sealed runner: {runner_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run(*, package_dir: Path, output_dir: Path) -> dict[str, object]:
    package_dir = package_dir.resolve()
    output_dir = output_dir.resolve()
    progress_dir = output_dir.with_name(f"{output_dir.name}.in_progress")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Refusing existing output: {output_dir}")
    if progress_dir.exists() or progress_dir.is_symlink():
        raise FileExistsError(f"Refusing existing progress output: {progress_dir}")
    progress_dir.mkdir(parents=True, exist_ok=False)

    runner = _load_runner(package_dir)
    job_path = package_dir / "jobs" / f"{EXECUTION_ID}.json"
    job, manifest, protocol, problem, temporary = runner._prepare(job_path)
    try:
        source_root = Path(temporary.name) / "source"
        original = Path.cwd()
        os.chdir(source_root)
        try:
            result, rounds = runner._execute(
                protocol=protocol,
                problem=problem,
                staging=progress_dir,
                maximum_rounds=runner.TARGET_HORIZON,
            )
        finally:
            os.chdir(original)

        result_payload = result.to_dict()
        summary_payload = result.run.paper_i_summary.to_dict()
        runner._write_json(progress_dir / "result.json", result_payload)
        runner._write_json(
            progress_dir / "paper_i_summary.json", summary_payload
        )
        science_payloads = runner._science_payload_bindings(progress_dir)
        terminal = summary_payload["accepted_error_trace"][-1]
        source_archive = manifest["source_archive"]
        receipt = runner.digested(
            {
                "schema": (
                    "paper_i_ra_adapt_threshold_local_macos_diagnostic_v1"
                ),
                "status": "passed_diagnostic_only",
                "wrapper_used": True,
                "execution_target": "local_macos_diagnostic",
                "paper_evidence_adopted": False,
                "source_value_anchor_claimed": False,
                "known_platform_limitation": (
                    "macos_did_not_reproduce_chtc_source_trajectory_v1"
                ),
                "package_id": runner.PACKAGE_ID,
                "package_manifest_sha256": manifest["sha256"],
                "source_archive_sha256": source_archive["sha256"],
                "job_spec_sha256": job["sha256"],
                "protocol_sha256": protocol.sha256,
                "route_contract_sha256": runner.ROUTE_CONTRACT_SHA256,
                "plateau_prior_mean_decrease_ratio_threshold": (
                    runner.PLATEAU_PRIOR_MEAN_RATIO_THRESHOLD
                ),
                "controller_rounds_completed": rounds,
                "terminal_same_cutoff_absolute_energy_error": terminal[
                    "absolute_energy_error"
                ],
                "terminal_projective_state_fingerprint": terminal[
                    "projective_state_fingerprint"
                ],
                "science_payloads": science_payloads,
            }
        )
        runner._write_json(progress_dir / "diagnostic_receipt.json", receipt)
        os.rename(progress_dir, output_dir)
        return receipt
    finally:
        temporary.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = run(
        package_dir=args.package_dir,
        output_dir=args.output_dir,
    )
    print(receipt["sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
