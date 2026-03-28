#!/usr/bin/env python3
"""Thin CLI wrapper for the staged HH noise-capable workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.hardcoded.hh_staged_cli_args import build_staged_hh_parser
from pipelines.hardcoded.hh_staged_noise_workflow import (
    resolve_staged_hh_noise_config,
    run_staged_hh_noise,
)


def build_parser() -> argparse.ArgumentParser:
    return build_staged_hh_parser(
        description=(
            "Historical/compatibility staged HH noise workflow: either (a) HF -> hh_hva_ptw warm-start -> staged ADAPT -> "
            "matched-family conventional replay -> final-only noisy/noiseless dynamics, or "
            "(b) imported lean ADAPT prepared-state/full-circuit fake-backend audits, or "
            "(c) imported fixed-lean noisy replay or pinned Marrakesh/Heron 6-term fixed-scaffold noisy replay "
            "with saved-theta initialization plus continuous optimization only, or "
            "(d) imported fixed-lean or fixed-scaffold gate-vs-readout attribution on one shared compiled circuit, or "
            "(e) imported fixed-lean or fixed-scaffold compile-control scouting on a locked local fake-backend circuit "
            "with requested-vs-observed transpile metadata surfaced in the outputs, or "
            "(f) imported fixed-scaffold Runtime energy-only baseline on the real Runtime path, or "
            "(g) imported fixed-scaffold raw-shot baseline with raw sidecars: sampler-backed on real Runtime and local "
            "fake-backend acquisition plus offline diagonal-only readout-then-symmetry postprocessing on the all-Z path. "
            "Fresh-stage staged noise also accepts opt-in phase3 oracle/raw-shot ADAPT scouting knobs. "
            "Kept for staged VQE->ADAPT->VQE reproduction and import-side audits; new HH default ADAPT work should use "
            "pipelines/hardcoded/adapt_pipeline.py with direct phase3_v1 continuation."
        ),
        include_noise=True,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = resolve_staged_hh_noise_config(args)
    payload = run_staged_hh_noise(cfg)
    print(f"workflow_json={payload['artifacts']['workflow']['output_json']}")
    if not bool(cfg.staged.artifacts.skip_pdf):
        print(f"workflow_pdf={payload['artifacts']['workflow']['output_pdf']}")
    import_source_json = payload.get("artifacts", {}).get("import_source_json", None)
    if import_source_json is not None:
        print(f"import_source_json={import_source_json}")
    fixed_scaffold_runtime_energy_only_json = payload.get("artifacts", {}).get(
        "fixed_scaffold_runtime_energy_only_json", None
    )
    if fixed_scaffold_runtime_energy_only_json is not None:
        print(f"fixed_scaffold_runtime_energy_only_json={fixed_scaffold_runtime_energy_only_json}")
    fixed_scaffold_runtime_raw_baseline_json = payload.get("artifacts", {}).get(
        "fixed_scaffold_runtime_raw_baseline_json", None
    )
    if fixed_scaffold_runtime_raw_baseline_json is not None:
        print(f"fixed_scaffold_runtime_raw_baseline_json={fixed_scaffold_runtime_raw_baseline_json}")
    intermediate = payload.get("artifacts", {}).get("intermediate", {})
    if isinstance(intermediate, dict) and intermediate.get("adapt_handoff_json", None) is not None:
        print(f"adapt_handoff_json={intermediate['adapt_handoff_json']}")
    if isinstance(intermediate, dict) and intermediate.get("replay_output_json", None) is not None:
        if str(payload.get("import_source", {}).get("mode", "")) != "imported_artifact":
            print(f"replay_json={intermediate['replay_output_json']}")


if __name__ == "__main__":
    main()
