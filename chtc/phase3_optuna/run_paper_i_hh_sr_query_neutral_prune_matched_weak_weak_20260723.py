#!/usr/bin/env python3
"""Run the Paper-I weak--weak prune study from its frozen parent source.

This launcher exists to prevent a local sensitivity candidate from silently
using the live checkout while its parent uses an immutable source archive.
The only scientific command change is the SR route profile.  Output paths,
segment identity, and an optional diagnostic horizon are operational changes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
ANCHOR_ID = (
    "paper_i_hh_sr_snake_no_overlap_trust_full_geometry_query_neutral_prune_"
    "parent_anchor_weak_weak_target_20260723_v3_chtc"
)
ANCHOR = ROOT / "chtc/phase3_optuna/input" / ANCHOR_ID
SOURCE_ARCHIVE = ANCHOR / "source_locked.tar.gz"
SOURCE_ARCHIVE_SHA256 = (
    "5747f73be5b6f4a050c5c33c12c87099db7e2edb57ca2eb9a41d9e4a783207e4"
)
MATCHED_PARENT_RESULT = Path(
    "/tmp/query_neutral_parent_anchor_source_locked_scope_repair_v2/result.json"
)
MATCHED_PARENT_LEDGER = Path(
    "/tmp/query_neutral_parent_anchor_source_locked_scope_repair_v2/ledger.json"
)
PARENT_ROUTE_ALIAS = (
    "sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1"
)
CANDIDATE_ROUTE_ALIAS = (
    "sr_snake_symmetric_cost_projected_phase3_no_overlap_trust_full_geometry_"
    "query_neutral_prune_v1"
)
PARENT_ROUTE = (
    "supported_projected_generalized_source_metric_no_overlap_trust_"
    "full_response_symmetric_cost_no_prune_v1"
)
PARENT_DIGEST = "fd5ec3fa2c98b2a9d1cbcc304241d723f57dbd6210f4ea2daf30753603a146c2"
CANDIDATE_ROUTE = (
    "supported_projected_generalized_source_metric_no_overlap_trust_"
    "full_response_symmetric_cost_query_neutral_fs_prune_v1"
)
CANDIDATE_DIGEST = (
    "326ae05091b24fcb580d33f86f25add4c1252bcdd64316b82ae14c14c6bb3372"
)
EXPECTED_FILTERED_POOL_SIZE = 102
FORBIDDEN_SOURCE_MARKERS = (
    "def _resolve_parent_sector_filter_policy(",
    "execution_deferred_to_runtime_projected_children",
)
SCOPE_REPAIR_OLD = """\
    phase1_prune_endpoint_overlap_policy_key = str(
        phase1_prune_endpoint_overlap_policy or "off"
    ).strip().lower()
    if (
        sr_controller_ablation_contract_key
"""
SCOPE_REPAIR_NEW = """\
    phase1_prune_endpoint_overlap_policy_key = str(
        phase1_prune_endpoint_overlap_policy or "off"
    ).strip().lower()
    phase1_prune_prefilter_policy_key = str(
        phase1_prune_prefilter_policy or PRUNE_PREFILTER_OFF
    ).strip().lower()
    phase1_prune_tolerance_mode_requested = str(
        phase1_prune_tolerance_mode or PRUNE_TOLERANCE_AUTO
    ).strip().lower()
    if (
        sr_controller_ablation_contract_key
"""
RUNTIME_RECEIPT_REPAIR_OLD = """\
                    "phase1_prune_fraction": float(phase1_prune_fraction),
                    "phase1_prune_min_candidates": int(
                        phase1_prune_min_candidates
                    ),
"""
RUNTIME_RECEIPT_REPAIR_NEW = """\
                    "phase1_prune_fraction": float(phase1_prune_fraction),
                    "phase1_prune_local_window_size": int(
                        phase1_prune_local_window_size
                    ),
                    "phase1_prune_max_candidates": int(
                        phase1_prune_max_candidates
                    ),
                    "phase1_prune_min_candidates": int(
                        phase1_prune_min_candidates
                    ),
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _replace_option(argv: list[str], option: str, value: str) -> None:
    occurrences = [index for index, item in enumerate(argv) if item == option]
    if len(occurrences) != 1:
        raise ValueError(f"expected one {option}, found {len(occurrences)}")
    index = occurrences[0]
    if index + 1 >= len(argv):
        raise ValueError(f"missing value after {option}")
    argv[index + 1] = value


def _remove_option(argv: list[str], option: str) -> None:
    occurrences = [index for index, item in enumerate(argv) if item == option]
    if len(occurrences) != 1:
        raise ValueError(f"expected one {option}, found {len(occurrences)}")
    index = occurrences[0]
    if index + 1 >= len(argv):
        raise ValueError(f"missing value after {option}")
    del argv[index : index + 2]


def build_candidate_argv(
    parent_argv: Sequence[str],
    *,
    output_dir: Path,
    max_rounds: int,
) -> list[str]:
    """Change only route identity plus operational output/horizon fields."""

    argv = list(parent_argv)
    _replace_option(argv, "--sr-route-profile", CANDIDATE_ROUTE_ALIAS)
    _replace_option(argv, "--adapt-max-depth", str(max_rounds))
    _replace_option(
        argv,
        "--adapt-segment-id",
        f"local-source-locked-query-neutral-prune-ww-r{max_rounds}-v1",
    )
    for option in (
        "--adapt-segment-target-controller-round",
        "--adapt-segment-target-depth",
        "--adapt-segment-max-new-admissions",
    ):
        _replace_option(argv, option, str(max_rounds))
    _replace_option(
        argv,
        "--adapt-current-json",
        str(output_dir / "current.json"),
    )
    _replace_option(
        argv,
        "--adapt-estimator-call-ledger-json",
        str(output_dir / "ledger.json"),
    )
    _replace_option(
        argv,
        "--output-json",
        str(output_dir / "result.json"),
    )
    for option, value in (
        ("--phase1-prune-max-candidates", "1"),
        ("--phase1-prune-local-window-size", "0"),
    ):
        if option in argv:
            raise ValueError(f"parent command unexpectedly defines {option}")
        argv.extend((option, value))
    return argv


def build_parent_argv(
    parent_argv: Sequence[str],
    *,
    output_dir: Path,
    max_rounds: int,
) -> list[str]:
    argv = build_candidate_argv(
        parent_argv,
        output_dir=output_dir,
        max_rounds=max_rounds,
    )
    _remove_option(argv, "--phase1-prune-max-candidates")
    _remove_option(argv, "--phase1-prune-local-window-size")
    _replace_option(argv, "--sr-route-profile", PARENT_ROUTE_ALIAS)
    _replace_option(
        argv,
        "--adapt-segment-id",
        f"local-source-locked-parent-scope-repair-ww-r{max_rounds}-v1",
    )
    return argv


def validate_matched_pool_surface(
    *,
    parent_result: Mapping[str, Any],
    candidate_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail when the candidate and parent did not expose the same pool."""

    parent = parent_result.get("adapt_vqe")
    candidate = candidate_result.get("adapt_vqe")
    if not isinstance(parent, Mapping) or not isinstance(candidate, Mapping):
        raise ValueError("missing ADAPT evidence")
    parent_sector = parent.get("generator_pool_sector_contract")
    candidate_sector = candidate.get("generator_pool_sector_contract")
    if not isinstance(parent_sector, Mapping) or not isinstance(
        candidate_sector, Mapping
    ):
        raise ValueError("missing generator-pool sector contract")
    parent_filter = parent_sector.get("filter")
    candidate_filter = candidate_sector.get("filter")
    if not isinstance(parent_filter, Mapping) or not isinstance(
        candidate_filter, Mapping
    ):
        raise ValueError("missing generator-pool filter receipt")

    exact_checks = {
        "pool_size": (
            int(parent.get("pool_size", -1)),
            int(candidate.get("pool_size", -2)),
        ),
        "generator_count": (
            int(parent_sector.get("generator_count", -1)),
            int(candidate_sector.get("generator_count", -2)),
        ),
        "filter_applied": (
            parent_filter.get("applied"),
            candidate_filter.get("applied"),
        ),
        "removed_count": (
            int(parent_filter.get("removed_count", -1)),
            int(candidate_filter.get("removed_count", -2)),
        ),
        "removed_labels": (
            list(parent_filter.get("removed_labels", ())),
            list(candidate_filter.get("removed_labels", ())),
        ),
        "shared_pauli_pool_ordered_label_hash": (
            parent.get("shared_pauli_pool_ordered_label_hash"),
            candidate.get("shared_pauli_pool_ordered_label_hash"),
        ),
        "shared_pauli_pool_ordered_pool_hash": (
            parent.get("shared_pauli_pool_ordered_pool_hash"),
            candidate.get("shared_pauli_pool_ordered_pool_hash"),
        ),
    }
    mismatches = {
        key: {"parent": values[0], "candidate": values[1]}
        for key, values in exact_checks.items()
        if values[0] != values[1]
    }
    if mismatches:
        raise ValueError(f"parent/candidate pool-surface drift: {mismatches}")
    if exact_checks["pool_size"][0] != EXPECTED_FILTERED_POOL_SIZE:
        raise ValueError("locked Paper-I weak--weak pool size drift")
    return {
        "schema": "paper_i_query_neutral_prune_matched_pool_surface_v1",
        "status": "pass",
        "filtered_pool_size": EXPECTED_FILTERED_POOL_SIZE,
        "removed_parent_count": int(parent_filter["removed_count"]),
        "shared_pauli_pool_ordered_label_hash": exact_checks[
            "shared_pauli_pool_ordered_label_hash"
        ][0],
        "shared_pauli_pool_ordered_pool_hash": exact_checks[
            "shared_pauli_pool_ordered_pool_hash"
        ][0],
    }


def apply_scope_repair(source_root: Path) -> dict[str, Any]:
    """Apply only the pre-route-invariant prune-key scope repair."""

    adapt_path = source_root / "pipelines/static_adapt/adapt_pipeline.py"
    adapt_text = adapt_path.read_text(encoding="utf-8")
    if adapt_text.count(SCOPE_REPAIR_OLD) != 1:
        raise ValueError("frozen prune-key scope-repair anchor drift")
    repaired = adapt_text.replace(SCOPE_REPAIR_OLD, SCOPE_REPAIR_NEW, 1)
    if repaired.count(RUNTIME_RECEIPT_REPAIR_OLD) != 1:
        raise ValueError("frozen prune runtime-receipt repair anchor drift")
    repaired = repaired.replace(
        RUNTIME_RECEIPT_REPAIR_OLD,
        RUNTIME_RECEIPT_REPAIR_NEW,
        1,
    )
    adapt_path.write_text(repaired, encoding="utf-8")
    return {
        "schema": "paper_i_query_neutral_prune_scope_repair_overlay_v1",
        "status": "pass",
        "parent_source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "adapt_pipeline_sha256": sha256(adapt_path),
        "scientific_setting_delta": 0,
        "algorithmic_query_delta": 0,
        "repair": (
            "normalize already-defined prune policy keys before route invariant "
            "validation and include the two locked prune cardinality fields in "
            "the runtime receipt"
        ),
    }


def validate_source_root(source_root: Path) -> dict[str, Any]:
    adapt_path = source_root / "pipelines/static_adapt/adapt_pipeline.py"
    adapt_text = adapt_path.read_text(encoding="utf-8")
    present = [
        marker for marker in FORBIDDEN_SOURCE_MARKERS if marker in adapt_text
    ]
    if present:
        raise ValueError(f"unrelated live-tree source drift present: {present}")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(source_root)
    command = (
        "from pipelines.static_adapt.sr_snake_route_profile import "
        "canonical_sr_snake_contract_sha256 as h;"
        f"assert h({PARENT_ROUTE_ALIAS!r})=={PARENT_DIGEST!r};"
        f"assert h({CANDIDATE_ROUTE_ALIAS!r})=={CANDIDATE_DIGEST!r}"
    )
    subprocess.run(
        [sys.executable, "-c", command],
        cwd=source_root,
        env=env,
        check=True,
    )
    return {
        "schema": "paper_i_query_neutral_prune_frozen_source_gate_v1",
        "status": "pass",
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "parent_route": PARENT_ROUTE,
        "parent_route_contract_sha256": PARENT_DIGEST,
        "candidate_route": CANDIDATE_ROUTE,
        "candidate_route_contract_sha256": CANDIDATE_DIGEST,
        "adapt_pipeline_sha256": sha256(adapt_path),
        "forbidden_live_tree_markers_absent": list(FORBIDDEN_SOURCE_MARKERS),
    }


def extract_source(output_dir: Path) -> Path:
    if sha256(SOURCE_ARCHIVE) != SOURCE_ARCHIVE_SHA256:
        raise ValueError("source-locked archive SHA-256 drift")
    source_root = output_dir / "source"
    if source_root.exists():
        raise FileExistsError(f"source extraction target exists: {source_root}")
    source_root.mkdir(parents=True)
    with tarfile.open(SOURCE_ARCHIVE, "r:gz") as archive:
        archive.extractall(source_root, filter="data")
    return source_root


def _load_candidate_validator():
    bundle_path = str(ANCHOR)
    if bundle_path not in sys.path:
        sys.path.insert(0, bundle_path)
    path = ROOT / "chtc/phase3_optuna/query_neutral_prune_evidence_validation.py"
    spec = importlib.util.spec_from_file_location(
        "matched_query_neutral_prune_evidence_validation",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load query-neutral evidence validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def recover_existing_candidate_validation(
    *,
    output_dir: Path,
    max_rounds: int,
) -> dict[str, Any]:
    """Validate already completed science without changing its query ledger."""

    validator = _load_candidate_validator()
    candidate_result = load(output_dir / "result.json")
    evidence = validator.validate_query_neutral_prune_evidence(
        result=candidate_result,
        current=load(output_dir / "current.json"),
        ledger_sidecar=load(output_dir / "ledger.json"),
        safety_cap=max_rounds,
    )
    pool_gate = validate_matched_pool_surface(
        parent_result=load(MATCHED_PARENT_RESULT),
        candidate_result=candidate_result,
    )
    receipt = {
        "schema": "paper_i_query_neutral_prune_reporting_only_recovery_v1",
        "status": "pass",
        "route": "candidate",
        "pool_surface_gate": pool_gate,
        "candidate_evidence": evidence,
        "validator_sha256": sha256(
            ROOT
            / "chtc/phase3_optuna/query_neutral_prune_evidence_validation.py"
        ),
        "result_sha256": sha256(output_dir / "result.json"),
        "ledger_sha256": sha256(output_dir / "ledger.json"),
        "current_sha256": sha256(output_dir / "current.json"),
        "science_rerun": False,
        "algorithmic_query_delta": 0,
    }
    dump(output_dir / "validation_receipt.json", receipt)
    return receipt


def run_route(
    *,
    output_dir: Path,
    max_rounds: int,
    full: bool,
    route: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"immutable output exists: {output_dir}")
    if route not in {"parent", "candidate"}:
        raise ValueError(f"unsupported route: {route}")
    if route == "candidate" and (
        not MATCHED_PARENT_RESULT.is_file() or not MATCHED_PARENT_LEDGER.is_file()
    ):
        raise FileNotFoundError("matched source-locked parent evidence is missing")
    output_dir.mkdir(parents=True)
    source_root = extract_source(output_dir)
    scope_repair = apply_scope_repair(source_root)
    source_gate = validate_source_root(source_root)
    parent_job = load(ANCHOR / "jobs/weak_weak.json")
    builder = build_parent_argv if route == "parent" else build_candidate_argv
    argv = builder(
        parent_job["command"]["argv"],
        output_dir=output_dir,
        max_rounds=max_rounds,
    )
    dump(
        output_dir / "run_manifest.json",
        {
            "schema": "paper_i_query_neutral_prune_matched_local_run_v1",
            "run_class": "diagnostic" if not full else "candidate",
            "route": route,
            "source_gate": source_gate,
            "scope_repair": scope_repair,
            "matched_parent_result": (
                str(MATCHED_PARENT_RESULT) if route == "candidate" else None
            ),
            "matched_parent_result_sha256": (
                sha256(MATCHED_PARENT_RESULT)
                if route == "candidate"
                else None
            ),
            "command_argv": argv,
            "max_rounds": max_rounds,
            "first_target_hit_stop": True,
        },
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source_root)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    with (output_dir / "run.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            argv,
            cwd=source_root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"candidate process exited {completed.returncode}")

    candidate_result = load(output_dir / "result.json")
    pool_gate = validate_matched_pool_surface(
        parent_result=(
            load(MATCHED_PARENT_RESULT)
            if route == "candidate"
            else load(
                Path("/tmp/query_neutral_parent_anchor_local_v1/result.json")
            )
        ),
        candidate_result=candidate_result,
    )
    evidence: dict[str, Any] | None = None
    if full:
        validator = _load_candidate_validator()
        validate = (
            validator.validate_parent_first_hit_evidence
            if route == "parent"
            else validator.validate_query_neutral_prune_evidence
        )
        evidence = validate(
            result=candidate_result,
            current=load(output_dir / "current.json"),
            ledger_sidecar=load(output_dir / "ledger.json"),
            safety_cap=max_rounds,
        )
    receipt = {
        "schema": "paper_i_query_neutral_prune_matched_local_receipt_v1",
        "status": "pass",
        "route": route,
        "source_gate": source_gate,
        "scope_repair": scope_repair,
        "pool_surface_gate": pool_gate,
        "candidate_evidence": evidence,
        "result_sha256": sha256(output_dir / "result.json"),
        "ledger_sha256": sha256(output_dir / "ledger.json"),
        "current_sha256": sha256(output_dir / "current.json"),
        "science_rerun": True,
        "route_change_only": True,
    }
    dump(output_dir / "validation_receipt.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-rounds", type=int, default=50)
    parser.add_argument(
        "--route",
        choices=("parent", "candidate"),
        default="candidate",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Require first-target-hit and complete prune/ledger validation.",
    )
    parser.add_argument(
        "--recover-validation-only",
        action="store_true",
        help="Validate an existing candidate output without rerunning science.",
    )
    args = parser.parse_args()
    if args.max_rounds <= 0 or args.max_rounds > 50:
        raise ValueError("max rounds must be in [1, 50]")
    if args.recover_validation_only:
        if args.route != "candidate" or not args.full:
            raise ValueError(
                "reporting-only recovery requires --route candidate --full"
            )
        receipt = recover_existing_candidate_validation(
            output_dir=args.output_dir.resolve(),
            max_rounds=args.max_rounds,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    receipt = run_route(
        output_dir=args.output_dir.resolve(),
        max_rounds=args.max_rounds,
        full=bool(args.full),
        route=str(args.route),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
